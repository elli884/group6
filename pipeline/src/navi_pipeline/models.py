"""Inference backends; importing this module never imports the inference stack."""

import gc
import hashlib
import json
import random
import re
import time
from copy import deepcopy
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from typing import Any, Protocol

from PIL import Image

from .config import ModelConfig, project_path
from .records import ModelRequest, ModelResponse


class Backend(Protocol):
    def generate(self, request: ModelRequest) -> ModelResponse: ...

    def metadata(self) -> dict[str, Any]: ...

    def close(self) -> None: ...


class FakeBackend:
    """Synthetic plumbing fixture, not an image classifier or quality baseline."""

    def __init__(self, label: str = "Forest") -> None:
        if not label.strip() or label != label.strip() or "\n" in label or "\r" in label:
            raise ValueError("Fake label must be a nonempty, stripped, single-line string")
        self.label = label
        self._closed = False

    def generate(self, request: ModelRequest) -> ModelResponse:
        if self._closed:
            raise RuntimeError("Backend is closed")
        text = (
            f"Label: {self.label}\nDescription: Synthetic image response for pipeline testing."
            if request.image is not None
            else "Synthetic dialogue response for pipeline testing."
        )
        return ModelResponse(text, {**self.metadata(), "seed": request.seed})

    def metadata(self) -> dict[str, Any]:
        return {"backend": "fake", "synthetic": True, "label": self.label}

    def close(self) -> None:
        self._closed = True


class TransformersBackend:
    def __init__(self, config: ModelConfig, root: Path) -> None:
        import numpy as np
        import torch
        import transformers

        self.config = config
        self._torch = torch
        self._numpy = np
        self._model = None
        self._processor = None
        self._device = torch.device(config.device)
        self._dtype = getattr(torch, config.dtype)
        if config.device == "cuda":
            if not torch.cuda.is_available():
                raise ValueError("CUDA was requested but is unavailable")
            self._device = torch.device("cuda", torch.cuda.current_device())
            if config.dtype == "bfloat16" and not torch.cuda.is_bf16_supported():
                raise ValueError("The selected CUDA device does not support bfloat16")
        if config.device == "mps" and not torch.backends.mps.is_available():
            raise ValueError("MPS was requested but is unavailable")

        cache = project_path(root, ".cache/huggingface")
        load_options = {
            "cache_dir": str(cache),
            "local_files_only": config.local_files_only,
            "trust_remote_code": False,
        }
        model_config = transformers.AutoConfig.from_pretrained(
            config.model_id, revision=config.revision, **load_options
        )
        commit = getattr(model_config, "_commit_hash", None)
        if not isinstance(commit, str) or not re.fullmatch(r"[0-9a-fA-F]{40}", commit):
            raise ValueError("Model config must resolve to an immutable Hugging Face commit")
        if re.fullmatch(r"[0-9a-fA-F]{40}", config.revision):
            if commit.lower() != config.revision.lower():
                raise ValueError("Resolved model revision does not match the requested commit")
        if getattr(model_config, "is_encoder_decoder", False):
            raise ValueError("Encoder-decoder models are not supported")
        load_options["revision"] = commit

        if model_config.model_type == "gemma3":
            model_class = transformers.Gemma3ForConditionalGeneration
        else:
            model_class = getattr(transformers, "AutoModelForImageTextToText", None)
            if model_class is None:
                raise ValueError("This Transformers version lacks AutoModelForImageTextToText")

        processor = transformers.AutoProcessor.from_pretrained(config.model_id, **load_options)
        tokenizer = getattr(processor, "tokenizer", None)
        observed_commits = {"config": commit}
        for name, component in (("processor", processor), ("tokenizer", tokenizer)):
            observed = getattr(component, "_commit_hash", None)
            if observed is None:
                observed = getattr(component, "init_kwargs", {}).get("_commit_hash")
            if observed is not None and observed != commit:
                raise ValueError(f"{name} resolved to a different model commit")
            observed_commits[name] = observed

        quantization = {"format": "none", "gguf": False}
        model_options = {"torch_dtype": self._dtype, "config": model_config,
                         "use_safetensors": True}
        if config.quantization != "none":
            quant_options = (
                {
                    "load_in_4bit": True,
                    "bnb_4bit_quant_type": "nf4",
                    "bnb_4bit_compute_dtype": self._dtype,
                    "bnb_4bit_use_double_quant": False,
                }
                if config.quantization == "4bit"
                else {"load_in_8bit": True}
            )
            quant_config = transformers.BitsAndBytesConfig(**quant_options)
            model_options["quantization_config"] = quant_config
            model_options["device_map"] = {"": str(self._device)}
            quantization = {
                "format": "bitsandbytes",
                "mode": config.quantization,
                "gguf": False,
                "config": quant_config.to_dict(),
            }
        if getattr(model_config, "quantization_config", None):
            raise ValueError("Use unquantized source weights; prequantized models are unsupported")

        adapter_metadata = None
        adapter = None
        if config.adapter_path is not None:
            adapter = project_path(root, config.adapter_path)
            if not adapter.is_dir() or not (adapter / "adapter_config.json").is_file():
                raise ValueError("adapter_path must be a local PEFT adapter directory")
            if not (adapter / "adapter_model.safetensors").is_file():
                raise ValueError("The local adapter must provide adapter_model.safetensors")
            hashes = {}
            for path in sorted(adapter.rglob("*")):
                if not path.resolve().is_relative_to(adapter):
                    raise ValueError("Adapter contains an escaping symlink")
                if path.is_file():
                    with path.open("rb") as source:
                        hashes[path.relative_to(adapter).as_posix()] = hashlib.file_digest(
                            source, "sha256"
                        ).hexdigest()
            adapter_metadata = {"path": config.adapter_path, "sha256_files": hashes}

        model = model_class.from_pretrained(config.model_id, **load_options, **model_options)
        if getattr(model.config, "_commit_hash", None) != commit:
            raise ValueError("Loaded model config does not match the pinned commit")
        if getattr(model.config, "is_encoder_decoder", False):
            raise ValueError("Encoder-decoder models are not supported")
        if config.quantization == "none":
            model = model.to(device=self._device, dtype=self._dtype)
        if adapter is not None:
            from peft import PeftModel

            model = PeftModel.from_pretrained(
                model, str(adapter), is_trainable=False,
                local_files_only=True, autocast_adapter_dtype=False,
            )
        model.eval()

        versions = {}
        for package in ("torch", "transformers", "numpy", "pillow", "accelerate",
                        "bitsandbytes", "peft", "huggingface-hub", "tokenizers"):
            try:
                versions[package] = version(package)
            except PackageNotFoundError:
                versions[package] = None
        processor_config = processor.to_dict()
        image_processor = getattr(processor, "image_processor", None)
        if image_processor is not None:
            processor_config["image_processor"] = image_processor.to_dict()
        chat_template = getattr(processor, "chat_template", None)
        if chat_template is None:
            chat_template = getattr(tokenizer, "chat_template", None)
        if chat_template is None:
            raise ValueError("The processor/tokenizer must provide a chat template")
        self._metadata = {
            "backend": "transformers", "synthetic": False,
            "model_id": config.model_id, "requested_revision": config.revision,
            "resolved_revision": commit, "asset_revision": commit,
            "observed_commits": observed_commits,
            "processor_class": type(processor).__name__,
            "model_class": type(model).__name__,
            "processor_config": processor_config,
            "processor_config_sha256": hashlib.sha256(
                json.dumps(processor_config, sort_keys=True).encode("utf-8")
            ).hexdigest(),
            "chat_template_sha256": hashlib.sha256(
                (chat_template if isinstance(chat_template, str)
                 else json.dumps(chat_template, sort_keys=True)).encode("utf-8")
            ).hexdigest(),
            "versions": versions, "device": str(self._device), "dtype": config.dtype,
            "device_name": (torch.cuda.get_device_name(self._device)
                            if config.device == "cuda" else config.device),
            "quantization": quantization, "adapter": adapter_metadata,
            "cache_dir": str(cache), "local_files_only": config.local_files_only,
            "generation": {
                "max_input_tokens": config.max_input_tokens,
                "max_new_tokens": config.max_new_tokens, "do_sample": config.do_sample,
                "temperature": config.temperature if config.do_sample else None,
            },
            "model_generation_defaults": model.generation_config.to_dict(),
        }
        self._model = model
        self._processor = processor

    def metadata(self) -> dict[str, Any]:
        return deepcopy(self._metadata)

    def generate(self, request: ModelRequest) -> ModelResponse:
        if self._model is None or self._processor is None:
            raise RuntimeError("Backend is closed")
        if not 0 <= request.seed <= 2**32 - 1:
            raise ValueError("Seed must be in the uint32 range")
        torch = self._torch
        random.seed(request.seed)
        self._numpy.random.seed(request.seed)
        torch.manual_seed(request.seed)
        if self.config.device == "cuda":
            torch.cuda.manual_seed_all(request.seed)
        if self.config.device == "mps":
            torch.mps.manual_seed(request.seed)

        content = []
        image = None
        if request.image is not None:
            # Rebuild pixels so filenames, EXIF, and image.info cannot reach the processor.
            image = Image.new("RGB", request.image.size)
            image.paste(request.image.convert("RGB"))
            content.append({"type": "image", "image": image})
        content.append({"type": "text", "text": request.prompt})
        rendered_prompt = self._processor.apply_chat_template(
            [{"role": "user", "content": content}],
            tokenize=False, add_generation_prompt=True,
        )
        processor_options = {"text": rendered_prompt, "return_tensors": "pt", "truncation": False,
                             "add_special_tokens": False}
        if image is not None:
            processor_options["images"] = image
        inputs = self._processor(**processor_options)
        input_tokens = inputs["input_ids"].shape[-1]
        if input_tokens > self.config.max_input_tokens:
            raise ValueError(
                f"Input has {input_tokens} tokens, exceeding max_input_tokens="
                f"{self.config.max_input_tokens}; input was not truncated"
            )
        text_config = getattr(self._model.config, "text_config", self._model.config)
        context_limit = getattr(text_config, "max_position_embeddings", None)
        if context_limit is not None and input_tokens + self.config.max_new_tokens > context_limit:
            raise ValueError("Input plus generation budget exceeds the model context window")
        inputs = {
            key: value.to(device=self._device, dtype=self._dtype)
            if value.is_floating_point() else value.to(self._device)
            for key, value in inputs.items()
        }
        options = {"max_new_tokens": self.config.max_new_tokens,
                   "do_sample": self.config.do_sample, "num_beams": 1,
                   "num_return_sequences": 1, "return_dict_in_generate": False}
        if self.config.do_sample:
            options["temperature"] = self.config.temperature
        if self.config.device == "cuda":
            torch.cuda.synchronize(self._device)
            torch.cuda.reset_peak_memory_stats(self._device)
        elif self.config.device == "mps":
            torch.mps.synchronize()
        started = time.perf_counter()
        with torch.inference_mode():
            output = self._model.generate(**inputs, **options)
        if self.config.device == "cuda":
            torch.cuda.synchronize(self._device)
        elif self.config.device == "mps":
            torch.mps.synchronize()
        elapsed = time.perf_counter() - started
        continuation = output[0, input_tokens:]
        text = self._processor.decode(continuation, skip_special_tokens=True)
        metadata = {
            "seed": request.seed, "input_tokens": input_tokens,
            "rendered_prompt": rendered_prompt,
            "generated_tokens": continuation.shape[-1], "elapsed_seconds": elapsed,
            "timing_scope": "generate_only", "device": str(self._device),
            "cuda_max_memory_allocated_bytes": None,
            "cuda_max_memory_reserved_bytes": None,
        }
        if self.config.device == "cuda":
            metadata["cuda_max_memory_allocated_bytes"] = torch.cuda.max_memory_allocated(
                self._device
            )
            metadata["cuda_max_memory_reserved_bytes"] = torch.cuda.max_memory_reserved(
                self._device
            )
        return ModelResponse(text, metadata)

    def close(self) -> None:
        self._model = None
        self._processor = None
        gc.collect()
        if self.config.device == "cuda":
            with self._torch.cuda.device(self._device):
                self._torch.cuda.empty_cache()
        elif self.config.device == "mps":
            self._torch.mps.empty_cache()


def create_backend(config: ModelConfig, root: Path) -> Backend:
    if config.backend == "fake":
        return FakeBackend(config.fake_label)
    if config.backend == "transformers":
        return TransformersBackend(config, root)
    raise ValueError(f"Unsupported backend: {config.backend}")
