"""Offline AP-PA-inspired patch training. Run outside the pipeline, with full
white-box access to a second, separately-loaded copy of the model.
"""
import torch
import torch.nn.functional as F
import transformers
from PIL import Image
import glob, random

MODEL_ID = "google/gemma-3-4b-it"
CACHE_DIR = ".cache/huggingface"
TARGET_LABEL = "Forest"
PATCH_SIZE = 96          # pixels, before placement
EPOCHS = 30
LR = 0.05
TV_WEIGHT = 1e-3

processor = transformers.AutoProcessor.from_pretrained(MODEL_ID, local_files_only=True, cache_dir=CACHE_DIR)
quant_config = transformers.BitsAndBytesConfig(
    load_in_4bit=True, bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.float32, bnb_4bit_use_double_quant=False,
)
model = transformers.Gemma3ForConditionalGeneration.from_pretrained(
    MODEL_ID, quantization_config=quant_config, dtype=torch.float32,
    local_files_only=True, cache_dir=CACHE_DIR, device_map={"": "cuda"},
)
model.eval()
for p in model.parameters():
    p.requires_grad_(False)   # freeze everything except the patch itself

patch = torch.rand(3, PATCH_SIZE, PATCH_SIZE, device="cuda", requires_grad=True)
optimizer = torch.optim.Adam([patch], lr=LR)

def total_variation(p):
    return (p[:, 1:, :] - p[:, :-1, :]).abs().sum() + (p[:, :, 1:] - p[:, :, :-1]).abs().sum()

def paste_patch(image_tensor, patch, x_frac=0.5, y_frac=0.5):
    # image_tensor: (3, H, W) float in [0, 255]. Pastes patch centered at the given fraction.
    _, h, w = image_tensor.shape
    x = int(x_frac * w - PATCH_SIZE / 2)
    y = int(y_frac * h - PATCH_SIZE / 2)
    x, y = max(0, min(w - PATCH_SIZE, x)), max(0, min(h - PATCH_SIZE, y))
    out = image_tensor.clone()
    out[:, y:y + PATCH_SIZE, x:x + PATCH_SIZE] = patch.clamp(0, 255)
    return out

# Train across a diverse, mixed-class sample -- NOT only airport images, so the
# patch generalizes rather than overfitting to one scene type (see note below).
train_paths = random.sample(glob.glob("data/aid/*/*.jpg"), 200)
target_text = f"Label: {TARGET_LABEL}\nDescription: placeholder"
target_ids = processor.tokenizer(target_text, return_tensors="pt").input_ids.to("cuda")

for epoch in range(EPOCHS):
    random.shuffle(train_paths)
    total_loss = 0.0
    for path in train_paths:
        image = Image.open(path).convert("RGB")
        arr = torch.from_numpy(
            __import__("numpy").array(image)
        ).permute(2, 0, 1).float().to("cuda")

        attacked = paste_patch(arr, patch)
        attacked_pil = Image.fromarray(
            attacked.clamp(0, 255).byte().permute(1, 2, 0).cpu().numpy()
        )

        prompt = processor.apply_chat_template(
            [{"role": "user", "content": [
                {"type": "image", "image": attacked_pil},
                {"type": "text", "text": "Classify this image."},
            ]}],
            tokenize=False, add_generation_prompt=True,
        )
        inputs = processor(text=prompt, images=attacked_pil, return_tensors="pt").to("cuda")
        # NOTE: this re-encodes the image through PIL, which detaches the graph
        # from `patch`. A working version needs the processor's pixel normalization
        # applied directly to the `attacked` tensor so gradients flow back to `patch` --
        # replicate the processor's resize+normalize as torch ops rather than round-
        # tripping through PIL. This is the main engineering gap to close before this
        # runs for real; treat the snippet above as the training *loop structure*,
        # not a drop-in working script.

        outputs = model(**inputs, labels=target_ids)
        loss = outputs.loss + TV_WEIGHT * total_variation(patch)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    print(f"epoch {epoch}: {total_loss / len(train_paths):.4f}")

final_patch = patch.clamp(0, 255).byte().permute(1, 2, 0).cpu().numpy()
Image.fromarray(final_patch).save("trained_patch.png")