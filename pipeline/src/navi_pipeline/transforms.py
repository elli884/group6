"""Input-stage boundaries. Only identity behavior is implemented in the clean pipeline.

Future attacks/defences receive model-visible input, never references or run storage.
Implementations must add validated threat-model configuration, saved artifacts, and
budget checks before being wired into the runner. These are not a security sandbox.
"""
import os
import hashlib
import json
import math
from dataclasses import asdict, dataclass
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from .records import ModelInput

_FONT_PATH = "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf" # Change font type

# -------------------------------------------------------------------------------------------

# Toggle and parameters for the typographic injection attack (7.1 / 7.10).
# Controlled by an environment variable so clean vs. attacked runs don't require
# touching ExperimentConfig's strict schema — see note below on why.
_TEXT_ENABLED = os.environ.get("NAVI_ATTACK_TEXT_INJECTION", "0") == "1"
_TEXT_LABEL = os.environ.get("NAVI_ATTACK_TARGET_LABEL", "Beach")


def _inject_label_text(image: Image.Image, text: str) -> Image.Image:
    """Paste a visible, high-contrast target label onto the image (typographic attack)."""
    img = image.copy()
    draw = ImageDraw.Draw(img)
    font_size = max(18, img.width // 8) # Change font size
    try:
        font = ImageFont.truetype(_FONT_PATH, font_size)
    except OSError:
        font = ImageFont.load_default()  # degrades gracefully, but log this if it happens
    draw.text(
        (img.width * 0.05, img.height * 0.05),
        text,
        fill=(255, 0, 0),
        font=font,
        stroke_width=max(1, font_size // 12),
        stroke_fill=(0, 0, 0),
    ) # Change colour through of text through fill and outline through stroke_fill
    return img

# ---------------------------------------------------------------------------------------------------------------
# Extension of text injection attack inspired from stenographic attacks, adds low-amplitude visible overlay instead of obvious text to the image
# Easy to look over at a glance
# Cannot actually implement single nudge of pixel colour as images get resized within the pipeline so a slight difference will not be visible to the model.

_STEGO_MODE = os.environ.get("NAVI_ATTACK_STEGO_MODE")  # "faint" | "lsb" | None
_STEGO_TEXT = os.environ.get("NAVI_ATTACK_STEGO_TEXT", "Forest")
_STEGO_ALPHA = float(os.environ.get("NAVI_ATTACK_STEGO_ALPHA", "0.08"))
_STEGO_BIT = int(os.environ.get("NAVI_ATTACK_STEGO_BIT", "0"))

if _STEGO_MODE not in (None, "faint", "lsb"):
    raise ValueError('NAVI_ATTACK_STEGO_MODE must be "faint" or "lsb"')


def _inject_faint_text(image: Image.Image, text: str, alpha: float) -> Image.Image:
    """Visible-in-principle, hard-to-notice text: real glyphs, low opacity.
        Alpha is the blend strength (0-1), 0.05-0.15 is "hard to notice at a glance".
        Far below that, it starts to approach true LSB territory and is unlikely to
        survive the model's resize at all.
        """
    img = image.convert("RGB")
    overlay = Image.new("RGB", img.size, (0, 0, 0))
    draw = ImageDraw.Draw(overlay)
    font_size = max(18, img.width // 8)
    try:
        font = ImageFont.truetype(_FONT_PATH, font_size)
    except OSError:
        font = ImageFont.load_default()
    draw.text((img.width * 0.05, img.height * 0.05), text, fill=(255, 0, 0), font=font)
    text_mask = overlay.convert("L").point(lambda p: int(p > 0) * int(255 * alpha))
    # Test return line, if image looks wrong replace it with Image.blend(img, overlay_with_text_only, alpha). 
    # Although this will wash the whole image toward black.
    return Image.composite(overlay, img, text_mask)


def _inject_lsb_text(image: Image.Image, text: str, bit_plane: int) -> Image.Image:
    """True bit-plane encoding: ASCII text packed into the chosen bit of the red
        channel. Likely to be destroyed by resizing but use this to test in-built defense.
        """
    arr = np.array(image.convert("RGB"), dtype=np.uint8)
    bits = "".join(f"{ord(c):08b}" for c in text) + "0" * 8
    flat_r = arr[:, :, 0].flatten()
    if len(bits) > flat_r.size:
        raise ValueError("Message too long for this image's capacity")
    mask = ~(1 << bit_plane) & 0xFF
    for i, bit in enumerate(bits):
        flat_r[i] = (flat_r[i] & mask) | (int(bit) << bit_plane)
    arr[:, :, 0] = flat_r.reshape(arr[:, :, 0].shape)
    return Image.fromarray(arr, mode="RGB")


def _apply_stego(image: Image.Image) -> Image.Image:
    if _STEGO_MODE == "faint":
        return _inject_faint_text(image, _STEGO_TEXT, _STEGO_ALPHA)
    return _inject_lsb_text(image, _STEGO_TEXT, _STEGO_BIT)


# -------------------------------------------------------------------------------------------------------------------

# Second type of visual injection attack. Creates an arbitrary shape of coloured pixels on top
# of the image to disrupt the VLM. 
_SHAPES = {"square", "rectangle", "circle", "ellipse"}

@dataclass(frozen=True)
class PatchSettings:
    shape: str = "square"              # Could set any of the shapes in _SHAPES
    size: float = 0.2                  # The width of the patch, calculated as a fraction of the image's shortest side.
    aspect: float = 1.0                # Height/width, used by the rectangle and ellipse
    x: float = 0.5                     # Fraction of the image width to indicate center of the patch (0 = left)
    y: float = 0.5                     # Fraction of the image height to indicate center of the patch (0 = top)
    color: str | tuple = "random"      # "random" colour to be applied or an [r,g,b] list with values 0-255
    cell: int = 1                      # random mode: size of the noise block in pixels (1 = per-pixel)
    seed: int = 0                      # random mode: noise seed
    per_image: bool = False            # random mode: False = same noise on every image

    def __post_init__(self):
        if self.shape not in _SHAPES:
            raise ValueError(f"Shape must be one of {sorted(_SHAPES)}")
        if not 0 < self.size <= 1:
            raise ValueError("Size must be in the range (0,1]")
        if self.aspect <= 0:
            raise ValueError("Aspect must be a positive number")
        if not (0 <= self.x <= 1 and 0 <= self.y <= 1):
            raise ValueError("The values of x and y must be in the range [0,1]")
        if self.cell < 1:
            raise ValueError("Cell must be at least 1")
        if self.color != "random" and not (isinstance(self.color, tuple) and len(self.color) == 3 and all(isinstance(c, int) and 0 <= c <= 255 for c in self.color)):
            raise ValueError("Chosen color must be random or [r,g,b] integers in 0-255")

def _load_patch_settings() -> PatchSettings | None:
    raw = os.environ.get("NAVI_ATTACK_PATCH")
    if not raw:
        return None
    params = json.loads(raw)
    if params.get("color", "random") != "random":
        params["color"] = tuple(params["color"])
    return PatchSettings(**params) # Unknown keys will raise TypeError

_PATCH = _load_patch_settings()

def _patch_seed(settings: PatchSettings, image_id: str) -> int:
    if not settings.per_image:
        return settings.seed
    digest = hashlib.sha256(f"{settings.seed}:{image_id}".encode()).digest()
    return int.from_bytes(digest[:4], "big")

def _apply_patch(image: Image.Image, s: PatchSettings, image_id: str) -> Image.Image:
    w, h = image.size
    pw = max(1, min(w, round(s.size * min(w,h))))
    ph = pw if s.shape in {"square", "circle"} else max(1, min(h, round(pw * s.aspect)))
    left = min(max(round(s.x * w - pw/2),0), w - pw) # Used to keep the entire patch within the image.
    top = min(max(round(s.y * h - ph/2),0), h - ph)

    if s.color == "random":
        rng = np.random.default_rng(_patch_seed(s, image_id))
        blocks = rng.integers(0, 256, size=(math.ceil(ph/s.cell), math.ceil(pw/s.cell), 3), dtype=np.uint8)
        pixels = blocks.repeat(s.cell, axis=0).repeat(s.cell, axis=1)[:ph,:pw]
    else:
        pixels = np.empty((ph,pw,3), dtype=np.uint8)
        pixels[:] = s.color

    mask = Image.new("L", (pw,ph), 0)
    drawer = ImageDraw.Draw(mask)
    if s.shape in {"square", "rectangle"}:
        drawer.rectangle([0,0,pw-1,ph-1], fill = 255)
    else:
        drawer.ellipse([0,0,pw-1,ph-1], fill=255)
    out = image.copy()
    out.paste(Image.fromarray(pixels), (left, top), mask)
    return out

# -----------------------------------------------------------------------------------------------------------------------------------------------------------------------------
# Extension of patch attack based on efficient AP-PA algorithm. 
# Not direct copy as implemented to an object detector using bounding-boxes unlike Gemma 3 which labels the whole image.
# Keeps ideas of iterative gradient optimization, total variation loss.
# requires white-box gradient access and true label to optimize towards, so needs to run an offline script this code will just load and paste from.
# Only implement if time permitting, maybe in conjunction with mission-log poisoning.

_TRAINED_PATCH_PATH = os.environ.get("NAVI_ATTACK_TRAINED_PATCH")
_TRAINED_PATCH_X = float(os.environ.get("NAVI_ATTACK_TRAINED_PATCH_X", "0.5"))
_TRAINED_PATCH_Y = float(os.environ.get("NAVI_ATTACK_TRAINED_PATCH_Y", "0.5"))
_TRAINED_PATCH = Image.open(_TRAINED_PATCH_PATH).convert("RGB") if _TRAINED_PATCH_PATH else None


def _apply_trained_patch(image: Image.Image) -> Image.Image:
    img = image.copy()
    pw, ph = _TRAINED_PATCH.size
    x = min(max(round(_TRAINED_PATCH_X * img.width - pw / 2), 0), img.width - pw)
    y = min(max(round(_TRAINED_PATCH_Y * img.height - ph / 2), 0), img.height - ph)
    img.paste(_TRAINED_PATCH, (x, y))
    return img


# ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

def transform_input(item: ModelInput) -> ModelInput:
    image = item.image
    if _TEXT_ENABLED:
        image = _inject_label_text(image, _TEXT_LABEL)
    if _PATCH is not None:
        image = _apply_patch(image, _PATCH, item.image_id)
    if _STEGO_MODE is not None:
        image = _apply_stego(image)
    if _TRAINED_PATCH is not None:
        image = _apply_trained_patch(image)
    if image is item.image:
        return item
    return ModelInput(item.image_id, image, item.prompt)


def defend_input(item: ModelInput) -> ModelInput:
    return item

# Following method allows run record to indicate what the active transformation elements are.
def transform_description():
    active = {}
    if _TEXT_ENABLED:
        active["text_injection"] = {"label": _TEXT_LABEL}
    if _PATCH is not None:
        active["random_patch"] = asdict(_PATCH)
    if _STEGO_MODE is not None:
        active["stego"] = {"mode": _STEGO_MODE, "text": _STEGO_TEXT,
                           "alpha": _STEGO_ALPHA if _STEGO_MODE == "faint" else None,
                           "bit_plane": _STEGO_BIT if _STEGO_MODE == "lsb" else None}
    if _TRAINED_PATCH is not None:
        active["trained_patch"] = {"path": _TRAINED_PATCH_PATH,
                                   "x": _TRAINED_PATCH_X, "y": _TRAINED_PATCH_Y}
    return active or "identity"
