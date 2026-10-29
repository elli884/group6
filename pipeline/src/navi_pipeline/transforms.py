"""Input-stage boundaries. Only identity behavior is implemented in the clean pipeline.

Future attacks/defences receive model-visible input, never references or run storage.
Implementations must add validated threat-model configuration, saved artifacts, and
budget checks before being wired into the runner. These are not a security sandbox.
"""
import os
from PIL import Image, ImageDraw, ImageFont
from .records import ModelInput

# Toggle and parameters for the typographic injection attack (7.1 / 7.10).
# Controlled by an environment variable so clean vs. attacked runs don't require
# touching ExperimentConfig's strict schema — see note below on why.
_ATTACK_ENABLED = os.environ.get("NAVI_ATTACK_TEXT_INJECTION", "0") == "1"
_ATTACK_LABEL = os.environ.get("NAVI_ATTACK_TARGET_LABEL", "Beach")
_FONT_PATH = "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf" # Change font type


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


def transform_input(item: ModelInput) -> ModelInput:
    if not _ATTACK_ENABLED:
        return item
    attacked_image = _inject_label_text(item.image, _ATTACK_LABEL)
    return ModelInput(item.image_id, attacked_image, item.prompt)


def defend_input(item: ModelInput) -> ModelInput:
    return item
