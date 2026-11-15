import numpy as np
from PIL import Image

# Can use function below to test whether the attack will survive the resize without having to run the full model.
# Get target size from the model.json of a previous run (processor_config -> image_processor). Look for size/crop_size.
def survives_resize(image: Image.Image, attacked: Image.Image, target_size: tuple[int, int]) -> float:
    """Returns mean absolute pixel difference after both images go through the
    same resize the model's processor will apply. If this is near zero, the
    perturbation has already been destroyed before the model sees it."""
    a = np.array(image.resize(target_size, Image.BICUBIC), dtype=np.float32)
    b = np.array(attacked.resize(target_size, Image.BICUBIC), dtype=np.float32)
    return float(np.abs(a - b).mean())