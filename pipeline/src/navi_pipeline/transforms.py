"""Input-stage boundaries. Only identity behavior is implemented in the clean pipeline.

Future attacks/defences receive model-visible input, never references or run storage.
Implementations must add validated threat-model configuration, saved artifacts, and
budget checks before being wired into the runner. These are not a security sandbox.
"""

from .records import ModelInput


def transform_input(item: ModelInput) -> ModelInput:
    return item


def defend_input(item: ModelInput) -> ModelInput:
    return item
