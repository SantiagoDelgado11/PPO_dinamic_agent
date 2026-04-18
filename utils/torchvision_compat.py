from __future__ import annotations

import torch


def patch_torchvision_fake_registration() -> None:
    """Avoid torchvision import crashes when optional ops are missing."""

    register_fake = getattr(torch.library, "register_fake", None)
    if register_fake is None or getattr(register_fake, "_stsiva_safe_patch", False):
        return

    def safe_register_fake(op_name, *args, **kwargs):
        decorator = register_fake(op_name, *args, **kwargs)

        def wrapper(fn):
            try:
                return decorator(fn)
            except RuntimeError as exc:
                if "does not exist" in str(exc):
                    return fn
                raise

        return wrapper

    safe_register_fake._stsiva_safe_patch = True
    torch.library.register_fake = safe_register_fake
