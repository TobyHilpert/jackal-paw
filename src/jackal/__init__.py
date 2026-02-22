from jax import config as _jax_config

_jax_config.update("jax_enable_x64", True)

from .version import __version__

__all__ = ["__version__"]
