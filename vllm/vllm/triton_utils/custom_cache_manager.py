import os

from vllm.logger import init_logger

logger = init_logger(__name__)

# The per-process cache-dir bug this module works around was fixed in
# triton-lang/triton/pull/4295 and is present in triton >= 3.1.0.
# On newer triton the old symbols no longer exist; import them only when
# available and skip the custom manager otherwise.
try:
    from triton.runtime.cache import (FileCacheManager, default_cache_dir,
                                      default_dump_dir, default_override_dir)
    _LEGACY_TRITON_CACHE_API = True
except ImportError:
    _LEGACY_TRITON_CACHE_API = False


def maybe_set_triton_cache_manager() -> None:
    """Set environment variable to tell Triton to use a custom cache manager.

    Only applied when running against triton < 3.1.0 where the per-process
    cache isolation bug exists. Newer triton handles this natively.
    """
    if not _LEGACY_TRITON_CACHE_API:
        return
    cache_manger = os.environ.get("TRITON_CACHE_MANAGER", None)
    if cache_manger is None:
        manager = "vllm.triton_utils.custom_cache_manager:CustomCacheManager"
        logger.info("Setting Triton cache manager to: %s", manager)
        os.environ["TRITON_CACHE_MANAGER"] = manager


if _LEGACY_TRITON_CACHE_API:
    class CustomCacheManager(FileCacheManager):  # type: ignore[misc]
        """Per-process triton cache dir — only needed on triton < 3.1.0.

        The underlying bug was fixed in triton-lang/triton/pull/4295.
        """

        def __init__(self, key, override=False, dump=False):
            self.key = key
            self.lock_path = None
            if dump:
                self.cache_dir = default_dump_dir()
                self.cache_dir = os.path.join(self.cache_dir, self.key)
                self.lock_path = os.path.join(self.cache_dir, "lock")
                os.makedirs(self.cache_dir, exist_ok=True)
            elif override:
                self.cache_dir = default_override_dir()
                self.cache_dir = os.path.join(self.cache_dir, self.key)
            else:
                self.cache_dir = os.getenv("TRITON_CACHE_DIR",
                                           "").strip() or default_cache_dir()
                if self.cache_dir:
                    self.cache_dir = f"{self.cache_dir}_{os.getpid()}"
                    self.cache_dir = os.path.join(self.cache_dir, self.key)
                    self.lock_path = os.path.join(self.cache_dir, "lock")
                    os.makedirs(self.cache_dir, exist_ok=True)
                else:
                    raise RuntimeError("Could not create or locate cache dir")
