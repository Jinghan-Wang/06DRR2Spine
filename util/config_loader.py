"""Load train/test args from YAML config files."""

import os
import sys
from pathlib import Path

try:
    import yaml
except ImportError as exc:
    raise ImportError("PyYAML is required to load config/*.yaml files.") from exc


class ConfigLoader:
    _mode_filenames = {
        "train": "trainConfig.yaml",
        "test": "test.yaml",
    }

    @classmethod
    def apply(cls, mode: str):
        """Load one mode-specific YAML file and apply env/sys.argv overrides."""
        path = cls._resolve_path(mode)
        block = cls._load(path)
        if not isinstance(block, dict):
            raise TypeError(f"Config file must contain a mapping: {path}")

        cls._apply_env(block.get("env", {}))
        cls._apply_argv(mode, block.get("args", {}))

    @classmethod
    def _resolve_path(cls, mode: str) -> Path:
        filename = cls._mode_filenames.get(mode)
        if filename is None:
            raise KeyError(f"Unsupported config mode: {mode}")

        configured = os.environ.get("CONFIG_PATH")
        if configured:
            base_path = Path(configured)
            path = base_path / filename if base_path.is_dir() else base_path
        else:
            if hasattr(sys, "_MEIPASS"):
                base = Path(sys._MEIPASS)
            else:
                base = Path(__file__).parent.parent
            path = base / "config" / filename

        if not path.is_file():
            raise FileNotFoundError(
                f"Config file not found: {path}\n"
                f"Set CONFIG_PATH to a YAML file or config directory containing {filename}."
            )
        return path

    @staticmethod
    def _load(path: Path) -> dict:
        with open(path, "r", encoding="utf-8") as handle:
            return yaml.safe_load(handle) or {}

    @staticmethod
    def _apply_env(env: dict):
        for key, value in env.items():
            os.environ[key] = str(value)

    @staticmethod
    def _apply_argv(mode: str, args: dict):
        """Build sys.argv from the YAML args section."""
        argv = [f"{mode}.py"]
        for flag, value in args.items():
            if isinstance(value, bool):
                if value:
                    argv.append(flag)
            else:
                argv.extend([flag, str(value)])
        sys.argv = argv
        print(f">>> [ConfigLoader] sys.argv = {sys.argv}")
