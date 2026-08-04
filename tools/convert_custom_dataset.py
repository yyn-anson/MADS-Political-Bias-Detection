"""Zero-install entry point for the custom dataset converter."""

import importlib
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
main = importlib.import_module("mads.custom_dataset").main


if __name__ == "__main__":
    raise SystemExit(main())
