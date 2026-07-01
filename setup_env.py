#!/usr/bin/env python3
"""Create a local virtual environment and install pyorica for benchmark/dev work.

Cross-platform (Windows/macOS/Linux) since it only uses the Python stdlib.

Usage:
    python setup_env.py          # venv at .venv, installs pyorica[full]
    python setup_env.py --dev    # also installs the dev extra (pytest, ruff)
"""
import subprocess
import sys
import venv
from pathlib import Path

VENV_DIR = Path(".venv")
MIN_PYTHON = (3, 11)


def venv_python(venv_dir: Path) -> Path:
    if sys.platform == "win32":
        return venv_dir / "Scripts" / "python.exe"
    return venv_dir / "bin" / "python"


def main() -> None:
    if sys.version_info < MIN_PYTHON:
        sys.exit(
            f"Error: pyorica requires Python >={MIN_PYTHON[0]}.{MIN_PYTHON[1]}, "
            f"found {sys.version_info.major}.{sys.version_info.minor}.\n"
            f"Re-run this script with a newer interpreter, e.g.: python3.12 setup_env.py"
        )

    print(f"Using interpreter: {sys.version.splitlines()[0]} ({sys.executable})")

    if not VENV_DIR.exists():
        print(f"Creating virtual environment in {VENV_DIR} ...")
        venv.create(VENV_DIR, with_pip=True)
    else:
        print(f"Reusing existing virtual environment in {VENV_DIR}")

    python = str(venv_python(VENV_DIR))

    def run(*args: str) -> None:
        subprocess.run([python, "-m", *args], check=True)

    run("pip", "install", "--upgrade", "pip")

    print("Installing pyorica[full] ...")
    run("pip", "install", "-e", ".[full]")

    if "--dev" in sys.argv[1:]:
        print("Installing dev extra ...")
        run("pip", "install", "-e", ".[dev]")

    activate_hint = (
        r".venv\Scripts\activate" if sys.platform == "win32" else "source .venv/bin/activate"
    )
    print()
    print("Done. Activate the environment with:")
    print(f"  {activate_hint}")


if __name__ == "__main__":
    main()
