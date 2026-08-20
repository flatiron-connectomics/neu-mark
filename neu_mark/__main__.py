"""``python -m neu_mark`` — the same entry point as the ``neu-mark`` script."""

from .cli import main

if __name__ == "__main__":
    raise SystemExit(main())
