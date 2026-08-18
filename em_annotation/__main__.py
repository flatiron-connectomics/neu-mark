"""``python -m em_annotation`` — the same entry point as the ``em-annot`` script."""

from .cli import main

if __name__ == "__main__":
    raise SystemExit(main())
