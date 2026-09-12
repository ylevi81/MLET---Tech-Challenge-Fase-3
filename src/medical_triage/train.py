"""Training entry point and stable imports for orchestration tools."""

from __future__ import annotations

import sys
from collections.abc import Sequence

from medical_triage.cli import main as cli_main
from medical_triage.model import train_and_save, train_model

__all__ = ["train_and_save", "train_model"]


def main(argv: Sequence[str] | None = None) -> int:
    """Run the CLI train command."""

    arguments = list(sys.argv[1:] if argv is None else argv)
    return cli_main(["train", *arguments])


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
