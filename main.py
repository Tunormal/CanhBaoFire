"""Entry point. Khong co tham so -> mo UI. Co --input -> chay headless (CLI)."""
import sys

from fire_safety.cli import build_parser, run_cli


def main() -> int:
    if len(sys.argv) > 1:
        args = build_parser().parse_args()
        return run_cli(args)
    from fire_safety.ui.app import launch_ui
    launch_ui()
    return 0


if __name__ == "__main__":
    sys.exit(main())
