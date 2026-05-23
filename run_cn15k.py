#!/usr/bin/env python3
"""Run the 10-seed CARE evaluation on CN15k."""

from scripts.run_10seed import main as run


if __name__ == "__main__":
    import sys

    sys.argv.extend(["--datasets", "cn15k"])
    run()
