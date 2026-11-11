"""Disable THP for this process and its descendants, then exec an unchanged command.

Usage: python launch_without_thp.py [--receipt /absolute/new.json] -- COMMAND [ARG ...]
       python launch_without_thp.py --check-only

Only the current process's prctl flag and NumPy environment setting are changed.
No machine-wide settings, project sources, framework imports, or RNG state are used.
Receipts attest to settings before exec, not successful command completion.
"""

from __future__ import annotations

import argparse
import ctypes
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import sys
from typing import Sequence


PR_SET_THP_DISABLE = 41
PR_GET_THP_DISABLE = 42


def disable_thp() -> int:
    """Set and verify the Linux per-process THP flag; fail closed on any error."""
    if sys.platform != "linux":
        raise RuntimeError("per-process THP disabling requires Linux")
    libc = ctypes.CDLL(None, use_errno=True)
    prctl = libc.prctl
    prctl.argtypes = [ctypes.c_int, *([ctypes.c_ulong] * 4)]
    prctl.restype = ctypes.c_int
    result = prctl(PR_SET_THP_DISABLE, 1, 0, 0, 0)
    if result == -1:
        raise OSError(ctypes.get_errno(), "PR_SET_THP_DISABLE failed")
    if result != 0:
        raise RuntimeError(f"PR_SET_THP_DISABLE returned unexpected value {result}")
    result = prctl(PR_GET_THP_DISABLE, 0, 0, 0, 0)
    if result == -1:
        raise OSError(ctypes.get_errno(), "PR_GET_THP_DISABLE failed")
    if result != 1:
        raise RuntimeError(f"THP disable verification failed: expected 1, got {result}")
    return result


def write_receipt(path: Path, receipt: dict[str, object]) -> None:
    """Create a private receipt without replacing any existing file or symlink."""
    if not path.is_absolute():
        raise ValueError("receipt path must be absolute")
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    with os.fdopen(descriptor, "w", encoding="utf-8", newline="\n") as handle:
        json.dump(receipt, handle, sort_keys=True, indent=2)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check-only", action="store_true", help="verify settings, print JSON, and exit")
    parser.add_argument("--receipt", type=Path, help="absolute path to a new, create-only JSON receipt")
    parser.add_argument("command", nargs=argparse.REMAINDER)
    args = parser.parse_args(argv)
    if args.receipt is not None and not args.receipt.is_absolute():
        parser.error("--receipt must be an absolute path")
    if args.check_only:
        if args.command:
            parser.error("--check-only does not accept a command")
        command: list[str] = []
    else:
        if len(args.command) < 2 or args.command[0] != "--" or not args.command[1]:
            parser.error("provide an executable and arguments after --")
        command = args.command[1:]

    try:
        os.environ["NUMPY_MADVISE_HUGEPAGE"] = "0"
        flag = disable_thp()
        receipt: dict[str, object] = {
            "schema": "launch_without_thp_v1",
            "status": "runtime_configured_before_exec" if command else "check_passed",
            "timestamp_utc": datetime.now(timezone.utc).isoformat(),
            "pid": os.getpid(),
            "command": command,
            "runtime_settings": {
                "pr_get_thp_disable": flag,
                "NUMPY_MADVISE_HUGEPAGE": os.environ["NUMPY_MADVISE_HUGEPAGE"],
                "scope": "current_process_and_descendants",
            },
        }
        if args.receipt is not None:
            write_receipt(args.receipt, receipt)
        print(json.dumps(receipt, sort_keys=True), flush=True)
        if command:
            os.execvpe(command[0], command, os.environ.copy())
        return 0
    except (OSError, RuntimeError, AttributeError) as error:
        print(f"launch_without_thp: refusing to launch: {error}", file=sys.stderr, flush=True)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
