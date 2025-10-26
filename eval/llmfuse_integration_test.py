#!/usr/bin/env python3
"""
Integration test for the LLM-backed FUSE filesystem.

Mounts the filesystem at a provided mountpoint, performs a sequence of basic
filesystem operations via shell commands (mkdir, touch, echo >>, ls, stat, cat,
rename, rm, rmdir), and asserts observable results. Exits with non-zero status
on failure.

Usage:
  python eval/llmfuse_integration_test.py /tmp/fuse_mount
"""
import os
import sys
import subprocess
import json
from pathlib import Path


def sh(cmd: str, cwd: str) -> tuple[int, str]:
    try:
        cp = subprocess.run(cmd, shell=True, cwd=cwd, capture_output=True, text=True, timeout=20)
        out = (cp.stdout + cp.stderr).strip()
        return cp.returncode, out
    except subprocess.TimeoutExpired:
        return 124, "timeout"


def require(cond: bool, msg: str):
    if not cond:
        print(f"FAIL: {msg}")
        sys.exit(1)


def main():
    if len(sys.argv) != 2:
        print("usage: llmfuse_integration_test.py <mountpoint>")
        sys.exit(2)
    mount_dir = sys.argv[1]
    Path(mount_dir).mkdir(parents=True, exist_ok=True)

    # Basic ops
    rc, out = sh("mkdir -p proj/docs", mount_dir)
    require(rc == 0, f"mkdir failed: {out}")

    rc, out = sh("touch proj/README.md", mount_dir)
    require(rc == 0, f"touch failed: {out}")

    rc, out = sh("echo 'hello world' > proj/README.md", mount_dir)
    require(rc == 0, f"write failed: {out}")

    rc, out = sh("ls -1 proj", mount_dir)
    require(rc == 0 and "README.md" in out.splitlines(), f"ls missing README.md: {out}")

    rc, out = sh("stat -c '%s' proj/README.md || stat -f '%z' proj/README.md", mount_dir)
    require(rc == 0 and out.strip().isdigit(), f"stat size not numeric: {out}")

    rc, out = sh("cat proj/README.md", mount_dir)
    require(rc == 0 and "hello world" in out, f"cat content mismatch: {out}")

    rc, out = sh("mv proj/README.md proj/README2.md", mount_dir)
    require(rc == 0, f"rename failed: {out}")

    rc, out = sh("test -f proj/README2.md; echo $?", mount_dir)
    require(out.strip() == "0", f"renamed file not present")

    rc, out = sh("rm proj/README2.md", mount_dir)
    require(rc == 0, f"rm failed: {out}")

    rc, out = sh("rmdir proj/docs", mount_dir)
    require(rc == 0, f"rmdir failed: {out}")

    # Final listing root
    rc, out = sh("ls -la .", mount_dir)
    require(rc == 0, f"final ls failed: {out}")

    print(json.dumps({"status": "ok"}))


if __name__ == "__main__":
    main()
