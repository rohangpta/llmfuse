#!/usr/bin/env python3
"""Utility to print the filesystem XML rendering for a given directory."""
import argparse
from llmfuse.fs_state import FSState


def main() -> None:
    parser = argparse.ArgumentParser(description="Render a directory tree as XML")
    parser.add_argument("root", help="Path to the root directory to serialize")
    args = parser.parse_args()

    state = FSState(args.root)
    state.sync_from_fs()
    print(state.to_xml_string())


if __name__ == "__main__":
    main()
