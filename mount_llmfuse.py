#!/usr/bin/env python3
"""
Standalone script to mount the LLM-driven FUSE filesystem.
"""

import os
import sys

# Add project root to Python path so we can import src as a package
project_root = os.path.dirname(__file__)
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from src import mount_fuse

if __name__ == '__main__':
    if len(sys.argv) != 2:
        print('Usage: %s <mountpoint>' % sys.argv[0])
        sys.exit(1)
    
    mountpoint = sys.argv[1]
    if not os.path.exists(mountpoint):
        os.makedirs(mountpoint)
    
    print(f"Mounting LLM-driven FUSE filesystem at {mountpoint}")
    print("Press Ctrl+C to unmount")
    
    mount_fuse(mountpoint) 