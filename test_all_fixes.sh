#!/bin/bash
set -e

echo "==========================================================="
echo "COMPREHENSIVE DATA QUALITY TESTS"
echo "==========================================================="

echo ""
echo "Running in Docker (requires FUSE)..."
echo ""

# Test 1: Generate 10 examples for quick validation
echo "Test 1: Generating 10 examples..."
docker-compose run --rm datagen python3 -m train.generate_data -n 10 --output_dir /app/data/test

# Test 2: Inspect the generated data
echo ""
echo "Test 2: Inspecting generated data..."
docker-compose run --rm datagen python3 <<'PYTHON'
import json
import sys

# Read the generated file
import glob
files = sorted(glob.glob('/app/data/test/fuse_*.jsonl'))
if not files:
    print("❌ No generated files found!")
    sys.exit(1)

latest = files[-1]
print(f"Reading: {latest}")

examples = []
with open(latest) as f:
    for line in f:
        examples.append(json.loads(line))

print(f"\n✅ Loaded {len(examples)} examples")

# Analyze examples
print("\n" + "="*60)
print("ANALYSIS")
print("="*60)

# Check for write operations with content
write_ops = [ex for ex in examples if '<W>' in ex['prompt'] and 'write(' in ex['prompt']]
print(f"\nWrite operations: {len(write_ops)}")
if write_ops:
    ex = write_ops[0]
    print("Sample write operation:")
    print(f"  Prompt (first 200 chars): {ex['prompt'][:200]}")

# Check for rename operations
rename_ops = [ex for ex in examples if 'rename(' in ex['prompt']]
print(f"\nRename operations: {len(rename_ops)}")
if rename_ops:
    ex = rename_ops[0]
    print("Sample rename operation:")
    print(f"  Prompt (first 200 chars): {ex['prompt'][:200]}")
    # Check if source file exists in initial state
    if '---' in ex['prompt']:
        parts = ex['prompt'].split('---')
        operation_part = parts[0]
        state_part = parts[1] if len(parts) > 1 else ''
        print(f"  State has {len(state_part)} chars")

# Check for read operations
read_ops = [ex for ex in examples if '<R>' in ex['prompt'] and 'read(' in ex['prompt']]
print(f"\nRead operations: {len(read_ops)}")

# Check for empty filesystem operations
empty_fs_ops = [ex for ex in examples if '/ dir 755' in ex['prompt'] and len(ex['prompt'].split('\n')) < 5]
print(f"\nEmpty filesystem operations: {len(empty_fs_ops)}")

print("\n" + "="*60)
print("SAMPLE EXAMPLES")
print("="*60)

for i, ex in enumerate(examples[:3]):
    print(f"\n--- Example {i+1} ---")
    print("Prompt:")
    print(ex['prompt'][:300])
    print("\nCompletion:")
    print(ex['completion'][:300])

print("\n🎉 DATA QUALITY TEST COMPLETE")
PYTHON

echo ""
echo "==========================================================="
echo "ALL TESTS COMPLETE"
echo "Review the output above for any issues"
echo "==========================================================="
