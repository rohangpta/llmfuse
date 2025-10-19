# Data Quality Fixes Summary

## Date: 2025-10-19

## Problems Identified

### 1. **Non-Deterministic Content** (Lines 389-392, 509, 556, 561, 572, 634, 644, 965, 1134)
- Same filepath generated different content across training examples
- Model couldn't learn to "copy" content from context
- Explained 52% exact match vs 78% accuracy gap

### 2. **Missing Files in Initial State**
- Write operations: Content not included in operation string
- Rename/Symlink: Compound commands created files inline (`touch X && mv X Y`)
- Truncate: Operated on files that didn't exist yet
- Read: Attempted to read files not in initial state

### 3. **Dead Code**
- `generate_batch()` function (lines 1097-1249) not used
- Caused FUSE mount deadlocks, explicitly avoided

## Fixes Implemented

### Fix #1: Deterministic Content Generation
- **Added**: `import hashlib` (line 12)
- **Renamed**: `PileContentSampler` → `DeterministicContentGenerator`
- **Changed**: Removed cache-based sampling, now generates on-the-fly
- **Added**: `_get_seeded_random(filepath)` method using SHA256 hash
- **Updated**: All content generation methods to use seeded RNG
- **Result**: Same filepath → same content (deterministic)

**Files Changed**:
- Lines 105-261: Complete class rewrite
- Lines 263-273: Updated global instance management
- Lines 389-392: Renamed `generate_random_content` → `generate_deterministic_content`

### Fix #2: Write Operations Include Content
- **Changed**: Lines 552-562 to ensure content is deterministic
- **Philosophy**: Model needs to see what's being written

### Fix #3: Rename/Symlink Only Use Existing Files
- **Changed**: Lines 513-525 (rename) - removed compound commands
- **Changed**: Lines 527-536 (symlink) - removed compound commands  
- **Changed**: Lines 590-610 (targeted operations) - removed inline file creation
- **Fallback**: If no files exist, generate different operation (mkdir/touch)

### Fix #4: Truncate Requires Existing Files
- **Changed**: Lines 501-511 - only truncate existing files
- **Fallback**: Generate mkdir if no files exist

### Fix #5: Read Only Existing Files
- **Changed**: Lines 564-573 - only read existing files
- **Fallback**: Generate touch if no files exist

### Fix #6: Remove Dead Code
- **Deleted**: Lines 1097-1249 (`generate_batch` function)
- **Reason**: Not used, caused FUSE deadlocks

## Verification

### Test #1: Content Determinism ✅
- Same filepath generates identical content across multiple calls
- Different filepaths generate different content
- Verified with simple hash-based test

### Test #2: Operation Completeness ✅
Generated 10 examples and verified:
- All rename/symlink operations reference files in initial state
- All read/write/chmod/chown/truncate operations reference existing files
- No operations create files inline with compound commands
- Empty FS operations handled correctly

**Results**:
```
Operation Distribution (10 examples):
✅ chmod: 1
✅ mkdir: 2
✅ read: 1
✅ rename: 3
✅ symlink: 1
✅ truncate: 1
✅ write: 1

Verification: 100% pass rate
- All 8 file-referencing operations verified
- All files exist in initial state
```

## Expected Impact on Model Performance

### Before Fixes:
- Accuracy: 78%
- Exact Match: 52%
- Gap: 26 percentage points

### After Fixes:
- **Deterministic content**: Model can now learn to copy content from context
- **Complete information**: All operations have necessary context
- **No impossible operations**: No operations reference non-existent files

### Predicted Results:
- Accuracy: Should remain ~78-80% (structural understanding already good)
- **Exact Match: Should improve to 75-95%** (can now copy content correctly)
- Gap: Should reduce to <10 percentage points

## Next Steps

1. ✅ Fixes implemented and verified
2. 🔄 Generate 5K training samples with fixes
3. 🔄 Retrain model on Modal (qwen3-4b, 3 epochs, distributed)
4. 🔄 Re-evaluate with same test set
5. 🔄 Compare exact match improvement (target: >75%)

## Files Modified

- `train/generate_data.py`: ~200 lines changed
  - Class refactor: DeterministicContentGenerator
  - Operation generation fixes
  - Dead code removal

## Philosophy Change

### Before:
```
(Random State, Operation) → (Random State)
```
- Content randomized
- Operations might reference non-existent files
- Model forced to hallucinate

### After:
```
(Deterministic State, Complete Operation) → (Deterministic State)
```
- Content deterministic (filepath-seeded)
- Operations only reference existing files
- Model can learn to copy/transform correctly

## Testing Artifacts

- `test_determinism_simple.py`: Basic determinism test
- `test_content_determinism.py`: Full content generation test
- `test_operation_completeness.py`: Operation validation test
- `test_all_fixes.sh`: Comprehensive Docker-based test
- `data/test/fuse_10_1760851987.jsonl`: Verified test data
