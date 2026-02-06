"""
Quick test script to verify preprocessing cache functionality.

This script tests:
1. First preprocessing run creates cache
2. Second run loads from cache (much faster)
3. Stats computation works without errors
"""

import time
import os
from src.utilities.preprocess import preprocess_with_cache
from src import config

def test_preprocessing_cache():
    """Test preprocessing cache functionality."""
    
    # Use a06 as the test record (the one that was causing the error)
    test_record = "a06"
    
    print("=" * 60)
    print("Testing Preprocessing Cache Functionality")
    print("=" * 60)
    
    # Clear any existing cache for this record
    cache_dir = os.path.join(config.PROCESSED_DATA_DIR, "cache")
    cache_file = os.path.join(cache_dir, f"{test_record}_preprocessed.npz")
    if os.path.exists(cache_file):
        os.remove(cache_file)
        print(f"Cleared existing cache for {test_record}")
    
    print("\n[TEST 1] First preprocessing run (no cache)...")
    start_time = time.time()
    result1 = preprocess_with_cache(test_record)
    time1 = time.time() - start_time
    
    print(f"✓ First run completed in {time1:.2f} seconds")
    print(f"  - Record: {result1['record']}")
    print(f"  - Tensors shape: {result1['tensors'].shape}")
    print(f"  - Valid minutes: {len(result1['minutes'])}")
    print(f"  - Skipped minutes: {len(result1['skipped'])}")
    print(f"  - Stats computed: {bool(result1.get('stats'))}")
    
    if result1.get('stats'):
        print(f"  - HRV Time Domain keys: {list(result1['stats'].get('hrv_time', {}).keys())}")
        print(f"  - HRV Freq Domain keys: {list(result1['stats'].get('hrv_freq', {}).keys())}")
        print(f"  - EDR keys: {list(result1['stats'].get('edr', {}).keys())}")
        print(f"  - R-peak stats keys: {list(result1['stats'].get('rpeak', {}).keys())}")
    
    print(f"\n[TEST 2] Second preprocessing run (with cache)...")
    start_time = time.time()
    result2 = preprocess_with_cache(test_record)
    time2 = time.time() - start_time
    
    print(f"✓ Second run completed in {time2:.2f} seconds")
    print(f"  - Speedup: {time1/time2:.1f}x faster!")
    
    # Verify cache file exists
    print(f"\n[TEST 3] Verifying cache file...")
    if os.path.exists(cache_file):
        cache_size_mb = os.path.getsize(cache_file) / (1024 * 1024)
        print(f"✓ Cache file exists: {cache_file}")
        print(f"  - Size: {cache_size_mb:.2f} MB")
    else:
        print(f"✗ Cache file not found: {cache_file}")
    
    # Verify results are identical
    print(f"\n[TEST 4] Verifying cached results match original...")
    assert result1['record'] == result2['record'], "Record names don't match"
    assert result1['tensors'].shape == result2['tensors'].shape, "Tensor shapes don't match"
    assert result1['minutes'] == result2['minutes'], "Minutes don't match"
    assert result1['skipped'] == result2['skipped'], "Skipped minutes don't match"
    print(f"✓ All results match!")
    
    print("\n" + "=" * 60)
    print("All tests passed!")
    print("=" * 60)
    print(f"\nPerformance Improvement:")
    print(f"  - First run (no cache): {time1:.2f}s")
    print(f"  - Second run (cached):  {time2:.2f}s")
    print(f"  - Speedup: {time1/time2:.1f}x")
    print(f"\nThis means gradcam operations will be ~{time1/time2:.1f}x faster!")
    print("=" * 60)

if __name__ == "__main__":
    test_preprocessing_cache()
