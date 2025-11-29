"""
Benchmark Bulk Lookup Strategies: Iterative vs Vectorized Merge
"""

import pandas as pd
import numpy as np
import time
from security_lookup import FastSecurityLookup, SecurityDataGenerator

def benchmark_bulk_strategies():
    print("="*80)
    print("BENCHMARKING BULK LOOKUP STRATEGIES (2M Records)")
    print("="*80)
    
    # 1. Load Lookup Data
    print("\n1. Loading Lookup Data (10M records)...")
    lookup = FastSecurityLookup('data/security_lookup.parquet')
    lookup.load()
    
    # 2. Generate Client Data (2M records)
    print("\n2. Generating Client Data (2M records)...")
    # We'll sample from the existing data to ensure hits, and add some randoms for misses
    
    # Take 1.5M real records
    real_samples = lookup.df.sample(1_500_000).reset_index()
    
    # Generate 0.5M fake records
    fake_generator = SecurityDataGenerator(num_records=500_000)
    fake_df = fake_generator.generate()
    # Modify values to ensure they don't match (mostly)
    fake_df['security_id_value'] = fake_df['security_id_value'] + "_FAKE"
    
    # Combine
    client_df = pd.concat([
        real_samples[['security_id_type', 'security_id_value']], 
        fake_df[['security_id_type', 'security_id_value']]
    ])
    
    # Shuffle
    client_df = client_df.sample(frac=1).reset_index(drop=True)
    
    print(f"   Client Data: {len(client_df):,} records")
    
    # Strategy A: Iterative Lookup (Current)
    print("\n3. Strategy A: Iterative Lookup (Current Code)...")
    records = list(zip(client_df['security_id_type'], client_df['security_id_value']))
    
    start = time.time()
    # We'll just do 100k for iterative to estimate, otherwise it takes too long for a quick test
    subset_size = 100_000
    _ = lookup.lookup_batch(records[:subset_size])
    elapsed = time.time() - start
    
    estimated_total = elapsed * (len(client_df) / subset_size)
    print(f"   Processed {subset_size:,} records in {elapsed:.2f}s")
    print(f"   Estimated time for 2M records: {estimated_total:.2f}s ({estimated_total/60:.1f} min)")
    print(f"   Speed: {subset_size/elapsed:,.0f} lookups/sec")
    
    # Strategy B: Vectorized Merge (Pandas Join)
    print("\n4. Strategy B: Vectorized Merge (Pandas Join)...")
    
    start = time.time()
    
    # Ensure client df has index set for join if needed, or just merge on columns
    # Merging on columns is standard
    # lookup.df is indexed by [type, value]
    
    # We join client_df with lookup.df
    # client_df columns: security_id_type, security_id_value
    # lookup.df index: security_id_type, security_id_value
    
    result = client_df.join(
        lookup.df, 
        on=['security_id_type', 'security_id_value'], 
        how='left'
    )
    
    elapsed = time.time() - start
    
    print(f"   Processed {len(client_df):,} records in {elapsed:.2f}s")
    print(f"   Speed: {len(client_df)/elapsed:,.0f} lookups/sec")
    
    # Comparison
    speedup = estimated_total / elapsed
    print("\n" + "="*80)
    print(f"RESULTS: Vectorized Merge is {speedup:.1f}x FASTER")
    print("="*80)
    
    return result

if __name__ == "__main__":
    benchmark_bulk_strategies()
