"""
High-Performance Security Identifier Lookup System
Generates 10M records and provides sub-millisecond lookups
"""

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import numpy as np
import time
from pathlib import Path

class SecurityDataGenerator:
    """Generate realistic security identifier data."""
    
    def __init__(self, num_records=10_000_000):
        self.num_records = num_records
        
    def generate(self):
        """Generate 10M security records with various identifier types."""
        print(f"Generating {self.num_records:,} security records...")
        start = time.time()
        
        # Security identifier types (realistic)
        id_types = ['CUSIP', 'ISIN', 'SEDOL', 'TICKER', 'FIGI', 'RIC']
        type_weights = [0.30, 0.25, 0.15, 0.20, 0.05, 0.05]  # Realistic distribution
        
        # Generate data in chunks to manage memory
        chunk_size = 1_000_000
        chunks = []
        
        for i in range(0, self.num_records, chunk_size):
            current_chunk_size = min(chunk_size, self.num_records - i)
            
            # Generate security types
            sec_types = np.random.choice(
                id_types, 
                size=current_chunk_size, 
                p=type_weights
            )
            
            # Generate security values (realistic formats)
            sec_values = []
            for sec_type in sec_types:
                if sec_type == 'CUSIP':
                    # 9-character alphanumeric
                    value = ''.join(np.random.choice(list('0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ'), 9))
                elif sec_type == 'ISIN':
                    # 12-character: 2-letter country + 10 alphanumeric
                    country = ''.join(np.random.choice(list('ABCDEFGHIJKLMNOPQRSTUVWXYZ'), 2))
                    rest = ''.join(np.random.choice(list('0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ'), 10))
                    value = country + rest
                elif sec_type == 'SEDOL':
                    # 7-character alphanumeric
                    value = ''.join(np.random.choice(list('0123456789BCDFGHJKLMNPQRSTVWXYZ'), 7))
                elif sec_type == 'TICKER':
                    # 1-5 character ticker
                    length = np.random.randint(1, 6)
                    value = ''.join(np.random.choice(list('ABCDEFGHIJKLMNOPQRSTUVWXYZ'), length))
                elif sec_type == 'FIGI':
                    # 12-character FIGI
                    value = 'BBG' + ''.join(np.random.choice(list('0123456789ABCDEFGHJKLMNPQRSTVWXYZ'), 9))
                else:  # RIC
                    value = ''.join(np.random.choice(list('ABCDEFGHIJKLMNOPQRSTUVWXYZ'), 4)) + '.N'
                
                sec_values.append(value)
            
            # Generate VINs (sequential for this chunk)
            vins = np.arange(i, i + current_chunk_size, dtype=np.int32)
            
            chunk_df = pd.DataFrame({
                'security_id_type': sec_types,
                'security_id_value': sec_values,
                'vin': vins
            })
            
            chunks.append(chunk_df)
            
            if (i + current_chunk_size) % 1_000_000 == 0:
                print(f"  Generated {i + current_chunk_size:,} records...")
        
        df = pd.concat(chunks, ignore_index=True)
        
        elapsed = time.time() - start
        print(f"✅ Generated {len(df):,} records in {elapsed:.2f}s")
        print(f"   Memory usage: {df.memory_usage(deep=True).sum() / 1024**2:.1f} MB")
        
        return df


class ParquetExporter:
    """Export data to optimized Parquet format."""
    
    def __init__(self, output_path='data/security_lookup.parquet'):
        self.output_path = Path(output_path)
        self.output_path.parent.mkdir(parents=True, exist_ok=True)
    
    def export(self, df):
        """Export DataFrame to Parquet with optimal compression."""
        print(f"\nExporting to Parquet: {self.output_path}")
        start = time.time()
        
        # Convert to PyArrow Table for better control
        table = pa.Table.from_pandas(df)
        
        # Write with optimal settings
        pq.write_table(
            table,
            self.output_path,
            compression='snappy',  # Fast compression/decompression
            use_dictionary=True,  # Efficient for repeated values
            write_statistics=True,  # Enable column statistics
            row_group_size=100000  # Optimize for query performance
        )
        
        elapsed = time.time() - start
        file_size = self.output_path.stat().st_size / 1024**2
        
        print(f"✅ Exported in {elapsed:.2f}s")
        print(f"   File size: {file_size:.1f} MB")
        print(f"   Compression ratio: {df.memory_usage(deep=True).sum() / 1024**2 / file_size:.1f}x")
        
        return self.output_path


class FastSecurityLookup:
    """High-performance security lookup engine."""
    
    def __init__(self, parquet_file='data/security_lookup.parquet'):
        self.parquet_file = Path(parquet_file)
        self.df = None
        self.index = None
        
    def load(self):
        """Load Parquet file into memory with indexing."""
        print(f"\nLoading lookup data from {self.parquet_file}...")
        start = time.time()
        
        # Read Parquet file
        self.df = pd.read_parquet(self.parquet_file)
        
        # Create multi-index for ultra-fast lookups
        self.df.set_index(['security_id_type', 'security_id_value'], inplace=True)
        self.df.sort_index(inplace=True)
        
        elapsed = time.time() - start
        print(f"✅ Loaded {len(self.df):,} records in {elapsed:.2f}s")
        print(f"   Load speed: {len(self.df) / elapsed:,.0f} records/sec")
        
    def lookup(self, security_type, security_value):
        """
        Fast lookup by security identifier.
        
        Args:
            security_type: e.g., 'CUSIP', 'ISIN', 'TICKER'
            security_value: The identifier value
            
        Returns:
            VIN (int) or None if not found
        """
        try:
            result = self.df.loc[(security_type, security_value), 'vin']
            if isinstance(result, pd.Series):
                return result.iloc[0]
            return result
        except KeyError:
            return None
    
    def lookup_batch(self, lookups):
        """
        Batch lookup for multiple securities.
        
        Args:
            lookups: List of (security_type, security_value) tuples
            
        Returns:
            List of VINs (or None for not found)
        """
        results = []
        for sec_type, sec_value in lookups:
            results.append(self.lookup(sec_type, sec_value))
        return results
    
    def benchmark(self, num_lookups=10000):
        """Benchmark lookup performance."""
        print(f"\nBenchmarking {num_lookups:,} random lookups...")
        
        # Get random samples from the dataset
        sample_indices = self.df.sample(n=num_lookups).index
        
        start = time.time()
        for sec_type, sec_value in sample_indices:
            _ = self.lookup(sec_type, sec_value)
        elapsed = time.time() - start
        
        per_lookup_us = (elapsed / num_lookups) * 1_000_000
        
        print(f"✅ Completed {num_lookups:,} lookups in {elapsed:.3f}s")
        print(f"   Average: {per_lookup_us:.2f} microseconds per lookup")
        print(f"   Throughput: {num_lookups / elapsed:,.0f} lookups/sec")


def simulate_database_export():
    """Simulate exporting from a database."""
    print("="*80)
    print("SIMULATING DATABASE EXPORT")
    print("="*80)
    
    # Generate data (simulates SELECT * FROM security_identifiers)
    generator = SecurityDataGenerator(num_records=10_000_000)
    df = generator.generate()
    
    # Export to Parquet (simulates ETL process)
    exporter = ParquetExporter('data/security_lookup.parquet')
    parquet_file = exporter.export(df)
    
    return parquet_file


def demo_lookup_system():
    """Demonstrate the lookup system."""
    print("\n" + "="*80)
    print("FAST SECURITY LOOKUP DEMO")
    print("="*80)
    
    # Initialize lookup engine
    lookup = FastSecurityLookup('data/security_lookup.parquet')
    lookup.load()
    
    # Single lookup example
    print("\n" + "-"*80)
    print("Single Lookup Example:")
    print("-"*80)
    
    # Get a sample security
    sample = lookup.df.sample(1)
    sec_type = sample.index[0][0]
    sec_value = sample.index[0][1]
    expected_vin = sample['vin'].values[0]
    
    start = time.time()
    result_vin = lookup.lookup(sec_type, sec_value)
    elapsed_us = (time.time() - start) * 1_000_000
    
    print(f"Lookup: ({sec_type}, {sec_value})")
    print(f"Result VIN: {result_vin}")
    print(f"Expected VIN: {expected_vin}")
    print(f"Match: {'✅' if result_vin == expected_vin else '❌'}")
    print(f"Time: {elapsed_us:.2f} microseconds")
    
    # Batch lookup example
    print("\n" + "-"*80)
    print("Batch Lookup Example:")
    print("-"*80)
    
    samples = lookup.df.sample(100)
    batch_lookups = [(idx[0], idx[1]) for idx in samples.index]
    
    start = time.time()
    results = lookup.lookup_batch(batch_lookups)
    elapsed_ms = (time.time() - start) * 1000
    
    print(f"Looked up {len(batch_lookups)} securities")
    print(f"Time: {elapsed_ms:.2f} milliseconds")
    print(f"Average: {elapsed_ms / len(batch_lookups):.3f} ms per lookup")
    
    # Performance benchmark
    lookup.benchmark(num_lookups=10000)
    
    # Not found example
    print("\n" + "-"*80)
    print("Not Found Example:")
    print("-"*80)
    
    result = lookup.lookup('CUSIP', 'NOTFOUND99')
    print(f"Lookup: (CUSIP, NOTFOUND99)")
    print(f"Result: {result}")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Security Identifier Lookup System')
    parser.add_argument('--generate', action='store_true', help='Generate and export data')
    parser.add_argument('--demo', action='store_true', help='Run demo lookups')
    parser.add_argument('--benchmark', action='store_true', help='Run performance benchmark')
    
    args = parser.parse_args()
    
    if args.generate:
        # Generate data and export
        parquet_file = simulate_database_export()
        print(f"\n✅ Data ready: {parquet_file.absolute()}")
        
    if args.demo or (not args.generate and not args.benchmark):
        # Run demo (default if no args)
        if not Path('data/security_lookup.parquet').exists():
            print("⚠️  Data file not found. Generating first...")
            simulate_database_export()
        
        demo_lookup_system()
        
    if args.benchmark:
        # Just benchmark
        if not Path('data/security_lookup.parquet').exists():
            print("⚠️  Data file not found. Generating first...")
            simulate_database_export()
        
        lookup = FastSecurityLookup('data/security_lookup.parquet')
        lookup.load()
        lookup.benchmark(num_lookups=100000)
