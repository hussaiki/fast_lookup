"""
Optimized Generic High-Performance Lookup System
Optimizations for 100M+ row datasets
"""

import pandas as pd
import pyarrow.parquet as pq
import yaml
import time
from pathlib import Path
from typing import List, Any, Union, Dict

class OptimizedLookupEngine:
    """
    Optimized lookup engine for 100M+ row datasets.
    Skips unnecessary sorting and uses PyArrow for faster loading.
    """
    
    def __init__(self, config_path: str = 'lookup_config.yaml'):
        self.config_path = config_path
        self.tables = {}
        self.configs = {}
        self._load_config()
        
    def _load_config(self):
        """Load YAML configuration."""
        if not Path(self.config_path).exists():
            raise FileNotFoundError(f"Config file not found: {self.config_path}")
            
        with open(self.config_path, 'r') as f:
            full_config = yaml.safe_load(f)
            
        self.configs = full_config.get('lookups', {})
        print(f"Loaded configuration for {len(self.configs)} lookup tables: {list(self.configs.keys())}")

    def load_table(self, table_name: str, assume_sorted: bool = True):
        """
        Load a specific table with optimizations for large datasets.
        
        Args:
            table_name: Name of the table from config
            assume_sorted: If True, skip sorting (assume Parquet already sorted by key)
        """
        if table_name not in self.configs:
            raise ValueError(f"Table '{table_name}' not defined in configuration.")
            
        config = self.configs[table_name]
        file_path = config['file_path']
        key_cols = config['key_columns']
        val_cols = config['value_columns']
        
        print(f"\nLoading table '{table_name}' from {file_path}...")
        total_start = time.time()
        
        # Optimization: Load only necessary columns
        columns = None
        if config.get('optimization') == 'memory':
            columns = list(set(key_cols + val_cols))
            
        # Load Data with PyArrow (faster than pandas for Parquet)
        t0 = time.time()
        if file_path.endswith('.parquet'):
            # Use PyArrow directly - much faster for large files
            table = pq.read_table(file_path, columns=columns)
            df = table.to_pandas()
            print(f"  - Read file (PyArrow): {time.time()-t0:.2f}s")
        elif file_path.endswith('.csv'):
            # For CSV, use dtype specification to avoid type inference overhead
            df = pd.read_csv(file_path, usecols=columns, low_memory=False)
            print(f"  - Read file (CSV): {time.time()-t0:.2f}s")
        else:
            raise ValueError("Unsupported file format. Use .parquet or .csv")
            
        # Create Index WITHOUT COPYING (faster)
        t0 = time.time()
        if len(key_cols) == 1:
            df.set_index(key_cols[0], inplace=True, drop=True)
        else:
            df.set_index(key_cols, inplace=True, drop=True)
        print(f"  - Set index: {time.time()-t0:.2f}s")
            
        # OPTIMIZATION: Skip sorting if data is already sorted
        if not assume_sorted:
            t0 = time.time()
            df.sort_index(inplace=True)
            print(f"  - Sort index: {time.time()-t0:.2f}s")
        else:
            print(f"  - Sort index: SKIPPED (assume_sorted=True)")
            # Verify if actually sorted (cheap check on index)
            if not df.index.is_monotonic_increasing:
                print("  ⚠️  WARNING: Index not sorted! Performance may degrade.")
                print("  💡 TIP: Sort data during Oracle export to avoid this.")
        
        self.tables[table_name] = df
        
        elapsed = time.time() - total_start
        mem_mb = df.memory_usage(deep=True).sum() / (1024**2)
        print(f"✅ Loaded {len(df):,} records in {elapsed:.2f}s")
        print(f"   Memory: {mem_mb:.1f} MB")
        print(f"   Keys: {key_cols}")
        print(f"   Values: {val_cols}")

    def get(self, table_name: str, keys: Union[List, Any]) -> Dict[str, Any]:
        """Lookup a single record."""
        if table_name not in self.tables:
            raise ValueError(f"Table '{table_name}' is not loaded.")
            
        df = self.tables[table_name]
        
        # Handle single vs multi-key
        lookup_key = keys
        if isinstance(keys, list) and len(keys) > 1:
            lookup_key = tuple(keys)
        elif isinstance(keys, list) and len(keys) == 1:
            lookup_key = keys[0]
            
        try:
            result = df.loc[lookup_key]
            
            # Handle duplicates (return first)
            if isinstance(result, pd.DataFrame):
                return result.iloc[0].to_dict()
            elif isinstance(result, pd.Series):
                return result.to_dict()
            else:
                return result
                
        except KeyError:
            return None

    def get_bulk(self, table_name: str, df: pd.DataFrame) -> pd.DataFrame:
        """
        Perform bulk lookup using merge (faster than join for large datasets).
        """
        if table_name not in self.tables:
            raise ValueError(f"Table '{table_name}' is not loaded.")
            
        config = self.configs[table_name]
        key_cols = config['key_columns']
        lookup_df = self.tables[table_name].reset_index()
        
        print(f"Performing bulk lookup on '{table_name}' ({len(df):,} records)...")
        start = time.time()
        
        # Use merge instead of join for better performance on large datasets
        result = pd.merge(
            df,
            lookup_df,
            on=key_cols,
            how='left'
        )
        
        elapsed = time.time() - start
        print(f"✅ Bulk lookup complete in {elapsed:.2f}s")
        print(f"   Speed: {len(df)/elapsed:,.0f} lookups/sec")
        
        return result


# Example usage
if __name__ == "__main__":
    print("="*80)
    print("OPTIMIZED LOOKUP DEMO (100M+ Rows)")
    print("="*80)
    
    engine = OptimizedLookupEngine('lookup_config.yaml')
    
    # Load table with assume_sorted=True to skip sorting
    # (Assumes Oracle export sorted the data)
    engine.load_table('securities', assume_sorted=True)
    
    # Single lookup test
    sample_key = engine.tables['securities'].index[1000]
    keys = list(sample_key) if isinstance(sample_key, tuple) else [sample_key]
    
    result = engine.get('securities', keys)
    print(f"\nSample lookup: {result}")
