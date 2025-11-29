"""
Generic High-Performance Lookup System
Driven by YAML configuration file.
"""

import pandas as pd
import yaml
import time
from pathlib import Path
from typing import List, Any, Union, Dict

class GenericLookupEngine:
    """
    Generic lookup engine that manages multiple lookup tables 
    based on a configuration file.
    """
    
    def __init__(self, config_path: str = 'lookup_config.yaml'):
        self.config_path = config_path
        self.tables = {}  # Stores loaded DataFrames
        self.configs = {} # Stores table configurations
        self._load_config()
        
    def _load_config(self):
        """Load YAML configuration."""
        if not Path(self.config_path).exists():
            raise FileNotFoundError(f"Config file not found: {self.config_path}")
            
        with open(self.config_path, 'r') as f:
            full_config = yaml.safe_load(f)
            
        self.configs = full_config.get('lookups', {})
        print(f"Loaded configuration for {len(self.configs)} lookup tables: {list(self.configs.keys())}")

    def load_table(self, table_name: str):
        """
        Load a specific table into memory and build index.
        """
        if table_name not in self.configs:
            raise ValueError(f"Table '{table_name}' not defined in configuration.")
            
        config = self.configs[table_name]
        file_path = config['file_path']
        key_cols = config['key_columns']
        val_cols = config['value_columns']
        
        print(f"\nLoading table '{table_name}' from {file_path}...")
        total_start = time.time()
        
        # Optimization: Load only necessary columns if specified
        columns = None
        if config.get('optimization') == 'memory':
            columns = list(set(key_cols + val_cols))
            
        # Load Data
        t0 = time.time()
        if file_path.endswith('.parquet'):
            df = pd.read_parquet(file_path, columns=columns)
        elif file_path.endswith('.csv'):
            df = pd.read_csv(file_path, usecols=columns)
        else:
            raise ValueError("Unsupported file format. Use .parquet or .csv")
        print(f"  - Read file: {time.time()-t0:.2f}s")
            
        # Create Index
        t0 = time.time()
        # If single key, use standard index. If multiple, use MultiIndex.
        if len(key_cols) == 1:
            df.set_index(key_cols[0], inplace=True)
        else:
            df.set_index(key_cols, inplace=True)
        print(f"  - Set index: {time.time()-t0:.2f}s")
            
        t0 = time.time()
        df.sort_index(inplace=True)
        print(f"  - Sort index: {time.time()-t0:.2f}s")
        
        self.tables[table_name] = df
        
        elapsed = time.time() - total_start
        print(f"✅ Loaded {len(df):,} records in {elapsed:.2f}s")
        print(f"   Keys: {key_cols}")
        print(f"   Values: {val_cols}")

    def get(self, table_name: str, keys: Union[List, Any]) -> Dict[str, Any]:
        """
        Lookup a single record.
        """
        if table_name not in self.tables:
            raise ValueError(f"Table '{table_name}' is not loaded. Call load_table() first.")
            
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
        Perform bulk lookup using a DataFrame input.
        """
        if table_name not in self.tables:
            raise ValueError(f"Table '{table_name}' is not loaded.")
            
        config = self.configs[table_name]
        key_cols = config['key_columns']
        lookup_df = self.tables[table_name]
        
        print(f"Performing bulk lookup on '{table_name}' ({len(df):,} records)...")
        start = time.time()
        
        # Perform join
        result = df.join(
            lookup_df,
            on=key_cols,
            how='left'
        )
        
        elapsed = time.time() - start
        print(f"✅ Bulk lookup complete in {elapsed:.2f}s")
        print(f"   Speed: {len(df)/elapsed:,.0f} lookups/sec")
        
        return result

def demo_generic():
    """Demonstrate the generic lookup system."""
    print("="*80)
    print("GENERIC LOOKUP DEMO")
    print("="*80)
    
    # Initialize
    engine = GenericLookupEngine('lookup_config.yaml')
    
    # Load the securities table defined in config
    try:
        engine.load_table('securities')
    except Exception as e:
        print(f"Error loading table: {e}")
        print("Make sure 'data/security_lookup.parquet' exists (run security_lookup.py --generate)")
        return

    # 1. Single Lookup
    print("\n1. Single Lookup Test")
    # We need a valid key. Let's peek at the file or just try a random one if we knew it.
    sample_idx = engine.tables['securities'].index[1000]
    print(f"   Looking up key: {sample_idx}")
    
    keys = list(sample_idx) if isinstance(sample_idx, tuple) else [sample_idx]
    
    start = time.time()
    result = engine.get('securities', keys)
    elapsed_us = (time.time() - start) * 1_000_000
    
    print(f"   Result: {result}")
    print(f"   Time: {elapsed_us:.2f} µs")
    
    # 2. Bulk Lookup Test
    print("\n2. Bulk Lookup Test (10k records)")
    # Optimize sampling: use integer indexing instead of full sample
    lookup_df = engine.tables['securities']
    
    # Take top 10k records for speed (avoid random sampling overhead on 10M rows)
    print("   Creating test batch (taking first 10k records)...")
    input_df = lookup_df.iloc[:10000].reset_index()[engine.configs['securities']['key_columns']].copy()
    
    result_df = engine.get_bulk('securities', input_df)
    print(f"   Input shape: {input_df.shape}")
    print(f"   Output shape: {result_df.shape}")
    print(f"   Columns added: {[c for c in result_df.columns if c not in input_df.columns]}")

if __name__ == "__main__":
    # Ensure pyyaml is installed
    try:
        import yaml
    except ImportError:
        print("Installing PyYAML...")
        import os
        os.system("pip install pyyaml")
        
    demo_generic()
