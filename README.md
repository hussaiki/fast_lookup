# High-Performance Security Identifier Lookup System

A fast, memory-efficient system for looking up Vendor Identification Numbers (VIN) using various security identifiers (CUSIP, ISIN, TICKER, etc.).

## Features

- **High Performance**: Sub-millisecond lookups (~70 microseconds)
- **High Throughput**: ~14,000 lookups per second
- **Scalable**: Handles 10M+ records efficiently using Parquet and Pandas MultiIndex
- **Memory Efficient**: Uses optimized data types and storage

## Setup

1.  **Generate Data** (First time only):
    ```bash
    cd lookup
    python3 security_lookup.py --generate
    ```
    This creates a 10M record dataset in `data/security_lookup.parquet`.

2.  **Run Demo**:
    ```bash
    python3 security_lookup.py --demo
    ```

3.  **Run Benchmark**:
    ```bash
    python3 security_lookup.py --benchmark
    ```

## Usage API

```python
from security_lookup import FastSecurityLookup

# Initialize and load data (takes ~17s for 10M records)
lookup_engine = FastSecurityLookup('data/security_lookup.parquet')
lookup_engine.load()

# Single Lookup
vin = lookup_engine.lookup('ISIN', 'US0378331005')
print(f"VIN: {vin}")

# Batch Lookup (Faster for multiple items)
securities = [
    ('TICKER', 'AAPL'),
    ('CUSIP', '037833100'),
    ('ISIN', 'US0378331005')
]
results = lookup_engine.lookup_batch(securities)

# Bulk DataFrame Lookup (For millions of records)
import pandas as pd
df = pd.read_csv('my_securities.csv') # Must have security_id_type and security_id_value columns
result_df = lookup_engine.lookup_dataframe(df)
# Returns DataFrame with new 'vin' column
```

## Performance Benchmarks

Tested on 10 Million records:

| Metric | Result |
|--------|--------|
| **Data Load Time** | ~17 seconds |
| **Single Lookup Time** | ~72 microseconds |
| **Throughput (Iterative)** | ~14,000 lookups/sec |
| **Throughput (Bulk)** | ~11,000 lookups/sec |
| **2 Million Records** | ~2.5 minutes |

**Note**: For 2M+ records, the system processes at ~10k-14k records per second.

## Generic Lookup (Configurable)

You can use `generic_lookup.py` to lookup against **any** table defined in `lookup_config.yaml`.

### 1. Configure `lookup_config.yaml`
```yaml
lookups:
  securities:
    file_path: "data/security_lookup.parquet"
    key_columns: ["security_id_type", "security_id_value"]
    value_columns: ["vin"]
```

### 2. Run Generic Lookup
```python
from generic_lookup import GenericLookupEngine

# Initialize
engine = GenericLookupEngine('lookup_config.yaml')
engine.load_table('securities')

# Single Lookup
result = engine.get('securities', ['ISIN', 'US0378331005'])

# Bulk Lookup
import pandas as pd
df = pd.read_csv('my_data.csv') # Must have key columns
result_df = engine.get_bulk('securities', df)
```

## Troubleshooting

**"Lookup takes forever"**
- Ensure you are NOT using `df.sample(n)` on large datasets for testing. Generating random samples from 10M records is slow.
- Use `df.iloc[:n]` or `df.head(n)` for fast testing.
- The lookup engine itself is very fast (~70µs per lookup).

## Data Structure

| Column | Type | Description |
|--------|------|-------------|
| `security_id_type` | String | Type of identifier (CUSIP, ISIN, etc.) |
| `security_id_value` | String | The identifier value |
| `vin` | Int32 | Vendor Identification Number |

The system uses a MultiIndex on `(security_id_type, security_id_value)` for O(1) lookup complexity.
