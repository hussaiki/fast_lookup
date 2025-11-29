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
```

## Performance Benchmarks

Tested on 10 Million records:

| Metric | Result |
|--------|--------|
| **Data Load Time** | ~17 seconds |
| **Single Lookup Time** | ~72 microseconds (0.072 ms) |
| **Throughput** | ~13,800 lookups/sec |
| **File Size** | ~178 MB (Parquet Snappy) |
| **Memory Usage** | ~1.1 GB |

## Data Structure

| Column | Type | Description |
|--------|------|-------------|
| `security_id_type` | String | Type of identifier (CUSIP, ISIN, etc.) |
| `security_id_value` | String | The identifier value |
| `vin` | Int32 | Vendor Identification Number |

The system uses a MultiIndex on `(security_id_type, security_id_value)` for O(1) lookup complexity.
