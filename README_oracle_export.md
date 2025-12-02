# Oracle to Parquet High-Performance Exporter

## Overview
Export 100M+ row Oracle tables to Parquet format with maximum performance using parallel processing, optimized fetching, and chunked writing.

## Features
- ✅ **Parallel Processing**: Multiple database connections fetch data simultaneously
- ✅ **Optimized Fetching**: Large `arraysize` for bulk network transfers
- ✅ **Configurable**: YAML-based configuration for tables, columns, filters
- ✅ **Memory Efficient**: Chunked processing avoids loading entire dataset
- ✅ **Compressed Output**: Snappy/GZIP/ZSTD compression options
- ✅ **Reusable**: Configuration-driven for any table export

## Installation

```bash
pip install -r requirements_oracle.txt
```

## Configuration

Edit `oracle_export_config.yaml`:

```yaml
database:
  user: "your_username"
  password: "your_password"
  dsn: "hostname:1521/service_name"

export:
  schema: "SMCAPP"
  table: "MASTER_VIN_LKUP"
  columns: ["pkcol", "pkval", "pkvin"]
  filter: "feed_cd = 'PROD'"  # WHERE clause
  parallel_workers: 8  # Adjust based on database/network capacity
  output_file: "output.parquet"
```

## Usage

```bash
python oracle_to_parquet_parallel.py oracle_export_config.yaml
```

## Performance Tuning

### Database Side (Oracle DBA should configure):
```sql
-- Enable parallel query execution
ALTER SESSION ENABLE PARALLEL DML;
ALTER SESSION ENABLE PARALLEL QUERY;

-- Increase SGA if needed for large queries
-- ALTER SYSTEM SET sga_target = 16G SCOPE=BOTH;
```

### Python Side (in config.yaml):

| Parameter | Recommended | Description |
|-----------|-------------|-------------|
| `parallel_workers` | 4-16 | More workers = faster, but limited by DB connections |
| `fetch_size` | 10000-50000 | Rows per network round-trip (larger = fewer trips) |
| `chunk_size` | 50000 | Rows processed in memory per batch |
| `row_group_size` | 100000 | Parquet row group size (affects compression) |

### Network Optimization:
- Run script on same network as database (minimize latency)
- Use 10Gbps+ network if available
- Consider compression (`snappy` is fast, `zstd` is smaller)

## Expected Performance

**100M row table (3 columns):**
- **8 workers**: ~5-10 minutes (depends on network/DB)
- **Network**: ~500MB-2GB transferred (compressed)
- **Output**: ~200-800MB Parquet file (Snappy compression)

## Advanced: Oracle Parallel Hints

For even faster exports, modify the script's query to use Oracle hints:

```python
query = f"SELECT /*+ PARALLEL(8) */ {cols} FROM {schema}.{table}"
```

This tells Oracle to use 8 parallel server processes.

## Troubleshooting

**"Too many connections"**
- Reduce `parallel_workers` in config

**"Out of memory"**
- Reduce `chunk_size` and `fetch_size`
- Process in multiple runs with different filters

**"Slow performance"**
- Check network bandwidth between app and DB
- Verify Oracle has sufficient CPU/memory
- Enable Oracle Instant Client (thick mode) for better performance

## Reading the Parquet File

```python
import pandas as pd
import pyarrow.parquet as pq

# Read entire file
df = pd.read_parquet('output.parquet')

# Read with PyArrow (faster, lower memory)
table = pq.read_table('output.parquet')
df = table.to_pandas()

# Read specific columns only
df = pd.read_parquet('output.parquet', columns=['pkcol', 'pkval'])
```
