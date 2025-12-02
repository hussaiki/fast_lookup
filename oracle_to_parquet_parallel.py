"""
High-Performance Oracle to Parquet Exporter
Supports parallel fetching, chunked processing, and configurable table/column exports
"""

import oracledb  # Modern Oracle driver (successor to cx_Oracle)
import pyarrow as pa
import pyarrow.parquet as pq
import pandas as pd
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import List, Dict, Optional
import yaml
import logging
from datetime import datetime
from pathlib import Path

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class OracleToParquetExporter:
    """Export Oracle table data to Parquet with parallel processing."""
    
    def __init__(self, config_path: str):
        """Initialize with configuration file."""
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        
        self.db_config = self.config['database']
        self.export_config = self.config['export']
        
    def get_connection(self):
        """Create Oracle connection with optimized settings."""
        conn = oracledb.connect(
            user=self.db_config['user'],
            password=self.db_config['password'],
            dsn=self.db_config['dsn']
        )
        
        # Enable thick mode for better performance (requires Oracle Instant Client)
        # oracledb.init_oracle_client(lib_dir="/path/to/instantclient")
        
        return conn
    
    def get_row_count(self, conn, table: str, filter_clause: str) -> int:
        """Get total row count with filter."""
        schema = self.export_config['schema']
        query = f"SELECT COUNT(*) FROM {schema}.{table}"
        if filter_clause:
            query += f" WHERE {filter_clause}"
        
        cursor = conn.cursor()
        cursor.execute(query)
        count = cursor.fetchone()[0]
        cursor.close()
        return count
    
    def get_partition_ranges(self, conn, table: str, num_partitions: int, filter_clause: str) -> List[tuple]:
        """
        Create partition ranges based on ROWID for parallel fetching.
        Uses Oracle's DBMS_ROWID for efficient range-based parallelism.
        """
        schema = self.export_config['schema']
        total_rows = self.get_row_count(conn, table, filter_clause)
        rows_per_partition = total_rows // num_partitions
        
        logger.info(f"Total rows: {total_rows:,}, Partitions: {num_partitions}, Rows/partition: {rows_per_partition:,}")
        
        # Get ROWID boundaries for each partition
        ranges = []
        for i in range(num_partitions):
            offset = i * rows_per_partition
            # Each partition will use OFFSET/FETCH for simplicity
            # More advanced: use actual ROWID ranges for true parallelism
            ranges.append((offset, rows_per_partition))
        
        return ranges
    
    def fetch_partition(
        self, 
        partition_id: int, 
        offset: int, 
        limit: int,
        table: str,
        columns: List[str],
        filter_clause: str
    ) -> pd.DataFrame:
        """Fetch a specific partition of data."""
        schema = self.export_config['schema']
        conn = self.get_connection()
        
        try:
            # Use arraysize for bulk fetching (critical for performance)
            cursor = conn.cursor()
            cursor.arraysize = self.export_config.get('fetch_size', 10000)
            
            # Build query with optional parallel hint
            cols = ", ".join(columns)
            parallel_degree = self.export_config.get('oracle_parallel_degree', 0)
            
            if parallel_degree > 0:
                # Use Oracle parallel query hint to utilize multiple CPUs
                query = f"SELECT /*+ PARALLEL({parallel_degree}) */ {cols} FROM {schema}.{table}"
            else:
                query = f"SELECT {cols} FROM {schema}.{table}"
            
            if filter_clause:
                query += f" WHERE {filter_clause}"
            
            # ORDER BY key columns for pre-sorted export (critical for fast lookups)
            # This makes set_index + sort_index unnecessary in Python
            sort_columns = self.export_config.get('sort_columns', [])
            if sort_columns:
                query += f" ORDER BY {', '.join(sort_columns)}"
            
            # Oracle 12c+ pagination (more efficient than ROWNUM)
            query += f" OFFSET {offset} ROWS FETCH NEXT {limit} ROWS ONLY"
            
            logger.info(f"Partition {partition_id}: Starting fetch (offset={offset:,}, limit={limit:,})")
            start_time = datetime.now()
            
            cursor.execute(query)
            
            # Fetch in chunks to avoid memory issues
            chunk_size = self.export_config.get('chunk_size', 50000)
            all_rows = []
            
            while True:
                rows = cursor.fetchmany(chunk_size)
                if not rows:
                    break
                all_rows.extend(rows)
                logger.debug(f"Partition {partition_id}: Fetched {len(all_rows):,} rows so far")
            
            elapsed = (datetime.now() - start_time).total_seconds()
            logger.info(f"Partition {partition_id}: Completed in {elapsed:.2f}s ({len(all_rows):,} rows)")
            
            # Convert to DataFrame
            df = pd.DataFrame(all_rows, columns=columns)
            
            cursor.close()
            conn.close()
            
            return df
            
        except Exception as e:
            logger.error(f"Partition {partition_id} failed: {e}")
            conn.close()
            raise
    
    def export_table(self, table_name: Optional[str] = None):
        """Export table to Parquet with parallel processing."""
        table = table_name or self.export_config['table']
        columns = self.export_config['columns']
        filter_clause = self.export_config.get('filter', '')
        num_workers = self.export_config.get('parallel_workers', 4)
        output_file = self.export_config.get('output_file', f'{table}.parquet')
        
        logger.info(f"Starting export: {self.export_config['schema']}.{table}")
        logger.info(f"Columns: {', '.join(columns)}")
        logger.info(f"Filter: {filter_clause or 'None'}")
        logger.info(f"Workers: {num_workers}")
        
        # Get connection to determine partitions
        conn = self.get_connection()
        partition_ranges = self.get_partition_ranges(conn, table, num_workers, filter_clause)
        conn.close()
        
        # Fetch partitions in parallel
        all_dataframes = []
        
        with ThreadPoolExecutor(max_workers=num_workers) as executor:
            futures = []
            
            for partition_id, (offset, limit) in enumerate(partition_ranges):
                future = executor.submit(
                    self.fetch_partition,
                    partition_id,
                    offset,
                    limit,
                    table,
                    columns,
                    filter_clause
                )
                futures.append(future)
            
            # Collect results as they complete
            for future in as_completed(futures):
                try:
                    df = future.result()
                    all_dataframes.append(df)
                    logger.info(f"Collected partition with {len(df):,} rows")
                except Exception as e:
                    logger.error(f"Partition failed: {e}")
                    raise
        
        # Combine all partitions
        logger.info("Combining partitions...")
        final_df = pd.concat(all_dataframes, ignore_index=True)
        logger.info(f"Total rows combined: {len(final_df):,}")
        
        # Write to Parquet with compression
        logger.info(f"Writing to Parquet: {output_file}")
        start_time = datetime.now()
        
        # Convert to PyArrow Table for efficient writing
        table_pa = pa.Table.from_pandas(final_df)
        
        pq.write_table(
            table_pa,
            output_file,
            compression=self.export_config.get('compression', 'snappy'),
            use_dictionary=True,  # Better compression for repeated values
            write_statistics=True,
            row_group_size=self.export_config.get('row_group_size', 100000)
        )
        
        elapsed = (datetime.now() - start_time).total_seconds()
        file_size_mb = Path(output_file).stat().st_size / (1024 * 1024)
        
        logger.info(f"✅ Export complete!")
        logger.info(f"   File: {output_file}")
        logger.info(f"   Size: {file_size_mb:.2f} MB")
        logger.info(f"   Rows: {len(final_df):,}")
        logger.info(f"   Write time: {elapsed:.2f}s")


def main():
    """Run export with configuration."""
    import sys
    
    if len(sys.argv) < 2:
        print("Usage: python oracle_to_parquet_parallel.py <config.yaml>")
        sys.exit(1)
    
    config_path = sys.argv[1]
    exporter = OracleToParquetExporter(config_path)
    exporter.export_table()


if __name__ == "__main__":
    main()
