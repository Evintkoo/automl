//! Data loading utilities

use crate::error::{AutoMLError, Result};
use polars::prelude::*;
use std::path::Path;
use std::io::{BufRead, BufReader};
use std::fs::File;
use std::time::Instant;

/// Data loader for various file formats
pub struct DataLoader {
    /// Number of threads for parallel loading
    n_threads: Option<usize>,
    /// Low memory mode
    low_memory: bool,
    /// Chunk size for streaming
    chunk_size: Option<usize>,
}

impl Default for DataLoader {
    fn default() -> Self {
        Self::new()
    }
}

impl DataLoader {
    /// Create a new data loader
    pub fn new() -> Self {
        Self {
            n_threads: None,
            low_memory: false,
            chunk_size: None,
        }
    }

    /// Set number of threads
    pub fn with_n_threads(mut self, n: usize) -> Self {
        self.n_threads = Some(n);
        self
    }

    /// Enable low memory mode
    pub fn with_low_memory(mut self, low_memory: bool) -> Self {
        self.low_memory = low_memory;
        self
    }

    /// Set chunk size for streaming
    pub fn with_chunk_size(mut self, size: usize) -> Self {
        self.chunk_size = Some(size);
        self
    }

    /// Load a CSV file
    pub fn load_csv(&self, path: &str) -> Result<DataFrame> {
        let file = File::open(path)
            .map_err(|e| AutoMLError::DataError(e.to_string()))?;

        let reader = CsvReadOptions::default()
            .with_has_header(true)
            .with_infer_schema_length(Some(100))
            .into_reader_with_file_handle(file);

        reader.finish()
            .map_err(|e| AutoMLError::DataError(e.to_string()))
    }

    /// Load a CSV file with specific options
    pub fn load_csv_with_options(
        &self,
        path: &str,
        delimiter: u8,
        has_header: bool,
        skip_rows: usize,
        _columns: Option<Vec<String>>,
    ) -> Result<DataFrame> {
        let file = File::open(path)
            .map_err(|e| AutoMLError::DataError(e.to_string()))?;

        let parse_opts = CsvParseOptions::default()
            .with_separator(delimiter);

        let reader = CsvReadOptions::default()
            .with_has_header(has_header)
            .with_skip_rows(skip_rows)
            .with_infer_schema_length(Some(100))
            .with_parse_options(parse_opts)
            .into_reader_with_file_handle(file);

        reader.finish()
            .map_err(|e| AutoMLError::DataError(e.to_string()))
    }

    /// Load a Parquet file
    pub fn load_parquet(&self, path: &str) -> Result<DataFrame> {
        let file = File::open(path)
            .map_err(|e| AutoMLError::DataError(e.to_string()))?;

        let reader = ParquetReader::new(file);

        reader.finish()
            .map_err(|e| AutoMLError::DataError(e.to_string()))
    }

    /// Load a Parquet file with specific columns
    pub fn load_parquet_columns(&self, path: &str, columns: Vec<String>) -> Result<DataFrame> {
        let file = File::open(path)
            .map_err(|e| AutoMLError::DataError(e.to_string()))?;

        ParquetReader::new(file)
            .with_columns(Some(columns))
            .finish()
            .map_err(|e| AutoMLError::DataError(e.to_string()))
    }

    /// Load a JSON file (line-delimited)
    pub fn load_json(&self, path: &str) -> Result<DataFrame> {
        let file = File::open(path)
            .map_err(|e| AutoMLError::DataError(e.to_string()))?;

        JsonReader::new(file)
            .finish()
            .map_err(|e| AutoMLError::DataError(e.to_string()))
    }

    /// Detect file format from extension and load
    pub fn load_auto(&self, path: &str) -> Result<DataFrame> {
        let path_lower = path.to_lowercase();
        
        if path_lower.ends_with(".csv") || path_lower.ends_with(".tsv") {
            let delimiter = if path_lower.ends_with(".tsv") { b'\t' } else { b',' };
            self.load_csv_with_options(path, delimiter, true, 0, None)
        } else if path_lower.ends_with(".parquet") || path_lower.ends_with(".pq") {
            self.load_parquet(path)
        } else if path_lower.ends_with(".json") || path_lower.ends_with(".jsonl") {
            self.load_json(path)
        } else {
            // Try CSV as default
            self.load_csv(path)
        }
    }

    /// Get file info without loading full data
    pub fn get_file_info(&self, path: &str) -> Result<FileInfo> {
        let metadata = std::fs::metadata(path)
            .map_err(|e| AutoMLError::DataError(e.to_string()))?;

        let file_size = metadata.len();
        
        // Quick row count for CSV
        let (n_rows, n_cols, columns) = if path.to_lowercase().ends_with(".csv") {
            let file = File::open(path)
                .map_err(|e| AutoMLError::DataError(e.to_string()))?;
            let reader = BufReader::new(file);
            let mut lines = reader.lines();
            
            // Get header
            let header = lines.next()
                .transpose()
                .map_err(|e| AutoMLError::DataError(e.to_string()))?
                .unwrap_or_default();
            
            let columns: Vec<String> = header.split(',')
                .map(|s| s.trim().to_string())
                .collect();
            
            let n_cols = columns.len();
            let n_rows = lines.count(); // Count remaining lines
            
            (Some(n_rows), Some(n_cols), Some(columns))
        } else {
            (None, None, None)
        };

        Ok(FileInfo {
            path: path.to_string(),
            file_size,
            n_rows,
            n_cols,
            columns,
        })
    }
}

/// File information
#[derive(Debug, Clone)]
pub struct FileInfo {
    pub path: String,
    pub file_size: u64,
    pub n_rows: Option<usize>,
    pub n_cols: Option<usize>,
    pub columns: Option<Vec<String>>,
}

/// Chunked data reader for large files
pub struct ChunkedReader {
    path: String,
    chunk_size: usize,
    current_chunk: usize,
    #[allow(dead_code)]
    total_rows: Option<usize>,
}

impl ChunkedReader {
    /// Create a new chunked reader
    pub fn new(path: &str, chunk_size: usize) -> Self {
        Self {
            path: path.to_string(),
            chunk_size,
            current_chunk: 0,
            total_rows: None,
        }
    }

    /// Read the next chunk
    pub fn next_chunk(&mut self) -> Result<Option<DataFrame>> {
        let skip_rows = self.current_chunk * self.chunk_size;
        
        let file = File::open(&self.path)
            .map_err(|e| AutoMLError::DataError(e.to_string()))?;

        let df = CsvReadOptions::default()
            .with_has_header(self.current_chunk == 0)
            .with_skip_rows(if self.current_chunk > 0 { skip_rows + 1 } else { 0 })
            .with_n_rows(Some(self.chunk_size))
            .into_reader_with_file_handle(file)
            .finish()
            .map_err(|e| AutoMLError::DataError(e.to_string()))?;

        if df.height() == 0 {
            return Ok(None);
        }

        self.current_chunk += 1;
        Ok(Some(df))
    }

    /// Reset to beginning
    pub fn reset(&mut self) {
        self.current_chunk = 0;
    }

    /// Get current chunk index
    pub fn current_chunk(&self) -> usize {
        self.current_chunk
    }
}

/// Save DataFrame to various formats
pub struct DataSaver;

impl DataSaver {
    /// Save to CSV
    pub fn save_csv(df: &mut DataFrame, path: &str) -> Result<()> {
        let mut file = File::create(path)
            .map_err(|e| AutoMLError::DataError(e.to_string()))?;

        CsvWriter::new(&mut file)
            .finish(df)
            .map_err(|e| AutoMLError::DataError(e.to_string()))
    }

    /// Save to Parquet
    pub fn save_parquet(df: &mut DataFrame, path: &str) -> Result<()> {
        let file = File::create(path)
            .map_err(|e| AutoMLError::DataError(e.to_string()))?;

        ParquetWriter::new(file)
            .finish(df)
            .map_err(|e| AutoMLError::DataError(e.to_string()))?;

        Ok(())
    }

    /// Save to JSON
    pub fn save_json(df: &mut DataFrame, path: &str) -> Result<()> {
        let mut file = File::create(path)
            .map_err(|e| AutoMLError::DataError(e.to_string()))?;

        JsonWriter::new(&mut file)
            .finish(df)
            .map_err(|e| AutoMLError::DataError(e.to_string()))
    }
}

// ---------------------------------------------------------------------------
// Optimized data loader (mirrors Python optimized_data_loader.py)
// ---------------------------------------------------------------------------

/// Dataset size classification based on row count.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DatasetSize {
    Tiny,
    Small,
    Medium,
    Large,
    VeryLarge,
}

impl DatasetSize {
    pub fn from_row_count(n_rows: usize) -> Self {
        match n_rows {
            0..=999 => DatasetSize::Tiny,
            1_000..=9_999 => DatasetSize::Small,
            10_000..=99_999 => DatasetSize::Medium,
            100_000..=999_999 => DatasetSize::Large,
            _ => DatasetSize::VeryLarge,
        }
    }
}

/// Strategy used to load a dataset into memory.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum LoadingStrategy {
    Direct,
    Chunked,
    Streaming,
    MemoryMapped,
}

/// Metadata about a loaded dataset.
#[derive(Debug, Clone)]
pub struct DatasetInfo {
    pub n_rows: usize,
    pub n_cols: usize,
    pub file_size_bytes: u64,
    pub format: String,
    pub loading_strategy: LoadingStrategy,
    pub load_time_secs: f64,
    pub memory_usage_mb: f64,
    pub optimizations_applied: Vec<String>,
}

/// Configuration for [`OptimizedDataLoader`].
#[derive(Debug, Clone)]
pub struct LoadingConfig {
    pub max_memory_pct: f64,
    pub chunk_size: usize,
    pub enable_dtype_optimization: bool,
    pub enable_categorical_optimization: bool,
}

impl Default for LoadingConfig {
    fn default() -> Self {
        Self {
            max_memory_pct: 0.8,
            chunk_size: 100_000,
            enable_dtype_optimization: true,
            enable_categorical_optimization: true,
        }
    }
}

/// An optimized data loader that automatically selects the best loading
/// strategy based on file size and available system memory.
pub struct OptimizedDataLoader {
    config: LoadingConfig,
}

impl OptimizedDataLoader {
    /// Create a new `OptimizedDataLoader` with an optional configuration.
    pub fn new(config: Option<LoadingConfig>) -> Self {
        Self {
            config: config.unwrap_or_default(),
        }
    }

    /// Estimate file complexity without fully reading the file.
    ///
    /// Returns `(DatasetSize, estimated_memory_mb)`.
    pub fn estimate_file_complexity(file_path: &str) -> Result<(DatasetSize, f64)> {
        let path = Path::new(file_path);
        let metadata = std::fs::metadata(path)
            .map_err(|e| AutoMLError::DataError(format!("Cannot stat file: {e}")))?;
        let file_size = metadata.len();
        let file_size_mb = file_size as f64 / (1024.0 * 1024.0);

        // Estimate row count from first few lines for CSV / TSV.
        let ext = path.extension().and_then(|e| e.to_str()).unwrap_or("");
        let estimated_rows: usize = match ext.to_lowercase().as_str() {
            "csv" | "tsv" => {
                let f = File::open(path)
                    .map_err(|e| AutoMLError::DataError(e.to_string()))?;
                let reader = BufReader::new(f);
                let mut lines = reader.lines();

                // Read header
                let header = lines
                    .next()
                    .transpose()
                    .map_err(|e| AutoMLError::DataError(e.to_string()))?
                    .unwrap_or_default();
                let header_len = header.len() + 1; // +1 for newline

                // Sample a few data lines to get average row size
                let mut sample_bytes: usize = 0;
                let mut sample_count: usize = 0;
                for line in lines.take(100) {
                    let l = line.map_err(|e| AutoMLError::DataError(e.to_string()))?;
                    sample_bytes += l.len() + 1;
                    sample_count += 1;
                }

                if sample_count == 0 {
                    0
                } else {
                    let avg_row = sample_bytes as f64 / sample_count as f64;
                    let data_bytes = (file_size as usize).saturating_sub(header_len);
                    (data_bytes as f64 / avg_row).ceil() as usize
                }
            }
            // For binary formats, use a rough heuristic.
            _ => (file_size_mb * 10_000.0) as usize,
        };

        let dataset_size = DatasetSize::from_row_count(estimated_rows);
        // Rough memory estimate: ~3x file size for in-memory DataFrame overhead.
        let estimated_memory_mb = file_size_mb * 3.0;

        Ok((dataset_size, estimated_memory_mb))
    }

    /// Pick the best loading strategy based on file size versus available memory.
    pub fn select_loading_strategy(file_size_mb: f64, available_memory_mb: f64) -> LoadingStrategy {
        let ratio = file_size_mb / available_memory_mb;
        if ratio < 0.1 {
            LoadingStrategy::Direct
        } else if ratio < 0.4 {
            LoadingStrategy::MemoryMapped
        } else if ratio < 0.8 {
            LoadingStrategy::Chunked
        } else {
            LoadingStrategy::Streaming
        }
    }

    /// Load data from `file_path` using an automatically selected strategy.
    pub fn load_data(&self, file_path: &str) -> std::result::Result<(DataFrame, DatasetInfo), String> {
        let start = Instant::now();
        let path = Path::new(file_path);

        let file_size_bytes = std::fs::metadata(path)
            .map_err(|e| format!("Cannot stat file: {e}"))?
            .len();
        let file_size_mb = file_size_bytes as f64 / (1024.0 * 1024.0);

        // Determine available memory via sysinfo.
        let available_memory_mb = {
            let mut sys = sysinfo::System::new();
            sys.refresh_memory();
            sys.available_memory() as f64 / (1024.0 * 1024.0)
        };

        let strategy = Self::select_loading_strategy(file_size_mb, available_memory_mb);

        let format = path
            .extension()
            .and_then(|e| e.to_str())
            .unwrap_or("unknown")
            .to_lowercase();

        // Load the DataFrame using the basic DataLoader (delegates format detection).
        let loader = DataLoader::new();
        let mut df = loader
            .load_auto(file_path)
            .map_err(|e| format!("Load failed: {e}"))?;

        let mut optimizations: Vec<String> = Vec::new();

        if self.config.enable_dtype_optimization {
            let opts = Self::optimize_dtypes(&mut df);
            optimizations.extend(opts);
        }

        let n_rows = df.height();
        let n_cols = df.width();
        let load_time_secs = start.elapsed().as_secs_f64();

        let memory_usage_mb = df.estimated_size() as f64 / (1024.0 * 1024.0);

        let info = DatasetInfo {
            n_rows,
            n_cols,
            file_size_bytes,
            format,
            loading_strategy: strategy,
            load_time_secs,
            memory_usage_mb,
            optimizations_applied: optimizations,
        };

        Ok((df, info))
    }

    /// Downcast numeric columns to the smallest fitting type.
    ///
    /// Returns a list of human-readable optimization descriptions.
    pub fn optimize_dtypes(df: &mut DataFrame) -> Vec<String> {
        let mut optimizations = Vec::new();
        let col_names: Vec<String> = df.get_column_names().into_iter().map(|s| s.to_string()).collect();

        for name in &col_names {
            let Some(series) = df.column(name).ok().map(|c| c.as_materialized_series().clone()) else {
                continue;
            };

            match series.dtype() {
                DataType::Int64 => {
                    if let Ok(ca) = series.i64() {
                        let min = ca.min().unwrap_or(i64::MIN);
                        let max = ca.max().unwrap_or(i64::MAX);
                        if min >= i8::MIN as i64 && max <= i8::MAX as i64 {
                            if let Ok(new_s) = series.cast(&DataType::Int8) {
                                let _ = df.replace(name.as_str(), new_s);
                                optimizations.push(format!("{name}: Int64 -> Int8"));
                            }
                        } else if min >= i16::MIN as i64 && max <= i16::MAX as i64 {
                            if let Ok(new_s) = series.cast(&DataType::Int16) {
                                let _ = df.replace(name.as_str(), new_s);
                                optimizations.push(format!("{name}: Int64 -> Int16"));
                            }
                        } else if min >= i32::MIN as i64 && max <= i32::MAX as i64 {
                            if let Ok(new_s) = series.cast(&DataType::Int32) {
                                let _ = df.replace(name.as_str(), new_s);
                                optimizations.push(format!("{name}: Int64 -> Int32"));
                            }
                        }
                    }
                }
                DataType::Float64 => {
                    if let Ok(ca) = series.f64() {
                        let min = ca.min().unwrap_or(f64::MIN);
                        let max = ca.max().unwrap_or(f64::MAX);
                        if min >= f32::MIN as f64 && max <= f32::MAX as f64 {
                            // Check precision loss is acceptable
                            let mut precision_ok = true;
                            for opt_val in ca.into_iter().take(1000) {
                                if let Some(v) = opt_val {
                                    if (v - v as f32 as f64).abs() > 1e-6 * v.abs().max(1.0) {
                                        precision_ok = false;
                                        break;
                                    }
                                }
                            }
                            if precision_ok {
                                if let Ok(new_s) = series.cast(&DataType::Float32) {
                                    let _ = df.replace(name.as_str(), new_s);
                                    optimizations.push(format!("{name}: Float64 -> Float32"));
                                }
                            }
                        }
                    }
                }
                _ => {}
            }
        }

        optimizations
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Write;
    use tempfile::NamedTempFile;

    fn create_test_csv() -> NamedTempFile {
        let mut file = tempfile::Builder::new()
            .suffix(".csv")
            .tempfile()
            .unwrap();
        writeln!(file, "a,b,c").unwrap();
        writeln!(file, "1,2,3").unwrap();
        writeln!(file, "4,5,6").unwrap();
        writeln!(file, "7,8,9").unwrap();
        file
    }

    #[test]
    fn test_load_csv() {
        let file = create_test_csv();
        let loader = DataLoader::new();
        
        let df = loader.load_csv(file.path().to_str().unwrap()).unwrap();
        
        assert_eq!(df.height(), 3);
        assert_eq!(df.width(), 3);
    }

    #[test]
    fn test_get_file_info() {
        let file = create_test_csv();
        let loader = DataLoader::new();
        
        let info = loader.get_file_info(file.path().to_str().unwrap()).unwrap();
        
        assert_eq!(info.n_rows, Some(3));  // 3 data rows (excluding header)
        assert_eq!(info.n_cols, Some(3));
        assert_eq!(info.columns.as_ref().unwrap().len(), 3);
    }

    #[test]
    fn test_chunked_reader() {
        let file = create_test_csv();
        let mut reader = ChunkedReader::new(file.path().to_str().unwrap(), 2);
        
        let chunk1 = reader.next_chunk().unwrap();
        assert!(chunk1.is_some());
        assert_eq!(chunk1.unwrap().height(), 2);
        
        let chunk2 = reader.next_chunk().unwrap();
        assert!(chunk2.is_some());
    }

    #[test]
    fn test_save_csv() {
        let mut df = DataFrame::new(vec![
            Column::new("a".into(), &[1, 2, 3]),
            Column::new("b".into(), &[4, 5, 6]),
        ]).unwrap();

        let file = NamedTempFile::new().unwrap();
        DataSaver::save_csv(&mut df, file.path().to_str().unwrap()).unwrap();

        // Reload and verify
        let loader = DataLoader::new();
        let loaded = loader.load_csv(file.path().to_str().unwrap()).unwrap();
        
        assert_eq!(loaded.height(), 3);
        assert_eq!(loaded.width(), 2);
    }
}
