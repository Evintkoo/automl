//! Device optimizer module for hardware detection and optimal configuration generation.
//!
//! Detects CPU capabilities, memory, disk, and environment to generate
//! hardware-optimized configurations for batch processing, preprocessing,
//! inference, training, and quantization.

use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::fs;
use std::path::Path;
use sysinfo::{Disks, System};

// ---------------------------------------------------------------------------
// Hardware info structs
// ---------------------------------------------------------------------------

/// CPU capability information.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CpuCapabilities {
    pub cores: usize,
    pub threads: usize,
    pub frequency_mhz: u64,
    pub l1_cache_kb: u64,
    pub l2_cache_kb: u64,
    pub l3_cache_kb: u64,
    pub has_sse: bool,
    pub has_sse2: bool,
    pub has_sse3: bool,
    pub has_sse41: bool,
    pub has_sse42: bool,
    pub has_avx: bool,
    pub has_avx2: bool,
    pub has_avx512: bool,
    pub has_neon: bool,
    pub has_fma: bool,
}

/// System memory information.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MemoryInfo {
    pub total_bytes: u64,
    pub available_bytes: u64,
    pub used_percentage: f64,
}

/// Disk information.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DiskInfo {
    pub total_bytes: u64,
    pub available_bytes: u64,
    pub is_ssd: bool,
}

/// Aggregated hardware information.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HardwareInfo {
    pub cpu: CpuCapabilities,
    pub memory: MemoryInfo,
    pub disk: DiskInfo,
    pub environment: String,
}

// ---------------------------------------------------------------------------
// Optimal config structs (local types independent of other crate modules)
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OptimalBatchConfig {
    pub max_batch_size: usize,
    pub num_workers: usize,
    pub enable_priority: bool,
    pub adaptive_sizing: bool,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OptimalPreprocessConfig {
    pub chunk_size: usize,
    pub num_workers: usize,
    pub normalization: String,
    pub enable_parallel: bool,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OptimalInferenceConfig {
    pub batch_size: usize,
    pub num_threads: usize,
    pub enable_quantization: bool,
    pub cache_size: usize,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OptimalTrainingConfig {
    pub n_estimators: usize,
    pub cv_folds: usize,
    pub max_depth: usize,
    pub learning_rate: f64,
    pub num_threads: usize,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OptimalQuantizationConfig {
    pub quantization_type: String,
    pub bits: u8,
    pub enable_calibration: bool,
}

/// All optimal configurations bundled together.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OptimalConfigs {
    pub batch: OptimalBatchConfig,
    pub preprocess: OptimalPreprocessConfig,
    pub inference: OptimalInferenceConfig,
    pub training: OptimalTrainingConfig,
    pub quantization: OptimalQuantizationConfig,
    pub hardware: HardwareInfo,
}

// ---------------------------------------------------------------------------
// DeviceOptimizer
// ---------------------------------------------------------------------------

/// Detects hardware capabilities and generates optimal configurations.
#[derive(Debug, Clone)]
pub struct DeviceOptimizer {
    pub hardware: HardwareInfo,
}

impl DeviceOptimizer {
    /// Create a new `DeviceOptimizer` by auto-detecting hardware.
    pub fn new() -> Self {
        let hardware = Self::detect_hardware();
        Self { hardware }
    }

    // -- Hardware detection --------------------------------------------------

    /// Detect all hardware characteristics.
    pub fn detect_hardware() -> HardwareInfo {
        let mut sys = System::new_all();
        sys.refresh_all();

        let cpu = Self::detect_cpu_capabilities(&sys);
        let memory = Self::detect_memory_info(&sys);
        let disk = Self::detect_disk_info();
        let environment = Self::detect_environment();

        HardwareInfo {
            cpu,
            memory,
            disk,
            environment,
        }
    }

    /// Detect CPU capabilities (cores, frequency, SIMD features).
    pub fn detect_cpu_capabilities(sys: &System) -> CpuCapabilities {
        let cpus = sys.cpus();
        let cores = sys.physical_core_count().unwrap_or(1);
        let threads = cpus.len().max(1);
        let frequency_mhz = cpus.first().map_or(0, |c| c.frequency());

        // Cache sizes are not directly available from sysinfo; use sensible
        // defaults scaled by core count.
        let l1_cache_kb = 32 * cores as u64;
        let l2_cache_kb = 256 * cores as u64;
        let l3_cache_kb = 2048 * (cores as u64).max(1);

        CpuCapabilities {
            cores,
            threads,
            frequency_mhz,
            l1_cache_kb,
            l2_cache_kb,
            l3_cache_kb,
            has_sse: cfg!(target_feature = "sse"),
            has_sse2: cfg!(target_feature = "sse2"),
            has_sse3: cfg!(target_feature = "sse3"),
            has_sse41: cfg!(target_feature = "sse4.1"),
            has_sse42: cfg!(target_feature = "sse4.2"),
            has_avx: cfg!(target_feature = "avx"),
            has_avx2: cfg!(target_feature = "avx2"),
            has_avx512: cfg!(target_feature = "avx512f"),
            has_neon: cfg!(target_arch = "aarch64"),
            has_fma: cfg!(target_feature = "fma"),
        }
    }

    fn detect_memory_info(sys: &System) -> MemoryInfo {
        let total = sys.total_memory();
        let available = sys.available_memory();
        let used_pct = if total > 0 {
            ((total - available) as f64 / total as f64) * 100.0
        } else {
            0.0
        };
        MemoryInfo {
            total_bytes: total,
            available_bytes: available,
            used_percentage: used_pct,
        }
    }

    fn detect_disk_info() -> DiskInfo {
        let disks = Disks::new_with_refreshed_list();
        let mut total: u64 = 0;
        let mut available: u64 = 0;
        let mut is_ssd = false;

        for disk in disks.list() {
            total += disk.total_space();
            available += disk.available_space();
            // Heuristic: SSD disks often report removable = false and have
            // kind != HDD. sysinfo::DiskKind::SSD is available on some
            // platforms.
            if format!("{:?}", disk.kind()).contains("SSD")
                || format!("{:?}", disk.kind()).contains("Ssd")
            {
                is_ssd = true;
            }
        }

        // If we could not determine type, assume SSD on macOS/modern systems.
        if !is_ssd && cfg!(target_os = "macos") {
            is_ssd = true;
        }

        DiskInfo {
            total_bytes: total,
            available_bytes: available,
            is_ssd,
        }
    }

    /// Detect the runtime environment (Docker, VM, bare-metal, cloud).
    pub fn detect_environment() -> String {
        // Docker detection
        if Path::new("/.dockerenv").exists() {
            return "docker".to_string();
        }
        if let Ok(cgroup) = fs::read_to_string("/proc/1/cgroup") {
            if cgroup.contains("docker") || cgroup.contains("kubepods") {
                return "docker".to_string();
            }
        }

        // VM detection via DMI on Linux
        if let Ok(product) = fs::read_to_string("/sys/class/dmi/id/product_name") {
            let lower = product.to_lowercase();
            if lower.contains("virtualbox")
                || lower.contains("vmware")
                || lower.contains("kvm")
                || lower.contains("qemu")
                || lower.contains("hyper-v")
                || lower.contains("xen")
            {
                return "vm".to_string();
            }
        }

        // Cloud detection – check common cloud metadata hints
        if let Ok(bios) = fs::read_to_string("/sys/class/dmi/id/bios_vendor") {
            let lower = bios.to_lowercase();
            if lower.contains("amazon")
                || lower.contains("google")
                || lower.contains("microsoft")
            {
                return "cloud".to_string();
            }
        }

        "bare-metal".to_string()
    }

    // -- Optimal configuration generators ------------------------------------

    /// Optimal batch processor configuration based on detected hardware.
    pub fn get_optimal_batch_config(&self) -> OptimalBatchConfig {
        let mem_gb = self.hardware.memory.total_bytes as f64 / 1_073_741_824.0;
        let cores = self.hardware.cpu.cores;

        let max_batch_size = if mem_gb >= 32.0 {
            10_000
        } else if mem_gb >= 16.0 {
            5_000
        } else if mem_gb >= 8.0 {
            2_000
        } else {
            500
        };

        let num_workers = (cores / 2).max(1);

        OptimalBatchConfig {
            max_batch_size,
            num_workers,
            enable_priority: cores >= 4,
            adaptive_sizing: mem_gb >= 8.0,
        }
    }

    /// Optimal preprocessing configuration.
    pub fn get_optimal_preprocessor_config(&self) -> OptimalPreprocessConfig {
        let mem_gb = self.hardware.memory.total_bytes as f64 / 1_073_741_824.0;
        let cores = self.hardware.cpu.cores;

        let chunk_size = if mem_gb >= 32.0 {
            100_000
        } else if mem_gb >= 16.0 {
            50_000
        } else if mem_gb >= 8.0 {
            25_000
        } else {
            10_000
        };

        let num_workers = (cores / 2).max(1);
        let normalization = if self.hardware.cpu.has_avx2 || self.hardware.cpu.has_neon {
            "standard".to_string()
        } else {
            "minmax".to_string()
        };

        OptimalPreprocessConfig {
            chunk_size,
            num_workers,
            normalization,
            enable_parallel: cores >= 2,
        }
    }

    /// Optimal inference configuration.
    pub fn get_optimal_inference_config(&self) -> OptimalInferenceConfig {
        let mem_gb = self.hardware.memory.total_bytes as f64 / 1_073_741_824.0;
        let threads = self.hardware.cpu.threads;

        let batch_size = if mem_gb >= 32.0 {
            256
        } else if mem_gb >= 16.0 {
            128
        } else if mem_gb >= 8.0 {
            64
        } else {
            32
        };

        let cache_mb = if mem_gb >= 32.0 {
            2048
        } else if mem_gb >= 16.0 {
            1024
        } else {
            512
        };

        OptimalInferenceConfig {
            batch_size,
            num_threads: threads.max(1),
            enable_quantization: mem_gb < 16.0,
            cache_size: cache_mb * 1_048_576, // bytes
        }
    }

    /// Optimal training configuration.
    pub fn get_optimal_training_config(&self) -> OptimalTrainingConfig {
        let mem_gb = self.hardware.memory.total_bytes as f64 / 1_073_741_824.0;
        let cores = self.hardware.cpu.cores;

        let n_estimators = if mem_gb >= 32.0 { 500 } else { 200 };
        let cv_folds = if cores >= 8 { 10 } else { 5 };
        let max_depth = if mem_gb >= 16.0 { 12 } else { 8 };
        let learning_rate = if mem_gb >= 32.0 { 0.01 } else { 0.05 };

        OptimalTrainingConfig {
            n_estimators,
            cv_folds,
            max_depth,
            learning_rate,
            num_threads: cores.max(1),
        }
    }

    /// Optimal quantization configuration.
    pub fn get_optimal_quantization_config(&self) -> OptimalQuantizationConfig {
        let mem_gb = self.hardware.memory.total_bytes as f64 / 1_073_741_824.0;
        let has_wide_simd = self.hardware.cpu.has_avx2 || self.hardware.cpu.has_neon;

        let (qtype, bits) = if mem_gb >= 32.0 && has_wide_simd {
            ("fp16".to_string(), 16u8)
        } else if mem_gb >= 16.0 {
            ("int8".to_string(), 8)
        } else {
            ("int4".to_string(), 4)
        };

        OptimalQuantizationConfig {
            quantization_type: qtype,
            bits,
            enable_calibration: mem_gb >= 8.0,
        }
    }

    // -- Aggregate helpers ---------------------------------------------------

    /// Generate all optimal configs at once.
    pub fn auto_tune_configs(&self) -> OptimalConfigs {
        OptimalConfigs {
            batch: self.get_optimal_batch_config(),
            preprocess: self.get_optimal_preprocessor_config(),
            inference: self.get_optimal_inference_config(),
            training: self.get_optimal_training_config(),
            quantization: self.get_optimal_quantization_config(),
            hardware: self.hardware.clone(),
        }
    }

    /// Return full system information as a JSON-compatible map.
    pub fn get_system_info(&self) -> HashMap<String, serde_json::Value> {
        let mut info = HashMap::new();

        info.insert(
            "cpu".to_string(),
            serde_json::to_value(&self.hardware.cpu).unwrap_or_default(),
        );
        info.insert(
            "memory".to_string(),
            serde_json::to_value(&self.hardware.memory).unwrap_or_default(),
        );
        info.insert(
            "disk".to_string(),
            serde_json::to_value(&self.hardware.disk).unwrap_or_default(),
        );
        info.insert(
            "environment".to_string(),
            serde_json::Value::String(self.hardware.environment.clone()),
        );

        let configs = self.auto_tune_configs();
        info.insert(
            "optimal_configs".to_string(),
            serde_json::to_value(&configs).unwrap_or_default(),
        );

        info
    }

    // -- Serialization -------------------------------------------------------

    /// Save all optimal configs to a JSON file.
    pub fn save_configs<P: AsRef<Path>>(&self, path: P) -> Result<(), String> {
        let configs = self.auto_tune_configs();
        let json = serde_json::to_string_pretty(&configs).map_err(|e| e.to_string())?;
        fs::write(path, json).map_err(|e| e.to_string())
    }

    /// Load optimal configs from a JSON file.
    pub fn load_configs<P: AsRef<Path>>(path: P) -> Result<OptimalConfigs, String> {
        let data = fs::read_to_string(path).map_err(|e| e.to_string())?;
        serde_json::from_str(&data).map_err(|e| e.to_string())
    }
}

impl Default for DeviceOptimizer {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_detect_hardware() {
        let hw = DeviceOptimizer::detect_hardware();
        assert!(hw.cpu.cores >= 1);
        assert!(hw.cpu.threads >= 1);
        assert!(hw.memory.total_bytes > 0);
    }

    #[test]
    fn test_auto_tune_configs() {
        let optimizer = DeviceOptimizer::new();
        let configs = optimizer.auto_tune_configs();
        assert!(configs.batch.max_batch_size > 0);
        assert!(configs.preprocess.num_workers >= 1);
        assert!(configs.inference.num_threads >= 1);
        assert!(configs.training.n_estimators > 0);
        assert!(configs.quantization.bits > 0);
    }

    #[test]
    fn test_save_load_configs() {
        let optimizer = DeviceOptimizer::new();
        let tmp = std::env::temp_dir().join("automl_test_device_configs.json");
        optimizer.save_configs(&tmp).expect("save failed");
        let loaded = DeviceOptimizer::load_configs(&tmp).expect("load failed");
        assert_eq!(loaded.batch.max_batch_size, optimizer.get_optimal_batch_config().max_batch_size);
        let _ = fs::remove_file(&tmp);
    }

    #[test]
    fn test_get_system_info() {
        let optimizer = DeviceOptimizer::new();
        let info = optimizer.get_system_info();
        assert!(info.contains_key("cpu"));
        assert!(info.contains_key("memory"));
        assert!(info.contains_key("environment"));
    }

    #[test]
    fn test_detect_environment() {
        let env = DeviceOptimizer::detect_environment();
        assert!(!env.is_empty());
    }
}
