//! Device optimization module — hardware detection and optimal configuration generation.
//!
//! Provides automatic detection of CPU capabilities (SIMD features like AVX2, AVX512, NEON),
//! memory topology, disk characteristics, and environment (Docker, VM, cloud, bare-metal).
//! Generates optimal batch, preprocessing, inference, training, and quantization configs
//! tuned to the detected hardware.
//!
//! # Example
//! ```no_run
//! use automl::device::DeviceOptimizer;
//!
//! let optimizer = DeviceOptimizer::new();
//! let hw = &optimizer.hardware;
//! println!("CPU cores: {}, AVX2: {}", hw.cpu.cores, hw.cpu.has_avx2);
//!
//! let configs = optimizer.auto_tune_configs();
//! println!("Recommended max batch size: {}", configs.batch.max_batch_size);
//! ```
pub mod optimizer;

pub use optimizer::{DeviceOptimizer, HardwareInfo, CpuCapabilities, MemoryInfo, DiskInfo, OptimalConfigs};
