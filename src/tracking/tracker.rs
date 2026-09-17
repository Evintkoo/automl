//! Experiment Tracker Implementation
//!
//! Track experiments, metrics, and artifacts.

use std::collections::HashMap;
use std::path::PathBuf;
use std::sync::RwLock;
use std::time::{SystemTime, UNIX_EPOCH};

use super::storage::{LocalStorage, StorageBackend};

/// Configuration for experiment tracking
#[derive(Debug, Clone)]
pub struct ExperimentConfig {
    /// Output directory for experiments
    pub output_dir: PathBuf,
    /// Experiment name
    pub experiment_name: String,
    /// Enable artifact logging
    pub enable_artifacts: bool,
    /// Enable metrics history
    pub enable_metrics_history: bool,
    /// Auto-save interval in seconds (0 to disable)
    pub auto_save_interval_secs: u64,
}

impl Default for ExperimentConfig {
    fn default() -> Self {
        Self {
            output_dir: PathBuf::from("./experiments"),
            experiment_name: format!("experiment_{}", current_timestamp()),
            enable_artifacts: true,
            enable_metrics_history: true,
            auto_save_interval_secs: 60,
        }
    }
}

/// A single metric value
#[derive(Debug, Clone)]
pub struct Metric {
    /// Metric name
    pub name: String,
    /// Metric value
    pub value: f64,
    /// Step/epoch number
    pub step: u64,
    /// Timestamp
    pub timestamp: u64,
}

impl Metric {
    /// Create a new metric
    pub fn new(name: impl Into<String>, value: f64, step: u64) -> Self {
        Self {
            name: name.into(),
            value,
            step,
            timestamp: current_timestamp(),
        }
    }
}

/// A run within an experiment
#[derive(Debug, Clone)]
pub struct Run {
    /// Run ID
    pub run_id: String,
    /// Run name
    pub run_name: String,
    /// Start time
    pub start_time: u64,
    /// End time (None if still running)
    pub end_time: Option<u64>,
    /// Parameters
    pub params: HashMap<String, String>,
    /// Latest metrics
    pub metrics: HashMap<String, f64>,
    /// Metrics history
    pub metrics_history: Vec<Metric>,
    /// Tags
    pub tags: HashMap<String, String>,
    /// Artifact paths
    pub artifacts: Vec<String>,
    /// Status
    pub status: RunStatus,
}

impl Run {
    /// Create a new run
    pub fn new(run_name: impl Into<String>) -> Self {
        let run_id = generate_run_id();
        Self {
            run_id,
            run_name: run_name.into(),
            start_time: current_timestamp(),
            end_time: None,
            params: HashMap::new(),
            metrics: HashMap::new(),
            metrics_history: Vec::new(),
            tags: HashMap::new(),
            artifacts: Vec::new(),
            status: RunStatus::Running,
        }
    }
    
    /// Get run duration in seconds
    pub fn duration_secs(&self) -> f64 {
        let end = self.end_time.unwrap_or_else(current_timestamp);
        end.saturating_sub(self.start_time) as f64
    }
}

/// Status of a run
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RunStatus {
    /// Run is currently running
    Running,
    /// Run completed successfully
    Finished,
    /// Run failed
    Failed,
    /// Run was killed/stopped
    Killed,
}

/// An experiment containing multiple runs
#[derive(Debug, Clone)]
pub struct Experiment {
    /// Experiment ID
    pub experiment_id: String,
    /// Experiment name
    pub name: String,
    /// Creation time
    pub created_at: u64,
    /// Runs in this experiment
    pub runs: Vec<Run>,
    /// Tags
    pub tags: HashMap<String, String>,
}

impl Experiment {
    /// Create a new experiment
    pub fn new(name: impl Into<String>) -> Self {
        Self {
            experiment_id: generate_experiment_id(),
            name: name.into(),
            created_at: current_timestamp(),
            runs: Vec::new(),
            tags: HashMap::new(),
        }
    }
    
    /// Get the best run by a metric (higher is better by default)
    pub fn best_run(&self, metric_name: &str, maximize: bool) -> Option<&Run> {
        self.runs.iter()
            .filter(|r| r.metrics.contains_key(metric_name))
            .max_by(|a, b| {
                let val_a = a.metrics.get(metric_name).unwrap_or(&0.0);
                let val_b = b.metrics.get(metric_name).unwrap_or(&0.0);
                
                if maximize {
                    val_a.partial_cmp(val_b).unwrap_or(std::cmp::Ordering::Equal)
                } else {
                    val_b.partial_cmp(val_a).unwrap_or(std::cmp::Ordering::Equal)
                }
            })
    }
}

/// Experiment tracker
pub struct ExperimentTracker {
    config: ExperimentConfig,
    storage: Box<dyn StorageBackend + Send + Sync>,
    
    // Current state
    current_experiment: RwLock<Option<Experiment>>,
    current_run: RwLock<Option<Run>>,
    
    // All experiments
    experiments: RwLock<HashMap<String, Experiment>>,
}

impl ExperimentTracker {
    /// Create a new experiment tracker
    pub fn new(config: ExperimentConfig) -> Self {
        let storage = Box::new(LocalStorage::new(config.output_dir.clone()));
        
        Self {
            config,
            storage,
            current_experiment: RwLock::new(None),
            current_run: RwLock::new(None),
            experiments: RwLock::new(HashMap::new()),
        }
    }
    
    /// Create with default configuration
    pub fn with_dir(output_dir: impl Into<PathBuf>) -> Self {
        let config = ExperimentConfig {
            output_dir: output_dir.into(),
            ..Default::default()
        };
        Self::new(config)
    }
    
    /// Create or get an experiment
    pub fn create_experiment(&self, name: impl Into<String>) -> String {
        let name = name.into();
        let mut experiment = Experiment::new(&name);

        if let Ok(mut experiments) = self.experiments.write() {
            // Guard against id collisions instead of silently overwriting an
            // existing experiment via `insert`. `generate_experiment_id` is
            // already collision-resistant (timestamp + random UUID-derived
            // suffix), so this loop is a defensive fallback rather than the
            // primary mechanism.
            while experiments.contains_key(&experiment.experiment_id) {
                experiment = Experiment::new(&name);
            }
            experiments.insert(experiment.experiment_id.clone(), experiment.clone());
        }

        let experiment_id = experiment.experiment_id.clone();

        if let Ok(mut current) = self.current_experiment.write() {
            *current = Some(experiment);
        }

        experiment_id
    }
    
    /// Start a new run
    pub fn start_run(&self, run_name: impl Into<String>) -> String {
        let run = Run::new(run_name);
        let run_id = run.run_id.clone();
        
        if let Ok(mut current) = self.current_run.write() {
            *current = Some(run);
        }
        
        run_id
    }
    
    /// Log a parameter
    pub fn log_param(&self, key: impl Into<String>, value: impl Into<String>) {
        if let Ok(mut run) = self.current_run.write() {
            if let Some(ref mut r) = *run {
                r.params.insert(key.into(), value.into());
            }
        }
    }
    
    /// Log multiple parameters
    pub fn log_params(&self, params: HashMap<String, String>) {
        if let Ok(mut run) = self.current_run.write() {
            if let Some(ref mut r) = *run {
                r.params.extend(params);
            }
        }
    }
    
    /// Log a metric
    pub fn log_metric(&self, name: impl Into<String>, value: f64, step: Option<u64>) {
        let name = name.into();
        let step = step.unwrap_or(0);
        
        if let Ok(mut run) = self.current_run.write() {
            if let Some(ref mut r) = *run {
                r.metrics.insert(name.clone(), value);
                
                if self.config.enable_metrics_history {
                    r.metrics_history.push(Metric::new(&name, value, step));
                }
            }
        }
    }
    
    /// Log multiple metrics
    pub fn log_metrics(&self, metrics: HashMap<String, f64>, step: Option<u64>) {
        for (name, value) in metrics {
            self.log_metric(name, value, step);
        }
    }
    
    /// Log a tag
    pub fn log_tag(&self, key: impl Into<String>, value: impl Into<String>) {
        if let Ok(mut run) = self.current_run.write() {
            if let Some(ref mut r) = *run {
                r.tags.insert(key.into(), value.into());
            }
        }
    }
    
    /// Log an artifact path
    pub fn log_artifact(&self, path: impl Into<String>) {
        if let Ok(mut run) = self.current_run.write() {
            if let Some(ref mut r) = *run {
                r.artifacts.push(path.into());
            }
        }
    }
    
    /// End the current run
    pub fn end_run(&self, status: RunStatus) {
        let completed_run = {
            let mut run_guard = match self.current_run.write() {
                Ok(g) => g,
                Err(_) => return,
            };
            
            if let Some(ref mut r) = *run_guard {
                r.end_time = Some(current_timestamp());
                r.status = status;
            }
            
            run_guard.take()
        };
        
        // Add to experiment
        if let Some(run) = completed_run {
            if let Ok(mut exp) = self.current_experiment.write() {
                if let Some(ref mut e) = *exp {
                    e.runs.push(run.clone());
                    
                    // Update in experiments map
                    if let Ok(mut experiments) = self.experiments.write() {
                        if let Some(stored_exp) = experiments.get_mut(&e.experiment_id) {
                            stored_exp.runs.push(run);
                        }
                    }
                }
            }
        }
    }
    
    /// End run as finished
    pub fn end_run_success(&self) {
        self.end_run(RunStatus::Finished);
    }
    
    /// End run as failed
    pub fn end_run_failed(&self) {
        self.end_run(RunStatus::Failed);
    }
    
    /// Get the current run
    pub fn current_run(&self) -> Option<Run> {
        self.current_run.read().ok().and_then(|r| r.clone())
    }
    
    /// Get the current experiment
    pub fn current_experiment(&self) -> Option<Experiment> {
        self.current_experiment.read().ok().and_then(|e| e.clone())
    }
    
    /// Get an experiment by ID
    pub fn get_experiment(&self, experiment_id: &str) -> Option<Experiment> {
        self.experiments.read()
            .ok()
            .and_then(|e| e.get(experiment_id).cloned())
    }
    
    /// List all experiments
    pub fn list_experiments(&self) -> Vec<Experiment> {
        self.experiments.read()
            .map(|e| e.values().cloned().collect())
            .unwrap_or_default()
    }
    
    /// Save current state to storage
    pub fn save(&self) -> Result<(), String> {
        let experiments: Vec<Experiment> = self.experiments.read()
            .map(|e| e.values().cloned().collect())
            .map_err(|e| e.to_string())?;
        
        self.storage.save_experiments(&experiments)
    }
    
    /// Load state from storage
    pub fn load(&self) -> Result<(), String> {
        let experiments = self.storage.load_experiments()?;
        
        if let Ok(mut stored) = self.experiments.write() {
            for exp in experiments {
                stored.insert(exp.experiment_id.clone(), exp);
            }
        }
        
        Ok(())
    }

    /// Generate a comprehensive text report for an experiment
    pub fn generate_report(&self, experiment_id: &str) -> Result<String, String> {
        let experiment = self.get_experiment(experiment_id)
            .ok_or_else(|| format!("Experiment '{}' not found", experiment_id))?;

        let mut report = String::new();

        // Header
        report.push_str(&format!("{:=<60}\n", ""));
        report.push_str("  Experiment Report\n");
        report.push_str(&format!("{:=<60}\n", ""));
        report.push_str(&format!("  Name: {}\n", experiment.name));
        report.push_str(&format!("  ID:   {}\n", experiment.experiment_id));
        report.push_str(&format!("  Created: {}\n", experiment.created_at));
        report.push_str(&format!("  Total Runs: {}\n\n", experiment.runs.len()));

        // Runs detail
        report.push_str(&format!("{:-<60}\n", ""));
        report.push_str("  Runs\n");
        report.push_str(&format!("{:-<60}\n", ""));

        for run in &experiment.runs {
            let status_str = match run.status {
                RunStatus::Running => "RUNNING",
                RunStatus::Finished => "FINISHED",
                RunStatus::Failed => "FAILED",
                RunStatus::Killed => "KILLED",
            };
            report.push_str(&format!("\n  Run: {} ({})\n", run.run_name, run.run_id));
            report.push_str(&format!("    Status: {}\n", status_str));
            report.push_str(&format!("    Duration: {:.2}s\n", run.duration_secs()));

            if !run.params.is_empty() {
                report.push_str("    Parameters:\n");
                let mut params: Vec<_> = run.params.iter().collect();
                params.sort_by_key(|(k, _)| (*k).clone());
                for (k, v) in &params {
                    report.push_str(&format!("      {}: {}\n", k, v));
                }
            }

            if !run.metrics.is_empty() {
                report.push_str("    Metrics:\n");
                let mut metrics: Vec<_> = run.metrics.iter().collect();
                metrics.sort_by_key(|(k, _)| (*k).clone());
                for (k, v) in &metrics {
                    report.push_str(&format!("      {}: {:.6}\n", k, v));
                }
            }
        }

        // Best run identification
        if !experiment.runs.is_empty() {
            // Collect all metric names
            let mut all_metrics: Vec<String> = experiment.runs.iter()
                .flat_map(|r| r.metrics.keys().cloned())
                .collect();
            all_metrics.sort();
            all_metrics.dedup();

            if !all_metrics.is_empty() {
                report.push_str(&format!("\n{:-<60}\n", ""));
                report.push_str("  Best Run (by first metric, maximize)\n");
                report.push_str(&format!("{:-<60}\n", ""));

                let primary_metric = &all_metrics[0];
                if let Some(best) = experiment.best_run(primary_metric, true) {
                    report.push_str(&format!("  Primary Metric: {}\n", primary_metric));
                    report.push_str(&format!("  Best Run: {} ({})\n", best.run_name, best.run_id));
                    if let Some(val) = best.metrics.get(primary_metric) {
                        report.push_str(&format!("  Value: {:.6}\n", val));
                    }
                }

                // Parameter summary
                report.push_str(&format!("\n{:-<60}\n", ""));
                report.push_str("  Parameter Summary\n");
                report.push_str(&format!("{:-<60}\n", ""));

                let mut all_params: Vec<String> = experiment.runs.iter()
                    .flat_map(|r| r.params.keys().cloned())
                    .collect();
                all_params.sort();
                all_params.dedup();

                for param in &all_params {
                    let values: Vec<&str> = experiment.runs.iter()
                        .filter_map(|r| r.params.get(param).map(|s| s.as_str()))
                        .collect();
                    let mut unique_values: Vec<&str> = values.clone();
                    unique_values.sort();
                    unique_values.dedup();
                    report.push_str(&format!("  {}: {} unique value(s) {:?}\n", param, unique_values.len(), unique_values));
                }

                // Metrics comparison table
                report.push_str(&format!("\n{:-<60}\n", ""));
                report.push_str("  Metrics Comparison\n");
                report.push_str(&format!("{:-<60}\n", ""));

                // Header row
                let run_col_width = 20;
                let metric_col_width = 14;
                report.push_str(&format!("  {:<width$}", "Run", width = run_col_width));
                for m in &all_metrics {
                    report.push_str(&format!("{:>width$}", m, width = metric_col_width));
                }
                report.push('\n');
                report.push_str(&format!("  {:-<width$}", "", width = run_col_width + all_metrics.len() * metric_col_width));
                report.push('\n');

                for run in &experiment.runs {
                    let name = if run.run_name.len() > run_col_width - 2 {
                        format!("{}…", &run.run_name[..run_col_width - 3])
                    } else {
                        run.run_name.clone()
                    };
                    report.push_str(&format!("  {:<width$}", name, width = run_col_width));
                    for m in &all_metrics {
                        match run.metrics.get(m) {
                            Some(v) => report.push_str(&format!("{:>width$.6}", v, width = metric_col_width)),
                            None => report.push_str(&format!("{:>width$}", "N/A", width = metric_col_width)),
                        }
                    }
                    report.push('\n');
                }
            }
        }

        report.push_str(&format!("\n{:=<60}\n", ""));
        Ok(report)
    }

    /// Compute and store a confusion matrix as an artifact
    pub fn log_confusion_matrix(
        &mut self,
        run_id: &str,
        y_true: &[i32],
        y_pred: &[i32],
        class_names: Option<&[String]>,
    ) -> Result<(), String> {
        if y_true.len() != y_pred.len() {
            return Err("y_true and y_pred must have the same length".to_string());
        }

        // Determine classes
        let mut classes: Vec<i32> = y_true.iter().chain(y_pred.iter()).copied().collect();
        classes.sort();
        classes.dedup();
        let n = classes.len();

        // Build class index map
        let class_idx: HashMap<i32, usize> = classes.iter().enumerate().map(|(i, &c)| (c, i)).collect();

        // Compute confusion matrix
        let mut matrix = vec![vec![0u64; n]; n];
        for (t, p) in y_true.iter().zip(y_pred.iter()) {
            let ti = class_idx[t];
            let pi = class_idx[p];
            matrix[ti][pi] += 1;
        }

        // Build artifact string
        let labels: Vec<String> = match class_names {
            Some(names) if names.len() == n => names.to_vec(),
            _ => classes.iter().map(|c| c.to_string()).collect(),
        };

        let mut artifact = String::from("Confusion Matrix\n");
        // Header
        artifact.push_str(&format!("{:<12}", ""));
        for label in &labels {
            artifact.push_str(&format!("{:>12}", label));
        }
        artifact.push('\n');
        // Rows
        for (i, label) in labels.iter().enumerate() {
            artifact.push_str(&format!("{:<12}", label));
            for j in 0..n {
                artifact.push_str(&format!("{:>12}", matrix[i][j]));
            }
            artifact.push('\n');
        }

        // Store as artifact on the matching run
        self.store_artifact_on_run(run_id, &format!("confusion_matrix_{}", run_id), &artifact)
    }

    /// Store feature importance data as an artifact
    pub fn log_feature_importance(
        &mut self,
        run_id: &str,
        feature_names: &[String],
        importance: &[f64],
    ) -> Result<(), String> {
        if feature_names.len() != importance.len() {
            return Err("feature_names and importance must have the same length".to_string());
        }

        // Sort by importance descending
        let mut pairs: Vec<(&String, &f64)> = feature_names.iter().zip(importance.iter()).collect();
        pairs.sort_by(|a, b| b.1.partial_cmp(a.1).unwrap_or(std::cmp::Ordering::Equal));

        let mut artifact = String::from("Feature Importance\n");
        artifact.push_str(&format!("{:<30} {:>12}\n", "Feature", "Importance"));
        artifact.push_str(&format!("{:-<42}\n", ""));
        for (name, &imp) in &pairs {
            artifact.push_str(&format!("{:<30} {:>12.6}\n", name, imp));
        }

        self.store_artifact_on_run(run_id, &format!("feature_importance_{}", run_id), &artifact)
    }

    /// Return a JSON-compatible summary of an experiment
    pub fn get_experiment_summary(
        &self,
        experiment_id: &str,
    ) -> Result<HashMap<String, serde_json::Value>, String> {
        let experiment = self.get_experiment(experiment_id)
            .ok_or_else(|| format!("Experiment '{}' not found", experiment_id))?;

        let mut summary = HashMap::new();
        summary.insert("experiment_id".to_string(), serde_json::Value::String(experiment.experiment_id.clone()));
        summary.insert("name".to_string(), serde_json::Value::String(experiment.name.clone()));
        summary.insert("created_at".to_string(), serde_json::json!(experiment.created_at));
        summary.insert("num_runs".to_string(), serde_json::json!(experiment.runs.len()));

        // Runs summary
        let runs_json: Vec<serde_json::Value> = experiment.runs.iter().map(|run| {
            let status_str = match run.status {
                RunStatus::Running => "RUNNING",
                RunStatus::Finished => "FINISHED",
                RunStatus::Failed => "FAILED",
                RunStatus::Killed => "KILLED",
            };
            serde_json::json!({
                "run_id": run.run_id,
                "run_name": run.run_name,
                "status": status_str,
                "duration_secs": run.duration_secs(),
                "params": run.params,
                "metrics": run.metrics,
                "artifacts": run.artifacts,
            })
        }).collect();
        summary.insert("runs".to_string(), serde_json::json!(runs_json));

        // Best run per metric
        let mut all_metrics: Vec<String> = experiment.runs.iter()
            .flat_map(|r| r.metrics.keys().cloned())
            .collect();
        all_metrics.sort();
        all_metrics.dedup();

        let mut best_runs = HashMap::new();
        for metric in &all_metrics {
            if let Some(best) = experiment.best_run(metric, true) {
                best_runs.insert(metric.clone(), serde_json::json!({
                    "run_id": best.run_id,
                    "run_name": best.run_name,
                    "value": best.metrics.get(metric),
                }));
            }
        }
        summary.insert("best_runs".to_string(), serde_json::json!(best_runs));

        // Tags
        summary.insert("tags".to_string(), serde_json::json!(experiment.tags));

        Ok(summary)
    }

    /// Internal helper: store an artifact string on a specific run by ID
    fn store_artifact_on_run(&mut self, run_id: &str, artifact_name: &str, content: &str) -> Result<(), String> {
        // Check current run first
        if let Ok(mut current) = self.current_run.write() {
            if let Some(ref mut r) = *current {
                if r.run_id == run_id {
                    r.artifacts.push(format!("{}:\n{}", artifact_name, content));
                    return Ok(());
                }
            }
        }

        // Search in experiments
        if let Ok(mut experiments) = self.experiments.write() {
            for exp in experiments.values_mut() {
                for run in &mut exp.runs {
                    if run.run_id == run_id {
                        run.artifacts.push(format!("{}:\n{}", artifact_name, content));
                        return Ok(());
                    }
                }
            }
        }

        Err(format!("Run '{}' not found", run_id))
    }
}

// Helper functions

fn current_timestamp() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_secs())
        .unwrap_or(0)
}

/// Short random suffix appended to timestamp-based ids so two ids generated
/// within the same second (`current_timestamp` only has second resolution)
/// don't collide. Backed by a UUID v4, so collisions are astronomically
/// unlikely even under heavy concurrent creation. Matches the
/// `uuid::Uuid::new_v4()`-derived short-id convention already used elsewhere
/// in this crate (see `src/server/handlers.rs`).
fn id_suffix() -> String {
    uuid::Uuid::new_v4()
        .to_string()
        .split('-')
        .next()
        .unwrap_or("0")
        .to_string()
}

fn generate_run_id() -> String {
    format!("run_{}_{}", current_timestamp(), id_suffix())
}

fn generate_experiment_id() -> String {
    format!("exp_{}_{}", current_timestamp(), id_suffix())
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::HashMap;
    
    #[test]
    fn test_tracker_basic() {
        let config = ExperimentConfig {
            output_dir: PathBuf::from("/tmp/test_experiments"),
            ..Default::default()
        };
        let tracker = ExperimentTracker::new(config);
        
        // Create experiment
        let exp_id = tracker.create_experiment("test_experiment");
        assert!(!exp_id.is_empty());
        
        // Start run
        let run_id = tracker.start_run("run_1");
        assert!(!run_id.is_empty());
        
        // Log params
        tracker.log_param("learning_rate", "0.01");
        tracker.log_param("batch_size", "32");
        
        // Log metrics
        tracker.log_metric("accuracy", 0.95, Some(1));
        tracker.log_metric("loss", 0.05, Some(1));
        
        // End run
        tracker.end_run_success();
        
        // Check experiment has the run
        let exp = tracker.current_experiment().unwrap();
        assert_eq!(exp.runs.len(), 1);
        assert_eq!(exp.runs[0].status, RunStatus::Finished);
    }
    
    #[test]
    fn test_metrics_history() {
        let config = ExperimentConfig {
            enable_metrics_history: true,
            ..Default::default()
        };
        let tracker = ExperimentTracker::new(config);
        
        tracker.create_experiment("test");
        tracker.start_run("run");
        
        tracker.log_metric("loss", 1.0, Some(0));
        tracker.log_metric("loss", 0.5, Some(1));
        tracker.log_metric("loss", 0.1, Some(2));
        
        let run = tracker.current_run().unwrap();
        assert_eq!(run.metrics_history.len(), 3);
        assert_eq!(run.metrics.get("loss"), Some(&0.1)); // Latest value
    }
    
    #[test]
    fn test_best_run() {
        let mut exp = Experiment::new("test");
        
        let mut run1 = Run::new("run1");
        run1.metrics.insert("accuracy".to_string(), 0.8);
        
        let mut run2 = Run::new("run2");
        run2.metrics.insert("accuracy".to_string(), 0.95);
        
        let mut run3 = Run::new("run3");
        run3.metrics.insert("accuracy".to_string(), 0.85);
        
        exp.runs = vec![run1, run2, run3];
        
        let best = exp.best_run("accuracy", true).unwrap();
        assert_eq!(best.metrics.get("accuracy"), Some(&0.95));
    }
}
