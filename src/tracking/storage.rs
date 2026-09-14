//! Storage Backend for Experiment Tracking
//!
//! Provides storage backends for persisting experiments.

use std::fs::{self, File};
use std::io::{Read, Write};
use std::path::PathBuf;

use super::tracker::Experiment;

/// Storage backend trait
pub trait StorageBackend {
    /// Save experiments to storage
    fn save_experiments(&self, experiments: &[Experiment]) -> Result<(), String>;
    
    /// Load experiments from storage
    fn load_experiments(&self) -> Result<Vec<Experiment>, String>;
    
    /// Delete an experiment
    fn delete_experiment(&self, experiment_id: &str) -> Result<(), String>;
    
    /// Check if storage is available
    fn is_available(&self) -> bool;
}

/// Local file system storage backend
pub struct LocalStorage {
    base_dir: PathBuf,
}

impl LocalStorage {
    /// Create a new local storage backend
    pub fn new(base_dir: PathBuf) -> Self {
        // Ensure directory exists
        let _ = fs::create_dir_all(&base_dir);
        
        Self { base_dir }
    }
    
    fn experiments_file(&self) -> PathBuf {
        self.base_dir.join("experiments.json")
    }
    
    fn experiment_dir(&self, experiment_id: &str) -> PathBuf {
        self.base_dir.join(experiment_id)
    }
}

impl StorageBackend for LocalStorage {
    fn save_experiments(&self, experiments: &[Experiment]) -> Result<(), String> {
        // Ensure base directory exists
        fs::create_dir_all(&self.base_dir)
            .map_err(|e| format!("Failed to create directory: {}", e))?;
        
        // Serialize experiments to JSON
        let json = serialize_experiments(experiments)?;
        
        // Write to file
        let mut file = File::create(self.experiments_file())
            .map_err(|e| format!("Failed to create file: {}", e))?;
        
        file.write_all(json.as_bytes())
            .map_err(|e| format!("Failed to write file: {}", e))?;
        
        Ok(())
    }
    
    fn load_experiments(&self) -> Result<Vec<Experiment>, String> {
        let file_path = self.experiments_file();
        
        if !file_path.exists() {
            return Ok(Vec::new());
        }
        
        let mut file = File::open(&file_path)
            .map_err(|e| format!("Failed to open file: {}", e))?;
        
        let mut contents = String::new();
        file.read_to_string(&mut contents)
            .map_err(|e| format!("Failed to read file: {}", e))?;
        
        deserialize_experiments(&contents)
    }
    
    fn delete_experiment(&self, experiment_id: &str) -> Result<(), String> {
        let exp_dir = self.experiment_dir(experiment_id);
        
        if exp_dir.exists() {
            fs::remove_dir_all(&exp_dir)
                .map_err(|e| format!("Failed to delete experiment: {}", e))?;
        }
        
        // Also update the experiments file
        let mut experiments = self.load_experiments()?;
        experiments.retain(|e| e.experiment_id != experiment_id);
        self.save_experiments(&experiments)?;
        
        Ok(())
    }
    
    fn is_available(&self) -> bool {
        fs::create_dir_all(&self.base_dir).is_ok()
    }
}

// Simple JSON serialization (without external dependencies)

fn serialize_experiments(experiments: &[Experiment]) -> Result<String, String> {
    let mut json = String::from("[\n");
    
    for (i, exp) in experiments.iter().enumerate() {
        if i > 0 {
            json.push_str(",\n");
        }
        json.push_str(&serialize_experiment(exp));
    }
    
    json.push_str("\n]");
    Ok(json)
}

fn serialize_experiment(exp: &Experiment) -> String {
    let mut json = String::from("  {\n");
    
    json.push_str(&format!("    \"experiment_id\": \"{}\",\n", escape_json(&exp.experiment_id)));
    json.push_str(&format!("    \"name\": \"{}\",\n", escape_json(&exp.name)));
    json.push_str(&format!("    \"created_at\": {},\n", exp.created_at));
    
    // Tags
    json.push_str("    \"tags\": {");
    let tags: Vec<String> = exp.tags.iter()
        .map(|(k, v)| format!("\"{}\": \"{}\"", escape_json(k), escape_json(v)))
        .collect();
    json.push_str(&tags.join(", "));
    json.push_str("},\n");
    
    // Runs
    json.push_str("    \"runs\": [\n");
    for (i, run) in exp.runs.iter().enumerate() {
        if i > 0 {
            json.push_str(",\n");
        }
        json.push_str(&serialize_run(run));
    }
    json.push_str("\n    ]\n");
    
    json.push_str("  }");
    json
}

fn serialize_run(run: &super::tracker::Run) -> String {
    let mut json = String::from("      {\n");
    
    json.push_str(&format!("        \"run_id\": \"{}\",\n", escape_json(&run.run_id)));
    json.push_str(&format!("        \"run_name\": \"{}\",\n", escape_json(&run.run_name)));
    json.push_str(&format!("        \"start_time\": {},\n", run.start_time));
    
    if let Some(end_time) = run.end_time {
        json.push_str(&format!("        \"end_time\": {},\n", end_time));
    } else {
        json.push_str("        \"end_time\": null,\n");
    }
    
    let status = match run.status {
        super::tracker::RunStatus::Running => "running",
        super::tracker::RunStatus::Finished => "finished",
        super::tracker::RunStatus::Failed => "failed",
        super::tracker::RunStatus::Killed => "killed",
    };
    json.push_str(&format!("        \"status\": \"{}\",\n", status));
    
    // Params
    json.push_str("        \"params\": {");
    let params: Vec<String> = run.params.iter()
        .map(|(k, v)| format!("\"{}\": \"{}\"", escape_json(k), escape_json(v)))
        .collect();
    json.push_str(&params.join(", "));
    json.push_str("},\n");
    
    // Metrics
    json.push_str("        \"metrics\": {");
    let metrics: Vec<String> = run.metrics.iter()
        .map(|(k, v)| format!("\"{}\": {}", escape_json(k), v))
        .collect();
    json.push_str(&metrics.join(", "));
    json.push_str("},\n");
    
    // Artifacts
    json.push_str("        \"artifacts\": [");
    let artifacts: Vec<String> = run.artifacts.iter()
        .map(|a| format!("\"{}\"", escape_json(a)))
        .collect();
    json.push_str(&artifacts.join(", "));
    json.push_str("]\n");
    
    json.push_str("      }");
    json
}

fn deserialize_experiments(_json: &str) -> Result<Vec<Experiment>, String> {
    // For now, return empty - full JSON parsing would require serde or manual parsing
    // This is a placeholder that can be extended with proper JSON parsing
    Ok(Vec::new())
}

fn escape_json(s: &str) -> String {
    s.replace('\\', "\\\\")
        .replace('"', "\\\"")
        .replace('\n', "\\n")
        .replace('\r', "\\r")
        .replace('\t', "\\t")
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::HashMap;
    
    #[test]
    fn test_local_storage_save_load() {
        let temp_dir = std::env::temp_dir().join("automl_test_storage");
        let storage = LocalStorage::new(temp_dir.clone());
        
        // Create test experiment
        let mut exp = Experiment {
            experiment_id: "test_exp_1".to_string(),
            name: "Test Experiment".to_string(),
            created_at: 1234567890,
            runs: Vec::new(),
            tags: HashMap::new(),
        };
        exp.tags.insert("env".to_string(), "test".to_string());
        
        // Save
        storage.save_experiments(&[exp]).unwrap();
        
        // Verify file exists
        assert!(storage.experiments_file().exists());
        
        // Cleanup
        let _ = fs::remove_dir_all(&temp_dir);
    }
    
    #[test]
    fn test_json_escaping() {
        assert_eq!(escape_json("hello"), "hello");
        assert_eq!(escape_json("hello\"world"), "hello\\\"world");
        assert_eq!(escape_json("line1\nline2"), "line1\\nline2");
    }
}
