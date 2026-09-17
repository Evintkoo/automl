//! Storage Backend for Experiment Tracking
//!
//! Provides storage backends for persisting experiments.

use std::collections::HashMap;
use std::fs::{self, File};
use std::io::{Read, Write};
use std::path::PathBuf;
use std::sync::Mutex;

use super::tracker::{Experiment, Run, RunStatus};

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
    /// Serializes the full load -> mutate -> save sequence across
    /// `save_experiments`/`load_experiments`/`delete_experiment`, so
    /// concurrent callers can't race on `experiments.json` and silently
    /// clobber each other's changes (last-writer-wins). Held for the
    /// duration of each call via the private `read_experiments_file`/
    /// `write_experiments_file` helpers, which do the raw I/O without
    /// locking themselves - so `delete_experiment` can hold the lock across
    /// its own read-modify-write without deadlocking.
    lock: Mutex<()>,
}

impl LocalStorage {
    /// Create a new local storage backend
    pub fn new(base_dir: PathBuf) -> Self {
        // Ensure directory exists
        let _ = fs::create_dir_all(&base_dir);

        Self { base_dir, lock: Mutex::new(()) }
    }

    fn experiments_file(&self) -> PathBuf {
        self.base_dir.join("experiments.json")
    }

    fn experiments_tmp_file(&self) -> PathBuf {
        self.base_dir.join("experiments.json.tmp")
    }

    fn experiment_dir(&self, experiment_id: &str) -> PathBuf {
        self.base_dir.join(experiment_id)
    }

    /// Write `experiments.json` atomically: serialize to a temp file in the
    /// same directory, flush it, then `rename` it over the live path.
    /// `rename` is atomic on the same filesystem, so a crash mid-write never
    /// leaves readers observing a truncated/corrupt `experiments.json` - the
    /// live file is either the old complete version or the new complete
    /// version, never a partial one. Does not itself take `self.lock`, so it
    /// can be called by callers (like `delete_experiment`) that already hold
    /// it as part of a larger read-modify-write.
    fn write_experiments_file(&self, experiments: &[Experiment]) -> Result<(), String> {
        // Ensure base directory exists
        fs::create_dir_all(&self.base_dir)
            .map_err(|e| format!("Failed to create directory: {}", e))?;

        // Serialize experiments to JSON
        let json = serialize_experiments(experiments)?;

        let tmp_path = self.experiments_tmp_file();
        {
            let mut file = File::create(&tmp_path)
                .map_err(|e| format!("Failed to create temp file: {}", e))?;

            file.write_all(json.as_bytes())
                .map_err(|e| format!("Failed to write temp file: {}", e))?;

            file.sync_all()
                .map_err(|e| format!("Failed to flush temp file: {}", e))?;
        }

        fs::rename(&tmp_path, self.experiments_file())
            .map_err(|e| format!("Failed to atomically replace experiments file: {}", e))?;

        Ok(())
    }

    /// Read and parse `experiments.json`. Does not itself take `self.lock`,
    /// for the same reason as `write_experiments_file`.
    fn read_experiments_file(&self) -> Result<Vec<Experiment>, String> {
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
}

impl StorageBackend for LocalStorage {
    fn save_experiments(&self, experiments: &[Experiment]) -> Result<(), String> {
        let _guard = self.lock.lock().map_err(|_| "Storage lock poisoned".to_string())?;
        self.write_experiments_file(experiments)
    }

    fn load_experiments(&self) -> Result<Vec<Experiment>, String> {
        let _guard = self.lock.lock().map_err(|_| "Storage lock poisoned".to_string())?;
        self.read_experiments_file()
    }

    fn delete_experiment(&self, experiment_id: &str) -> Result<(), String> {
        // Hold the lock across the entire delete-dir + read-modify-write of
        // experiments.json, so a concurrent save_experiments()/
        // delete_experiment() call serializes instead of racing with this
        // one and silently losing changes.
        let _guard = self.lock.lock().map_err(|_| "Storage lock poisoned".to_string())?;

        let exp_dir = self.experiment_dir(experiment_id);

        if exp_dir.exists() {
            fs::remove_dir_all(&exp_dir)
                .map_err(|e| format!("Failed to delete experiment: {}", e))?;
        }

        // Also update the experiments file
        let mut experiments = self.read_experiments_file()?;
        experiments.retain(|e| e.experiment_id != experiment_id);
        self.write_experiments_file(&experiments)?;

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

    // Metrics history (per-step values) - previously dropped on persist,
    // silently losing all step-by-step history across a save/load cycle.
    json.push_str("        \"metrics_history\": [");
    let metrics_history: Vec<String> = run.metrics_history.iter()
        .map(|m| format!(
            "{{\"name\": \"{}\", \"value\": {}, \"step\": {}, \"timestamp\": {}}}",
            escape_json(&m.name), m.value, m.step, m.timestamp
        ))
        .collect();
    json.push_str(&metrics_history.join(", "));
    json.push_str("],\n");

    // Tags - previously dropped on persist, same issue as metrics_history.
    json.push_str("        \"tags\": {");
    let tags: Vec<String> = run.tags.iter()
        .map(|(k, v)| format!("\"{}\": \"{}\"", escape_json(k), escape_json(v)))
        .collect();
    json.push_str(&tags.join(", "));
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

fn deserialize_experiments(json: &str) -> Result<Vec<Experiment>, String> {
    let value = parse_json(json)?;
    let items = match value {
        JsonValue::Array(items) => items,
        other => return Err(format!("Expected top-level JSON array of experiments, found {:?}", other)),
    };

    items.into_iter().map(parse_experiment).collect()
}

fn parse_experiment(value: JsonValue) -> Result<Experiment, String> {
    let obj = as_object(value)?;

    let experiment_id = get_string(&obj, "experiment_id")?;
    let name = get_string(&obj, "name")?;
    let created_at = get_u64(&obj, "created_at")?;
    let tags = get_string_map(&obj, "tags")?;
    let runs = get_array(&obj, "runs")?
        .into_iter()
        .map(parse_run)
        .collect::<Result<Vec<_>, _>>()?;

    Ok(Experiment {
        experiment_id,
        name,
        created_at,
        runs,
        tags,
    })
}

fn parse_run(value: JsonValue) -> Result<Run, String> {
    let obj = as_object(value)?;

    let run_id = get_string(&obj, "run_id")?;
    let run_name = get_string(&obj, "run_name")?;
    let start_time = get_u64(&obj, "start_time")?;
    let end_time = match get_field(&obj, "end_time")? {
        JsonValue::Null => None,
        JsonValue::Number(n) => Some(n as u64),
        other => return Err(format!("Field 'end_time' expected number or null, found {:?}", other)),
    };

    let status_str = get_string(&obj, "status")?;
    let status = match status_str.as_str() {
        "running" => RunStatus::Running,
        "finished" => RunStatus::Finished,
        "failed" => RunStatus::Failed,
        "killed" => RunStatus::Killed,
        other => return Err(format!("Unknown run status: '{}'", other)),
    };

    let params = get_string_map(&obj, "params")?;
    let metrics = get_f64_map(&obj, "metrics")?;
    let metrics_history = get_array(&obj, "metrics_history")?
        .into_iter()
        .map(parse_metric)
        .collect::<Result<Vec<_>, _>>()?;
    let tags = get_string_map(&obj, "tags")?;
    let artifacts = get_array(&obj, "artifacts")?
        .into_iter()
        .map(|v| match v {
            JsonValue::String(s) => Ok(s),
            other => Err(format!("Expected string artifact entry, found {:?}", other)),
        })
        .collect::<Result<Vec<_>, _>>()?;

    Ok(Run {
        run_id,
        run_name,
        start_time,
        end_time,
        params,
        metrics,
        metrics_history,
        tags,
        artifacts,
        status,
    })
}

fn parse_metric(value: JsonValue) -> Result<super::tracker::Metric, String> {
    let obj = as_object(value)?;

    Ok(super::tracker::Metric {
        name: get_string(&obj, "name")?,
        value: get_f64(&obj, "value")?,
        step: get_u64(&obj, "step")?,
        timestamp: get_u64(&obj, "timestamp")?,
    })
}

// Minimal JSON parser matching the hand-rolled format produced by
// `serialize_experiments`/`serialize_experiment`/`serialize_run` above.
// Kept dependency-free and local to this module rather than pulling in serde
// derives for `Experiment`/`Run`, since those types are serialized by hand.

#[derive(Debug, Clone)]
enum JsonValue {
    Null,
    Bool(bool),
    Number(f64),
    String(String),
    Array(Vec<JsonValue>),
    Object(Vec<(String, JsonValue)>),
}

struct JsonParser {
    chars: Vec<char>,
    pos: usize,
}

impl JsonParser {
    fn new(input: &str) -> Self {
        Self {
            chars: input.chars().collect(),
            pos: 0,
        }
    }

    fn peek(&self) -> Option<char> {
        self.chars.get(self.pos).copied()
    }

    fn advance(&mut self) -> Option<char> {
        let c = self.peek();
        if c.is_some() {
            self.pos += 1;
        }
        c
    }

    fn skip_whitespace(&mut self) {
        while let Some(c) = self.peek() {
            if c.is_whitespace() {
                self.pos += 1;
            } else {
                break;
            }
        }
    }

    fn expect(&mut self, expected: char) -> Result<(), String> {
        match self.advance() {
            Some(c) if c == expected => Ok(()),
            other => Err(format!("Expected '{}', found {:?}", expected, other)),
        }
    }

    fn parse_value(&mut self) -> Result<JsonValue, String> {
        self.skip_whitespace();
        match self.peek() {
            Some('{') => self.parse_object(),
            Some('[') => self.parse_array(),
            Some('"') => self.parse_string().map(JsonValue::String),
            Some('t') | Some('f') => self.parse_bool(),
            Some('n') => self.parse_null(),
            Some(c) if c == '-' || c.is_ascii_digit() => self.parse_number(),
            other => Err(format!("Unexpected character in JSON: {:?}", other)),
        }
    }

    fn parse_object(&mut self) -> Result<JsonValue, String> {
        self.expect('{')?;
        let mut entries = Vec::new();
        self.skip_whitespace();
        if self.peek() == Some('}') {
            self.advance();
            return Ok(JsonValue::Object(entries));
        }
        loop {
            self.skip_whitespace();
            let key = self.parse_string()?;
            self.skip_whitespace();
            self.expect(':')?;
            let value = self.parse_value()?;
            entries.push((key, value));
            self.skip_whitespace();
            match self.advance() {
                Some(',') => continue,
                Some('}') => break,
                other => return Err(format!("Expected ',' or '}}' in object, found {:?}", other)),
            }
        }
        Ok(JsonValue::Object(entries))
    }

    fn parse_array(&mut self) -> Result<JsonValue, String> {
        self.expect('[')?;
        let mut items = Vec::new();
        self.skip_whitespace();
        if self.peek() == Some(']') {
            self.advance();
            return Ok(JsonValue::Array(items));
        }
        loop {
            let value = self.parse_value()?;
            items.push(value);
            self.skip_whitespace();
            match self.advance() {
                Some(',') => continue,
                Some(']') => break,
                other => return Err(format!("Expected ',' or ']' in array, found {:?}", other)),
            }
        }
        Ok(JsonValue::Array(items))
    }

    fn parse_string(&mut self) -> Result<String, String> {
        self.skip_whitespace();
        self.expect('"')?;
        let mut s = String::new();
        loop {
            match self.advance() {
                Some('"') => break,
                Some('\\') => match self.advance() {
                    Some('"') => s.push('"'),
                    Some('\\') => s.push('\\'),
                    Some('/') => s.push('/'),
                    Some('n') => s.push('\n'),
                    Some('r') => s.push('\r'),
                    Some('t') => s.push('\t'),
                    Some('u') => {
                        let mut code = 0u32;
                        for _ in 0..4 {
                            let c = self.advance().ok_or("Unexpected end of input in unicode escape")?;
                            let digit = c.to_digit(16).ok_or("Invalid unicode escape digit")?;
                            code = code * 16 + digit;
                        }
                        if let Some(ch) = char::from_u32(code) {
                            s.push(ch);
                        }
                    }
                    other => return Err(format!("Invalid escape sequence: {:?}", other)),
                },
                Some(c) => s.push(c),
                None => return Err("Unterminated string in JSON".to_string()),
            }
        }
        Ok(s)
    }

    fn parse_bool(&mut self) -> Result<JsonValue, String> {
        if self.chars[self.pos..].starts_with(&['t', 'r', 'u', 'e']) {
            self.pos += 4;
            Ok(JsonValue::Bool(true))
        } else if self.chars[self.pos..].starts_with(&['f', 'a', 'l', 's', 'e']) {
            self.pos += 5;
            Ok(JsonValue::Bool(false))
        } else {
            Err("Invalid literal, expected 'true' or 'false'".to_string())
        }
    }

    fn parse_null(&mut self) -> Result<JsonValue, String> {
        if self.chars[self.pos..].starts_with(&['n', 'u', 'l', 'l']) {
            self.pos += 4;
            Ok(JsonValue::Null)
        } else {
            Err("Invalid literal, expected 'null'".to_string())
        }
    }

    fn parse_number(&mut self) -> Result<JsonValue, String> {
        let start = self.pos;
        if self.peek() == Some('-') {
            self.pos += 1;
        }
        while let Some(c) = self.peek() {
            if c.is_ascii_digit() {
                self.pos += 1;
            } else {
                break;
            }
        }
        if self.peek() == Some('.') {
            self.pos += 1;
            while let Some(c) = self.peek() {
                if c.is_ascii_digit() {
                    self.pos += 1;
                } else {
                    break;
                }
            }
        }
        if matches!(self.peek(), Some('e') | Some('E')) {
            self.pos += 1;
            if matches!(self.peek(), Some('+') | Some('-')) {
                self.pos += 1;
            }
            while let Some(c) = self.peek() {
                if c.is_ascii_digit() {
                    self.pos += 1;
                } else {
                    break;
                }
            }
        }
        let s: String = self.chars[start..self.pos].iter().collect();
        s.parse::<f64>()
            .map(JsonValue::Number)
            .map_err(|e| format!("Invalid number '{}': {}", s, e))
    }
}

fn parse_json(input: &str) -> Result<JsonValue, String> {
    let mut parser = JsonParser::new(input);
    let value = parser.parse_value()?;
    parser.skip_whitespace();
    Ok(value)
}

fn as_object(value: JsonValue) -> Result<Vec<(String, JsonValue)>, String> {
    match value {
        JsonValue::Object(entries) => Ok(entries),
        other => Err(format!("Expected JSON object, found {:?}", other)),
    }
}

fn get_field(obj: &[(String, JsonValue)], key: &str) -> Result<JsonValue, String> {
    obj.iter()
        .find(|(k, _)| k == key)
        .map(|(_, v)| v.clone())
        .ok_or_else(|| format!("Missing field '{}'", key))
}

fn get_string(obj: &[(String, JsonValue)], key: &str) -> Result<String, String> {
    match get_field(obj, key)? {
        JsonValue::String(s) => Ok(s),
        other => Err(format!("Field '{}' expected string, found {:?}", key, other)),
    }
}

fn get_u64(obj: &[(String, JsonValue)], key: &str) -> Result<u64, String> {
    match get_field(obj, key)? {
        JsonValue::Number(n) => Ok(n as u64),
        other => Err(format!("Field '{}' expected number, found {:?}", key, other)),
    }
}

fn get_f64(obj: &[(String, JsonValue)], key: &str) -> Result<f64, String> {
    match get_field(obj, key)? {
        JsonValue::Number(n) => Ok(n),
        other => Err(format!("Field '{}' expected number, found {:?}", key, other)),
    }
}

fn get_array(obj: &[(String, JsonValue)], key: &str) -> Result<Vec<JsonValue>, String> {
    match get_field(obj, key)? {
        JsonValue::Array(items) => Ok(items),
        other => Err(format!("Field '{}' expected array, found {:?}", key, other)),
    }
}

fn get_string_map(obj: &[(String, JsonValue)], key: &str) -> Result<HashMap<String, String>, String> {
    match get_field(obj, key)? {
        JsonValue::Object(entries) => {
            let mut map = HashMap::new();
            for (k, v) in entries {
                match v {
                    JsonValue::String(s) => {
                        map.insert(k, s);
                    }
                    other => return Err(format!("Field '{}' expected string values, found {:?}", key, other)),
                }
            }
            Ok(map)
        }
        other => Err(format!("Field '{}' expected object, found {:?}", key, other)),
    }
}

fn get_f64_map(obj: &[(String, JsonValue)], key: &str) -> Result<HashMap<String, f64>, String> {
    match get_field(obj, key)? {
        JsonValue::Object(entries) => {
            let mut map = HashMap::new();
            for (k, v) in entries {
                match v {
                    JsonValue::Number(n) => {
                        map.insert(k, n);
                    }
                    other => return Err(format!("Field '{}' expected number values, found {:?}", key, other)),
                }
            }
            Ok(map)
        }
        other => Err(format!("Field '{}' expected object, found {:?}", key, other)),
    }
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
