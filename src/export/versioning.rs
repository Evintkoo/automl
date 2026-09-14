//! Model versioning and registry
//!
//! Provides version tracking and model registry for managing
//! multiple versions of trained models.

use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::fs::{self, File};
use std::io::{Read, Write, BufReader, BufWriter};
use std::path::{Path, PathBuf};

use crate::error::{AutoMLError, Result};
use super::serializer::ModelMetadata;

/// Semantic version
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
pub struct ModelVersion {
    pub major: u32,
    pub minor: u32,
    pub patch: u32,
}

impl ModelVersion {
    /// Create new version
    pub fn new(major: u32, minor: u32, patch: u32) -> Self {
        Self { major, minor, patch }
    }

    /// Parse from string (e.g., "1.2.3")
    pub fn parse(s: &str) -> Result<Self> {
        let parts: Vec<&str> = s.split('.').collect();
        if parts.len() != 3 {
            return Err(AutoMLError::ValidationError(
                format!("Invalid version format: {}", s)
            ));
        }

        let major = parts[0].parse().map_err(|_| {
            AutoMLError::ValidationError(format!("Invalid major version: {}", parts[0]))
        })?;
        let minor = parts[1].parse().map_err(|_| {
            AutoMLError::ValidationError(format!("Invalid minor version: {}", parts[1]))
        })?;
        let patch = parts[2].parse().map_err(|_| {
            AutoMLError::ValidationError(format!("Invalid patch version: {}", parts[2]))
        })?;

        Ok(Self { major, minor, patch })
    }

    /// Bump major version
    pub fn bump_major(&self) -> Self {
        Self::new(self.major + 1, 0, 0)
    }

    /// Bump minor version
    pub fn bump_minor(&self) -> Self {
        Self::new(self.major, self.minor + 1, 0)
    }

    /// Bump patch version
    pub fn bump_patch(&self) -> Self {
        Self::new(self.major, self.minor, self.patch + 1)
    }
}

impl std::fmt::Display for ModelVersion {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}.{}.{}", self.major, self.minor, self.patch)
    }
}

impl Default for ModelVersion {
    fn default() -> Self {
        Self::new(1, 0, 0)
    }
}

/// Versioned model wrapper
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct VersionedModel<M> {
    /// Model version
    pub version: ModelVersion,
    /// Model metadata
    pub metadata: ModelMetadata,
    /// The model itself
    pub model: M,
    /// Tags for this version
    pub tags: Vec<String>,
    /// Description of changes
    pub description: String,
}

impl<M: Clone> VersionedModel<M> {
    /// Create new versioned model
    pub fn new(model: M, metadata: ModelMetadata) -> Self {
        Self {
            version: ModelVersion::default(),
            metadata,
            model,
            tags: Vec::new(),
            description: String::new(),
        }
    }

    /// Set version
    pub fn with_version(mut self, version: ModelVersion) -> Self {
        self.version = version;
        self
    }

    /// Add tag
    pub fn with_tag(mut self, tag: impl Into<String>) -> Self {
        self.tags.push(tag.into());
        self
    }

    /// Set description
    pub fn with_description(mut self, desc: impl Into<String>) -> Self {
        self.description = desc.into();
        self
    }

    /// Create next patch version
    pub fn next_patch(&self, model: M) -> Self {
        Self {
            version: self.version.bump_patch(),
            metadata: self.metadata.clone(),
            model,
            tags: Vec::new(),
            description: String::new(),
        }
    }

    /// Create next minor version
    pub fn next_minor(&self, model: M) -> Self {
        Self {
            version: self.version.bump_minor(),
            metadata: self.metadata.clone(),
            model,
            tags: Vec::new(),
            description: String::new(),
        }
    }

    /// Create next major version
    pub fn next_major(&self, model: M) -> Self {
        Self {
            version: self.version.bump_major(),
            metadata: self.metadata.clone(),
            model,
            tags: Vec::new(),
            description: String::new(),
        }
    }
}

/// Model registry entry (metadata only, without model data)
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RegistryEntry {
    /// Model name
    pub name: String,
    /// Model version
    pub version: ModelVersion,
    /// Model metadata
    pub metadata: ModelMetadata,
    /// Tags
    pub tags: Vec<String>,
    /// Description
    pub description: String,
    /// File path relative to registry root
    pub path: String,
    /// Registration timestamp
    pub registered_at: String,
}

/// Registry index
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct RegistryIndex {
    /// All models by name
    pub models: HashMap<String, Vec<RegistryEntry>>,
    /// Tag index
    pub tags: HashMap<String, Vec<(String, ModelVersion)>>,
}

/// Model registry for managing versioned models
pub struct ModelRegistry {
    /// Root directory
    root: PathBuf,
    /// Registry index
    index: RegistryIndex,
}

impl ModelRegistry {
    /// Create or open registry at path
    pub fn open(path: impl AsRef<Path>) -> Result<Self> {
        let root = path.as_ref().to_path_buf();
        
        // Create directory if needed
        if !root.exists() {
            fs::create_dir_all(&root).map_err(|e| {
                AutoMLError::DataError(format!("Failed to create registry: {}", e))
            })?;
        }

        // Load or create index
        let index_path = root.join("index.json");
        let index = if index_path.exists() {
            let file = File::open(&index_path).map_err(|e| {
                AutoMLError::DataError(format!("Failed to open index: {}", e))
            })?;
            serde_json::from_reader(BufReader::new(file)).map_err(|e| {
                AutoMLError::SerializationError(format!("Failed to read index: {}", e))
            })?
        } else {
            RegistryIndex::default()
        };

        Ok(Self { root, index })
    }

    /// Save registry index
    fn save_index(&self) -> Result<()> {
        let index_path = self.root.join("index.json");
        let file = File::create(&index_path).map_err(|e| {
            AutoMLError::DataError(format!("Failed to create index: {}", e))
        })?;
        
        serde_json::to_writer_pretty(BufWriter::new(file), &self.index).map_err(|e| {
            AutoMLError::SerializationError(format!("Failed to write index: {}", e))
        })?;

        Ok(())
    }

    /// Register a model
    pub fn register<M: Serialize>(
        &mut self,
        name: &str,
        versioned: &VersionedModel<M>,
    ) -> Result<String> {
        // Create model directory
        let model_dir = self.root.join(name);
        if !model_dir.exists() {
            fs::create_dir_all(&model_dir).map_err(|e| {
                AutoMLError::DataError(format!("Failed to create model dir: {}", e))
            })?;
        }

        // Create version file path
        let file_name = format!("v{}.bin", versioned.version);
        let file_path = model_dir.join(&file_name);
        let relative_path = format!("{}/{}", name, file_name);

        // Serialize model
        let bytes = bincode::serialize(versioned).map_err(|e| {
            AutoMLError::SerializationError(format!("Failed to serialize: {}", e))
        })?;

        let mut file = File::create(&file_path).map_err(|e| {
            AutoMLError::DataError(format!("Failed to create model file: {}", e))
        })?;
        file.write_all(&bytes).map_err(|e| {
            AutoMLError::DataError(format!("Failed to write model: {}", e))
        })?;

        // Create registry entry
        let entry = RegistryEntry {
            name: name.to_string(),
            version: versioned.version.clone(),
            metadata: versioned.metadata.clone(),
            tags: versioned.tags.clone(),
            description: versioned.description.clone(),
            path: relative_path.clone(),
            registered_at: String::new(),
        };

        // Update index
        self.index.models
            .entry(name.to_string())
            .or_default()
            .push(entry);

        // Update tag index
        for tag in &versioned.tags {
            self.index.tags
                .entry(tag.clone())
                .or_default()
                .push((name.to_string(), versioned.version.clone()));
        }

        self.save_index()?;

        Ok(relative_path)
    }

    /// Get latest version of a model
    pub fn get_latest<M: for<'de> Deserialize<'de>>(
        &self,
        name: &str,
    ) -> Result<VersionedModel<M>> {
        let entries = self.index.models.get(name).ok_or_else(|| {
            AutoMLError::DataError(format!("Model not found: {}", name))
        })?;

        let latest = entries.iter().max_by(|a, b| a.version.cmp(&b.version))
            .ok_or_else(|| {
                AutoMLError::DataError(format!("No versions found: {}", name))
            })?;

        self.load(&latest.path)
    }

    /// Get specific version of a model
    pub fn get_version<M: for<'de> Deserialize<'de>>(
        &self,
        name: &str,
        version: &ModelVersion,
    ) -> Result<VersionedModel<M>> {
        let entries = self.index.models.get(name).ok_or_else(|| {
            AutoMLError::DataError(format!("Model not found: {}", name))
        })?;

        let entry = entries.iter().find(|e| &e.version == version)
            .ok_or_else(|| {
                AutoMLError::DataError(format!("Version not found: {}", version))
            })?;

        self.load(&entry.path)
    }

    /// Load model from path
    fn load<M: for<'de> Deserialize<'de>>(&self, rel_path: &str) -> Result<VersionedModel<M>> {
        let path = self.root.join(rel_path);
        let mut file = File::open(&path).map_err(|e| {
            AutoMLError::DataError(format!("Failed to open model: {}", e))
        })?;

        let mut bytes = Vec::new();
        file.read_to_end(&mut bytes).map_err(|e| {
            AutoMLError::DataError(format!("Failed to read model: {}", e))
        })?;

        bincode::deserialize(&bytes).map_err(|e| {
            AutoMLError::SerializationError(format!("Failed to deserialize: {}", e))
        })
    }

    /// List all models
    pub fn list_models(&self) -> Vec<String> {
        self.index.models.keys().cloned().collect()
    }

    /// List versions of a model
    pub fn list_versions(&self, name: &str) -> Vec<ModelVersion> {
        self.index.models.get(name)
            .map(|entries| entries.iter().map(|e| e.version.clone()).collect())
            .unwrap_or_default()
    }

    /// Find models by tag
    pub fn find_by_tag(&self, tag: &str) -> Vec<(String, ModelVersion)> {
        self.index.tags.get(tag).cloned().unwrap_or_default()
    }

    /// Delete a model version
    pub fn delete(&mut self, name: &str, version: &ModelVersion) -> Result<()> {
        let entries = self.index.models.get_mut(name).ok_or_else(|| {
            AutoMLError::DataError(format!("Model not found: {}", name))
        })?;

        let idx = entries.iter().position(|e| &e.version == version)
            .ok_or_else(|| {
                AutoMLError::DataError(format!("Version not found: {}", version))
            })?;

        let entry = entries.remove(idx);

        // Remove file
        let path = self.root.join(&entry.path);
        if path.exists() {
            fs::remove_file(&path).map_err(|e| {
                AutoMLError::DataError(format!("Failed to delete file: {}", e))
            })?;
        }

        // Update tag index
        for tag in &entry.tags {
            if let Some(tagged) = self.index.tags.get_mut(tag) {
                tagged.retain(|(n, v)| n != name || v != version);
            }
        }

        // Clean up empty entries
        if entries.is_empty() {
            self.index.models.remove(name);
        }

        self.save_index()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_version_parsing() {
        let v = ModelVersion::parse("1.2.3").unwrap();
        assert_eq!(v.major, 1);
        assert_eq!(v.minor, 2);
        assert_eq!(v.patch, 3);
    }

    #[test]
    fn test_version_bumping() {
        let v = ModelVersion::new(1, 2, 3);
        
        let patch = v.bump_patch();
        assert_eq!(patch.to_string(), "1.2.4");
        
        let minor = v.bump_minor();
        assert_eq!(minor.to_string(), "1.3.0");
        
        let major = v.bump_major();
        assert_eq!(major.to_string(), "2.0.0");
    }

    #[test]
    fn test_version_ordering() {
        let v1 = ModelVersion::new(1, 0, 0);
        let v2 = ModelVersion::new(1, 1, 0);
        let v3 = ModelVersion::new(2, 0, 0);
        
        assert!(v1 < v2);
        assert!(v2 < v3);
        assert!(v1 < v3);
    }

    #[test]
    fn test_versioned_model() {
        #[derive(Clone, Serialize, Deserialize)]
        struct DummyModel {
            value: i32,
        }

        let model = DummyModel { value: 42 };
        let metadata = ModelMetadata::new("test");
        
        let versioned = VersionedModel::new(model, metadata)
            .with_tag("production")
            .with_description("Initial version");
        
        assert_eq!(versioned.version, ModelVersion::new(1, 0, 0));
        assert_eq!(versioned.tags, vec!["production"]);
    }
}
