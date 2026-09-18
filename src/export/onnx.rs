//! ONNX export functionality
//!
//! Provides export of trained models to ONNX format for interoperability
//! with other ML frameworks and deployment platforms.

use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::fs::File;
use std::io::{Write, BufWriter};
use std::path::Path;

use crate::error::{AutoMLError, Result};

/// ONNX configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ONNXConfig {
    /// ONNX opset version
    pub opset_version: i64,
    /// Producer name
    pub producer_name: String,
    /// Producer version
    pub producer_version: String,
    /// Model description
    pub description: String,
}

impl Default for ONNXConfig {
    fn default() -> Self {
        Self {
            opset_version: 15,
            producer_name: "AutoML".to_string(),
            producer_version: env!("CARGO_PKG_VERSION").to_string(),
            description: String::new(),
        }
    }
}

/// ONNX data types
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ONNXDataType {
    Float = 1,
    Double = 11,
    Int32 = 6,
    Int64 = 7,
    String = 8,
    Bool = 9,
}

/// ONNX tensor shape dimension
#[derive(Debug, Clone)]
pub enum Dimension {
    /// Fixed size dimension
    Fixed(i64),
    /// Dynamic dimension with name
    Dynamic(String),
}

/// ONNX tensor specification
#[derive(Debug, Clone)]
pub struct TensorSpec {
    /// Tensor name
    pub name: String,
    /// Data type
    pub dtype: ONNXDataType,
    /// Shape dimensions
    pub shape: Vec<Dimension>,
}

impl TensorSpec {
    /// Create new tensor spec
    pub fn new(name: impl Into<String>, dtype: ONNXDataType, shape: Vec<Dimension>) -> Self {
        Self {
            name: name.into(),
            dtype,
            shape,
        }
    }

    /// Create float tensor with given shape
    pub fn float(name: impl Into<String>, shape: Vec<Dimension>) -> Self {
        Self::new(name, ONNXDataType::Float, shape)
    }

    /// Create int64 tensor with given shape
    pub fn int64(name: impl Into<String>, shape: Vec<Dimension>) -> Self {
        Self::new(name, ONNXDataType::Int64, shape)
    }
}

/// ONNX operator node
#[derive(Debug, Clone)]
pub struct ONNXNode {
    /// Node name
    pub name: String,
    /// Operator type (e.g., "MatMul", "Add", "Relu")
    pub op_type: String,
    /// Input tensor names
    pub inputs: Vec<String>,
    /// Output tensor names
    pub outputs: Vec<String>,
    /// Attributes
    pub attributes: HashMap<String, ONNXAttribute>,
}

impl ONNXNode {
    /// Create new node
    pub fn new(
        name: impl Into<String>,
        op_type: impl Into<String>,
        inputs: Vec<String>,
        outputs: Vec<String>,
    ) -> Self {
        Self {
            name: name.into(),
            op_type: op_type.into(),
            inputs,
            outputs,
            attributes: HashMap::new(),
        }
    }

    /// Add attribute
    pub fn with_attribute(mut self, key: impl Into<String>, value: ONNXAttribute) -> Self {
        self.attributes.insert(key.into(), value);
        self
    }
}

/// ONNX attribute value
#[derive(Debug, Clone)]
pub enum ONNXAttribute {
    Int(i64),
    Float(f64),
    String(String),
    Ints(Vec<i64>),
    Floats(Vec<f64>),
    Strings(Vec<String>),
}

/// ONNX initializer (constant tensor)
#[derive(Debug, Clone)]
pub struct ONNXInitializer {
    /// Tensor name
    pub name: String,
    /// Data type
    pub dtype: ONNXDataType,
    /// Shape
    pub dims: Vec<i64>,
    /// Raw data
    pub data: InitializerData,
}

/// Initializer data variants
#[derive(Debug, Clone)]
pub enum InitializerData {
    Float(Vec<f32>),
    Double(Vec<f64>),
    Int32(Vec<i32>),
    Int64(Vec<i64>),
}

impl ONNXInitializer {
    /// Create float initializer
    pub fn float(name: impl Into<String>, dims: Vec<i64>, data: Vec<f32>) -> Self {
        Self {
            name: name.into(),
            dtype: ONNXDataType::Float,
            dims,
            data: InitializerData::Float(data),
        }
    }

    /// Create double initializer
    pub fn double(name: impl Into<String>, dims: Vec<i64>, data: Vec<f64>) -> Self {
        Self {
            name: name.into(),
            dtype: ONNXDataType::Double,
            dims,
            data: InitializerData::Double(data),
        }
    }
}

/// ONNX graph representation
#[derive(Debug, Clone)]
pub struct ONNXGraph {
    /// Graph name
    pub name: String,
    /// Input tensors
    pub inputs: Vec<TensorSpec>,
    /// Output tensors
    pub outputs: Vec<TensorSpec>,
    /// Nodes
    pub nodes: Vec<ONNXNode>,
    /// Initializers (weights, biases)
    pub initializers: Vec<ONNXInitializer>,
}

impl ONNXGraph {
    /// Create new graph
    pub fn new(name: impl Into<String>) -> Self {
        Self {
            name: name.into(),
            inputs: Vec::new(),
            outputs: Vec::new(),
            nodes: Vec::new(),
            initializers: Vec::new(),
        }
    }

    /// Add input
    pub fn add_input(mut self, spec: TensorSpec) -> Self {
        self.inputs.push(spec);
        self
    }

    /// Add output
    pub fn add_output(mut self, spec: TensorSpec) -> Self {
        self.outputs.push(spec);
        self
    }

    /// Add node
    pub fn add_node(mut self, node: ONNXNode) -> Self {
        self.nodes.push(node);
        self
    }

    /// Add initializer
    pub fn add_initializer(mut self, init: ONNXInitializer) -> Self {
        self.initializers.push(init);
        self
    }
}

/// Trait for models that can be exported to ONNX
pub trait ONNXExportable {
    /// Convert model to ONNX graph
    fn to_onnx_graph(&self) -> Result<ONNXGraph>;
}

/// ONNX model exporter
pub struct ONNXExporter {
    config: ONNXConfig,
}

impl ONNXExporter {
    /// Create new exporter with default config
    pub fn new() -> Self {
        Self {
            config: ONNXConfig::default(),
        }
    }

    /// Create with custom config
    pub fn with_config(config: ONNXConfig) -> Self {
        Self { config }
    }

    /// Export model to ONNX JSON representation
    /// (Full binary ONNX export would require protobuf)
    pub fn export_json<M: ONNXExportable>(
        &self,
        model: &M,
        path: impl AsRef<Path>,
    ) -> Result<()> {
        let graph = model.to_onnx_graph()?;
        
        #[derive(Serialize)]
        struct ONNXModel {
            ir_version: i64,
            opset_import: Vec<OpsetImport>,
            producer_name: String,
            producer_version: String,
            model_version: i64,
            graph: GraphJson,
        }

        #[derive(Serialize)]
        struct OpsetImport {
            domain: String,
            version: i64,
        }

        #[derive(Serialize)]
        struct GraphJson {
            name: String,
            inputs: Vec<TensorSpecJson>,
            outputs: Vec<TensorSpecJson>,
            nodes: Vec<NodeJson>,
            initializers: Vec<InitializerJson>,
        }

        #[derive(Serialize)]
        struct TensorSpecJson {
            name: String,
            dtype: i32,
            shape: Vec<String>,
        }

        #[derive(Serialize)]
        struct NodeJson {
            name: String,
            op_type: String,
            inputs: Vec<String>,
            outputs: Vec<String>,
            attributes: HashMap<String, serde_json::Value>,
        }

        #[derive(Serialize)]
        struct InitializerJson {
            name: String,
            dtype: i32,
            dims: Vec<i64>,
            data_location: String,
        }

        let onnx_model = ONNXModel {
            ir_version: 8,
            opset_import: vec![OpsetImport {
                domain: String::new(),
                version: self.config.opset_version,
            }],
            producer_name: self.config.producer_name.clone(),
            producer_version: self.config.producer_version.clone(),
            model_version: 1,
            graph: GraphJson {
                name: graph.name,
                inputs: graph.inputs.iter().map(|i| TensorSpecJson {
                    name: i.name.clone(),
                    dtype: i.dtype as i32,
                    shape: i.shape.iter().map(|d| match d {
                        Dimension::Fixed(n) => n.to_string(),
                        Dimension::Dynamic(s) => s.clone(),
                    }).collect(),
                }).collect(),
                outputs: graph.outputs.iter().map(|o| TensorSpecJson {
                    name: o.name.clone(),
                    dtype: o.dtype as i32,
                    shape: o.shape.iter().map(|d| match d {
                        Dimension::Fixed(n) => n.to_string(),
                        Dimension::Dynamic(s) => s.clone(),
                    }).collect(),
                }).collect(),
                nodes: graph.nodes.iter().map(|n| NodeJson {
                    name: n.name.clone(),
                    op_type: n.op_type.clone(),
                    inputs: n.inputs.clone(),
                    outputs: n.outputs.clone(),
                    attributes: n.attributes.iter().map(|(k, v)| {
                        let json_val = match v {
                            ONNXAttribute::Int(i) => serde_json::json!(i),
                            ONNXAttribute::Float(f) => serde_json::json!(f),
                            ONNXAttribute::String(s) => serde_json::json!(s),
                            ONNXAttribute::Ints(is) => serde_json::json!(is),
                            ONNXAttribute::Floats(fs) => serde_json::json!(fs),
                            ONNXAttribute::Strings(ss) => serde_json::json!(ss),
                        };
                        (k.clone(), json_val)
                    }).collect(),
                }).collect(),
                initializers: graph.initializers.iter().map(|i| InitializerJson {
                    name: i.name.clone(),
                    dtype: i.dtype as i32,
                    dims: i.dims.clone(),
                    data_location: "external".to_string(),
                }).collect(),
            },
        };

        let file = File::create(path.as_ref()).map_err(|e| {
            AutoMLError::DataError(format!("Failed to create file: {}", e))
        })?;
        let mut writer = BufWriter::new(file);

        serde_json::to_writer_pretty(&mut writer, &onnx_model).map_err(|e| {
            AutoMLError::SerializationError(format!("Failed to write ONNX JSON: {}", e))
        })?;

        // Ensure buffered bytes actually reach the OS before reporting success;
        // BufWriter's Drop impl flushes but silently discards any I/O error.
        writer.flush().map_err(|e| {
            AutoMLError::DataError(format!("Failed to flush file: {}", e))
        })?;

        Ok(())
    }

    /// Export initializer data to separate file
    pub fn export_weights(
        &self,
        graph: &ONNXGraph,
        path: impl AsRef<Path>,
    ) -> Result<()> {
        let file = File::create(path.as_ref()).map_err(|e| {
            AutoMLError::DataError(format!("Failed to create weights file: {}", e))
        })?;
        let mut writer = BufWriter::new(file);

        // Build one contiguous byte buffer per initializer and issue a
        // single `write_all` instead of one per value — each call previously
        // paid its own function-call/bounds-check overhead independent of
        // BufWriter's internal buffering, which adds up for models with
        // millions of weights.
        for init in &graph.initializers {
            match &init.data {
                InitializerData::Float(data) => {
                    let mut buf = Vec::with_capacity(data.len() * 4);
                    for &val in data {
                        buf.extend_from_slice(&val.to_le_bytes());
                    }
                    writer.write_all(&buf).map_err(|e| {
                        AutoMLError::DataError(format!("Failed to write: {}", e))
                    })?;
                }
                InitializerData::Double(data) => {
                    let mut buf = Vec::with_capacity(data.len() * 8);
                    for &val in data {
                        buf.extend_from_slice(&val.to_le_bytes());
                    }
                    writer.write_all(&buf).map_err(|e| {
                        AutoMLError::DataError(format!("Failed to write: {}", e))
                    })?;
                }
                InitializerData::Int32(data) => {
                    let mut buf = Vec::with_capacity(data.len() * 4);
                    for &val in data {
                        buf.extend_from_slice(&val.to_le_bytes());
                    }
                    writer.write_all(&buf).map_err(|e| {
                        AutoMLError::DataError(format!("Failed to write: {}", e))
                    })?;
                }
                InitializerData::Int64(data) => {
                    let mut buf = Vec::with_capacity(data.len() * 8);
                    for &val in data {
                        buf.extend_from_slice(&val.to_le_bytes());
                    }
                    writer.write_all(&buf).map_err(|e| {
                        AutoMLError::DataError(format!("Failed to write: {}", e))
                    })?;
                }
            }
        }

        // Ensure buffered bytes actually reach the OS before reporting success;
        // BufWriter's Drop impl flushes but silently discards any I/O error.
        writer.flush().map_err(|e| {
            AutoMLError::DataError(format!("Failed to flush file: {}", e))
        })?;

        Ok(())
    }
}

impl Default for ONNXExporter {
    fn default() -> Self {
        Self::new()
    }
}

/// Helper to create linear regression ONNX graph
#[allow(dead_code)]
pub fn linear_regression_to_onnx(
    weights: &[f64],
    bias: f64,
    n_features: usize,
) -> ONNXGraph {
    let mut graph = ONNXGraph::new("linear_regression")
        .add_input(TensorSpec::float(
            "X",
            vec![Dimension::Dynamic("batch".to_string()), Dimension::Fixed(n_features as i64)],
        ))
        .add_output(TensorSpec::float(
            "Y",
            vec![Dimension::Dynamic("batch".to_string()), Dimension::Fixed(1)],
        ));

    // Add weight initializer
    graph = graph.add_initializer(ONNXInitializer::float(
        "weights",
        vec![n_features as i64, 1],
        weights.iter().map(|&w| w as f32).collect(),
    ));

    // Add bias initializer
    graph = graph.add_initializer(ONNXInitializer::float(
        "bias",
        vec![1],
        vec![bias as f32],
    ));

    // MatMul node: X @ weights
    graph = graph.add_node(ONNXNode::new(
        "matmul",
        "MatMul",
        vec!["X".to_string(), "weights".to_string()],
        vec!["matmul_out".to_string()],
    ));

    // Add node: matmul_out + bias
    graph = graph.add_node(ONNXNode::new(
        "add",
        "Add",
        vec!["matmul_out".to_string(), "bias".to_string()],
        vec!["Y".to_string()],
    ));

    graph
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_onnx_graph_builder() {
        let graph = ONNXGraph::new("test_graph")
            .add_input(TensorSpec::float("input", vec![
                Dimension::Dynamic("batch".to_string()),
                Dimension::Fixed(10),
            ]))
            .add_output(TensorSpec::float("output", vec![
                Dimension::Dynamic("batch".to_string()),
                Dimension::Fixed(1),
            ]));

        assert_eq!(graph.name, "test_graph");
        assert_eq!(graph.inputs.len(), 1);
        assert_eq!(graph.outputs.len(), 1);
    }

    #[test]
    fn test_linear_regression_onnx() {
        let weights = vec![1.0, 2.0, 3.0];
        let bias = 0.5;
        
        let graph = linear_regression_to_onnx(&weights, bias, 3);
        
        assert_eq!(graph.inputs.len(), 1);
        assert_eq!(graph.outputs.len(), 1);
        assert_eq!(graph.nodes.len(), 2);
        assert_eq!(graph.initializers.len(), 2);
    }

    #[test]
    fn test_onnx_node_attributes() {
        let node = ONNXNode::new(
            "relu",
            "Relu",
            vec!["input".to_string()],
            vec!["output".to_string()],
        )
        .with_attribute("alpha", ONNXAttribute::Float(0.01));

        assert!(node.attributes.contains_key("alpha"));
    }
}
