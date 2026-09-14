//! PMML (Predictive Model Markup Language) export
//!
//! Provides export of models to PMML format for compatibility with
//! enterprise ML deployment platforms.

use serde::{Deserialize, Serialize};
use std::fs::File;
use std::io::{Write, BufWriter};
use std::path::Path;

use crate::error::{AutoMLError, Result};

/// PMML data type
#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub enum PMMLDataType {
    Double,
    Float,
    Integer,
    String,
    Boolean,
}

impl PMMLDataType {
    fn as_str(&self) -> &'static str {
        match self {
            PMMLDataType::Double => "double",
            PMMLDataType::Float => "float",
            PMMLDataType::Integer => "integer",
            PMMLDataType::String => "string",
            PMMLDataType::Boolean => "boolean",
        }
    }
}

/// PMML field usage type
#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub enum FieldUsageType {
    Active,
    Target,
    Predicted,
    Supplementary,
}

impl FieldUsageType {
    fn as_str(&self) -> &'static str {
        match self {
            FieldUsageType::Active => "active",
            FieldUsageType::Target => "target",
            FieldUsageType::Predicted => "predicted",
            FieldUsageType::Supplementary => "supplementary",
        }
    }
}

/// PMML data field definition
#[derive(Debug, Clone)]
pub struct DataField {
    pub name: String,
    pub data_type: PMMLDataType,
    pub op_type: String,
    pub values: Option<Vec<String>>,
}

impl DataField {
    /// Create continuous numeric field
    pub fn continuous(name: impl Into<String>) -> Self {
        Self {
            name: name.into(),
            data_type: PMMLDataType::Double,
            op_type: "continuous".to_string(),
            values: None,
        }
    }

    /// Create categorical field
    pub fn categorical(name: impl Into<String>, values: Vec<String>) -> Self {
        Self {
            name: name.into(),
            data_type: PMMLDataType::String,
            op_type: "categorical".to_string(),
            values: Some(values),
        }
    }
}

/// PMML mining field
#[derive(Debug, Clone)]
pub struct MiningField {
    pub name: String,
    pub usage_type: FieldUsageType,
    pub importance: Option<f64>,
}

impl MiningField {
    pub fn active(name: impl Into<String>) -> Self {
        Self {
            name: name.into(),
            usage_type: FieldUsageType::Active,
            importance: None,
        }
    }

    pub fn target(name: impl Into<String>) -> Self {
        Self {
            name: name.into(),
            usage_type: FieldUsageType::Target,
            importance: None,
        }
    }

    pub fn with_importance(mut self, importance: f64) -> Self {
        self.importance = Some(importance);
        self
    }
}

/// Linear regression coefficient
#[derive(Debug, Clone)]
pub struct NumericPredictor {
    pub name: String,
    pub coefficient: f64,
    pub exponent: i32,
}

impl NumericPredictor {
    pub fn new(name: impl Into<String>, coefficient: f64) -> Self {
        Self {
            name: name.into(),
            coefficient,
            exponent: 1,
        }
    }
}

/// PMML regression model
#[derive(Debug, Clone)]
pub struct RegressionModel {
    pub model_name: String,
    pub function_name: String,
    pub algorithm_name: Option<String>,
    pub intercept: f64,
    pub predictors: Vec<NumericPredictor>,
}

/// Decision tree node
#[derive(Debug, Clone)]
pub struct TreeNode {
    pub id: String,
    pub score: Option<f64>,
    pub record_count: Option<f64>,
    pub predicate: TreePredicate,
    pub children: Vec<TreeNode>,
}

/// Tree predicate for splits
#[derive(Debug, Clone)]
pub enum TreePredicate {
    True,
    SimplePredicate {
        field: String,
        operator: String,
        value: String,
    },
    CompoundPredicate {
        boolean_operator: String,
        predicates: Vec<TreePredicate>,
    },
}

/// PMML tree model
#[derive(Debug, Clone)]
pub struct TreeModel {
    pub model_name: String,
    pub function_name: String,
    pub algorithm_name: Option<String>,
    pub split_characteristic: String,
    pub root: TreeNode,
}

/// PMML model types
#[derive(Debug, Clone)]
pub enum PMMLModel {
    Regression(RegressionModel),
    Tree(TreeModel),
}

/// Complete PMML document
#[derive(Debug, Clone)]
pub struct PMMLDocument {
    pub version: String,
    pub header: PMMLHeader,
    pub data_dictionary: Vec<DataField>,
    pub mining_schema: Vec<MiningField>,
    pub model: PMMLModel,
}

/// PMML header
#[derive(Debug, Clone)]
pub struct PMMLHeader {
    pub copyright: Option<String>,
    pub description: Option<String>,
    pub application_name: String,
    pub application_version: String,
}

impl Default for PMMLHeader {
    fn default() -> Self {
        Self {
            copyright: None,
            description: None,
            application_name: "AutoML".to_string(),
            application_version: env!("CARGO_PKG_VERSION").to_string(),
        }
    }
}

/// PMML exporter
pub struct PMMLExporter;

impl PMMLExporter {
    /// Create new exporter
    pub fn new() -> Self {
        Self
    }

    /// Export PMML document to file
    pub fn export(&self, doc: &PMMLDocument, path: impl AsRef<Path>) -> Result<()> {
        let file = File::create(path.as_ref()).map_err(|e| {
            AutoMLError::DataError(format!("Failed to create file: {}", e))
        })?;
        let mut writer = BufWriter::new(file);

        self.write_pmml(&mut writer, doc)
    }

    /// Export to string
    pub fn export_to_string(&self, doc: &PMMLDocument) -> Result<String> {
        let mut buffer = Vec::new();
        self.write_pmml(&mut buffer, doc)?;
        String::from_utf8(buffer).map_err(|e| {
            AutoMLError::SerializationError(format!("Invalid UTF-8: {}", e))
        })
    }

    fn write_pmml<W: Write>(&self, writer: &mut W, doc: &PMMLDocument) -> Result<()> {
        writeln!(writer, "<?xml version=\"1.0\" encoding=\"UTF-8\"?>").map_err(Self::io_err)?;
        writeln!(
            writer,
            "<PMML xmlns=\"http://www.dmg.org/PMML-4_4\" version=\"{}\">",
            doc.version
        ).map_err(Self::io_err)?;

        self.write_header(writer, &doc.header)?;
        self.write_data_dictionary(writer, &doc.data_dictionary)?;

        match &doc.model {
            PMMLModel::Regression(reg) => {
                self.write_regression_model(writer, reg, &doc.mining_schema)?;
            }
            PMMLModel::Tree(tree) => {
                self.write_tree_model(writer, tree, &doc.mining_schema)?;
            }
        }

        writeln!(writer, "</PMML>").map_err(Self::io_err)?;
        Ok(())
    }

    fn write_header<W: Write>(&self, writer: &mut W, header: &PMMLHeader) -> Result<()> {
        writeln!(writer, "  <Header>").map_err(Self::io_err)?;
        
        if let Some(ref copyright) = header.copyright {
            writeln!(writer, "    <Copyright>{}</Copyright>", Self::escape_xml(copyright))
                .map_err(Self::io_err)?;
        }
        
        if let Some(ref desc) = header.description {
            writeln!(writer, "    <Description>{}</Description>", Self::escape_xml(desc))
                .map_err(Self::io_err)?;
        }
        
        writeln!(
            writer,
            "    <Application name=\"{}\" version=\"{}\"/>",
            Self::escape_xml(&header.application_name),
            Self::escape_xml(&header.application_version)
        ).map_err(Self::io_err)?;
        
        writeln!(writer, "  </Header>").map_err(Self::io_err)?;
        Ok(())
    }

    fn write_data_dictionary<W: Write>(&self, writer: &mut W, fields: &[DataField]) -> Result<()> {
        writeln!(writer, "  <DataDictionary numberOfFields=\"{}\">", fields.len())
            .map_err(Self::io_err)?;
        
        for field in fields {
            write!(
                writer,
                "    <DataField name=\"{}\" dataType=\"{}\" optype=\"{}\"",
                Self::escape_xml(&field.name),
                field.data_type.as_str(),
                Self::escape_xml(&field.op_type)
            ).map_err(Self::io_err)?;
            
            if let Some(ref values) = field.values {
                writeln!(writer, ">").map_err(Self::io_err)?;
                for val in values {
                    writeln!(writer, "      <Value value=\"{}\"/>", Self::escape_xml(val))
                        .map_err(Self::io_err)?;
                }
                writeln!(writer, "    </DataField>").map_err(Self::io_err)?;
            } else {
                writeln!(writer, "/>").map_err(Self::io_err)?;
            }
        }
        
        writeln!(writer, "  </DataDictionary>").map_err(Self::io_err)?;
        Ok(())
    }

    fn write_mining_schema<W: Write>(&self, writer: &mut W, fields: &[MiningField]) -> Result<()> {
        writeln!(writer, "    <MiningSchema>").map_err(Self::io_err)?;
        
        for field in fields {
            write!(
                writer,
                "      <MiningField name=\"{}\" usageType=\"{}\"",
                Self::escape_xml(&field.name),
                field.usage_type.as_str()
            ).map_err(Self::io_err)?;
            
            if let Some(imp) = field.importance {
                write!(writer, " importance=\"{}\"", imp).map_err(Self::io_err)?;
            }
            
            writeln!(writer, "/>").map_err(Self::io_err)?;
        }
        
        writeln!(writer, "    </MiningSchema>").map_err(Self::io_err)?;
        Ok(())
    }

    fn write_regression_model<W: Write>(
        &self,
        writer: &mut W,
        model: &RegressionModel,
        schema: &[MiningField],
    ) -> Result<()> {
        write!(
            writer,
            "  <RegressionModel modelName=\"{}\" functionName=\"{}\"",
            Self::escape_xml(&model.model_name),
            Self::escape_xml(&model.function_name)
        ).map_err(Self::io_err)?;
        
        if let Some(ref algo) = model.algorithm_name {
            write!(writer, " algorithmName=\"{}\"", Self::escape_xml(algo))
                .map_err(Self::io_err)?;
        }
        
        writeln!(writer, ">").map_err(Self::io_err)?;
        
        self.write_mining_schema(writer, schema)?;
        
        writeln!(
            writer,
            "    <RegressionTable intercept=\"{}\">",
            model.intercept
        ).map_err(Self::io_err)?;
        
        for pred in &model.predictors {
            writeln!(
                writer,
                "      <NumericPredictor name=\"{}\" coefficient=\"{}\" exponent=\"{}\"/>",
                Self::escape_xml(&pred.name),
                pred.coefficient,
                pred.exponent
            ).map_err(Self::io_err)?;
        }
        
        writeln!(writer, "    </RegressionTable>").map_err(Self::io_err)?;
        writeln!(writer, "  </RegressionModel>").map_err(Self::io_err)?;
        Ok(())
    }

    fn write_tree_model<W: Write>(
        &self,
        writer: &mut W,
        model: &TreeModel,
        schema: &[MiningField],
    ) -> Result<()> {
        write!(
            writer,
            "  <TreeModel modelName=\"{}\" functionName=\"{}\" splitCharacteristic=\"{}\"",
            Self::escape_xml(&model.model_name),
            Self::escape_xml(&model.function_name),
            Self::escape_xml(&model.split_characteristic)
        ).map_err(Self::io_err)?;
        
        if let Some(ref algo) = model.algorithm_name {
            write!(writer, " algorithmName=\"{}\"", Self::escape_xml(algo))
                .map_err(Self::io_err)?;
        }
        
        writeln!(writer, ">").map_err(Self::io_err)?;
        
        self.write_mining_schema(writer, schema)?;
        self.write_tree_node(writer, &model.root, 4)?;
        
        writeln!(writer, "  </TreeModel>").map_err(Self::io_err)?;
        Ok(())
    }

    fn write_tree_node<W: Write>(
        &self,
        writer: &mut W,
        node: &TreeNode,
        indent: usize,
    ) -> Result<()> {
        let pad = " ".repeat(indent);
        
        write!(writer, "{}<Node id=\"{}\"", pad, Self::escape_xml(&node.id))
            .map_err(Self::io_err)?;
        
        if let Some(score) = node.score {
            write!(writer, " score=\"{}\"", score).map_err(Self::io_err)?;
        }
        
        if let Some(count) = node.record_count {
            write!(writer, " recordCount=\"{}\"", count).map_err(Self::io_err)?;
        }
        
        writeln!(writer, ">").map_err(Self::io_err)?;
        
        self.write_predicate(writer, &node.predicate, indent + 2)?;
        
        for child in &node.children {
            self.write_tree_node(writer, child, indent + 2)?;
        }
        
        writeln!(writer, "{}</Node>", pad).map_err(Self::io_err)?;
        Ok(())
    }

    fn write_predicate<W: Write>(
        &self,
        writer: &mut W,
        predicate: &TreePredicate,
        indent: usize,
    ) -> Result<()> {
        let pad = " ".repeat(indent);
        
        match predicate {
            TreePredicate::True => {
                writeln!(writer, "{}<True/>", pad).map_err(Self::io_err)?;
            }
            TreePredicate::SimplePredicate { field, operator, value } => {
                writeln!(
                    writer,
                    "{}<SimplePredicate field=\"{}\" operator=\"{}\" value=\"{}\"/>",
                    pad,
                    Self::escape_xml(field),
                    Self::escape_xml(operator),
                    Self::escape_xml(value)
                ).map_err(Self::io_err)?;
            }
            TreePredicate::CompoundPredicate { boolean_operator, predicates } => {
                writeln!(
                    writer,
                    "{}<CompoundPredicate booleanOperator=\"{}\">",
                    pad,
                    Self::escape_xml(boolean_operator)
                ).map_err(Self::io_err)?;
                
                for pred in predicates {
                    self.write_predicate(writer, pred, indent + 2)?;
                }
                
                writeln!(writer, "{}</CompoundPredicate>", pad).map_err(Self::io_err)?;
            }
        }
        
        Ok(())
    }

    fn escape_xml(s: &str) -> String {
        s.replace('&', "&amp;")
            .replace('<', "&lt;")
            .replace('>', "&gt;")
            .replace('"', "&quot;")
            .replace('\'', "&apos;")
    }

    fn io_err(e: std::io::Error) -> AutoMLError {
        AutoMLError::DataError(format!("Failed to write PMML: {}", e))
    }
}

impl Default for PMMLExporter {
    fn default() -> Self {
        Self::new()
    }
}

/// Helper to create linear regression PMML
#[allow(dead_code)]
pub fn linear_regression_to_pmml(
    model_name: &str,
    feature_names: &[String],
    weights: &[f64],
    intercept: f64,
    target_name: &str,
) -> PMMLDocument {
    let mut data_fields: Vec<DataField> = feature_names
        .iter()
        .map(|name| DataField::continuous(name))
        .collect();
    data_fields.push(DataField::continuous(target_name));

    let mut mining_fields: Vec<MiningField> = feature_names
        .iter()
        .map(|name| MiningField::active(name))
        .collect();
    mining_fields.push(MiningField::target(target_name));

    let predictors: Vec<NumericPredictor> = feature_names
        .iter()
        .zip(weights.iter())
        .map(|(name, &coef)| NumericPredictor::new(name, coef))
        .collect();

    PMMLDocument {
        version: "4.4".to_string(),
        header: PMMLHeader::default(),
        data_dictionary: data_fields,
        mining_schema: mining_fields,
        model: PMMLModel::Regression(RegressionModel {
            model_name: model_name.to_string(),
            function_name: "regression".to_string(),
            algorithm_name: Some("linearRegression".to_string()),
            intercept,
            predictors,
        }),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_linear_regression_pmml() {
        let features = vec!["x1".to_string(), "x2".to_string()];
        let weights = vec![1.5, -0.5];
        let doc = linear_regression_to_pmml("test_model", &features, &weights, 0.1, "y");

        let exporter = PMMLExporter::new();
        let pmml_str = exporter.export_to_string(&doc).unwrap();

        assert!(pmml_str.contains("RegressionModel"));
        assert!(pmml_str.contains("x1"));
        assert!(pmml_str.contains("1.5"));
    }

    #[test]
    fn test_xml_escaping() {
        let escaped = PMMLExporter::escape_xml("<test & \"value\">");
        assert_eq!(escaped, "&lt;test &amp; &quot;value&quot;&gt;");
    }

    #[test]
    fn test_data_field_continuous() {
        let field = DataField::continuous("feature1");
        assert_eq!(field.name, "feature1");
        assert_eq!(field.op_type, "continuous");
    }

    #[test]
    fn test_data_field_categorical() {
        let field = DataField::categorical("color", vec!["red".to_string(), "blue".to_string()]);
        assert_eq!(field.values.as_ref().unwrap().len(), 2);
    }
}
