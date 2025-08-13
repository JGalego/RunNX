//! Conversion utilities between internal representations and ONNX protobuf format
//!
//! This module provides conversion functions to bridge between the RunNX internal
//! data structures and the ONNX protobuf format, enabling support for loading
//! binary .onnx files.

use crate::{
    error::{OnnxError, Result},
    graph::{Graph, Node, TensorSpec},
    model::{Model, ModelMetadata},
    proto,
    tensor::Tensor,
};
use prost::Message;
use std::collections::HashMap;
use std::fs;
use std::path::Path;

/// Convert ONNX protobuf ModelProto to internal Model representation
pub fn from_model_proto(model_proto: &proto::ModelProto) -> Result<Model> {
    // Extract metadata
    let graph_proto = model_proto
        .graph
        .as_ref()
        .ok_or_else(|| OnnxError::model_load_error("Model proto missing graph"))?;

    let metadata = ModelMetadata {
        name: graph_proto.name.clone().unwrap_or_default(),
        version: model_proto
            .model_version
            .map(|v| v.to_string())
            .unwrap_or_default(),
        description: model_proto.doc_string.clone().unwrap_or_default(),
        producer: model_proto.producer_name.clone().unwrap_or_default(),
        onnx_version: format!("IR_VERSION_{}", model_proto.ir_version.unwrap_or(0)),
        domain: model_proto.domain.clone().unwrap_or_default(),
    };

    // Convert graph
    let graph = from_graph_proto(graph_proto)?;

    Ok(Model::with_metadata(metadata, graph))
}

/// Convert ONNX protobuf GraphProto to internal Graph representation
pub fn from_graph_proto(graph_proto: &proto::GraphProto) -> Result<Graph> {
    // Convert nodes
    let mut nodes = Vec::new();
    for node_proto in &graph_proto.node {
        nodes.push(from_node_proto(node_proto)?);
    }

    // Convert inputs
    let mut inputs = Vec::new();
    for input_proto in &graph_proto.input {
        inputs.push(from_value_info_proto(input_proto)?);
    }

    // Convert outputs
    let mut outputs = Vec::new();
    for output_proto in &graph_proto.output {
        outputs.push(from_value_info_proto(output_proto)?);
    }

    // Convert initializers (weights/constants)
    let mut initializers = HashMap::new();
    for tensor_proto in &graph_proto.initializer {
        let name = tensor_proto.name.clone().unwrap_or_default();
        let tensor = from_tensor_proto(tensor_proto)?;
        initializers.insert(name, tensor);
    }

    Ok(Graph {
        name: graph_proto.name.clone().unwrap_or_default(),
        nodes,
        inputs,
        outputs,
        initializers,
    })
}

/// Convert ONNX protobuf NodeProto to internal Node representation
pub fn from_node_proto(node_proto: &proto::NodeProto) -> Result<Node> {
    // Filter out empty string inputs (optional inputs in ONNX)
    let inputs: Vec<String> = node_proto
        .input
        .iter()
        .filter(|input| !input.is_empty())
        .cloned()
        .collect();

    let mut node = Node::new(
        node_proto.name.clone().unwrap_or_default(),
        node_proto.op_type.clone().unwrap_or_default(),
        inputs,
        node_proto.output.clone(),
    );

    // Convert attributes
    for attr_proto in &node_proto.attribute {
        let name = attr_proto.name.clone().unwrap_or_default();
        let value = from_attribute_proto(attr_proto)?;
        node.add_attribute(name, value);
    }

    Ok(node)
}

/// Convert ONNX protobuf ValueInfoProto to internal TensorSpec representation
pub fn from_value_info_proto(value_info_proto: &proto::ValueInfoProto) -> Result<TensorSpec> {
    let name = value_info_proto.name.clone().unwrap_or_default();

    let type_proto = value_info_proto
        .r#type
        .as_ref()
        .ok_or_else(|| OnnxError::model_load_error("ValueInfo missing type information"))?;

    let tensor_type = match &type_proto.value {
        Some(proto::type_proto::Value::TensorType(tensor)) => tensor,
        _ => {
            return Err(OnnxError::model_load_error(
                "Type proto missing tensor type",
            ))
        }
    };

    // Extract shape
    let mut dimensions = Vec::new();
    if let Some(shape_proto) = &tensor_type.shape {
        for dim in &shape_proto.dim {
            match &dim.value {
                Some(proto::tensor_shape_proto::dimension::Value::DimValue(dim_value)) => {
                    dimensions.push(Some(*dim_value as usize));
                }
                Some(proto::tensor_shape_proto::dimension::Value::DimParam(_)) => {
                    // Handle symbolic dimensions as None (dynamic)
                    dimensions.push(None);
                }
                None => {
                    // Unknown dimension, treat as dynamic
                    dimensions.push(None);
                }
            }
        }
    }

    // Convert data type
    let dtype = match proto::tensor_proto::DataType::try_from(tensor_type.elem_type.unwrap_or(0)) {
        Ok(proto::tensor_proto::DataType::Float) => "float32",
        Ok(proto::tensor_proto::DataType::Double) => "float64",
        Ok(proto::tensor_proto::DataType::Int32) => "int32",
        Ok(proto::tensor_proto::DataType::Int64) => "int64",
        _ => {
            return Err(OnnxError::unsupported_operation(format!(
                "Unsupported data type: {}",
                tensor_type.elem_type.unwrap_or(0)
            )))
        }
    };

    Ok(TensorSpec {
        name,
        dimensions,
        dtype: dtype.to_string(),
    })
}

/// Convert ONNX protobuf TensorProto to internal Tensor representation
pub fn from_tensor_proto(tensor_proto: &proto::TensorProto) -> Result<Tensor> {
    // Extract shape
    let shape: Vec<usize> = tensor_proto.dims.iter().map(|&dim| dim as usize).collect();

    // Convert data based on type
    let data_type = proto::tensor_proto::DataType::try_from(tensor_proto.data_type.unwrap_or(0))
        .map_err(|_| {
            OnnxError::unsupported_operation(format!(
                "Unsupported tensor data type: {}",
                tensor_proto.data_type.unwrap_or(0)
            ))
        })?;

    match data_type {
        proto::tensor_proto::DataType::Float => {
            let data = if !tensor_proto.float_data.is_empty() {
                tensor_proto.float_data.clone()
            } else if let Some(ref raw_data) = tensor_proto.raw_data {
                if !raw_data.is_empty() {
                    // Parse raw bytes as f32
                    if raw_data.len() % 4 != 0 {
                        return Err(OnnxError::model_load_error(
                            "Invalid raw data length for float32",
                        ));
                    }
                    let mut floats = Vec::with_capacity(raw_data.len() / 4);
                    for chunk in raw_data.chunks_exact(4) {
                        let bytes = [chunk[0], chunk[1], chunk[2], chunk[3]];
                        floats.push(f32::from_le_bytes(bytes));
                    }
                    floats
                } else {
                    return Err(OnnxError::model_load_error("Tensor missing data"));
                }
            } else {
                return Err(OnnxError::model_load_error("Tensor missing data"));
            };

            Tensor::from_shape_vec(&shape, data)
        }
        proto::tensor_proto::DataType::Int64 => {
            let data = if !tensor_proto.int64_data.is_empty() {
                // Convert i64 to f32 for now (simplified approach)
                tensor_proto.int64_data.iter().map(|&x| x as f32).collect()
            } else if let Some(ref raw_data) = tensor_proto.raw_data {
                if !raw_data.is_empty() {
                    // Parse raw bytes as i64, then convert to f32
                    if raw_data.len() % 8 != 0 {
                        return Err(OnnxError::model_load_error(
                            "Invalid raw data length for int64",
                        ));
                    }
                    let mut floats = Vec::with_capacity(raw_data.len() / 8);
                    for chunk in raw_data.chunks_exact(8) {
                        let bytes = [
                            chunk[0], chunk[1], chunk[2], chunk[3], chunk[4], chunk[5], chunk[6],
                            chunk[7],
                        ];
                        let int_val = i64::from_le_bytes(bytes);
                        floats.push(int_val as f32);
                    }
                    floats
                } else {
                    return Err(OnnxError::model_load_error("Tensor missing data"));
                }
            } else {
                return Err(OnnxError::model_load_error("Tensor missing data"));
            };

            Tensor::from_shape_vec(&shape, data)
        }
        proto::tensor_proto::DataType::Int32 => {
            let data = if !tensor_proto.int32_data.is_empty() {
                // Convert i32 to f32 for now (simplified approach)
                tensor_proto.int32_data.iter().map(|&x| x as f32).collect()
            } else if let Some(ref raw_data) = tensor_proto.raw_data {
                if !raw_data.is_empty() {
                    // Parse raw bytes as i32, then convert to f32
                    if raw_data.len() % 4 != 0 {
                        return Err(OnnxError::model_load_error(
                            "Invalid raw data length for int32",
                        ));
                    }
                    let mut floats = Vec::with_capacity(raw_data.len() / 4);
                    for chunk in raw_data.chunks_exact(4) {
                        let bytes = [chunk[0], chunk[1], chunk[2], chunk[3]];
                        let int_val = i32::from_le_bytes(bytes);
                        floats.push(int_val as f32);
                    }
                    floats
                } else {
                    return Err(OnnxError::model_load_error("Tensor missing data"));
                }
            } else {
                return Err(OnnxError::model_load_error("Tensor missing data"));
            };

            Tensor::from_shape_vec(&shape, data)
        }
        _ => Err(OnnxError::unsupported_operation(format!(
            "Unsupported tensor data type: {data_type:?}"
        ))),
    }
}

/// Convert ONNX protobuf AttributeProto to string representation
pub fn from_attribute_proto(attr_proto: &proto::AttributeProto) -> Result<String> {
    // For simplicity, convert all attributes to string
    // In a full implementation, you'd want to preserve types
    if let Some(ref s) = attr_proto.s {
        Ok(String::from_utf8_lossy(s).to_string())
    } else if let Some(i) = attr_proto.i {
        Ok(i.to_string())
    } else if let Some(f) = attr_proto.f {
        Ok(f.to_string())
    } else if !attr_proto.ints.is_empty() {
        Ok(format!("{:?}", attr_proto.ints))
    } else if !attr_proto.floats.is_empty() {
        Ok(format!("{:?}", attr_proto.floats))
    } else {
        Ok(String::new())
    }
}

/// Load an ONNX model from a binary .onnx file
pub fn load_onnx_model<P: AsRef<Path>>(path: P) -> Result<Model> {
    let path = path.as_ref();
    let bytes = fs::read(path).map_err(|e| {
        OnnxError::model_load_error(format!(
            "Failed to read ONNX file '{}': {}",
            path.display(),
            e
        ))
    })?;

    let model_proto = proto::ModelProto::decode(&bytes[..]).map_err(|e| {
        OnnxError::model_load_error(format!(
            "Failed to parse ONNX file '{}': {}",
            path.display(),
            e
        ))
    })?;

    from_model_proto(&model_proto)
}

/// Convert internal Model to ONNX protobuf ModelProto
pub fn to_model_proto(model: &Model) -> Result<proto::ModelProto> {
    let model_proto = proto::ModelProto {
        ir_version: Some(7i64), // IR_VERSION_2020_5_8
        producer_name: Some(model.metadata.producer.clone()),
        producer_version: Some(model.metadata.version.clone()),
        domain: Some(model.metadata.domain.clone()),
        doc_string: Some(model.metadata.description.clone()),
        graph: Some(to_graph_proto(&model.graph)?),
        model_version: None,
        opset_import: vec![],
        metadata_props: vec![],
        training_info: vec![],
        functions: vec![],
        configuration: vec![],
    };

    Ok(model_proto)
}

/// Convert internal Graph to ONNX protobuf GraphProto
pub fn to_graph_proto(graph: &Graph) -> Result<proto::GraphProto> {
    let mut nodes = Vec::new();
    for node in &graph.nodes {
        nodes.push(to_node_proto(node)?);
    }

    let mut inputs = Vec::new();
    for input in &graph.inputs {
        inputs.push(to_value_info_proto(input)?);
    }

    let mut outputs = Vec::new();
    for output in &graph.outputs {
        outputs.push(to_value_info_proto(output)?);
    }

    let mut initializers = Vec::new();
    for (name, tensor) in &graph.initializers {
        initializers.push(to_tensor_proto(name, tensor)?);
    }

    let graph_proto = proto::GraphProto {
        node: nodes,
        name: Some(graph.name.clone()),
        initializer: initializers,
        sparse_initializer: vec![],
        doc_string: None,
        input: inputs,
        output: outputs,
        value_info: vec![],
        quantization_annotation: vec![],
        metadata_props: vec![],
    };

    Ok(graph_proto)
}

/// Convert internal Node to ONNX protobuf NodeProto
pub fn to_node_proto(node: &Node) -> Result<proto::NodeProto> {
    let node_proto = proto::NodeProto {
        input: node.inputs.clone(),
        output: node.outputs.clone(),
        name: Some(node.name.clone()),
        op_type: Some(node.op_type.clone()),
        domain: None,
        attribute: vec![], // For simplicity, skip attributes in conversion back
        doc_string: None,
        overload: None,
        metadata_props: vec![],
        device_configurations: vec![],
    };

    Ok(node_proto)
}

/// Convert internal TensorSpec to ONNX protobuf ValueInfoProto
pub fn to_value_info_proto(spec: &TensorSpec) -> Result<proto::ValueInfoProto> {
    // Set element type
    let elem_type = match spec.dtype.as_str() {
        "float32" => proto::tensor_proto::DataType::Float as i32,
        "float64" => proto::tensor_proto::DataType::Double as i32,
        "int32" => proto::tensor_proto::DataType::Int32 as i32,
        "int64" => proto::tensor_proto::DataType::Int64 as i32,
        _ => {
            return Err(OnnxError::unsupported_operation(format!(
                "Unsupported data type: {}",
                spec.dtype
            )))
        }
    };

    // Set shape
    let mut dims = Vec::new();
    for dim_opt in &spec.dimensions {
        match dim_opt {
            Some(dim_size) => {
                dims.push(proto::tensor_shape_proto::Dimension {
                    value: Some(proto::tensor_shape_proto::dimension::Value::DimValue(
                        *dim_size as i64,
                    )),
                    denotation: None,
                });
            }
            None => {
                // Dynamic dimension, use dimension parameter
                dims.push(proto::tensor_shape_proto::Dimension {
                    value: Some(proto::tensor_shape_proto::dimension::Value::DimParam(
                        "dynamic".to_string(),
                    )),
                    denotation: None,
                });
            }
        }
    }

    let tensor_type = proto::type_proto::Tensor {
        elem_type: Some(elem_type),
        shape: Some(proto::TensorShapeProto { dim: dims }),
    };

    let type_proto = proto::TypeProto {
        denotation: None,
        value: Some(proto::type_proto::Value::TensorType(tensor_type)),
    };

    let value_info = proto::ValueInfoProto {
        name: Some(spec.name.clone()),
        r#type: Some(type_proto),
        doc_string: None,
        metadata_props: vec![],
    };

    Ok(value_info)
}

/// Convert internal Tensor to ONNX protobuf TensorProto
pub fn to_tensor_proto(name: &str, tensor: &Tensor) -> Result<proto::TensorProto> {
    let tensor_proto = proto::TensorProto {
        dims: tensor.shape().iter().map(|&dim| dim as i64).collect(),
        data_type: Some(proto::tensor_proto::DataType::Float as i32),
        segment: None,
        float_data: tensor.data().iter().cloned().collect(),
        int32_data: vec![],
        string_data: vec![],
        int64_data: vec![],
        name: Some(name.to_string()),
        doc_string: None,
        raw_data: None,
        external_data: vec![],
        data_location: None,
        double_data: vec![],
        uint64_data: vec![],
        metadata_props: vec![],
    };

    Ok(tensor_proto)
}

/// Save a model to binary .onnx format
pub fn save_onnx_model<P: AsRef<Path>>(model: &Model, path: P) -> Result<()> {
    let path = path.as_ref();
    let model_proto = to_model_proto(model)?;
    let mut bytes = Vec::new();
    model_proto
        .encode(&mut bytes)
        .map_err(|e| OnnxError::other(format!("Failed to serialize model: {e}")))?;

    fs::write(path, bytes).map_err(|e| {
        OnnxError::other(format!(
            "Failed to write ONNX file '{}': {}",
            path.display(),
            e
        ))
    })
}

#[cfg(test)]
mod converter_tests {
    use super::*;
    use crate::graph::TensorSpec;
    use crate::{Graph, Model, Node};

    #[test]
    fn test_converter_round_trip() {
        // Create a simple model
        let mut graph = Graph::new("test_converter".to_string());

        // Add input and output specifications
        let input_spec = TensorSpec::new("input".to_string(), vec![Some(1), Some(3)]);
        graph.add_input(input_spec);

        let output_spec = TensorSpec::new("output".to_string(), vec![Some(1), Some(3)]);
        graph.add_output(output_spec);

        // Add a node
        let node = Node::new(
            "relu".to_string(),
            "Relu".to_string(),
            vec!["input".to_string()],
            vec!["output".to_string()],
        );
        graph.add_node(node);

        // Create model
        let original_model = Model::new(graph);

        // Convert to protobuf and back
        let proto = to_model_proto(&original_model).expect("Failed to convert to proto");
        let converted_model = from_model_proto(&proto).expect("Failed to convert from proto");

        // Verify the conversion preserved the model structure
        assert_eq!(original_model.graph.name, converted_model.graph.name);
        assert_eq!(
            original_model.graph.nodes.len(),
            converted_model.graph.nodes.len()
        );
        assert_eq!(
            original_model.graph.inputs.len(),
            converted_model.graph.inputs.len()
        );
        assert_eq!(
            original_model.graph.outputs.len(),
            converted_model.graph.outputs.len()
        );

        // Check node details
        let orig_node = &original_model.graph.nodes[0];
        let conv_node = &converted_model.graph.nodes[0];
        assert_eq!(orig_node.name, conv_node.name);
        assert_eq!(orig_node.op_type, conv_node.op_type);
        assert_eq!(orig_node.inputs, conv_node.inputs);
        assert_eq!(orig_node.outputs, conv_node.outputs);
    }

    #[test]
    fn test_tensor_spec_conversion() {
        // Test tensor spec with fixed dimensions
        let spec = TensorSpec::new("test".to_string(), vec![Some(2), Some(4)]);
        let value_info = to_value_info_proto(&spec).expect("Failed to convert TensorSpec");
        let converted_spec = from_value_info_proto(&value_info).expect("Failed to convert back");

        assert_eq!(spec.name, converted_spec.name);
        assert_eq!(spec.dimensions, converted_spec.dimensions);
        assert_eq!(spec.dtype, converted_spec.dtype);

        // Test tensor spec with dynamic dimensions
        let dynamic_spec = TensorSpec::new("dynamic".to_string(), vec![None, Some(3), None]);
        let dynamic_value_info =
            to_value_info_proto(&dynamic_spec).expect("Failed to convert dynamic TensorSpec");
        let converted_dynamic =
            from_value_info_proto(&dynamic_value_info).expect("Failed to convert back dynamic");

        assert_eq!(dynamic_spec.name, converted_dynamic.name);
        assert_eq!(dynamic_spec.dimensions, converted_dynamic.dimensions);
        assert_eq!(dynamic_spec.dtype, converted_dynamic.dtype);
    }

    #[test]
    fn test_load_save_onnx_model() {
        use std::env;
        use std::fs;

        // Create a simple model
        let mut graph = Graph::new("test_save_load".to_string());

        let input_spec = TensorSpec::new("x".to_string(), vec![Some(1), Some(2)]);
        graph.add_input(input_spec);

        let output_spec = TensorSpec::new("y".to_string(), vec![Some(1), Some(2)]);
        graph.add_output(output_spec);

        let node = Node::new(
            "sigmoid".to_string(),
            "Sigmoid".to_string(),
            vec!["x".to_string()],
            vec!["y".to_string()],
        );
        graph.add_node(node);

        let model = Model::new(graph);

        // Use platform-independent temporary directory
        let temp_dir = env::temp_dir();
        let test_path = temp_dir.join("test_converter.onnx");

        // Test save
        save_onnx_model(&model, &test_path).expect("Failed to save ONNX model");

        // Test load
        let loaded_model = load_onnx_model(&test_path).expect("Failed to load ONNX model");

        // Verify loaded model
        assert_eq!(model.graph.name, loaded_model.graph.name);
        assert_eq!(model.graph.nodes.len(), loaded_model.graph.nodes.len());
        assert_eq!(model.graph.inputs.len(), loaded_model.graph.inputs.len());
        assert_eq!(model.graph.outputs.len(), loaded_model.graph.outputs.len());

        // Cleanup
        let _ = fs::remove_file(&test_path);
    }

    #[test]
    fn test_invalid_tensor_type() {
        // Create a mock ValueInfoProto with missing tensor type
        let value_info = proto::ValueInfoProto {
            name: Some("invalid".to_string()),
            r#type: None,
            doc_string: None,
            metadata_props: vec![],
        };

        let result = from_value_info_proto(&value_info);
        assert!(result.is_err());
        assert!(result
            .unwrap_err()
            .to_string()
            .contains("missing type information"));
    }
}
