fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("cargo:rerun-if-env-changed=RUNNX_REGENERATE_ONNX_PROTO");
    println!("cargo:rerun-if-changed=src/onnx.rs");

    if std::env::var_os("RUNNX_REGENERATE_ONNX_PROTO").is_none() {
        return Ok(());
    }

    let proto_path = "third_party/onnx/onnx/onnx.proto";
    if !std::path::Path::new(proto_path).is_file() {
        return Err(format!(
            "{proto_path} is unavailable; initialize submodules before regenerating bindings"
        )
        .into());
    }

    println!("cargo:rerun-if-changed={proto_path}");
    prost_build::Config::new()
        .out_dir("src/")
        .compile_protos(&[proto_path], &["third_party/onnx/"])?;

    Ok(())
}
