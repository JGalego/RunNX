fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Note: This build script requires the Protocol Buffers compiler (protoc) to be installed
    // - Ubuntu/Debian: sudo apt-get install protobuf-compiler
    // - macOS: brew install protobuf
    // - Windows: choco install protoc
    // See CI workflows for automated installation examples

    // Only run if proto files exist (for development)
    // The generated onnx.rs is already included in the published package
    let proto_path = "third_party/onnx/onnx/onnx.proto";

    if std::path::Path::new(proto_path).exists() {
        // Tell cargo to rerun build script if proto files change
        println!("cargo:rerun-if-changed={proto_path}");

        prost_build::Config::new()
            .out_dir("src/")
            .compile_protos(&[proto_path], &["third_party/onnx/"])?;

        println!("cargo:warning=Regenerated protobuf definitions from {proto_path}");
    } else {
        // For published packages, proto files may not be available
        // The pre-generated onnx.rs file should be used instead
        println!("cargo:warning=Proto files not found, using pre-generated onnx.rs");
    }

    Ok(())
}
