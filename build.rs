fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Note: This build script requires the Protocol Buffers compiler (protoc) to be installed
    // - Ubuntu/Debian: sudo apt-get install protobuf-compiler
    // - macOS: brew install protobuf
    // - Windows: choco install protoc
    // See CI workflows for automated installation examples

    // Tell cargo to rerun build script if proto files change
    println!("cargo:rerun-if-changed=third_party/onnx/onnx/onnx.proto");

    prost_build::Config::new().out_dir("src/").compile_protos(
        &["third_party/onnx/onnx/onnx.proto"],
        &["third_party/onnx/"],
    )?;

    Ok(())
}
