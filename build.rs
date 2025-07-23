fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Tell cargo to rerun build script if proto files change
    println!("cargo:rerun-if-changed=third_party/onnx/onnx/onnx.proto");

    prost_build::Config::new().out_dir("src/").compile_protos(
        &["third_party/onnx/onnx/onnx.proto"],
        &["third_party/onnx/"],
    )?;

    Ok(())
}
