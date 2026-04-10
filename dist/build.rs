use flate2::write::GzEncoder;
use flate2::Compression;
use std::env;
use std::fs;
use std::io::Write;
use std::path::Path;

fn compress_file(input: &Path, output: &Path) {
    let data = fs::read(input).unwrap_or_else(|e| panic!("Failed to read {}: {}", input.display(), e));
    let mut encoder = GzEncoder::new(Vec::new(), Compression::best());
    encoder.write_all(&data).expect("compression failed");
    let compressed = encoder.finish().expect("compression finish failed");
    fs::write(output, &compressed).unwrap_or_else(|e| panic!("Failed to write {}: {}", output.display(), e));

    eprintln!(
        "Compressed {} -> {} ({:.1}MB -> {:.1}MB)",
        input.display(),
        output.display(),
        data.len() as f64 / 1_048_576.0,
        compressed.len() as f64 / 1_048_576.0,
    );
}

fn main() {
    let out_dir = env::var("OUT_DIR").unwrap();
    let out = Path::new(&out_dir);

    let server_bin = env::var("TELEMUZE_SERVER_BIN")
        .expect("Set TELEMUZE_SERVER_BIN to the path of the compiled telemuze server binary");
    let sherpa_so = env::var("TELEMUZE_SHERPA_SO")
        .expect("Set TELEMUZE_SHERPA_SO to the path of libsherpa-onnx-c-api.so");
    let onnxruntime_so = env::var("TELEMUZE_ONNXRUNTIME_SO")
        .expect("Set TELEMUZE_ONNXRUNTIME_SO to the path of libonnxruntime.so");

    compress_file(Path::new(&server_bin), &out.join("telemuze.gz"));
    compress_file(Path::new(&sherpa_so), &out.join("libsherpa-onnx-c-api.so.gz"));
    compress_file(Path::new(&onnxruntime_so), &out.join("libonnxruntime.so.gz"));

    // Version hash from file sizes — cheap staleness check
    let version = format!(
        "{}-{}-{}",
        fs::metadata(&server_bin).unwrap().len(),
        fs::metadata(&sherpa_so).unwrap().len(),
        fs::metadata(&onnxruntime_so).unwrap().len(),
    );
    fs::write(out.join("version_hash.txt"), &version).unwrap();

    println!("cargo:rerun-if-env-changed=TELEMUZE_SERVER_BIN");
    println!("cargo:rerun-if-env-changed=TELEMUZE_SHERPA_SO");
    println!("cargo:rerun-if-env-changed=TELEMUZE_ONNXRUNTIME_SO");
    println!("cargo:rerun-if-changed={server_bin}");
    println!("cargo:rerun-if-changed={sherpa_so}");
    println!("cargo:rerun-if-changed={onnxruntime_so}");
}
