use flate2::read::GzDecoder;
use std::env;
use std::ffi::CString;
use std::fs;
use std::io::Read;
use std::os::unix::fs::PermissionsExt;
use std::path::{Path, PathBuf};

const SERVER_GZ: &[u8] = include_bytes!(concat!(env!("OUT_DIR"), "/telemuze.gz"));
const SHERPA_SO_GZ: &[u8] = include_bytes!(concat!(env!("OUT_DIR"), "/libsherpa-onnx-c-api.so.gz"));
const ONNXRUNTIME_SO_GZ: &[u8] = include_bytes!(concat!(env!("OUT_DIR"), "/libonnxruntime.so.gz"));
const VERSION_HASH: &str = include_str!(concat!(env!("OUT_DIR"), "/version_hash.txt"));

fn decompress(data: &[u8]) -> Vec<u8> {
    let mut decoder = GzDecoder::new(data);
    let mut out = Vec::new();
    decoder.read_to_end(&mut out).expect("decompression failed");
    out
}

fn cache_dir() -> PathBuf {
    if let Ok(xdg) = env::var("XDG_CACHE_HOME") {
        PathBuf::from(xdg).join("telemuze")
    } else if let Ok(home) = env::var("HOME") {
        PathBuf::from(home).join(".cache").join("telemuze")
    } else {
        PathBuf::from("/tmp/telemuze-cache")
    }
}

fn needs_extract(cache: &Path) -> bool {
    let version_file = cache.join(".version");
    match fs::read_to_string(&version_file) {
        Ok(v) => v != VERSION_HASH,
        Err(_) => true,
    }
}

fn extract(cache: &Path) {
    let bin_dir = cache.join("bin");
    let lib_dir = cache.join("lib");
    fs::create_dir_all(&bin_dir).expect("failed to create cache bin dir");
    fs::create_dir_all(&lib_dir).expect("failed to create cache lib dir");

    eprintln!("Extracting runtime libraries...");

    let server_path = bin_dir.join("telemuze");
    fs::write(&server_path, decompress(SERVER_GZ)).expect("failed to write server binary");
    fs::set_permissions(&server_path, fs::Permissions::from_mode(0o755))
        .expect("failed to set server binary permissions");

    fs::write(
        lib_dir.join("libsherpa-onnx-c-api.so"),
        decompress(SHERPA_SO_GZ),
    )
    .expect("failed to write libsherpa-onnx-c-api.so");

    fs::write(
        lib_dir.join("libonnxruntime.so"),
        decompress(ONNXRUNTIME_SO_GZ),
    )
    .expect("failed to write libonnxruntime.so");

    fs::write(cache.join(".version"), VERSION_HASH).expect("failed to write version file");

    eprintln!("Done.");
}

fn main() {
    let cache = cache_dir();

    if needs_extract(&cache) {
        extract(&cache);
    }

    let server_bin = cache.join("bin").join("telemuze");
    let lib_dir = cache.join("lib");

    // Prepend our lib dir to LD_LIBRARY_PATH
    let ld_path = match env::var("LD_LIBRARY_PATH") {
        Ok(existing) => format!("{}:{}", lib_dir.display(), existing),
        Err(_) => lib_dir.display().to_string(),
    };

    // Collect args (skip argv[0], pass the rest through)
    let args: Vec<CString> = std::iter::once(CString::new(server_bin.to_str().unwrap()).unwrap())
        .chain(env::args().skip(1).map(|a| CString::new(a).unwrap()))
        .collect();

    // Build env, replacing/adding LD_LIBRARY_PATH
    let mut envs: Vec<CString> = env::vars()
        .filter(|(k, _)| k != "LD_LIBRARY_PATH")
        .map(|(k, v)| CString::new(format!("{k}={v}")).unwrap())
        .collect();
    envs.push(CString::new(format!("LD_LIBRARY_PATH={ld_path}")).unwrap());

    let arg_ptrs: Vec<*const libc::c_char> = args.iter().map(|a| a.as_ptr()).chain(std::iter::once(std::ptr::null())).collect();
    let env_ptrs: Vec<*const libc::c_char> = envs.iter().map(|e| e.as_ptr()).chain(std::iter::once(std::ptr::null())).collect();

    unsafe {
        libc::execve(arg_ptrs[0], arg_ptrs.as_ptr(), env_ptrs.as_ptr());
    }

    // execve only returns on error
    eprintln!(
        "Failed to exec {}: {}",
        server_bin.display(),
        std::io::Error::last_os_error()
    );
    std::process::exit(1);
}
