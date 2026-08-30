use std::path::PathBuf;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    // tonic-build / prost-build needs the `protoc` binary. It honours the
    // `PROTOC` env var, then PATH. On developer machines protoc is often
    // installed by pip (`pip install protobuf`) or a package manager into a
    // location that is NOT on PATH; probe the common ones and export `PROTOC`
    // so `cargo build` just works without manual setup.
    if std::env::var_os("PROTOC").is_none() {
        if let Some(path) = find_protoc() {
            std::env::set_var("PROTOC", &path);
            println!("cargo:warning=protoc not on PATH; using {}", path.display());
        }
    }

    tonic_build::compile_protos("proto/inference.proto")?;
    Ok(())
}

/// Fallback locations checked when `PROTOC` is unset and `protoc` is not on
/// PATH. Covers pip (`pip install protobuf`), anaconda, and Homebrew installs.
fn find_protoc() -> Option<PathBuf> {
    let home = std::env::var_os("HOME")
        .or_else(|| std::env::var_os("USERPROFILE"))
        .map(PathBuf::from)?;

    const RELATIVE: &[&str] = &[
        ".local/bin/bin/protoc.exe",      // Windows, pip --user
        ".local/bin/protoc",              // Linux/macOS, pip --user
        "anaconda3/bin/protoc",           // anaconda (Linux/macOS)
        "Anaconda3/Library/bin/protoc.exe", // anaconda (Windows)
        "Miniconda3/Library/bin/protoc.exe", // miniconda (Windows)
    ];

    RELATIVE
        .iter()
        .map(|rel| home.join(rel))
        .find(|candidate| candidate.is_file())
}