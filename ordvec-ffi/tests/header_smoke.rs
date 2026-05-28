use std::io::Write;
use std::process::Command;

fn temp_source(ext: &str, body: &str) -> std::path::PathBuf {
    let mut path = std::env::temp_dir();
    path.push(format!(
        "ordvec_header_smoke_{}_{}.{}",
        std::process::id(),
        std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_nanos(),
        ext
    ));
    std::fs::File::create(&path)
        .unwrap()
        .write_all(body.as_bytes())
        .unwrap();
    path
}

#[test]
fn header_compiles_as_c11() {
    let manifest = std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    let include = manifest.join("include");
    let src = temp_source(
        "c",
        r#"#include "ordvec.h"
#include "ordvec.h"

_Static_assert(sizeof(ordvec_index_info_t) == 128, "ordvec_index_info_t size");
_Static_assert(sizeof(ordvec_search_params_t) == 128, "ordvec_search_params_t size");
_Static_assert(sizeof(ordvec_hit_t) == 24, "ordvec_hit_t size");
_Static_assert(sizeof(ordvec_search_stats_t) == 184, "ordvec_search_stats_t size");

int main(void) {
    ordvec_search_params_t params;
    ordvec_search_params_init(&params);
    return (int)ordvec_abi_version() - 1;
}
"#,
    );
    let obj = src.with_extension("o");
    let status = Command::new("cc")
        .arg("-std=c11")
        .arg("-I")
        .arg(&include)
        .arg("-c")
        .arg(&src)
        .arg("-o")
        .arg(&obj)
        .status();
    std::fs::remove_file(&src).ok();
    std::fs::remove_file(&obj).ok();
    match status {
        Ok(status) => assert!(status.success(), "ordvec.h did not compile as C11"),
        Err(err) if err.kind() == std::io::ErrorKind::NotFound => {}
        Err(err) => panic!("failed to spawn C compiler: {err}"),
    }
}

#[test]
fn header_compiles_as_cpp() {
    let manifest = std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    let include = manifest.join("include");
    let src = temp_source(
        "cc",
        r#"#include "ordvec.h"
#include "ordvec.h"

static_assert(sizeof(ordvec_index_info_t) == 128, "ordvec_index_info_t size");
static_assert(sizeof(ordvec_search_params_t) == 128, "ordvec_search_params_t size");
static_assert(sizeof(ordvec_hit_t) == 24, "ordvec_hit_t size");
static_assert(sizeof(ordvec_search_stats_t) == 184, "ordvec_search_stats_t size");

int main() {
    ordvec_search_stats_t stats;
    ordvec_search_stats_init(&stats);
    return static_cast<int>(ordvec_abi_version()) - 1;
}
"#,
    );
    let obj = src.with_extension("o");
    let compiler = Command::new("c++")
        .arg("-std=c++17")
        .arg("-I")
        .arg(&include)
        .arg("-c")
        .arg(&src)
        .arg("-o")
        .arg(&obj)
        .status();
    std::fs::remove_file(&src).ok();
    std::fs::remove_file(&obj).ok();
    match compiler {
        Ok(status) => assert!(status.success(), "ordvec.h did not compile as C++17"),
        Err(err) if err.kind() == std::io::ErrorKind::NotFound => {}
        Err(err) => panic!("failed to spawn C++ compiler: {err}"),
    }
}
