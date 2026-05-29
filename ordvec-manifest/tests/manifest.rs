use ordvec::{Bitmap, Rank, RankQuant, SignBitmap};
use ordvec_manifest::{
    create_manifest_for_index, create_manifest_for_index_with_options, load_manifest_file,
    sha256_file, verify_index_manifest, verify_manifest_with_base, CreateManifestOptions,
    CreateRowIdentity, ManifestIndexParams, RowIdentity, VerifyOptions,
};
use serde_json::json;
use std::fs;
use std::io::Write;
use std::path::{Path, PathBuf};
use std::process::Command;

fn write_index(dir: &Path) -> PathBuf {
    let path = dir.join("index.tvrq");
    let mut index = RankQuant::new(16, 2);
    let docs: Vec<f32> = (0..32).map(|i| i as f32 - 12.0).collect();
    index.add(&docs);
    index.write(&path).unwrap();
    path
}

#[derive(Clone, Copy)]
enum FixtureKind {
    Rank,
    RankQuant,
    Bitmap,
    SignBitmap,
}

fn write_index_kind(dir: &Path, kind: FixtureKind) -> PathBuf {
    match kind {
        FixtureKind::Rank => {
            let path = dir.join("index.tvr");
            let mut index = Rank::new(8);
            index.add(&[
                1.0, 3.0, 2.0, 4.0, 8.0, 7.0, 6.0, 5.0, 8.0, 6.0, 7.0, 5.0, 1.0, 2.0, 3.0, 4.0,
            ]);
            index.write(&path).unwrap();
            path
        }
        FixtureKind::RankQuant => write_index(dir),
        FixtureKind::Bitmap => {
            let path = dir.join("index.tvbm");
            let mut index = Bitmap::new(64, 16);
            let docs: Vec<f32> = (0..128).map(|i| ((i * 17) % 31) as f32).collect();
            index.add(&docs);
            index.write(&path).unwrap();
            path
        }
        FixtureKind::SignBitmap => {
            let path = dir.join("index.tvsb");
            let mut index = SignBitmap::new(64);
            let docs: Vec<f32> = (0usize..128)
                .map(|i| if i.is_multiple_of(3) { 1.0 } else { -1.0 })
                .collect();
            index.add(&docs);
            index.write(&path).unwrap();
            path
        }
    }
}

fn write_row_map(path: &Path, rows: &[(&str, Option<&str>)]) {
    let mut file = fs::File::create(path).unwrap();
    for (row_id, (db_id, parent_id)) in rows.iter().enumerate() {
        let value = if let Some(parent_id) = parent_id {
            json!({"row_id": row_id, "db_id": db_id, "parent_id": parent_id})
        } else {
            json!({"row_id": row_id, "db_id": db_id})
        };
        writeln!(file, "{value}").unwrap();
    }
}

fn identity_manifest(dir: &Path) -> (tempfile::TempDir, ordvec_manifest::IndexManifest, PathBuf) {
    let temp = tempfile::tempdir_in(dir).unwrap();
    let index = write_index(temp.path());
    let manifest_path = temp.path().join("manifest.json");
    let manifest = create_manifest_for_index(
        &index,
        CreateRowIdentity::RowIdIdentity,
        "test-embedding",
        &manifest_path,
    )
    .unwrap();
    (temp, manifest, manifest_path)
}

#[test]
fn create_then_verify_identity_manifest_for_all_persisted_formats() {
    let temp = tempfile::tempdir().unwrap();
    for (kind, expected) in [
        (FixtureKind::Rank, ordvec_manifest::ManifestIndexKind::Rank),
        (
            FixtureKind::RankQuant,
            ordvec_manifest::ManifestIndexKind::RankQuant,
        ),
        (
            FixtureKind::Bitmap,
            ordvec_manifest::ManifestIndexKind::Bitmap,
        ),
        (
            FixtureKind::SignBitmap,
            ordvec_manifest::ManifestIndexKind::SignBitmap,
        ),
    ] {
        let case = tempfile::tempdir_in(temp.path()).unwrap();
        let index = write_index_kind(case.path(), kind);
        let manifest_path = case.path().join("manifest.json");
        let manifest = create_manifest_for_index(
            &index,
            CreateRowIdentity::RowIdIdentity,
            "test-embedding",
            &manifest_path,
        )
        .unwrap();

        let report = verify_manifest_with_base(manifest, case.path(), VerifyOptions::default());
        assert!(report.ok, "{:?}", report.errors);
        assert_eq!(report.skipped_checks, ["attestations_absent"]);
        assert_eq!(report.artifact.metadata.unwrap().kind, expected);
    }
}

#[test]
fn create_manifest_creates_output_parent_for_programmatic_callers() {
    let temp = tempfile::tempdir().unwrap();
    let index = write_index(temp.path());
    let manifest_path = temp.path().join("nested").join("manifest.json");

    let manifest = create_manifest_for_index_with_options(
        &index,
        CreateRowIdentity::RowIdIdentity,
        "test-embedding",
        &manifest_path,
        CreateManifestOptions {
            allow_path_escape: true,
            ..CreateManifestOptions::default()
        },
    )
    .unwrap();

    assert!(manifest_path.parent().unwrap().is_dir());
    assert_eq!(manifest.row_identity.row_count(), 2);
}

#[test]
fn schema_rejects_unknown_fields_and_bad_extension_keys() {
    let root = tempfile::tempdir().unwrap();
    let (temp, mut manifest, _manifest_path) = identity_manifest(root.path());

    let mut value = serde_json::to_value(&manifest).unwrap();
    value
        .as_object_mut()
        .unwrap()
        .insert("unknown".to_string(), json!(true));
    let parsed = serde_json::from_value::<ordvec_manifest::IndexManifest>(value);
    assert!(
        parsed.is_err(),
        "schema-owned structs must reject unknown fields"
    );

    manifest
        .extensions
        .insert("policy".to_string(), json!({"decision": "deny"}));
    let report = verify_manifest_with_base(manifest.clone(), temp.path(), VerifyOptions::default());
    assert!(report
        .errors
        .iter()
        .any(|issue| issue.code == "extension_key_not_namespaced"));

    manifest.extensions.clear();
    manifest.extensions.insert(
        "com.example.policy".to_string(),
        json!({"decision": "allow"}),
    );
    let report = verify_manifest_with_base(manifest, temp.path(), VerifyOptions::default());
    assert!(report.ok, "{:?}", report.errors);
}

#[test]
fn schema_enforces_lowercase_sha256_and_optional_field_shapes() {
    let root = tempfile::tempdir().unwrap();
    let (temp, mut manifest, _manifest_path) = identity_manifest(root.path());
    manifest.artifact.sha256 = manifest.artifact.sha256.to_ascii_uppercase();
    manifest.row_identity = RowIdentity::Jsonl {
        path: "rows.jsonl".to_string(),
        sha256: "A".repeat(64),
        row_count: 2,
        id_kind: "uuid".to_string(),
        db: None,
    };
    manifest.embedding.model_revision = Some("".to_string());
    manifest.embedding.corpus_digest = Some("A".repeat(64));
    manifest.embedding.embedding_matrix_digest = Some("not-a-digest".to_string());
    manifest.embedding.normalization = Some("".to_string());
    manifest.build.as_mut().unwrap().source_repo = Some("".to_string());

    let report = verify_manifest_with_base(manifest, temp.path(), VerifyOptions::default());
    for code in [
        "artifact_sha256_invalid",
        "row_identity_sha256_invalid",
        "embedding_model_revision_empty",
        "embedding_corpus_digest_invalid",
        "embedding_matrix_digest_invalid",
        "embedding_normalization_empty",
        "build_source_repo_empty",
    ] {
        assert!(
            report.errors.iter().any(|issue| issue.code == code),
            "missing {code}: {:?}",
            report.errors
        );
    }
}

#[test]
fn artifact_metadata_mismatches_are_reported_with_stable_codes() {
    let root = tempfile::tempdir().unwrap();
    let (temp, mut manifest, _manifest_path) = identity_manifest(root.path());
    manifest.artifact.dim += 1;
    manifest.embedding.dim += 1;

    let report = verify_manifest_with_base(manifest, temp.path(), VerifyOptions::default());
    assert!(!report.ok);
    assert!(report
        .errors
        .iter()
        .any(|issue| issue.code == "artifact_dim_mismatch"));

    let (temp, mut manifest, _manifest_path) = identity_manifest(root.path());
    manifest.artifact.params = ManifestIndexParams::RankQuant { bits: 4 };
    let report = verify_manifest_with_base(manifest, temp.path(), VerifyOptions::default());
    assert!(report
        .errors
        .iter()
        .any(|issue| issue.code == "artifact_params_mismatch"));

    let case = tempfile::tempdir_in(root.path()).unwrap();
    let bitmap = write_index_kind(case.path(), FixtureKind::Bitmap);
    let manifest_path = case.path().join("bitmap.manifest.json");
    let mut manifest = create_manifest_for_index(
        &bitmap,
        CreateRowIdentity::RowIdIdentity,
        "test-embedding",
        &manifest_path,
    )
    .unwrap();
    manifest.artifact.params = ManifestIndexParams::Bitmap { n_top: 8 };
    let report = verify_manifest_with_base(manifest, case.path(), VerifyOptions::default());
    assert!(report
        .errors
        .iter()
        .any(|issue| issue.code == "artifact_params_mismatch"));
}

#[test]
fn missing_artifact_and_row_count_mismatch_are_reported() {
    let root = tempfile::tempdir().unwrap();
    let (temp, mut manifest, _manifest_path) = identity_manifest(root.path());
    manifest.row_identity = RowIdentity::RowIdIdentity { row_count: 1 };
    let report = verify_manifest_with_base(manifest.clone(), temp.path(), VerifyOptions::default());
    assert!(report
        .errors
        .iter()
        .any(|issue| issue.code == "artifact_row_count_mismatch"));

    manifest.row_identity = RowIdentity::RowIdIdentity { row_count: 2 };
    fs::remove_file(temp.path().join(&manifest.artifact.path)).unwrap();
    let report = verify_manifest_with_base(manifest, temp.path(), VerifyOptions::default());
    assert!(report
        .errors
        .iter()
        .any(|issue| issue.code == "artifact_path_unavailable"));
}

#[test]
fn path_policy_rejects_escapes_and_absolute_paths_by_default() {
    let root = tempfile::tempdir().unwrap();
    let base = root.path().join("manifests");
    fs::create_dir(&base).unwrap();
    let index = write_index(root.path());
    let manifest_path = base.join("manifest.json");
    let mut manifest = create_manifest_for_index_with_options(
        &index,
        CreateRowIdentity::RowIdIdentity,
        "test-embedding",
        &manifest_path,
        CreateManifestOptions {
            allow_path_escape: true,
            ..CreateManifestOptions::default()
        },
    )
    .unwrap();

    manifest.artifact.path = "../index.tvrq".to_string();
    let report = verify_manifest_with_base(manifest.clone(), &base, VerifyOptions::default());
    assert!(report
        .errors
        .iter()
        .any(|issue| issue.code == "artifact_path_escape_rejected"));

    let report = verify_manifest_with_base(
        manifest.clone(),
        &base,
        VerifyOptions {
            allow_path_escape: true,
            ..VerifyOptions::default()
        },
    );
    assert!(report.ok, "{:?}", report.errors);

    manifest.artifact.path = index.display().to_string();
    let report = verify_manifest_with_base(manifest.clone(), &base, VerifyOptions::default());
    assert!(report
        .errors
        .iter()
        .any(|issue| issue.code == "artifact_absolute_path_rejected"));

    let report = verify_manifest_with_base(
        manifest,
        &base,
        VerifyOptions {
            allow_absolute_paths: true,
            allow_path_escape: true,
            ..VerifyOptions::default()
        },
    );
    assert!(report.ok, "{:?}", report.errors);
}

#[cfg(unix)]
#[test]
fn symlink_escape_reports_observed_canonical_path() {
    use std::os::unix::fs::symlink;

    let root = tempfile::tempdir().unwrap();
    let base = root.path().join("base");
    let outside = root.path().join("outside");
    fs::create_dir(&base).unwrap();
    fs::create_dir(&outside).unwrap();
    let index = write_index(&outside);
    symlink(&index, base.join("link.tvrq")).unwrap();
    let manifest_path = base.join("manifest.json");
    let mut manifest = create_manifest_for_index_with_options(
        &index,
        CreateRowIdentity::RowIdIdentity,
        "test-embedding",
        &manifest_path,
        CreateManifestOptions {
            allow_path_escape: true,
            ..CreateManifestOptions::default()
        },
    )
    .unwrap();
    manifest.artifact.path = "link.tvrq".to_string();

    let report = verify_manifest_with_base(manifest.clone(), &base, VerifyOptions::default());
    assert!(report
        .errors
        .iter()
        .any(|issue| issue.code == "artifact_path_escape_rejected"));

    let report = verify_manifest_with_base(
        manifest,
        &base,
        VerifyOptions {
            allow_path_escape: true,
            ..VerifyOptions::default()
        },
    );
    assert!(report.ok, "{:?}", report.errors);
    assert_eq!(
        PathBuf::from(report.artifact.canonical_path.unwrap()),
        fs::canonicalize(index).unwrap()
    );
}

#[test]
fn jsonl_row_identity_is_strict_and_duplicate_ids_need_opt_in() {
    let temp = tempfile::tempdir().unwrap();
    let index = write_index(temp.path());
    let rows = temp.path().join("rows.jsonl");
    write_row_map(
        &rows,
        &[
            ("00000000-0000-0000-0000-000000000001", None),
            ("00000000-0000-0000-0000-000000000001", None),
        ],
    );
    let row_hash = sha256_file(&rows).unwrap();
    let manifest_path = temp.path().join("manifest.json");
    let mut manifest = create_manifest_for_index(
        &index,
        CreateRowIdentity::RowIdIdentity,
        "test-embedding",
        &manifest_path,
    )
    .unwrap();
    manifest.row_identity = RowIdentity::Jsonl {
        path: "rows.jsonl".to_string(),
        sha256: row_hash.sha256,
        row_count: 2,
        id_kind: "uuid".to_string(),
        db: None,
    };

    let report = verify_manifest_with_base(manifest.clone(), temp.path(), VerifyOptions::default());
    assert!(report
        .errors
        .iter()
        .any(|issue| issue.code == "row_identity_duplicate_db_id"));

    let report = verify_manifest_with_base(
        manifest,
        temp.path(),
        VerifyOptions {
            allow_duplicate_db_ids: true,
            ..VerifyOptions::default()
        },
    );
    assert!(report.ok, "{:?}", report.errors);

    fs::write(
        &rows,
        "{\"row_id\":1,\"db_id\":\"\"}\n{\"row_id\":1,\"db_id\":\"ok\",\"extra\":true}\n",
    )
    .unwrap();
    let row_hash = sha256_file(&rows).unwrap();
    let mut manifest = create_manifest_for_index(
        &index,
        CreateRowIdentity::RowIdIdentity,
        "test-embedding",
        &manifest_path,
    )
    .unwrap();
    manifest.row_identity = RowIdentity::Jsonl {
        path: "rows.jsonl".to_string(),
        sha256: row_hash.sha256,
        row_count: 2,
        id_kind: "uuid".to_string(),
        db: None,
    };
    let report = verify_manifest_with_base(manifest, temp.path(), VerifyOptions::default());
    assert!(report
        .errors
        .iter()
        .any(|issue| issue.code == "row_identity_jsonl_invalid_json"));
    assert!(report
        .errors
        .iter()
        .any(|issue| issue.code == "row_identity_row_id_mismatch"));
}

#[test]
fn attestation_shape_requires_matching_subject_sha256() {
    let root = tempfile::tempdir().unwrap();
    let (temp, mut manifest, _manifest_path) = identity_manifest(root.path());
    manifest.attestations.push(json!({
        "predicateType": "https://slsa.dev/provenance/v1",
        "predicate": {"builder": {"id": "builder"}},
        "subject": [{"name": "index.tvrq", "digest": {"sha256": "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa"}}]
    }));

    let report = verify_manifest_with_base(manifest.clone(), temp.path(), VerifyOptions::default());
    assert!(report
        .errors
        .iter()
        .any(|issue| issue.code == "attestation_subject_sha256_mismatch"));

    let sha = manifest.artifact.sha256.clone();
    manifest.attestations[0]["subject"][0]["digest"]["sha256"] = json!(sha);
    let report = verify_manifest_with_base(manifest, temp.path(), VerifyOptions::default());
    assert!(report.ok, "{:?}", report.errors);
    assert_eq!(
        report.attestation_shape_checks[0].predicate_type.as_deref(),
        Some("https://slsa.dev/provenance/v1")
    );
}

#[test]
fn cli_create_verify_and_exit_codes() {
    let temp = tempfile::tempdir().unwrap();
    let index = write_index(temp.path());
    let manifest = temp.path().join("manifest.json");
    let bin = env!("CARGO_BIN_EXE_ordvec-manifest");

    let output = Command::new(bin)
        .args([
            "create",
            "--index",
            index.to_str().unwrap(),
            "--row-id-is-identity",
            "--embedding-model",
            "test-embedding",
            "--out",
            manifest.to_str().unwrap(),
        ])
        .output()
        .unwrap();
    assert!(
        output.status.success(),
        "{}",
        String::from_utf8_lossy(&output.stderr)
    );

    let output = Command::new(bin)
        .args(["verify", "--manifest", manifest.to_str().unwrap(), "--json"])
        .output()
        .unwrap();
    assert!(
        output.status.success(),
        "{}",
        String::from_utf8_lossy(&output.stderr)
    );

    let mut document = load_manifest_file(&manifest).unwrap();
    document.manifest.artifact.sha256 =
        "bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb".to_string();
    fs::write(
        &manifest,
        serde_json::to_string_pretty(&document.manifest).unwrap(),
    )
    .unwrap();
    let output = Command::new(bin)
        .args(["verify", "--manifest", manifest.to_str().unwrap()])
        .output()
        .unwrap();
    assert_eq!(output.status.code(), Some(1));

    let output = Command::new(bin)
        .args([
            "create",
            "--index",
            index.to_str().unwrap(),
            "--embedding-model",
            "test-embedding",
            "--out",
            manifest.to_str().unwrap(),
        ])
        .output()
        .unwrap();
    assert_eq!(output.status.code(), Some(2));
}

#[test]
fn create_outside_manifest_dir_requires_explicit_path_policy() {
    let temp = tempfile::tempdir().unwrap();
    let outside = temp.path().join("outside");
    let manifests = temp.path().join("manifests");
    fs::create_dir(&outside).unwrap();
    let index = write_index(&outside);
    let manifest_path = manifests.join("manifest.json");

    let err = create_manifest_for_index(
        &index,
        CreateRowIdentity::RowIdIdentity,
        "test-embedding",
        &manifest_path,
    )
    .unwrap_err();
    assert!(err.to_string().contains("outside manifest directory"));

    let bin = env!("CARGO_BIN_EXE_ordvec-manifest");
    let output = Command::new(bin)
        .args([
            "create",
            "--index",
            index.to_str().unwrap(),
            "--row-id-is-identity",
            "--embedding-model",
            "test-embedding",
            "--out",
            manifest_path.to_str().unwrap(),
        ])
        .output()
        .unwrap();
    assert_eq!(output.status.code(), Some(2));

    let output = Command::new(bin)
        .args([
            "create",
            "--index",
            index.to_str().unwrap(),
            "--row-id-is-identity",
            "--embedding-model",
            "test-embedding",
            "--out",
            manifest_path.to_str().unwrap(),
            "--allow-path-escape",
        ])
        .output()
        .unwrap();
    assert!(
        output.status.success(),
        "{}",
        String::from_utf8_lossy(&output.stderr)
    );

    let output = Command::new(bin)
        .args(["verify", "--manifest", manifest_path.to_str().unwrap()])
        .output()
        .unwrap();
    assert_eq!(output.status.code(), Some(1));

    let output = Command::new(bin)
        .args([
            "verify",
            "--manifest",
            manifest_path.to_str().unwrap(),
            "--allow-path-escape",
        ])
        .output()
        .unwrap();
    assert!(
        output.status.success(),
        "{}",
        String::from_utf8_lossy(&output.stderr)
    );
}

#[test]
fn verify_index_manifest_uses_explicit_index_override() {
    let temp = tempfile::tempdir().unwrap();
    let index = write_index(temp.path());
    let manifest_path = temp.path().join("manifest.json");
    let mut manifest = create_manifest_for_index(
        &index,
        CreateRowIdentity::RowIdIdentity,
        "test-embedding",
        &manifest_path,
    )
    .unwrap();
    manifest.artifact.path = "missing.tvrq".to_string();
    fs::write(
        &manifest_path,
        serde_json::to_string_pretty(&manifest).unwrap(),
    )
    .unwrap();

    let report = verify_index_manifest(
        PathBuf::from("index.tvrq"),
        &manifest_path,
        VerifyOptions::default(),
    )
    .unwrap();
    assert!(report.ok, "{:?}", report.errors);
}

#[cfg(feature = "sqlite")]
#[test]
fn sqlite_cache_is_explicit_and_activation_reverifies_by_default() {
    use rusqlite::Connection;
    use std::fs::OpenOptions;

    let temp = tempfile::tempdir().unwrap();
    let index = write_index(temp.path());
    let manifest_path = temp.path().join("manifest.json");
    let manifest = create_manifest_for_index(
        &index,
        CreateRowIdentity::RowIdIdentity,
        "test-embedding",
        &manifest_path,
    )
    .unwrap();
    fs::write(
        &manifest_path,
        serde_json::to_string_pretty(&manifest).unwrap(),
    )
    .unwrap();
    let document = load_manifest_file(&manifest_path).unwrap();
    let db = temp.path().join("registry.sqlite");

    let report = ordvec_manifest::sqlite::verify_with_registry(
        &db,
        &document,
        &manifest_path,
        VerifyOptions::default(),
        true,
    )
    .unwrap();
    assert!(report.ok, "{:?}", report.errors);

    let second_fresh = ordvec_manifest::sqlite::verify_with_registry(
        &db,
        &document,
        &manifest_path,
        VerifyOptions::default(),
        false,
    )
    .unwrap();
    assert!(second_fresh.ok, "{:?}", second_fresh.errors);

    let conn = Connection::open(&db).unwrap();
    let count: i64 = conn
        .query_row("SELECT COUNT(*) FROM verification_reports", [], |row| {
            row.get(0)
        })
        .unwrap();
    assert!(
        count >= 2,
        "rapid verifications must preserve report history"
    );

    OpenOptions::new()
        .append(true)
        .open(&index)
        .unwrap()
        .write_all(b"\0")
        .unwrap();

    let cached = ordvec_manifest::sqlite::verify_with_registry(
        &db,
        &document,
        &manifest_path,
        VerifyOptions::default(),
        true,
    )
    .unwrap();
    assert!(
        !cached.ok,
        "cache key mismatch must force fresh verification"
    );

    let fresh = ordvec_manifest::sqlite::verify_with_registry(
        &db,
        &document,
        &manifest_path,
        VerifyOptions::default(),
        false,
    )
    .unwrap();
    assert!(!fresh.ok);

    let activation = ordvec_manifest::sqlite::activate(
        &db,
        &document,
        &manifest_path,
        VerifyOptions::default(),
        false,
    )
    .unwrap();
    assert!(!activation.ok);

    let forced = ordvec_manifest::sqlite::activate(
        &db,
        &document,
        &manifest_path,
        VerifyOptions::default(),
        true,
    )
    .unwrap();
    assert!(!forced.ok);
    assert!(forced
        .warnings
        .iter()
        .any(|issue| issue.code == "sqlite_activation_forced"));

    let bin = env!("CARGO_BIN_EXE_ordvec-manifest");
    let output = Command::new(bin)
        .args([
            "sqlite",
            "activate",
            "--db",
            db.to_str().unwrap(),
            "--manifest",
            manifest_path.to_str().unwrap(),
            "--force",
            "--json",
        ])
        .output()
        .unwrap();
    assert_eq!(output.status.code(), Some(0));
    let forced_report: ordvec_manifest::VerificationReport =
        serde_json::from_slice(&output.stdout).unwrap();
    assert!(!forced_report.ok);
    assert!(forced_report
        .warnings
        .iter()
        .any(|issue| issue.code == "sqlite_activation_forced"));
}
