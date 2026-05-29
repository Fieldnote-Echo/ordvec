use crate::{verify_manifest, ManifestDocument, ManifestError, VerificationReport, VerifyOptions};
use chrono::{SecondsFormat, Utc};
use rusqlite::{params, Connection, OptionalExtension};
use std::path::Path;

pub fn verify_with_registry(
    db_path: impl AsRef<Path>,
    document: &ManifestDocument,
    manifest_path: impl AsRef<Path>,
    options: VerifyOptions,
    use_cache: bool,
) -> Result<VerificationReport, ManifestError> {
    let mut conn = Connection::open(db_path).map_err(sqlite_err)?;
    init(&conn)?;
    if use_cache {
        return load_cached_report(&conn, &document.manifest.manifest_id)?.ok_or_else(|| {
            ManifestError::invalid(format!(
                "no cached verification report for manifest_id {}",
                document.manifest.manifest_id
            ))
        });
    }

    let report = verify_manifest(document, options);
    store_report(&mut conn, document, manifest_path.as_ref(), &report)?;
    Ok(report)
}

pub fn activate(
    db_path: impl AsRef<Path>,
    document: &ManifestDocument,
    manifest_path: impl AsRef<Path>,
    options: VerifyOptions,
    force: bool,
) -> Result<VerificationReport, ManifestError> {
    let mut conn = Connection::open(db_path).map_err(sqlite_err)?;
    init(&conn)?;
    let report = verify_manifest(document, options);
    store_report(&mut conn, document, manifest_path.as_ref(), &report)?;
    if !report.ok && !force {
        return Ok(report);
    }

    conn.execute(
        "INSERT INTO active_manifest(id, manifest_id, manifest_path, activated_at, forced)
         VALUES(1, ?1, ?2, ?3, ?4)
         ON CONFLICT(id) DO UPDATE SET
           manifest_id=excluded.manifest_id,
           manifest_path=excluded.manifest_path,
           activated_at=excluded.activated_at,
           forced=excluded.forced",
        params![
            document.manifest.manifest_id,
            manifest_path.as_ref().display().to_string(),
            Utc::now().to_rfc3339_opts(SecondsFormat::Secs, true),
            i64::from(force),
        ],
    )
    .map_err(sqlite_err)?;
    Ok(report)
}

fn init(conn: &Connection) -> Result<(), ManifestError> {
    conn.execute_batch(
        "CREATE TABLE IF NOT EXISTS verification_reports(
            manifest_id TEXT NOT NULL,
            manifest_path TEXT NOT NULL,
            checked_at TEXT NOT NULL,
            ok INTEGER NOT NULL,
            report_json TEXT NOT NULL,
            PRIMARY KEY(manifest_id, checked_at)
        );
        CREATE TABLE IF NOT EXISTS active_manifest(
            id INTEGER PRIMARY KEY CHECK(id = 1),
            manifest_id TEXT NOT NULL,
            manifest_path TEXT NOT NULL,
            activated_at TEXT NOT NULL,
            forced INTEGER NOT NULL
        );",
    )
    .map_err(sqlite_err)?;
    Ok(())
}

fn store_report(
    conn: &mut Connection,
    document: &ManifestDocument,
    manifest_path: &Path,
    report: &VerificationReport,
) -> Result<(), ManifestError> {
    let tx = conn.transaction().map_err(sqlite_err)?;
    let report_json = serde_json::to_string(report)?;
    tx.execute(
        "INSERT OR REPLACE INTO verification_reports(
            manifest_id, manifest_path, checked_at, ok, report_json
         ) VALUES(?1, ?2, ?3, ?4, ?5)",
        params![
            document.manifest.manifest_id,
            manifest_path.display().to_string(),
            report.checked_at,
            i64::from(report.ok),
            report_json,
        ],
    )
    .map_err(sqlite_err)?;
    tx.commit().map_err(sqlite_err)?;
    Ok(())
}

fn load_cached_report(
    conn: &Connection,
    manifest_id: &str,
) -> Result<Option<VerificationReport>, ManifestError> {
    let report_json: Option<String> = conn
        .query_row(
            "SELECT report_json
             FROM verification_reports
             WHERE manifest_id = ?1
             ORDER BY checked_at DESC
             LIMIT 1",
            params![manifest_id],
            |row| row.get(0),
        )
        .optional()
        .map_err(sqlite_err)?;
    report_json
        .map(|json| serde_json::from_str(&json).map_err(ManifestError::from))
        .transpose()
}

fn sqlite_err(err: rusqlite::Error) -> ManifestError {
    ManifestError::invalid(format!("sqlite error: {err}"))
}
