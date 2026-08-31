use anyhow::{Context, Result};
use glmrt_core::{ModelFacts, TensorCatalog};
use serde::{Deserialize, Serialize};
use std::env;
use std::fs;
use std::path::{Path, PathBuf};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SnapshotResolution {
    pub model_id: String,
    pub cache_root: PathBuf,
    pub model_cache: PathBuf,
    pub snapshot_path: Option<PathBuf>,
    pub snapshots: Vec<PathBuf>,
}

pub fn default_hf_home() -> PathBuf {
    env::var_os("HF_HOME")
        .map(PathBuf::from)
        .or_else(|| env::var_os("HOME").map(|home| PathBuf::from(home).join(".cache/huggingface")))
        .unwrap_or_else(|| PathBuf::from("/root/.cache/huggingface"))
}

pub fn model_cache_dir(hf_home: &Path, model_id: &str) -> PathBuf {
    hf_home
        .join("hub")
        .join(format!("models--{}", model_id.replace('/', "--")))
}

pub fn resolve_snapshot(model_id: &str, hf_home: Option<&Path>) -> Result<SnapshotResolution> {
    let cache_root = hf_home
        .map(Path::to_path_buf)
        .unwrap_or_else(default_hf_home);
    let model_cache = model_cache_dir(&cache_root, model_id);
    let snapshots_root = model_cache.join("snapshots");
    let mut snapshots = Vec::new();
    if snapshots_root.is_dir() {
        for entry in fs::read_dir(&snapshots_root)
            .with_context(|| format!("reading {}", snapshots_root.display()))?
        {
            let entry = entry?;
            if entry.file_type()?.is_dir() {
                snapshots.push(entry.path());
            }
        }
    }
    snapshots.sort();
    let main_ref = model_cache.join("refs/main");
    let snapshot_path = match fs::read_to_string(&main_ref) {
        Ok(revision) => {
            let revision = revision.trim();
            anyhow::ensure!(
                !revision.is_empty(),
                "Hugging Face main ref is empty: {}",
                main_ref.display()
            );
            Some(
                snapshots
                    .iter()
                    .find(|snapshot| {
                        snapshot.file_name().and_then(|name| name.to_str()) == Some(revision)
                    })
                    .cloned()
                    .with_context(|| {
                        format!(
                            "Hugging Face main ref {} selects missing snapshot {revision}",
                            main_ref.display()
                        )
                    })?,
            )
        }
        Err(error) if error.kind() == std::io::ErrorKind::NotFound => snapshots.last().cloned(),
        Err(error) => return Err(error).with_context(|| format!("reading {}", main_ref.display())),
    };
    Ok(SnapshotResolution {
        model_id: model_id.to_owned(),
        cache_root,
        model_cache,
        snapshot_path,
        snapshots,
    })
}

#[cfg(test)]
mod tests {
    use super::{model_cache_dir, resolve_snapshot};

    #[test]
    fn main_ref_selects_snapshot_instead_of_lexicographic_order() {
        let temporary = tempfile::tempdir().unwrap();
        let model_id = "owner/model";
        let cache = model_cache_dir(temporary.path(), model_id);
        std::fs::create_dir_all(cache.join("snapshots/aaa")).unwrap();
        std::fs::create_dir_all(cache.join("snapshots/zzz")).unwrap();
        std::fs::create_dir_all(cache.join("refs")).unwrap();
        std::fs::write(cache.join("refs/main"), "aaa\n").unwrap();

        let resolution = resolve_snapshot(model_id, Some(temporary.path())).unwrap();
        assert_eq!(
            resolution.snapshot_path.unwrap(),
            cache.join("snapshots/aaa")
        );
    }

    #[test]
    fn missing_main_ref_preserves_legacy_latest_snapshot_fallback() {
        let temporary = tempfile::tempdir().unwrap();
        let model_id = "owner/model";
        let cache = model_cache_dir(temporary.path(), model_id);
        std::fs::create_dir_all(cache.join("snapshots/aaa")).unwrap();
        std::fs::create_dir_all(cache.join("snapshots/zzz")).unwrap();

        let resolution = resolve_snapshot(model_id, Some(temporary.path())).unwrap();
        assert_eq!(
            resolution.snapshot_path.unwrap(),
            cache.join("snapshots/zzz")
        );
    }

    #[test]
    fn dangling_main_ref_is_rejected() {
        let temporary = tempfile::tempdir().unwrap();
        let model_id = "owner/model";
        let cache = model_cache_dir(temporary.path(), model_id);
        std::fs::create_dir_all(cache.join("snapshots/aaa")).unwrap();
        std::fs::create_dir_all(cache.join("refs")).unwrap();
        std::fs::write(cache.join("refs/main"), "missing\n").unwrap();

        let error = resolve_snapshot(model_id, Some(temporary.path())).unwrap_err();
        assert!(error
            .to_string()
            .contains("selects missing snapshot missing"));
    }
}

pub fn empty_catalog_for_snapshot(model_id: &str, snapshot_path: &Path) -> TensorCatalog {
    TensorCatalog {
        model_id: model_id.to_owned(),
        snapshot_path: snapshot_path.display().to_string(),
        facts: ModelFacts {
            model_id: model_id.to_owned(),
            ..ModelFacts::default()
        },
        tensors: Vec::new(),
    }
}
