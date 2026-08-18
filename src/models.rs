//! In-memory model registry.
//!
//! Used by `GET /models` for discovery and by `POST /infer` for early
//! request validation (unknown model names are rejected before they are
//! forwarded to the inference core).
//!
//! Two modes:
//! - Static snapshot (`Default`): used with the legacy `infer` protocol,
//!   when no inference core is reachable or queried. This matches the
//!   original MVP behaviour.
//! - Dynamic (`replace_all`): in `llama-chat` mode the gateway runs a
//!   background sync task that periodically pulls the real model list
//!   from llama-server's `GET /v1/models` and swaps it in, so the
//!   registry always mirrors what the inference core actually serves.

use serde::Serialize;
use std::collections::HashSet;
use std::sync::RwLock;

/// Public view of a registered model, matching the documented
/// `GET /models` wire contract (plus `backend`).
#[derive(Serialize, Clone)]
pub struct ModelInfo {
    pub name: String,
    pub status: String,
    pub device: String,
    pub backend: String,
}

/// Snapshot of the registry contents. Kept behind a `RwLock` so the
/// background sync task can replace it while handlers read it.
struct RegistryInner {
    models: Vec<ModelInfo>,
    /// Case-insensitive set of accepted names. Includes aliases when
    /// the list was fetched from llama-server.
    known_names: HashSet<String>,
}

/// In-memory registry of models the gateway knows about.
pub struct ModelRegistry {
    inner: RwLock<RegistryInner>,
}

impl ModelRegistry {
    /// List all registered models (newest snapshot).
    pub fn list(&self) -> Vec<ModelInfo> {
        self.inner
            .read()
            .expect("registry read lock poisoned")
            .models
            .clone()
    }

    /// Case-insensitive membership check used for request validation.
    pub fn contains(&self, name: &str) -> bool {
        self.inner
            .read()
            .expect("registry read lock poisoned")
            .known_names
            .contains(&name.to_ascii_lowercase())
    }

    /// Replace the whole registry with a fresh snapshot fetched from
    /// the inference core. Every model name plus the given aliases
    /// become accepted request names. Used only by the dynamic sync
    /// task; a failed sync leaves the previous snapshot untouched.
    pub fn replace_all(&self, models: Vec<ModelInfo>, aliases: impl IntoIterator<Item = String>) {
        let mut known_names: HashSet<String> = models
            .iter()
            .map(|model| model.name.to_ascii_lowercase())
            .collect();
        known_names.extend(aliases.into_iter().map(|alias| alias.to_ascii_lowercase()));
        let mut inner = self
            .inner
            .write()
            .expect("registry write lock poisoned");
        inner.models = models;
        inner.known_names = known_names;
    }
}

impl Default for ModelRegistry {
    fn default() -> Self {
        // Static fallback snapshot for the legacy `infer` protocol.
        // The first entry is the model actually deployed on the GPU
        // (loaded by llama-server, backend llama.cpp with CUDA). The
        // remaining entries are registered for discovery but not
        // currently loaded — the gateway validates request names against
        // the full list, while the inference core only serves the loaded
        // model. A dynamic registry (llama-chat mode) replaces this
        // snapshot with the real list from the core.
        let models = vec![
            ModelInfo {
                name: "qwen2.5-0.5b-instruct".to_string(),
                status: "loaded".to_string(),
                device: "gpu0".to_string(),
                backend: "llama.cpp-cuda".to_string(),
            },
            ModelInfo {
                name: "llama-7b".to_string(),
                status: "available".to_string(),
                device: "gpu0".to_string(),
                backend: "cuda".to_string(),
            },
            ModelInfo {
                name: "llama-13b".to_string(),
                status: "available".to_string(),
                device: "gpu0".to_string(),
                backend: "cuda".to_string(),
            },
            ModelInfo {
                name: "mistral-7b".to_string(),
                status: "available".to_string(),
                device: "gpu0".to_string(),
                backend: "cuda".to_string(),
            },
        ];
        let known_names = models
            .iter()
            .map(|model| model.name.to_ascii_lowercase())
            .collect();
        Self {
            inner: RwLock::new(RegistryInner {
                models,
                known_names,
            }),
        }
    }
}