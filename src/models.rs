//! Simple in-memory model registry.
//!
//! Used by `GET /models` for discovery and by `POST /infer` for early
//! request validation (unknown model names are rejected before they are
//! forwarded to the inference core). A real deployment would load this
//! registry from storage or query the C++ core; here it is a small
//! static snapshot that can grow into a dynamic implementation.

use serde::Serialize;
use std::collections::HashSet;

/// Public view of a registered model, matching the documented
/// `GET /models` wire contract (plus `backend`).
#[derive(Serialize, Clone)]
pub struct ModelInfo {
    pub name: String,
    pub status: String,
    pub device: String,
    pub backend: String,
}

/// In-memory registry of models the gateway knows about.
pub struct ModelRegistry {
    models: Vec<ModelInfo>,
    known_names: HashSet<String>,
}

impl ModelRegistry {
    /// List all registered models (newest snapshot).
    pub fn list(&self) -> Vec<ModelInfo> {
        self.models.clone()
    }

    /// Case-insensitive membership check used for request validation.
    pub fn contains(&self, name: &str) -> bool {
        self.known_names.contains(&name.to_ascii_lowercase())
    }
}

impl Default for ModelRegistry {
    fn default() -> Self {
        // The first entry is the model actually deployed on the GPU
        // (loaded by llama-server, backend llama.cpp with CUDA). The
        // remaining entries are registered for discovery but not
        // currently loaded — the gateway validates request names against
        // the full list, while the inference core only serves the loaded
        // model. A dynamic registry (Phase 3) will replace this snapshot.
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
            models,
            known_names,
        }
    }
}
