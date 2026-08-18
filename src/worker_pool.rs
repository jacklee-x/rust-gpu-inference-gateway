//! Adaptive worker pool.
//!
//! Replaces the fixed-size semaphore of the dispatcher with a pool whose
//! concurrency limit grows and shrinks with the observed queue length:
//! - Idle (empty queue): the pool settles on `min_workers`.
//! - Growing backlog: the pool adds permits up to `max_workers` so more
//!   jobs are dispatched concurrently instead of piling up in the queue.
//!
//! The pool is a thin wrapper around a `tokio::sync::Semaphore` plus an
//! atomic counter of the currently allowed permit count. Growing adds
//! permits; shrinking forgets surplus permits (workers already running
//! are never interrupted, they just do not get successors until the
//! pool grows again). All counters are relaxed atomics; exactness is
//! not required for scaling decisions.

use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::Arc;
use tokio::sync::Semaphore;

/// Adaptive semaphore-backed worker pool.
pub struct AdaptiveWorkerPool {
    semaphore: Arc<Semaphore>,
    /// Currently allowed concurrent workers (the pool size).
    current: AtomicUsize,
    min_workers: usize,
    max_workers: usize,
}

impl AdaptiveWorkerPool {
    /// Create a pool with the given bounds, starting at `min_workers`.
    /// `min_workers` must be >= 1 and <= `max_workers`; out-of-range
    /// values are clamped (min is never below 1, and never above max).
    pub fn new(min_workers: usize, max_workers: usize) -> Self {
        let max_workers = max_workers.max(1);
        let min_workers = min_workers.min(max_workers).max(1);
        Self {
            semaphore: Arc::new(Semaphore::new(min_workers)),
            current: AtomicUsize::new(min_workers),
            min_workers,
            max_workers,
        }
    }

    /// Acquire an owned permit, waiting until one is available.
    pub async fn acquire(&self) -> tokio::sync::OwnedSemaphorePermit {
        self.semaphore.clone().acquire_owned().await.unwrap()
    }

    /// Re-evaluate the desired pool size from the current queue length
    /// and apply the delta (grow or shrink permits). Returns the new
    /// pool size. Called by the dispatcher once per dispatched job, so
    /// the pool reacts to backpressure within one queue cycle.
    pub fn reconfigure(&self, queue_len: usize) -> usize {
        let desired = self.plan(queue_len);
        let mut current = self.current.load(Ordering::Relaxed);

        // Loop only as long as another reconfigure raced us; in the
        // common case this exits after one iteration.
        loop {
            if desired == current {
                return desired;
            }
            let delta = desired as isize - current as isize;
            if delta > 0 {
                self.semaphore.add_permits(delta as usize);
            } else {
                self.semaphore.forget_permits((-delta) as usize);
            }
            // If the counter moved since we read it, recompute against
            // the new value so we never add/forget more than needed.
            match self.current.compare_exchange(
                current,
                desired,
                Ordering::Relaxed,
                Ordering::Relaxed,
            ) {
                Ok(_) => return desired,
                Err(actual) => current = actual,
            }
        }
    }

    /// Scaling policy: one worker handles about two queued jobs before
    /// the pool grows by one, capped by the configured bounds. Idle
    /// queues collapse back to `min_workers` on the next dispatch.
    fn plan(&self, queue_len: usize) -> usize {
        let base = self.min_workers;
        let extra = queue_len / 2;
        (base + extra).clamp(self.min_workers, self.max_workers)
    }

    /// Current pool size (for logging/metrics).
    pub fn current_size(&self) -> usize {
        self.current.load(Ordering::Relaxed)
    }

    /// Configured upper bound (for logging/metrics).
    pub fn max_size(&self) -> usize {
        self.max_workers
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn starts_at_min_workers() {
        let pool = AdaptiveWorkerPool::new(1, 8);
        assert_eq!(pool.current_size(), 1);
        assert_eq!(pool.max_size(), 8);
    }

    #[test]
    fn clamps_bounds() {
        // min < 1 is clamped up, min > max is clamped down to max.
        let pool = AdaptiveWorkerPool::new(0, 0);
        assert_eq!(pool.current_size(), 1);
        let pool = AdaptiveWorkerPool::new(9, 4);
        assert_eq!(pool.current_size(), 4);
    }

    #[test]
    fn grows_and_shrinks_with_queue() {
        let pool = AdaptiveWorkerPool::new(1, 4);
        // Empty queue: stay at the minimum.
        assert_eq!(pool.reconfigure(0), 1);
        // Deep queue: grow one worker per two queued jobs.
        assert_eq!(pool.reconfigure(6), 4);
        // Backlog gone: shrink back to the minimum.
        assert_eq!(pool.reconfigure(0), 1);
        assert_eq!(pool.current_size(), 1);
    }

    #[test]
    fn never_exceeds_max() {
        let pool = AdaptiveWorkerPool::new(1, 3);
        assert_eq!(pool.reconfigure(1000), 3);
    }

    #[tokio::test]
    async fn permits_match_pool_size() {
        let pool = AdaptiveWorkerPool::new(1, 4);
        pool.reconfigure(6);
        // Four concurrent acquires must all succeed immediately.
        let mut permits = Vec::new();
        for _ in 0..4 {
            permits.push(pool.acquire().await);
        }
        drop(permits);
        // After shrinking, only the minimum permits are available. Hold
        // the first permit so the second acquisition must fail.
        pool.reconfigure(0);
        let _held = pool.semaphore.clone().try_acquire_owned().unwrap();
        assert!(pool.semaphore.clone().try_acquire_owned().is_err());
    }
}