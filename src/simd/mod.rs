//! SIMD加速模块 - 统一管理SIMD优化

pub mod searcher;
pub mod avx2;
pub mod sse4;
//pub mod neon;
pub mod scalar;
pub mod hybrid;

pub use searcher::{SimdSearcher, SimdSearcherType, SimdStrategy};
pub use avx2::Avx2Searcher;
pub use sse4::Sse41Searcher;
//pub use neon::NeonSearcher;
pub use scalar::ScalarSearcher;
pub use hybrid::HybridSearcher;

//#[cfg(target_arch = "x86_64")]
//pub use avx2::Avx512Searcher;

use std::sync::{Arc, RwLock};
use crate::types::Fingerprint;

/// 全局SIMD策略
pub static SIMD_STRATEGY: once_cell::sync::Lazy<RwLock<SimdStrategy>> = 
    once_cell::sync::Lazy::new(|| {
        RwLock::new(SimdStrategy::auto_detect())
    });

/// 获取全局SIMD搜索器
pub fn global_searcher() -> Arc<dyn SimdSearcher> {
    let strategy = SIMD_STRATEGY.read().expect("RwLock poisoned");
    strategy.get_searcher()
}

/// 更新全局策略
pub fn update_global_strategy() {
    let mut strategy = SIMD_STRATEGY.write().expect("RwLock poisoned");
    *strategy = SimdStrategy::auto_detect();
}

/// SIMD加速搜索
pub fn simd_search(bucket: &[Fingerprint], needle: Fingerprint) -> Option<usize> {
    let searcher = global_searcher();
    unsafe { searcher.search(bucket, needle) }
}

/// SIMD加速批量搜索
pub fn simd_search_batch(bucket: &[Fingerprint], needles: &[Fingerprint]) -> Vec<Option<usize>> {
    let searcher = global_searcher();
    unsafe { searcher.search_batch(bucket, needles) }
}

/// 检测SIMD支持
pub fn simd_support_level() -> &'static str {
    #[cfg(target_arch = "x86_64")]
    {
        if is_x86_feature_detected!("avx512f") && is_x86_feature_detected!("avx512bw") {
            return "AVX-512";
        }
        
        if is_x86_feature_detected!("avx2") {
            return "AVX2";
        }
        
        if is_x86_feature_detected!("sse4.1") {
            return "SSE4.1";
        }
    }
    
    #[cfg(target_arch = "aarch64")]
    {
        if is_aarch64_feature_detected!("neon") {
            return "NEON";
        }
    }
    
    "Scalar"
}