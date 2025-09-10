//! 哈希表核心模块 - 实现Cuckoo哈希表及其组件

pub mod cuckoo_map;
pub mod bucket;


use std::sync::Arc;

pub use cuckoo_map::{CuckooMap, CuckooMapConfig, CuckooMapStats};
pub use bucket::{Bucket, BucketGuard};

use once_cell::sync::Lazy;

use crate::{HashStrategy, Key, MemoryAllocator, SimdSearcher, StatsRecorder, Value};

/// 全局默认配置
pub static DEFAULT_CONFIG: Lazy<CuckooMapConfig> = Lazy::new(CuckooMapConfig::default);

/// 创建新CuckooMap


/// 从配置创建新映射
pub fn from_config<K: Key+Clone+ Send + Sync, V: Value+Clone+ Send + Sync>(
    config: CuckooMapConfig,
    hasher: Box<dyn HashStrategy>,
    memory_pool: Arc<dyn MemoryAllocator>,
    simd_searcher: Arc<dyn SimdSearcher>,
    stats_recorder: Arc<dyn StatsRecorder>,
) -> CuckooMap<K, V> {
    CuckooMap::from_components(config, hasher, memory_pool, simd_searcher, stats_recorder)
}

// 预定义的桶大小常量
pub const DEFAULT_BUCKET_SIZE: usize = 4;
pub const MAX_BUCKET_SIZE: usize = 8;