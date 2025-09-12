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

// 预定义的桶大小常量
pub const DEFAULT_BUCKET_SIZE: usize = 4;
pub const MAX_BUCKET_SIZE: usize = 8;