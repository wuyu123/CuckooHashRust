//! 哈希模块 - 统一管理哈希相关功能

pub mod strategy;
pub mod double_hash;
pub mod linear_probe;
pub mod fingerprint;

pub use strategy::{HashStrategy, HashStrategyType};
pub use double_hash::DoubleHashStrategy;
pub use linear_probe::LinearProbeStrategy;
pub use fingerprint::{FingerprintGenerator, DefaultFingerprintGenerator, SimdFingerprintGenerator};

use crate::hash::strategy::HashAlgorithm;





/// 默认哈希策略
pub fn default_hash_strategy(capacity: usize) -> Box<dyn HashStrategy> {
    Box::new(DoubleHashStrategy::new(capacity,HashAlgorithm::AHash))
}

/// SIMD加速哈希策略
#[cfg(target_arch = "x86_64")]
pub fn simd_hash_strategy(capacity: usize) -> Box<dyn HashStrategy> {
    Box::new(DoubleHashStrategy::with_simd(capacity,HashAlgorithm::AHash))
}

/// 哈希工具函数
pub fn calculate_bucket(hash: u64, capacity: usize) -> usize {
    (hash as usize) % capacity
}

/// 计算备用桶位置
pub fn alternate_bucket(hash1: usize, hash2: usize, capacity: usize) -> usize {
    (hash1.wrapping_add(hash2)) % capacity
}

// 单元测试
#[cfg(test)]
mod tests {
    use crate::Key;

    use super::*;
    use std::fmt;
    #[derive(Debug, PartialEq, Clone)]
    struct TestKey(Vec<u8>);
    
    impl Key for TestKey {
        fn as_bytes(&self) -> &[u8] {
            &self.0
        }
        
        fn clone_key(&self) -> Box<dyn Key> {
            Box::new(self.clone())
        }
        
        fn eq_key(&self, other: &dyn Key) -> bool {
            if let Some(other) = other.as_any().downcast_ref::<TestKey>() {
                self == other
            } else {
                false
            }
        }
        
        fn hash_key(&self) -> u64 {
            // 简单哈希，测试用
            use std::collections::hash_map::DefaultHasher;
            use std::hash::{Hash, Hasher};
            let mut hasher = DefaultHasher::new();
            self.0.hash(&mut hasher);
            hasher.finish()
        }
        
        fn as_any(&self) -> &dyn std::any::Any {
            self
        }
        
        fn from_bytes(bytes: &[u8]) -> Option<Self> where Self: Sized {
            todo!()
        }
    }
    
    impl fmt::Display for TestKey {
        fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
            write!(f, "TestKey({:?})", self.0)
        }
    }
    #[test]
    fn test_default_hash_strategy() {
        let strategy = default_hash_strategy(100);
        let key = TestKey(b"test".to_vec());
        let (b1, b2) = strategy.locate_buckets(&key);
        assert_ne!(b1, b2);
        assert!(b1 < 100);
        assert!(b2 < 100);
    }
    
    #[test]
    fn test_bucket_calculation() {
        assert_eq!(calculate_bucket(123, 100), 23);
        assert_eq!(alternate_bucket(50, 30, 100), 80);
    }
}