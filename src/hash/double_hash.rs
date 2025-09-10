//! 双哈希策略 - 使用两个独立哈希函数定位桶

use crate::{
    hash::{fingerprint::FingerprintGenerator, strategy::{HashAlgorithm, HashStrategy, HasherFunction}, DefaultFingerprintGenerator, HashStrategyType},
    types::{Fingerprint, Key},
};
use ahash::RandomState;
use std::{
    any::Any,
    hash::{Hash, Hasher,BuildHasher},
    sync::Arc,
};



/// 双哈希策略
#[derive(Clone)]
pub struct DoubleHashStrategy {
    primary_hasher: Arc<dyn HasherFunction>,
    secondary_hasher: Arc<dyn HasherFunction>,
    fingerprint_generator: Arc<dyn FingerprintGenerator>,
    capacity: usize,
  
}

impl DoubleHashStrategy {
    /// 创建新双哈希策略
    pub fn new(capacity: usize, algorithm: HashAlgorithm) -> Self {
        Self::new_with_generator(capacity, Arc::new(DefaultFingerprintGenerator), algorithm)
    }
    
    /// 使用指定指纹生成器创建
    pub fn new_with_generator(
        capacity: usize,
        fingerprint_generator: Arc<dyn FingerprintGenerator>,
        algorithm: HashAlgorithm,
    ) -> Self {
        // 使用不同种子创建两个哈希函数
        let primary_hasher = Self::build_hasher_function(algorithm, 42);
        let secondary_hasher = Self::build_hasher_function(algorithm, 123);
        
        Self {
            primary_hasher,
            secondary_hasher,
            fingerprint_generator,
            capacity,
           
        }
    }
    
    /// 构建哈希函数
    fn build_hasher_function(algorithm: HashAlgorithm, seed: usize) -> Arc<dyn HasherFunction> {
        match algorithm {
            HashAlgorithm::AHash => {
                let state = RandomState::with_seed(seed);
                Arc::new(move |data: &[u8]| {
                    let mut hasher = state.build_hasher();
                    data.hash(&mut hasher);
                    hasher.finish()
                })
            }
            HashAlgorithm::XxHash => {
                let seed = seed as u64;
                Arc::new(move |data: &[u8]| {
                    let mut hasher = twox_hash::XxHash64::with_seed(seed);
                    data.hash(&mut hasher);
                    hasher.finish()
                })
            }
            HashAlgorithm::Default => {
                Arc::new(|data: &[u8]| {
                    let mut hasher = std::collections::hash_map::DefaultHasher::new();
                    data.hash(&mut hasher);
                    hasher.finish()
                })
            }
        }
    }
    
    /// 创建SIMD加速策略
    #[cfg(target_arch = "x86_64")]
    pub fn with_simd(capacity: usize, algorithm: HashAlgorithm) -> Self {
        use crate::hash::fingerprint::SimdFingerprintGenerator;
        Self::new_with_generator(capacity, Arc::new(SimdFingerprintGenerator), algorithm)
    }
}

impl HashStrategy for DoubleHashStrategy {
    fn locate_buckets(&self, key: &dyn Key) -> (usize, usize) {
        let key_bytes = key.as_bytes();
        let h1 = self.primary_hasher.hash_bytes(key_bytes) as usize % self.capacity;
        let h2 = self.secondary_hasher.hash_bytes(key_bytes) as usize % self.capacity;
        
        // 确保两个桶位置不同
        if h1 == h2 {
            (h1, (h2 + 1) % self.capacity)
        } else {
            (h1, h2)
        }
    }

    fn locate_buckets_with_capacity(&self, key: &dyn Key, capacity: usize) -> (usize, usize) {
        let key_bytes = key.as_bytes();
        let h1 = self.primary_hasher.hash_bytes(key_bytes) as usize % capacity;
        let h2 = self.secondary_hasher.hash_bytes(key_bytes) as usize % capacity;
        
        // 确保两个桶位置不同
        if h1 == h2 {
            (h1, (h2 + 1) % capacity)
        } else {
            (h1, h2)
        }
    }
    
    fn fingerprint(&self, key: &dyn Key) -> Fingerprint {
        self.fingerprint_generator.generate(key.as_bytes())
    }
    
    fn update_capacity(&mut self, new_capacity: usize) {
        self.capacity = new_capacity;
    }
    
    fn strategy_type(&self) -> HashStrategyType {
        HashStrategyType::DoubleHash
    }
    
    fn as_any(&self) -> &dyn Any {
        self
    }
    
    fn as_any_mut(&mut self) -> &mut dyn Any {
        self
    }

    fn clone_box(&self) -> Box<dyn HashStrategy> {
        Box::new(self.clone())
    }
}

// 单元测试
#[cfg(test)]
mod tests {
    use super::*;
    use crate::types::Key;
    use std::{any::Any, fmt};
    
    // 测试用Key实现
    #[derive(Debug)]
    struct TestKey(Vec<u8>);
    
    impl Key for TestKey {
        fn as_bytes(&self) -> &[u8] {
            &self.0
        }
        
        fn clone_key(&self) -> Box<dyn Key> {
            Box::new(TestKey(self.0.clone()))
        }
        
        fn eq_key(&self, other: &dyn Key) -> bool {
            if let Some(other_key) = other.as_any().downcast_ref::<TestKey>() {
                self.0 == other_key.0
            } else {
                false
            }
        }
        
        fn hash_key(&self) -> u64 {
            let mut hasher = std::collections::hash_map::DefaultHasher::new();
            self.0.hash(&mut hasher);
            hasher.finish()
        }
        
        fn as_any(&self) -> &dyn Any {
            self
        }
        
        fn from_bytes(bytes: &[u8]) -> Option<Self> where Self: Sized {
            Some(TestKey(bytes.to_vec()))
        }
    }
    
    impl fmt::Display for TestKey {
        fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
            write!(f, "{:?}", self.0)
        }
    }
    
    #[test]
    fn test_double_hash_ahash() {
        test_double_hash_algorithm(HashAlgorithm::AHash);
    }
    
    #[test]
    fn test_double_hash_xxhash() {
        test_double_hash_algorithm(HashAlgorithm::XxHash);
    }
    
    #[test]
    fn test_double_hash_default() {
        test_double_hash_algorithm(HashAlgorithm::Default);
    }
    
    fn test_double_hash_algorithm(algorithm: HashAlgorithm) {
        let strategy = DoubleHashStrategy::new(100, algorithm);
        let key = TestKey(b"test_key".to_vec());
        
        let (b1, b2) = strategy.locate_buckets(&key);
        assert_ne!(b1, b2, "两个哈希值应该不同");
        assert!(b1 < 100, "桶索引应在容量范围内");
        assert!(b2 < 100, "桶索引应在容量范围内");
        
        let fp = strategy.fingerprint(&key);
        assert_ne!(fp.as_u16(), 0, "指纹不应为零");
    }
    
    #[test]
    fn test_capacity_update() {
        let mut strategy = DoubleHashStrategy::new(100, HashAlgorithm::AHash);
        strategy.update_capacity(200);
        
        let key = TestKey(b"test".to_vec());
        let (b1, b2) = strategy.locate_buckets(&key);
        assert!(b1 < 200, "桶索引应在更新后的容量范围内");
        assert!(b2 < 200, "桶索引应在更新后的容量范围内");
    }
    
    #[test]
    fn test_same_key_same_buckets() {
        let strategy = DoubleHashStrategy::new(100, HashAlgorithm::AHash);
        let key1 = TestKey(b"consistent_key".to_vec());
        let key2 = TestKey(b"consistent_key".to_vec());
        
        let (b1_1, b1_2) = strategy.locate_buckets(&key1);
        let (b2_1, b2_2) = strategy.locate_buckets(&key2);
        
        assert_eq!(b1_1, b2_1, "相同键应有相同的主桶索引");
        assert_eq!(b1_2, b2_2, "相同键应有相同的备桶索引");
    }
    
    #[test]
    fn test_different_keys_different_buckets() {
        let strategy = DoubleHashStrategy::new(100, HashAlgorithm::AHash);
        let key1 = TestKey(b"key_one".to_vec());
        let key2 = TestKey(b"key_two".to_vec());
        
        let (b1_1, b1_2) = strategy.locate_buckets(&key1);
        let (b2_1, b2_2) = strategy.locate_buckets(&key2);
        
        // 可能碰撞，但概率很低
        assert!(
            b1_1 != b2_1 || b1_2 != b2_2,
            "不同键应有不同的桶索引（可能有碰撞，但概率很低）"
        );
    }
    
    #[test]
    fn test_fingerprint_consistency() {
        let strategy = DoubleHashStrategy::new(100, HashAlgorithm::AHash);
        let key = TestKey(b"fingerprint_test".to_vec());
        
        let fp1 = strategy.fingerprint(&key);
        let fp2 = strategy.fingerprint(&key);
        
        assert_eq!(fp1, fp2, "相同键应有相同的指纹");
    }
    
    #[test]
    fn test_fingerprint_range() {
        let strategy = DoubleHashStrategy::new(100, HashAlgorithm::AHash);
        let key = TestKey(b"test_key".to_vec());
        
        let fp = strategy.fingerprint(&key);
        // 确保指纹在0-1023范围内（10位）
        assert!(fp.as_u16() <= 0x3FF, "指纹应在0-1023范围内");
    }
    
    #[test]
    #[cfg(target_arch = "x86_64")]
    fn test_simd_fingerprint() {
        let strategy = DoubleHashStrategy::with_simd(100, HashAlgorithm::AHash);
        let key = TestKey(b"test_key".to_vec());
        
        let fp = strategy.fingerprint(&key);
        assert_ne!(fp.as_u16(), 0, "SIMD指纹不应为零");
    }
}