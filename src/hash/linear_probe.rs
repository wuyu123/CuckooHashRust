//! 线性探测策略 - 使用线性探测解决冲突

use crate::{
    hash::{fingerprint::FingerprintGenerator, strategy::{HashAlgorithm, HashStrategy}, DefaultFingerprintGenerator, HashStrategyType},
    types::{Fingerprint, Key},
};
use std::sync::Arc;
use ahash::RandomState;
use std::hash::{Hash, Hasher, BuildHasher};
use std::any::Any;

/// 哈希函数特征
trait HasherFunction: Send + Sync {
    fn hash_bytes(&self, data: &[u8]) -> u64;
}

impl<T> HasherFunction for T
where
    T: Fn(&[u8]) -> u64 + Send + Sync,
{
    fn hash_bytes(&self, data: &[u8]) -> u64 {
        self(data)
    }
}


/// 线性探测策略
#[derive(Clone)]
pub struct LinearProbeStrategy {
    hasher: Arc<dyn HasherFunction>,
    fingerprint_generator: Arc<dyn FingerprintGenerator>,
    capacity: usize,
   
}

impl LinearProbeStrategy {
    /// 创建新线性探测策略
    pub fn new(capacity: usize, algorithm: HashAlgorithm) -> Self {
        Self::new_with_generator(capacity, Arc::new(DefaultFingerprintGenerator), algorithm)
    }
    
    /// 使用指定指纹生成器创建
    pub fn new_with_generator(
        capacity: usize,
        fingerprint_generator: Arc<dyn FingerprintGenerator>,
        algorithm: HashAlgorithm,
    ) -> Self {
        let hasher = Self::build_hasher_function(algorithm, 42);
        
        Self {
            hasher,
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

impl HashStrategy for LinearProbeStrategy {
    fn locate_buckets(&self, key: &dyn Key) -> (usize, usize) {
        let key_bytes = key.as_bytes();
        
        // 计算哈希值
        let h1 = self.hasher.hash_bytes(key_bytes) as usize % self.capacity;
        
        // 线性探测策略只返回一个位置
        (h1, (h1 + 1) % self.capacity)
    }
    
    fn fingerprint(&self, key: &dyn Key) -> Fingerprint {
        self.fingerprint_generator.generate(key.as_bytes())
    }
    
    fn update_capacity(&mut self, new_capacity: usize) {
        self.capacity = new_capacity;
    }
    
    fn strategy_type(&self) -> HashStrategyType {
        HashStrategyType::LinearProbe
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
    
    fn locate_buckets_with_capacity(&self, key: &dyn Key, capacity: usize) -> (usize, usize) {
        let key_bytes = key.as_bytes();
        
        // 计算哈希值
        let h1 = self.hasher.hash_bytes(key_bytes) as usize % capacity;
        
        // 线性探测策略只返回一个位置
        (h1, (h1 + 1) % capacity)
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
    fn test_linear_probe() {
        let strategy = LinearProbeStrategy::new(100,HashAlgorithm::XxHash);
        let key = TestKey(b"test_key".to_vec());
        
        let (b1, b2) = strategy.locate_buckets(&key);
        assert_eq!(b2, (b1 + 1) % 100, "备桶应是主桶的下一个位置");
        assert!(b1 < 100, "桶索引应在容量范围内");
        assert!(b2 < 100, "桶索引应在容量范围内");
        
        let fp = strategy.fingerprint(&key);
        assert_ne!(fp.as_u16(), 0, "指纹不应为零");
    }
    
    #[test]
    fn test_capacity_update() {
        let mut strategy = LinearProbeStrategy::new(100,HashAlgorithm::XxHash);
        strategy.update_capacity(200);
        
        let key = TestKey(b"test".to_vec());
        let (b1, b2) = strategy.locate_buckets(&key);
        assert!(b1 < 200, "桶索引应在更新后的容量范围内");
        assert!(b2 < 200, "桶索引应在更新后的容量范围内");
    }
    
    #[test]
    fn test_same_key_same_buckets() {
        let strategy = LinearProbeStrategy::new(100,HashAlgorithm::XxHash);
        let key1 = TestKey(b"consistent_key".to_vec());
        let key2 = TestKey(b"consistent_key".to_vec());
        
        let (b1_1, b1_2) = strategy.locate_buckets(&key1);
        let (b2_1, b2_2) = strategy.locate_buckets(&key2);
        
        assert_eq!(b1_1, b2_1, "相同键应有相同的主桶索引");
        assert_eq!(b1_2, b2_2, "相同键应有相同的备桶索引");
    }
    
    #[test]
    fn test_fingerprint_consistency() {
        let strategy = LinearProbeStrategy::new(100,HashAlgorithm::XxHash);
        let key = TestKey(b"fingerprint_test".to_vec());
        
        let fp1 = strategy.fingerprint(&key);
        let fp2 = strategy.fingerprint(&key);
        
        assert_eq!(fp1, fp2, "相同键应有相同的指纹");
    }
    
    #[test]
    fn test_fingerprint_range() {
        let strategy = LinearProbeStrategy::new(100,HashAlgorithm::XxHash);
        let key = TestKey(b"test_key".to_vec());
        
        let fp = strategy.fingerprint(&key);
        // 确保指纹在0-1023范围内（10位）
        assert!(fp.as_u16() <= 0x3FF, "指纹应在0-1023范围内");
    }
    
    #[test]
    #[cfg(target_arch = "x86_64")]
    fn test_simd_fingerprint() {
        let strategy = LinearProbeStrategy::with_simd(100,HashAlgorithm::XxHash);
        let key = TestKey(b"test_key".to_vec());
        
        let fp = strategy.fingerprint(&key);
        assert_ne!(fp.as_u16(), 0, "SIMD指纹不应为零");
    }
}