//! 哈希策略模块 - 定义桶定位策略

use crate::{
    hash::{fingerprint::FingerprintGenerator, DefaultFingerprintGenerator},
    types::{Fingerprint, Key}, DoubleHashStrategy, LinearProbeStrategy,
};
use ahash::RandomState;
use std::{
    any::Any,
    hash::{BuildHasher, Hash, Hasher},
    sync::{atomic::AtomicUsize, Arc},
};

/// 哈希策略类型
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum HashStrategyType {
    DoubleHash,
    LinearProbe,
    //Adaptive,
}

/// 哈希算法选择
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum HashAlgorithm {
    AHash,
    XxHash,
    Default,
}

/// 哈希策略特征
pub trait HashStrategy: Send + Sync {
    /// 获取键对应的桶位置
    fn locate_buckets(&self, key: &dyn Key) -> (usize, usize);
    /// 使用指定容量计算桶位置
    fn locate_buckets_with_capacity(&self, key: &dyn Key, capacity: usize) -> (usize, usize);
    /// 获取键的指纹
    fn fingerprint(&self, key: &dyn Key) -> Fingerprint;
    
    /// 更新容量
    fn update_capacity(&mut self, new_capacity: usize);
     fn get_capacity(&self) -> usize;
    /// 获取策略类型
    fn strategy_type(&self) -> HashStrategyType;
    
    /// 批量定位桶位置
    fn locate_batch(&self, keys: &[&dyn Key]) -> Vec<(usize, usize)> {
        keys.iter().map(|key| self.locate_buckets(*key)).collect()
    }
    
    /// 作为Any类型，支持向下转型
    fn as_any(&self) -> &dyn Any;
    
    /// 作为可变Any类型
    fn as_any_mut(&mut self) -> &mut dyn Any;
    
    // 克隆策略
   // fn clone_box(&self) -> Box<dyn HashStrategy>;
}

// 为 Box<dyn HashStrategy> 实现 Clone
// impl Clone for Box<dyn HashStrategy> {
//     fn clone(&self) -> Self {
//         self.clone_box()
//     }
// }


/// 哈希函数特征
pub trait HasherFunction: Send + Sync {
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



/// 哈希策略工厂
pub struct HashStrategyFactory {
    strategy_type: HashStrategyType,
    fingerprint_generator: Arc<dyn FingerprintGenerator>,
    hash_algorithm: HashAlgorithm,
}

impl HashStrategyFactory {
    /// 创建新工厂
    pub fn new(
        strategy_type: HashStrategyType,
        fingerprint_generator: Arc<dyn FingerprintGenerator>,
        hash_algorithm: HashAlgorithm,
    ) -> Self {
        Self {
            strategy_type,
            fingerprint_generator,
            hash_algorithm,
        }
    }
    
    /// 创建哈希策略
    pub fn create_strategy(&self, capacity: usize) -> Box<dyn HashStrategy> {
        match self.strategy_type {
            HashStrategyType::DoubleHash => Box::new(DoubleHashStrategy::new_with_generator(
                AtomicUsize::new(capacity),
                self.fingerprint_generator.clone(),
                self.hash_algorithm,
            )),
            HashStrategyType::LinearProbe => Box::new(LinearProbeStrategy::new_with_generator(
                capacity,
                self.fingerprint_generator.clone(),
                self.hash_algorithm,
            )),
           
        }
    }
}

// 单元测试
#[cfg(test)]
mod tests {
    use super::*;
    use crate::types::Key;
    use std::{any::Any, fmt, sync::atomic::AtomicUsize};
    
    // 测试用Key实现
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
            write!(f, "TestKey({:?})", self.0)
        }
    }
    fn compacity()->AtomicUsize{
        AtomicUsize::new(100)
    }
    /// 测试双哈希策略
    fn test_double_hash_strategy(algorithm: HashAlgorithm) {
        let strategy = DoubleHashStrategy::new(compacity(), algorithm);
        let key = TestKey(b"test_key".to_vec());
        
        // 成功锁定（空槽位）
        let (h1, h2) = strategy.locate_buckets(&key);
        assert_ne!(h1, h2, "两个桶位置应该不同");
        assert!(h1 < 100, "桶索引应在容量范围内");
        assert!(h2 < 100, "桶索引应在容量范围内");
        
        // 获取指纹
        let fp = strategy.fingerprint(&key);
        assert_ne!(fp.as_u16(), 0, "指纹不应为零");
    }

    #[test]
    fn test_double_hash_ahash() {
        test_double_hash_strategy(HashAlgorithm::AHash);
    }

    #[test]
    fn test_double_hash_xxhash() {
        test_double_hash_strategy(HashAlgorithm::XxHash);
    }

    #[test]
    fn test_double_hash_default() {
        test_double_hash_strategy(HashAlgorithm::Default);
    }

    /// 测试线性探测策略
    fn test_linear_probe_strategy(algorithm: HashAlgorithm) {
        let strategy = LinearProbeStrategy::new(100, algorithm);
        let key = TestKey(b"test_key".to_vec());
        
        // 定位桶位置
        let (h1, h2) = strategy.locate_buckets(&key);
        assert_eq!(h2, (h1 + 1) % 100, "备桶应是主桶的下一个位置");
        assert!(h1 < 100, "桶索引应在容量范围内");
        assert!(h2 < 100, "桶索引应在容量范围内");
        
        // 获取指纹
        let fp = strategy.fingerprint(&key);
        assert_ne!(fp.as_u16(), 0, "指纹不应为零");
    }

    #[test]
    fn test_linear_probe_ahash() {
        test_linear_probe_strategy(HashAlgorithm::AHash);
    }

    #[test]
    fn test_linear_probe_xxhash() {
        test_linear_probe_strategy(HashAlgorithm::XxHash);
    }

    #[test]
    fn test_linear_probe_default() {
        test_linear_probe_strategy(HashAlgorithm::Default);
    }

   

    #[test]
    fn test_strategy_factory() {
        let fingerprint = Arc::new(DefaultFingerprintGenerator);
        
        // 测试双哈希策略工厂
        let factory = HashStrategyFactory::new(
            HashStrategyType::DoubleHash,
            fingerprint.clone(),
            HashAlgorithm::AHash,
        );
        let strategy = factory.create_strategy(100);
        assert_eq!(strategy.strategy_type(), HashStrategyType::DoubleHash);
        
        // 测试线性探测策略工厂
        let factory = HashStrategyFactory::new(
            HashStrategyType::LinearProbe,
            fingerprint.clone(),
            HashAlgorithm::XxHash,
        );
        let strategy = factory.create_strategy(100);
        assert_eq!(strategy.strategy_type(), HashStrategyType::LinearProbe);
        
       
    }
}