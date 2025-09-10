//! 哈希策略模块 - 定义桶定位策略

use crate::{
    hash::{fingerprint::FingerprintGenerator, DefaultFingerprintGenerator},
    types::{Fingerprint, Key}, DoubleHashStrategy, LinearProbeStrategy,
};
use ahash::RandomState;
use std::{
    any::Any,
    hash::{Hash, Hasher, BuildHasher},
    sync::Arc,
};

/// 哈希策略类型
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum HashStrategyType {
    DoubleHash,
    LinearProbe,
    Adaptive,
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
    
    /// 克隆策略
    fn clone_box(&self) -> Box<dyn HashStrategy>;
}

// 为 Box<dyn HashStrategy> 实现 Clone
impl Clone for Box<dyn HashStrategy> {
    fn clone(&self) -> Self {
        self.clone_box()
    }
}


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

/// 自适应哈希策略
#[derive(Clone)]
pub struct AdaptiveHashStrategy {
    primary: Box<dyn HashStrategy>,
    secondary: Box<dyn HashStrategy>,
    current: Box<dyn HashStrategy>,
    collision_rate: f32,
    threshold: f32,
}

impl AdaptiveHashStrategy {
    /// 创建新自适应策略
    pub fn new(
        primary: Box<dyn HashStrategy>,
        secondary: Box<dyn HashStrategy>,
    ) -> Self {
        let current = primary.clone_box();
        Self {
            primary,
            secondary,
            current,
            collision_rate: 0.0,
            threshold: 0.3, // 30%碰撞率阈值
        }
    }
    
    /// 更新碰撞率
    pub fn update_collision_rate(&mut self, rate: f32) {
        self.collision_rate = rate;
        
        if rate > self.threshold {
            self.current = self.secondary.clone_box();
        } else {
            self.current = self.primary.clone_box();
        }
    }
}

impl HashStrategy for AdaptiveHashStrategy {
    fn locate_buckets(&self, key: &dyn Key) -> (usize, usize) {
        self.current.locate_buckets(key)
    }
    
    fn fingerprint(&self, key: &dyn Key) -> Fingerprint {
        self.current.fingerprint(key)
    }
    
    fn update_capacity(&mut self, new_capacity: usize) {
        self.primary.update_capacity(new_capacity);
        self.secondary.update_capacity(new_capacity);
        self.current.update_capacity(new_capacity);
    }
    
    fn strategy_type(&self) -> HashStrategyType {
        HashStrategyType::Adaptive
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
        self.current.locate_buckets_with_capacity(key, capacity)
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
                capacity,
                self.fingerprint_generator.clone(),
                self.hash_algorithm,
            )),
            HashStrategyType::LinearProbe => Box::new(LinearProbeStrategy::new_with_generator(
                capacity,
                self.fingerprint_generator.clone(),
                self.hash_algorithm,
            )),
            HashStrategyType::Adaptive => {
                // 自适应策略结合双哈希和线性探测
                let primary = DoubleHashStrategy::new_with_generator(
                    capacity,
                    self.fingerprint_generator.clone(),
                    self.hash_algorithm,
                );
                let secondary = LinearProbeStrategy::new_with_generator(
                    capacity,
                    self.fingerprint_generator.clone(),
                    self.hash_algorithm,
                );
                Box::new(AdaptiveHashStrategy::new(
                    Box::new(primary),
                    Box::new(secondary),
                ))
            }
        }
    }
}

// 单元测试
#[cfg(test)]
mod tests {
    use super::*;
    use crate::types::Key;
    use std::{any::Any, fmt};
    
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

    /// 测试双哈希策略
    fn test_double_hash_strategy(algorithm: HashAlgorithm) {
        let strategy = DoubleHashStrategy::new(100, algorithm);
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
    fn test_adaptive_strategy() {
        let fingerprint = Arc::new(DefaultFingerprintGenerator);
        let factory = HashStrategyFactory::new(
            HashStrategyType::Adaptive,
            fingerprint,
            HashAlgorithm::AHash,
        );
        
        let mut strategy = factory.create_strategy(100);
        
        let key = TestKey(b"test_key".to_vec());
        
        // 初始使用主策略（双哈希）
        let (b1, b2) = strategy.locate_buckets(&key);
        assert_ne!(b1, b2, "双哈希策略应返回不同的桶位置");
        
        // 更新碰撞率高于阈值，切换到备选策略（线性探测）
        if let Some(adaptive) = strategy.as_any_mut().downcast_mut::<AdaptiveHashStrategy>() {
            adaptive.update_collision_rate(0.4);
        }
        
        // 切换后应使用线性探测策略
        let (b3, b4) = strategy.locate_buckets(&key);
        assert_eq!(b4, (b3 + 1) % 100, "线性探测策略应返回相邻桶位置");
        
        // 更新碰撞率低于阈值，切换回主策略
        if let Some(adaptive) = strategy.as_any_mut().downcast_mut::<AdaptiveHashStrategy>() {
            adaptive.update_collision_rate(0.2);
        }
        
        // 应回到双哈希策略
        let (b5, b6) = strategy.locate_buckets(&key);
        assert_ne!(b6, (b5 + 1) % 100, "双哈希策略应返回不同的桶位置");
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
        
        // 测试自适应策略工厂
        let factory = HashStrategyFactory::new(
            HashStrategyType::Adaptive,
            fingerprint,
            HashAlgorithm::Default,
        );
        let strategy = factory.create_strategy(100);
        assert_eq!(strategy.strategy_type(), HashStrategyType::Adaptive);
    }
}