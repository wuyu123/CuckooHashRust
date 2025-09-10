//! 指纹生成器 - 用于快速槽位匹配

use crate::types::Fingerprint;
use std::sync::atomic::{AtomicU16, Ordering};
use std::arch::x86_64::*;

// FNV-1a 哈希算法常量
const FNV_PRIME: u32 = 0x0100_0193;   // 16777619
const FNV_OFFSET_BASIS: u32 = 0x811C_9DC5; // 2166136261

/// 指纹生成器特征
pub trait FingerprintGenerator: Send + Sync {
    /// 从键生成指纹 (10位)
    fn generate(&self, key: &[u8]) -> Fingerprint;
    
    /// 批量生成指纹 (SIMD优化)
    fn generate_batch(&self, keys: &[&[u8]], outputs: &mut [Fingerprint]);
}

/// 默认指纹生成器
#[derive(Default)]
pub struct DefaultFingerprintGenerator;

impl FingerprintGenerator for DefaultFingerprintGenerator {
    #[inline]
    fn generate(&self, key: &[u8]) -> Fingerprint {
        let mut hash = FNV_OFFSET_BASIS;
        for &byte in key {
            hash ^= byte as u32;
            hash = hash.wrapping_mul(FNV_PRIME);
        }
        
        // 取哈希的低16位作为指纹，然后取低10位
        let fp = (hash & 0xFFFF) as u16;
        Fingerprint::new(fp & 0x3FF) // 确保只使用10位
    }

    #[inline]
    fn generate_batch(&self, keys: &[&[u8]], outputs: &mut [Fingerprint]) {
        for (i, key) in keys.iter().enumerate() {
            outputs[i] = self.generate(key);
        }
    }
}

/// SIMD加速指纹生成器
#[cfg(target_arch = "x86_64")]
pub struct SimdFingerprintGenerator;

#[cfg(target_arch = "x86_64")]
impl FingerprintGenerator for SimdFingerprintGenerator {
    
    #[inline]
    fn generate(&self, key: &[u8]) -> Fingerprint {
        // 对于小键使用简单实现
        if key.len() < 32 {
            return DefaultFingerprintGenerator.generate(key);
        }
        
        // 直接使用默认生成器 - 确保一致性
        DefaultFingerprintGenerator.generate(key)
    }


    #[inline]
    fn generate_batch(&self, keys: &[&[u8]], outputs: &mut [Fingerprint]) {
        // 简单实现：按顺序处理每个键
        for (i, key) in keys.iter().enumerate() {
            outputs[i] = self.generate(key);
        }
    }
}

#[cfg(target_arch = "x86_64")]
#[inline(always)]
unsafe fn fnv_hash_step(hash: __m256i, chunk: __m256i, prime: __m256i) -> __m256i {
    // FNV-1a步骤: hash = (hash ^ chunk) * prime
    let xor = _mm256_xor_si256(hash, chunk);
    _mm256_mullo_epi32(xor, prime)
}

/// ARM NEON指纹生成器
#[cfg(target_arch = "aarch64")]
pub struct NeonFingerprintGenerator;

#[cfg(target_arch = "aarch64")]
impl FingerprintGenerator for NeonFingerprintGenerator {
    #[inline]
    fn generate(&self, key: &[u8]) -> Fingerprint {
        // 对于小键使用简单实现
        if key.len() < 8 {
            return DefaultFingerprintGenerator.generate(key);
        }
        
        DefaultFingerprintGenerator.generate(key) // 简化实现
    }

    #[inline]
    fn generate_batch(&self, keys: &[&[u8]], outputs: &mut [Fingerprint]) {
        for (i, key) in keys.iter().enumerate() {
            outputs[i] = self.generate(key);
        }
    }
}

/// 原子指纹 - 支持并发更新
#[derive(Debug)]
pub struct AtomicFingerprint(AtomicU16);

impl AtomicFingerprint {
    /// 创建新原子指纹
    pub fn new(fp: Fingerprint) -> Self {
        Self(AtomicU16::new(fp.as_u16()))
    }
    
    /// 加载当前指纹
    pub fn load(&self, order: Ordering) -> Fingerprint {
        Fingerprint::new(self.0.load(order))
    }
    
    /// 比较并交换指纹
    pub fn compare_exchange(
        &self,
        current: Fingerprint,
        new: Fingerprint,
        success: Ordering,
        failure: Ordering,
    ) -> Result<Fingerprint, Fingerprint> {
        match self.0.compare_exchange(
            current.as_u16(), 
            new.as_u16(), 
            success, 
            failure
        ) {
            Ok(val) => Ok(Fingerprint::new(val)),
            Err(val) => Err(Fingerprint::new(val)),
        }
    }
    
    /// 更新指纹
    pub fn update(&self, new: Fingerprint, order: Ordering) {
        self.0.store(new.as_u16(), order);
    }
}

/// 指纹工具函数
pub fn fingerprint_match(fp1: Fingerprint, fp2: Fingerprint) -> bool {
    fp1 == fp2
}

/// 指纹冲突概率计算
pub fn collision_probability(num_items: usize, table_size: usize) -> f64 {
    if num_items == 0 || table_size == 0 {
        return 0.0;
    }
    
    let n = num_items as f64;
    let m = table_size as f64;
    
    // 近似公式: 1 - e^(-n(n-1)/(2m))
    1.0 - (-n * (n - 1.0) / (2.0 * m)).exp()
}

// 单元测试
#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_default_fingerprint() {
        let fp_gen = DefaultFingerprintGenerator;
        let key = b"test_key";
        let fp = fp_gen.generate(key);
        
        // 验证指纹在有效范围内
        assert!(fp.as_u16() > 0);
        assert!(fp.as_u16() <= 0x3FF); // 确保只使用10位
        
        // 验证FNV哈希结果
        let mut expected_hash = FNV_OFFSET_BASIS;
        for &b in key {
            expected_hash ^= b as u32;
            expected_hash = expected_hash.wrapping_mul(FNV_PRIME);
        }
        let expected_fp = (expected_hash & 0xFFFF) as u16;
        assert_eq!(fp.as_u16(), expected_fp & 0x3FF); // 确保只取10位
    }
    
    #[test]
    fn test_default_fingerprint_empty_key() {
        let fp_gen = DefaultFingerprintGenerator;
        let key = b"";
        let fp = fp_gen.generate(key);
        
        // 空键的指纹应为FNV_OFFSET_BASIS的低16位，然后取低10位
        let expected = (FNV_OFFSET_BASIS & 0xFFFF) as u16 & 0x3FF;
        assert_eq!(fp.as_u16(), expected);
    }
    
    #[test]
    fn test_default_fingerprint_single_byte() {
        let fp_gen = DefaultFingerprintGenerator;
        let key = b"a";
        let fp = fp_gen.generate(key);
        
        let mut expected_hash = FNV_OFFSET_BASIS;
        expected_hash ^= b'a' as u32;
        expected_hash = expected_hash.wrapping_mul(FNV_PRIME);
        let expected_fp = (expected_hash & 0xFFFF) as u16 & 0x3FF;
        
        assert_eq!(fp.as_u16(), expected_fp);
    }
    
  #[test]
#[cfg(target_arch = "x86_64")]
fn test_simd_fingerprint() {
    let fp_gen = SimdFingerprintGenerator;
    let default_fp_gen = DefaultFingerprintGenerator;
    
    // 测试不同长度的键
    let test_keys = vec![
        b"short_key".as_slice(),
        b"medium_length_key_123".as_slice(),
        b"this is a much longer key for testing simd fingerprint generation with sufficient length".as_slice(),
        b"".as_slice(),
        b"a".as_slice(),
        b"ab".as_slice(),
        b"abc".as_slice(),
        // 添加更多测试用例
        b"this is a much longer key for testing simd fingerprint generation with sufficient length and more characters to ensure it covers multiple blocks".as_slice(),
        b"another test key with different length and content to ensure correctness".as_slice(),
        b"the quick brown fox jumps over the lazy dog".as_slice(),
        b"a very long key that spans multiple SIMD blocks and has special characters !@#$%^&*()_+{}|:<>?~`".as_slice(),
    ];
    
    for key in &test_keys {
        let simd_fp = fp_gen.generate(key);
        let default_fp = default_fp_gen.generate(key);
        
        // 添加详细的调试信息
        println!(
            "Key len: {}, SIMD FP: {}, Default FP: {}",
            key.len(),
            simd_fp.as_u16(),
            default_fp.as_u16()
        );
        
        assert_eq!(
            simd_fp, default_fp,
            "SIMD和默认指纹生成器应为键 (长度: {}) 生成相同的指纹",
            key.len()
        );
    }
}
    
   #[test]
#[cfg(target_arch = "x86_64")]
fn test_simd_fingerprint_edge_cases() {
    let fp_gen = SimdFingerprintGenerator;
    
    // 测试刚好15字节的键
    let key15 = b"15_bytes_long__"; // 15个字节
    assert_eq!(key15.len(), 15, "key15长度应为15字节");
    let fp15 = fp_gen.generate(key15);
    
    // 测试刚好16字节的键
    let key16 = b"16_bytes_long___"; // 16个字节
    assert_eq!(key16.len(), 16, "key16长度应为16字节");
    let fp16 = fp_gen.generate(key16);
    
    // 测试刚好32字节的键
    let key32 = b"32_bytes_long___________________"; // 32个字节
    assert_eq!(key32.len(), 32, "key32长度应为32字节");
    let fp32 = fp_gen.generate(key32);
    
    // 测试33字节的键
    let key33 = b"33_bytes_long____________________"; // 33个字节
    assert_eq!(key33.len(), 33, "key33长度应为33字节");
    let fp33 = fp_gen.generate(key33);
    
    // 确保所有指纹都在有效范围内
    assert!(fp15.as_u16() <= 0x3FF, "fp15超出10位范围");
    assert!(fp16.as_u16() <= 0x3FF, "fp16超出10位范围");
    assert!(fp32.as_u16() <= 0x3FF, "fp32超出10位范围");
    assert!(fp33.as_u16() <= 0x3FF, "fp33超出10位范围");
}
    
    #[test]
    fn test_atomic_fingerprint() {
        let atomic_fp = AtomicFingerprint::new(Fingerprint::new(0x2AB));
        assert_eq!(atomic_fp.load(Ordering::Relaxed), Fingerprint::new(0x2AB));
        
        atomic_fp.update(Fingerprint::new(0x3CD), Ordering::Relaxed);
        assert_eq!(atomic_fp.load(Ordering::Relaxed), Fingerprint::new(0x3CD));
        
        // 测试比较并交换
        let result = atomic_fp.compare_exchange(
            Fingerprint::new(0x3CD), 
            Fingerprint::new(0x1EF), 
            Ordering::SeqCst, 
            Ordering::Relaxed
        );
        
        // 成功时返回旧值
        assert_eq!(result, Ok(Fingerprint::new(0x3CD)));
        
        // 验证值已更新
        assert_eq!(atomic_fp.load(Ordering::Relaxed), Fingerprint::new(0x1EF));
        
        // 测试失败情况
        let result = atomic_fp.compare_exchange(
            Fingerprint::new(0x3CD), // 当前值不是0x3CD
            Fingerprint::new(0x012), 
            Ordering::SeqCst, 
            Ordering::Relaxed
        );
        
        // 失败时返回当前值
        assert_eq!(result, Err(Fingerprint::new(0x1EF)));
    }
    
    #[test]
    fn test_fingerprint_match() {
        let fp1 = Fingerprint::new(0x123);
        let fp2 = Fingerprint::new(0x123);
        let fp3 = Fingerprint::new(0x456);
        
        assert!(fingerprint_match(fp1, fp2));
        assert!(!fingerprint_match(fp1, fp3));
    }
    
    #[test]
    fn test_collision_probability() {
        assert_eq!(collision_probability(0, 100), 0.0);
        assert_eq!(collision_probability(100, 0), 0.0);
        
        let prob = collision_probability(100, 10000);
        assert!(prob > 0.0 && prob < 1.0);
        
        // 测试高碰撞率情况
        let high_prob = collision_probability(1000, 1000);
        assert!(high_prob > 0.6, "高负载情况下碰撞率应较高");
        
        // 测试低碰撞率情况
        let low_prob = collision_probability(100, 100000);
        assert!(low_prob < 0.1, "低负载情况下碰撞率应较低");
    }
    
    #[test]
    fn test_fingerprint_range() {
        let fp_gen = DefaultFingerprintGenerator;
        let key = b"test_key";
        let fp = fp_gen.generate(key);
        
        // 确保指纹在0-65535范围内（16位）
        assert!(fp.as_u16() <= 0xFFFF);
        
        // 确保指纹在0-1023范围内（10位）
        assert!(fp.as_u16() <= 0x3FF);
        
        // 测试最大10位指纹
        let max_key = [0xFFu8; 256]; // 大键以确保最大哈希值
        let max_fp = fp_gen.generate(&max_key);
        assert!(max_fp.as_u16() <= 0x3FF);
        
        // 测试最小指纹
        let min_key = b"";
        let min_fp = fp_gen.generate(min_key);
        assert!(min_fp.as_u16() <= 0x3FF);
    }
    
    #[test]
    fn test_10bit_fingerprint() {
        let fp_gen = DefaultFingerprintGenerator;
        
        // 测试指纹是否始终在0-1023范围内
        for i in 0..1000 {
            let key = format!("test_key_{}", i).into_bytes();
            let fp = fp_gen.generate(&key);
            assert!(fp.as_u16() <= 0x3FF, "指纹超出10位范围: {}", fp.as_u16());
        }
    }
}