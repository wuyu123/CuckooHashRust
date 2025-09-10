// src/simd/searcher.rs
//! SIMD搜索器接口 - 定义SIMD加速搜索功能

use crate::types::Fingerprint;
use std::sync::Arc;

/// SIMD搜索器特征
pub trait SimdSearcher:Send + Sync{
    /// 在桶中搜索空槽位
  //  unsafe fn find_empty_slot(&self, bucket: &[Fingerprint]) -> Option<usize>;
    /// 在桶中搜索指纹
    unsafe fn search(&self, bucket: &[Fingerprint], needle: Fingerprint) -> Option<usize>;
    
    /// 查找所有匹配指纹的槽位索引
    unsafe fn search_all(&self, bucket: &[Fingerprint], needle: Fingerprint) -> Vec<usize>;
    /// 批量搜索多个指纹
    unsafe fn search_batch(&self, bucket: &[Fingerprint], needles: &[Fingerprint]) -> Vec<Option<usize>>;
    
    /// 获取搜索器类型
    fn searcher_type(&self) -> SimdSearcherType;
}

/// SIMD搜索器类型
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SimdSearcherType {
    Avx2,
    Sse41,
    Avx512,
    Neon,
    Scalar,
    Hybrid,
}

/// SIMD策略
pub struct SimdStrategy {
    searcher: Arc<dyn SimdSearcher + Send + Sync>,
}

impl SimdStrategy {
    /// 自动检测最佳搜索器
    pub fn auto_detect() -> Self {
        #[cfg(target_arch = "x86_64")]
        {
            if is_x86_feature_detected!("avx2") {
                return Self::new(SimdSearcherType::Avx2);
            }
            
            if is_x86_feature_detected!("sse4.1") {
                return Self::new(SimdSearcherType::Sse41);
            }
        }
        
        #[cfg(target_arch = "aarch64")]
        {
            if is_aarch64_feature_detected!("neon") {
                return Self::new(SimdSearcherType::Neon);
            }
        }
        
        Self::new(SimdSearcherType::Scalar)
    }
    
    /// 创建指定类型的策略
    pub fn new(searcher_type: SimdSearcherType) -> Self {
        let searcher: Arc<dyn SimdSearcher> = match searcher_type {
            SimdSearcherType::Avx2 => Arc::new(Avx2Searcher),
            SimdSearcherType::Sse41 => Arc::new(Sse41Searcher),
            SimdSearcherType::Scalar => Arc::new(ScalarSearcher),
            _ => Arc::new(ScalarSearcher),
        };
        
        Self { searcher }
    }
    
    /// 获取SIMD搜索器
    pub fn get_searcher(&self) -> Arc<dyn SimdSearcher + Send + Sync+'static> {
        self.searcher.clone()
    }
}
unsafe impl Send for SimdStrategy {}
unsafe impl Sync for SimdStrategy {}
/// AVX2搜索器实现
#[cfg(target_arch = "x86_64")]
pub struct Avx2Searcher;

#[cfg(target_arch = "x86_64")]
impl SimdSearcher for Avx2Searcher {
    unsafe fn search(&self, bucket: &[Fingerprint], needle: Fingerprint) -> Option<usize> {
        // AVX2实现
        None
    }
    
    unsafe fn search_batch(&self, bucket: &[Fingerprint], needles: &[Fingerprint]) -> Vec<Option<usize>> {
        // AVX2实现
        vec![]
    }
    
    fn searcher_type(&self) -> SimdSearcherType {
        SimdSearcherType::Avx2
    }
    
    unsafe fn search_all(&self, bucket: &[Fingerprint], needle: Fingerprint) -> Vec<usize> {
        todo!()
    }
    
  
}

/// SSE4.1搜索器实现
#[cfg(target_arch = "x86_64")]
pub struct Sse41Searcher;

#[cfg(target_arch = "x86_64")]
impl SimdSearcher for Sse41Searcher {
    unsafe fn search(&self, bucket: &[Fingerprint], needle: Fingerprint) -> Option<usize> {
        // SSE4.1实现
        None
    }
    
    unsafe fn search_batch(&self, bucket: &[Fingerprint], needles: &[Fingerprint]) -> Vec<Option<usize>> {
        // SSE4.1实现
        vec![]
    }
    
    fn searcher_type(&self) -> SimdSearcherType {
        SimdSearcherType::Sse41
    }
    
    unsafe fn search_all(&self, bucket: &[Fingerprint], needle: Fingerprint) -> Vec<usize> {
        todo!()
    }
}

/// 标量回退实现
pub struct ScalarSearcher;

impl SimdSearcher for ScalarSearcher {
    unsafe fn search(&self, bucket: &[Fingerprint], needle: Fingerprint) -> Option<usize> {
        bucket.iter().position(|&fp| fp == needle)
    }

    unsafe fn search_batch(&self, bucket: &[Fingerprint], needles: &[Fingerprint]) -> Vec<Option<usize>> {
        needles.iter()
            .map(|&needle| self.search(bucket, needle))
            .collect()
    }
    
    fn searcher_type(&self) -> SimdSearcherType {
        SimdSearcherType::Scalar
    }
    
    unsafe fn search_all(&self, bucket: &[Fingerprint], needle: Fingerprint) -> Vec<usize> {
         bucket.iter()
            .enumerate()
            .filter_map(|(i, &fp)| if fp == needle { Some(i) } else { None })
            .collect()
    }
}

// 单元测试
#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_scalar_search() {
        let searcher = ScalarSearcher;
        let bucket = [Fingerprint::new(1), Fingerprint::new(2), Fingerprint::new(3)];
        unsafe {
            assert_eq!(searcher.search(&bucket, Fingerprint::new(2)), Some(1));
            assert_eq!(searcher.search(&bucket, Fingerprint::new(4)), None);
        }
    }
    
    #[test]
    fn test_scalar_search_batch() {
        let searcher = ScalarSearcher;
        let bucket = [Fingerprint::new(1), Fingerprint::new(2), Fingerprint::new(3)];
        let needles = [Fingerprint::new(2), Fingerprint::new(3), Fingerprint::new(4)];
        unsafe {
            let results = searcher.search_batch(&bucket, &needles);
            assert_eq!(results, vec![Some(1), Some(2), None]);
        }
    }
}