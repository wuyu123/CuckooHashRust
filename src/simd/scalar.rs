//! 标量回退实现 - 无SIMD加速的搜索

use crate::{
    simd::searcher::{SimdSearcher, SimdSearcherType},
    types::Fingerprint,
};

/// 标量搜索器
pub struct ScalarSearcher;

impl SimdSearcher for ScalarSearcher {
    #[inline]
  unsafe  fn search(&self, bucket: &[Fingerprint], needle: Fingerprint) -> Option<usize> {
        bucket.iter().position(|&fp| fp == needle)
    }

    #[inline]
  unsafe  fn search_batch(&self, bucket: &[Fingerprint], needles: &[Fingerprint]) -> Vec<Option<usize>> {
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
    // 创建测试指纹
    fn fp(val: u16) -> Fingerprint {
        Fingerprint::new(val)
    }
   #[test]
    fn test_scalar_search() {
        let searcher = ScalarSearcher;
        let bucket = [fp(1), fp(2), fp(3), fp(4), fp(5)];
        unsafe {
        assert_eq!(searcher.search(&bucket, fp(3)), Some(2));
        assert_eq!(searcher.search(&bucket, fp(6)), None);
    }
    }
    
    #[test]
    fn test_scalar_search_batch() {
        let searcher = ScalarSearcher;
        let bucket = [fp(1), fp(2), fp(3), fp(4), fp(5)];
        let needles = [fp(2), fp(4), fp(6)];
        let results = unsafe { searcher.search_batch(&bucket, &needles) };
        assert_eq!(results, vec![Some(1), Some(3), None]);
    }
    
    #[test]
    fn test_scalar_search_empty() {
        let searcher = ScalarSearcher;
        let bucket = [];
        unsafe {
        assert_eq!(searcher.search(&bucket, fp(1)), None);}
    }
    
    #[test]
    fn test_scalar_search_single() {
        let searcher = ScalarSearcher;
        let bucket = [fp(42)];
        unsafe {
        assert_eq!(searcher.search(&bucket, fp(42)), Some(0));
        assert_eq!(searcher.search(&bucket, fp(1)), None);
        }
    }
    
    #[test]
    fn test_scalar_search_duplicates() {
        let searcher = ScalarSearcher;
        let bucket = [fp(1), fp(2), fp(2), fp(3)];
        unsafe {
        assert_eq!(searcher.search(&bucket, fp(2)), Some(1)); // 返回第一个匹配项
        }
    }
}