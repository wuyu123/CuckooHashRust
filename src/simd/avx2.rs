//! AVX2加速搜索实现

use crate::{
    simd::searcher::{SimdSearcher, SimdSearcherType},
    types::Fingerprint,
};
use std::arch::x86_64::*;

/// AVX2搜索器
#[cfg(target_arch = "x86_64")]
pub struct Avx2Searcher;

#[cfg(target_arch = "x86_64")]
impl SimdSearcher for Avx2Searcher {
    #[inline]
    #[target_feature(enable = "avx2")]
    unsafe fn search(&self, bucket: &[Fingerprint], needle: Fingerprint) -> Option<usize> {
         // 正确转换Fingerprint为i8
        let needle_i16 = needle.as_i16();
        let needle_vec = _mm256_set1_epi16(needle_i16);
        
        let mut index = 0;
        
        // 每次处理32个指纹
        for chunk in bucket.chunks_exact(32) {
            let data = unsafe { _mm256_loadu_si256(chunk.as_ptr() as *const __m256i) };
            let cmp = _mm256_cmpeq_epi8(needle_vec, data);
            let mask = _mm256_movemask_epi8(cmp) as u32;
            
            if mask != 0 {
                return Some(index + mask.trailing_zeros() as usize);
            }
            
            index += 32;
        }
        
        // 处理剩余部分
        for (i, &fp) in bucket.chunks_exact(32).remainder().iter().enumerate() {
            if fp == needle {
                return Some(index + i);
            }
        }
        
        None
    }

    #[inline]
    #[target_feature(enable = "avx2")]
    unsafe fn search_batch(&self, bucket: &[Fingerprint], needles: &[Fingerprint]) -> Vec<Option<usize>> {
        let mut results = Vec::with_capacity(needles.len());
        
        for &needle in needles {
            results.push(self.search(bucket, needle));
        }
        
        results
    }
    
    fn searcher_type(&self) -> SimdSearcherType {
        SimdSearcherType::Avx2
    }
    
    unsafe fn search_all(&self, bucket: &[Fingerprint], needle: Fingerprint) -> Vec<usize> {
        todo!()
    }
}


// 单元测试
#[cfg(test)]
#[cfg(target_arch = "x86_64")]
mod tests {
    use super::*;
    use crate::types::Fingerprint;
    use std::arch::is_x86_feature_detected;
    
    // 创建测试指纹
    fn fp(val: u16) -> Fingerprint {
        Fingerprint::new(val)
    }
    
    #[test]
    fn test_avx2_search() {
        if !is_x86_feature_detected!("avx2") {
            return; // 跳过测试
        }
        
        let searcher = Avx2Searcher;
        let mut bucket = [fp(0); 64]; // 64个指纹
        bucket[32] = fp(42);
        bucket[63] = fp(42);
        
        unsafe {
            assert_eq!(searcher.search(&bucket, fp(42)), Some(32));
            assert_eq!(searcher.search(&bucket, fp(0)), Some(0));
            assert_eq!(searcher.search(&bucket, fp(255)), None);
        }
    }
    
    #[test]
    fn test_avx2_search_batch() {
        if !is_x86_feature_detected!("avx2") {
            return; // 跳过测试
        }
        
        let searcher = Avx2Searcher;
        let bucket = [fp(1), fp(2), fp(3)];
        let needles = [fp(2), fp(3), fp(4)];
        
        unsafe {
            let results = searcher.search_batch(&bucket, &needles);
            assert_eq!(results, vec![Some(1), Some(2), None]);
        }
    }
    
    #[test]
    fn test_avx2_search_partial_chunk() {
        if !is_x86_feature_detected!("avx2") {
            return; // 跳过测试
        }
        
        let searcher = Avx2Searcher;
        let mut bucket = [fp(0); 35]; // 35个指纹
        bucket[32] = fp(42); // 第33个元素
        bucket[34] = fp(42); // 最后一个元素
        
        unsafe {
            assert_eq!(searcher.search(&bucket, fp(42)), Some(32));
            assert_eq!(searcher.search(&bucket, fp(0)), Some(0));
        }
    }
    
    #[test]
    fn test_avx2_search_empty() {
        if !is_x86_feature_detected!("avx2") {
            return; // 跳过测试
        }
        
        let searcher = Avx2Searcher;
        let bucket = [];
        
        unsafe {
            assert_eq!(searcher.search(&bucket, fp(42)), None);
        }
    }
}