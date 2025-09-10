//! SSE4.1加速搜索实现

use crate::{
    simd::searcher::{SimdSearcher, SimdSearcherType},
    types::Fingerprint,
};
use std::arch::x86_64::*;

/// SSE4.1搜索器
#[cfg(target_arch = "x86_64")]
pub struct Sse41Searcher;

#[cfg(target_arch = "x86_64")]
impl SimdSearcher for Sse41Searcher {
    #[inline]
    #[target_feature(enable = "sse4.1")]
    unsafe fn search(&self, bucket: &[Fingerprint], needle: Fingerprint) -> Option<usize> {
         // 正确转换Fingerprint为i8
        let needle_i16 = needle.as_i16();
        let needle_vec = _mm_set1_epi16(needle_i16);
        let mut index = 0;
        
        // 每次处理16个指纹
        for chunk in bucket.chunks_exact(16) {
            let data = unsafe { _mm_loadu_si128(chunk.as_ptr() as *const __m128i) };
            let cmp = _mm_cmpeq_epi8(needle_vec, data);
            let mask = _mm_movemask_epi8(cmp) as u32;
            
            if mask != 0 {
                return Some(index + mask.trailing_zeros() as usize);
            }
            
            index += 16;
        }
        
        // 处理剩余部分
        for (i, &fp) in bucket.chunks_exact(16).remainder().iter().enumerate() {
            if fp == needle {
                return Some(index + i);
            }
        }
        
        None
    }

    #[inline]
    #[target_feature(enable = "sse4.1")]
    unsafe fn search_batch(&self, bucket: &[Fingerprint], needles: &[Fingerprint]) -> Vec<Option<usize>> {
        let mut results = Vec::with_capacity(needles.len());
        
        for &needle in needles {
            results.push(self.search(bucket, needle));
        }
        
        results
    }
    
    fn searcher_type(&self) -> SimdSearcherType {
        SimdSearcherType::Sse41
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
    #[test]
fn test_sse41_search() {
    use std::arch::is_x86_feature_detected;
    
    if !is_x86_feature_detected!("sse4.1") {
        return; // 跳过测试
    }
    
    let searcher = Sse41Searcher;
    // 创建 Fingerprint 数组
    let mut bucket = [Fingerprint::new(0); 32]; // 32个指纹
    bucket[16] = Fingerprint::new(42);
    bucket[31] = Fingerprint::new(42);
    
    unsafe {
        assert_eq!(searcher.search(&bucket, Fingerprint::new(42)), Some(16));
        assert_eq!(searcher.search(&bucket, Fingerprint::new(0)), Some(0));
        assert_eq!(searcher.search(&bucket, Fingerprint::new(255)), None);
    }
}
}