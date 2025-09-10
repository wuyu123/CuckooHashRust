//! 混合搜索策略 - 结合不同SIMD指令集

#[cfg(target_arch = "x86_64")]
use crate::{
    simd::searcher::{SimdSearcher, SimdSearcherType},
    types::Fingerprint,
};

/// 混合搜索器
#[cfg(target_arch = "x86_64")]
pub struct HybridSearcher {
    avx2_searcher: super::avx2::Avx2Searcher,
    sse_searcher: super::sse4::Sse41Searcher,
    scalar_searcher: super::scalar::ScalarSearcher,
}

#[cfg(target_arch = "x86_64")]
impl HybridSearcher {
    pub fn new() -> Self {
        Self {
            avx2_searcher: super::avx2::Avx2Searcher,
            sse_searcher: super::sse4::Sse41Searcher,
            scalar_searcher: super::scalar::ScalarSearcher,
        }
    }
}

#[cfg(target_arch = "x86_64")]
impl SimdSearcher for HybridSearcher {
    #[inline]
   unsafe fn search(&self, bucket: &[Fingerprint], needle: Fingerprint) -> Option<usize> {
        // 根据大小选择策略
        if bucket.len() >= 64 && is_x86_feature_detected!("avx2") {
            unsafe { self.avx2_searcher.search(bucket, needle) }
        } else if bucket.len() >= 32 && is_x86_feature_detected!("sse4.1") {
            unsafe { self.sse_searcher.search(bucket, needle) }
        } else {
            self.scalar_searcher.search(bucket, needle)
        }
    }

    #[inline]
   unsafe fn search_batch(&self, bucket: &[Fingerprint], needles: &[Fingerprint]) -> Vec<Option<usize>> {
        // 根据平均大小选择策略
        let avg_len = needles.iter().map(|_| bucket.len()).sum::<usize>() / needles.len().max(1);
        
        if avg_len >= 64 && is_x86_feature_detected!("avx2") {
            unsafe { self.avx2_searcher.search_batch(bucket, needles) }
        } else if avg_len >= 32 && is_x86_feature_detected!("sse4.1") {
            unsafe { self.sse_searcher.search_batch(bucket, needles) }
        } else {
            self.scalar_searcher.search_batch(bucket, needles)
        }
    }
    
    fn searcher_type(&self) -> SimdSearcherType {
        SimdSearcherType::Hybrid
    }
    
    unsafe fn search_all(&self, bucket: &[Fingerprint], needle: Fingerprint) -> Vec<usize> {
        todo!()
    }
}