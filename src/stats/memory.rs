//! 内存统计 - 跟踪哈希表内存使用情况

use std::sync::atomic::{AtomicUsize, AtomicU64, Ordering};

/// 内存统计接口
pub trait MemoryRecorder: Send + Sync {
    /// 记录内存分配
    fn record_allocation(&self, size: usize);
    
    /// 记录内存释放
    fn record_deallocation(&self, size: usize);
    
    /// 记录内存池利用率
    fn record_pool_utilization(&self, used: usize, total: usize);
    
    /// 获取内存统计快照
    fn snapshot(&self) -> MemoryStatsSnapshot;
    
    /// 重置统计
    fn reset(&self);
    
    /// 导出Prometheus格式指标
    fn export_prometheus(&self) -> String;
}

/// 内存统计快照（非原子）
#[derive(Debug, Default, Clone)]
pub struct MemoryStatsSnapshot {
    pub total_allocated: usize,
    pub current_used: usize,
    pub peak_used: usize,
    pub allocation_count: u64,
    pub deallocation_count: u64,
    pub pool_utilization: f64, // 存储为百分比 (0.0-100.0)
}

/// 原子内存统计
#[derive(Debug, Default)]
pub struct AtomicMemoryStats {
    total_allocated: AtomicUsize,
    current_used: AtomicUsize,
    peak_used: AtomicUsize,
    allocation_count: AtomicU64,
    deallocation_count: AtomicU64,
    pool_utilization: AtomicUsize, // 存储为百分比 * 100 (避免浮点)
}

impl AtomicMemoryStats {
    /// 创建新统计
    pub fn new() -> Self {
        Self::default()
    }
}

impl MemoryRecorder for AtomicMemoryStats {
    fn record_allocation(&self, size: usize) {
        self.allocation_count.fetch_add(1, Ordering::Relaxed);
        let prev = self.current_used.fetch_add(size, Ordering::AcqRel);
        let new = prev + size;
        
        // 更新峰值
        let mut current_peak = self.peak_used.load(Ordering::Acquire);
        while new > current_peak {
            match self.peak_used.compare_exchange_weak(
                current_peak,
                new,
                Ordering::AcqRel,
                Ordering::Acquire,
            ) {
                Ok(_) => break,
                Err(peak) => current_peak = peak,
            }
        }
        
        // 更新总分配量
        self.total_allocated.fetch_add(size, Ordering::Relaxed);
    }
    
    fn record_deallocation(&self, size: usize) {
        self.deallocation_count.fetch_add(1, Ordering::Relaxed);
        self.current_used.fetch_sub(size, Ordering::AcqRel);
    }
    
    fn record_pool_utilization(&self, used: usize, total: usize) {
        let ratio = if total > 0 {
            (used as f64 / total as f64 * 10000.0) as usize // 存储为百分比 * 100
        } else {
            0
        };
        self.pool_utilization.store(ratio, Ordering::Release);
    }
    
    fn snapshot(&self) -> MemoryStatsSnapshot {
        MemoryStatsSnapshot {
            total_allocated: self.total_allocated.load(Ordering::Relaxed),
            current_used: self.current_used.load(Ordering::Relaxed),
            peak_used: self.peak_used.load(Ordering::Relaxed),
            allocation_count: self.allocation_count.load(Ordering::Relaxed),
            deallocation_count: self.deallocation_count.load(Ordering::Relaxed),
            pool_utilization: self.pool_utilization.load(Ordering::Relaxed) as f64 / 100.0,
        }
    }
    
    fn reset(&self) {
        self.total_allocated.store(0, Ordering::Relaxed);
        self.current_used.store(0, Ordering::Relaxed);
        self.peak_used.store(0, Ordering::Relaxed);
        self.allocation_count.store(0, Ordering::Relaxed);
        self.deallocation_count.store(0, Ordering::Relaxed);
        self.pool_utilization.store(0, Ordering::Relaxed);
    }
    
    fn export_prometheus(&self) -> String {
        let snapshot = self.snapshot();
        let mut output = String::new();
        
        output.push_str("# HELP cuckoo_memory_total_allocated Total allocated memory (bytes)\n");
        output.push_str("# TYPE cuckoo_memory_total_allocated counter\n");
        output.push_str(&format!(
            "cuckoo_memory_total_allocated {}\n",
            snapshot.total_allocated
        ));
        
        output.push_str("# HELP cuckoo_memory_current_used Current memory usage (bytes)\n");
        output.push_str("# TYPE cuckoo_memory_current_used gauge\n");
        output.push_str(&format!(
            "cuckoo_memory_current_used {}\n",
            snapshot.current_used
        ));
        
        output.push_str("# HELP cuckoo_memory_peak_used Peak memory usage (bytes)\n");
        output.push_str("# TYPE cuckoo_memory_peak_used gauge\n");
        output.push_str(&format!(
            "cuckoo_memory_peak_used {}\n",
            snapshot.peak_used
        ));
        
        output.push_str("# HELP cuckoo_memory_allocation_count Memory allocation count\n");
        output.push_str("# TYPE cuckoo_memory_allocation_count counter\n");
        output.push_str(&format!(
            "cuckoo_memory_allocation_count {}\n",
            snapshot.allocation_count
        ));
        
        output.push_str("# HELP cuckoo_memory_deallocation_count Memory deallocation count\n");
        output.push_str("# TYPE cuckoo_memory_deallocation_count counter\n");
        output.push_str(&format!(
            "cuckoo_memory_deallocation_count {}\n",
            snapshot.deallocation_count
        ));
        
        output.push_str("# HELP cuckoo_memory_pool_utilization Memory pool utilization percentage\n");
        output.push_str("# TYPE cuckoo_memory_pool_utilization gauge\n");
        output.push_str(&format!(
            "cuckoo_memory_pool_utilization {:.2}\n",
            snapshot.pool_utilization
        ));
        
        output
    }
}

/// 禁用内存统计
#[derive(Default)]
pub struct DisabledMemoryRecorder;

impl MemoryRecorder for DisabledMemoryRecorder {
    fn record_allocation(&self, _size: usize) {}
    fn record_deallocation(&self, _size: usize) {}
    fn record_pool_utilization(&self, _used: usize, _total: usize) {}
    fn snapshot(&self) -> MemoryStatsSnapshot {
        MemoryStatsSnapshot::default()
    }
    fn reset(&self) {}
    fn export_prometheus(&self) -> String {
        String::new()
    }
}