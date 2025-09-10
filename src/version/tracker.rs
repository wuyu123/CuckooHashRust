//! 版本跟踪器 - 实现无锁读操作检测并发修改

use crate::{
    hash::DefaultFingerprintGenerator, types::OperationType, FingerprintGenerator
    
};
use parking_lot::RwLock;
use std::{
    fmt, sync::{
        atomic::{AtomicU64, Ordering},
        Arc,
    }, time::{Duration, Instant}
};



/// 版本保护器 - 用于检测读操作期间的并发修改
#[derive(Debug, Clone)]
pub struct VersionGuard {
    version: u64,
    start_time: Instant,
    operation_type: OperationType,
}

impl VersionGuard {
    /// 创建新版本保护器
    pub fn new(version: u64, operation_type: OperationType) -> Self {
        Self {
            version,
            start_time: Instant::now(),
            operation_type,
        }
    }
    
    /// 获取开始时的版本号
    pub fn version(&self) -> u64 {
        self.version
    }
    
    /// 获取操作已耗时
    pub fn elapsed(&self) -> Duration {
        self.start_time.elapsed()
    }
    
    /// 获取操作类型
    pub fn operation_type(&self) -> OperationType {
        self.operation_type
    }
    
    /// 验证版本是否仍然有效
    pub fn is_valid(&self, current_version: u64) -> bool {
        self.version == current_version
    }
}

/// 版本统计
#[derive(Debug, Default)]
struct VersionStats {
    success_count: AtomicU64,
    conflict_count: AtomicU64,
    increment_count: AtomicU64,
    current_operations: RwLock<Vec<VersionGuard>>,
}

impl VersionStats {
    fn record_success(&self, operation_type: OperationType, duration: Duration) {
        self.success_count.fetch_add(1, Ordering::Relaxed);
        let mut ops = self.current_operations.write();
        ops.retain(|op| op.operation_type() != operation_type);
    }
    
    fn record_conflict(&self, operation_type: OperationType, duration: Duration) {
        self.conflict_count.fetch_add(1, Ordering::Relaxed);
        let mut ops = self.current_operations.write();
        ops.retain(|op| op.operation_type() != operation_type);
    }
    
    fn record_increment(&self) {
        self.increment_count.fetch_add(1, Ordering::Relaxed);
    }
    
    fn record_operation(&self, guard: VersionGuard) {
        let mut ops = self.current_operations.write();
        ops.push(guard);
    }
    
    fn check_long_operations(&self, current_time: Instant, threshold: Duration) -> Vec<VersionGuard> {
        let mut long_ops = Vec::new();
        let mut ops = self.current_operations.write();
        
        let mut i = 0;
        while i < ops.len() {
            if current_time.duration_since(ops[i].start_time) > threshold {
                long_ops.push(ops.remove(i));
            } else {
                i += 1;
            }
        }
        
        long_ops
    }
}

/// 版本跟踪器 - 管理并发访问的版本控制

pub struct VersionTracker {
    version: AtomicU64,
    fingerprint_generator: Arc<dyn FingerprintGenerator>,
    stats: VersionStats,
    emergency_threshold: Duration,
}

impl VersionTracker {
    /// 创建新版本跟踪器
    pub fn new() -> Self {
        Self {
            version: AtomicU64::new(0),
            fingerprint_generator: Arc::new(DefaultFingerprintGenerator),
            stats: VersionStats::default(),
            emergency_threshold: Duration::from_millis(100),
        }
    }
    
    /// 创建带指纹生成器的版本跟踪器
    pub fn with_fingerprint(fingerprint_generator: Arc<dyn FingerprintGenerator>) -> Self {
        Self {
            version: AtomicU64::new(0),
            fingerprint_generator,
            stats: VersionStats::default(),
            emergency_threshold: Duration::from_millis(100),
        }
    }
    
    /// 开始一个新操作
    pub fn begin_operation(&self, operation_type: OperationType) -> VersionGuard {
        let version = self.version.load(Ordering::Acquire);
        let guard = VersionGuard::new(version, operation_type);
        self.stats.record_operation(guard.clone());
        guard
    }
    
    /// 验证操作期间版本未变
    pub fn validate(&self, guard: &VersionGuard) -> bool {
        let current_version = self.version.load(Ordering::Acquire);
        let valid = guard.version() == current_version;
        
        // 记录验证统计
        if valid {
            self.stats.record_success(guard.operation_type(), guard.elapsed());
        } else {
            self.stats.record_conflict(guard.operation_type(), guard.elapsed());
        }
        
        valid
    }
    
    /// 递增版本号（在写操作结束时调用）
    pub fn increment(&self) {
        self.version.fetch_add(1, Ordering::Release);
        self.stats.record_increment();
    }
    
    /// 获取当前版本号
    pub fn current_version(&self) -> u64 {
        self.version.load(Ordering::Acquire)
    }
    
    /// 获取指纹生成器
    pub fn fingerprint_generator(&self) -> Arc<dyn FingerprintGenerator> {
        self.fingerprint_generator.clone()
    }
    
    /// 检查长时操作
    pub fn check_long_operations(&self) -> Vec<VersionGuard> {
        let current_time = Instant::now();
        self.stats.check_long_operations(current_time, self.emergency_threshold)
    }

    
}
impl fmt::Debug for VersionTracker {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("VersionTracker")
            .field("version", &self.version.load(Ordering::Relaxed))
            .field("stats", &self.stats)
            .field("emergency_threshold", &self.emergency_threshold)
            .field("fingerprint_generator", &"<dyn FingerprintGenerator>")
            .finish()
    }
}
// 单元测试
#[cfg(test)]
mod tests {
    use super::*;
    use std::thread;
    
    #[test]
    fn test_version_tracking() {
        let tracker = Arc::new(VersionTracker::new());
        
        // 初始版本为0
        assert_eq!(tracker.current_version(), 0);
        
        // 开始读操作
        let read_guard = tracker.begin_operation(OperationType::Read);
        assert_eq!(read_guard.version(), 0);
        
        // 验证读操作开始后版本未变
        assert!(tracker.validate(&read_guard));
        
        // 递增版本
        tracker.increment();
        assert_eq!(tracker.current_version(), 1);
        
        // 验证读操作已失效
        assert!(!tracker.validate(&read_guard));
        
        // 开始新的读操作
        let new_read_guard = tracker.begin_operation(OperationType::Read);
        assert_eq!(new_read_guard.version(), 1);
        assert!(tracker.validate(&new_read_guard));
    }
    
    #[test]
    fn test_concurrent_validation() {
        let tracker = Arc::new(VersionTracker::new());
        let n_workers = 8;
        let barrier = Arc::new(std::sync::Barrier::new(n_workers));
        
        let handles: Vec<_> = (0..n_workers)
            .map(|i| {
                let tracker = tracker.clone();
                let barrier = barrier.clone();
                thread::spawn(move || {
                    barrier.wait();
                    
                    for _ in 0..100 {
                        let guard = tracker.begin_operation(OperationType::Read);
                        
                        // 模拟一些工作
                        thread::sleep(Duration::from_micros(10));
                        
                        if !tracker.validate(&guard) {
                            return Err(());
                        }
                    }
                    Ok(())
                })
            })
            .collect();
        
        // 在工作线程运行期间递增版本
        thread::spawn({
            let tracker = tracker.clone();
            move || {
                for _ in 0..50 {
                    thread::sleep(Duration::from_micros(20));
                    tracker.increment();
                }
            }
        });
        
        // 检查结果
        let mut success_count = 0;
        for handle in handles {
            if handle.join().unwrap().is_ok() {
                success_count += 1;
            }
        }
        
        // 验证大多数读操作成功（由于并发写，部分会失败）
        assert!(success_count > n_workers / 2);
    }
    
    #[test]
    fn test_long_operation_detection() {
        let tracker = VersionTracker::new();
        
        let long_op = VersionGuard::new(
            tracker.current_version(),
            OperationType::Migration
        );
        
        // 记录操作
        tracker.stats.record_operation(long_op.clone());
        
        // 等待超过阈值
        thread::sleep(Duration::from_millis(150));
        
        let long_ops = tracker.check_long_operations();
        assert_eq!(long_ops.len(), 1);
        assert_eq!(long_ops[0].operation_type(), OperationType::Migration);
    }
}