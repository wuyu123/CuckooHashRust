//! 迁移统计 - 跟踪哈希表迁移性能

use std::{
    sync::atomic::{AtomicU64, AtomicUsize, Ordering}, 
    time::Duration
};

/// 迁移统计详情
#[derive(Debug, Clone, Copy)]
pub struct MigrationSnapshot {
    pub migrated_items: u64,
    pub skipped_items: u64,
    pub failed_items: u64,
    pub duration: Duration,
}

/// 迁移统计接口
pub trait MigrationRecorder: Send + Sync {
    /// 记录迁移开始
    fn start_migration(&self);
    
    /// 记录迁移完成
    fn record_migration(&self, stats: MigrationSnapshot);
    
    /// 获取迁移统计快照
    fn snapshot(&self) -> MigrationAccumulatedSnapshot;
    
    /// 重置统计
    fn reset(&self);
    
    /// 导出Prometheus格式指标
    fn export_prometheus(&self) -> String;
}

/// 累积迁移统计
#[derive(Debug, Default)]
pub struct MigrationStats {
    count: AtomicU64,
    migrated_items: AtomicU64,
    skipped_items: AtomicU64,
    failed_items: AtomicU64,
    duration_sum: AtomicU64, // 纳秒
    rate: AtomicUsize,       // 迁移速率 (items/sec * 100)
    failure_count: AtomicU64,
}

/// 累积统计快照
#[derive(Debug, Default, Clone, Copy)]
pub struct MigrationAccumulatedSnapshot {
    pub count: u64,
    pub migrated_items: u64,
    pub skipped_items: u64,
    pub failed_items: u64,
    pub duration_sum: Duration,
    pub rate: usize,
    pub failure_count: u64,
}

impl MigrationStats {
    /// 创建新统计
    pub fn new() -> Self {
        Self::default()
    }
}

impl MigrationRecorder for MigrationStats {
    fn start_migration(&self) {
        self.count.fetch_add(1, Ordering::Relaxed);
    }
    
    fn record_migration(&self, stats: MigrationSnapshot) {
        self.migrated_items.fetch_add(
            stats.migrated_items, 
            Ordering::Relaxed
        );
        
        self.skipped_items.fetch_add(
            stats.skipped_items, 
            Ordering::Relaxed
        );
        
        self.failed_items.fetch_add(
            stats.failed_items, 
            Ordering::Relaxed
        );
        
        self.duration_sum.fetch_add(
            stats.duration.as_nanos() as u64,
            Ordering::Relaxed
        );
        
        if stats.failed_items > 0 {
            self.failure_count.fetch_add(1, Ordering::Relaxed);
        }
        
        // 计算并存储速率
        if stats.duration.as_secs() > 0 {
            let rate = (stats.migrated_items as f64 / stats.duration.as_secs_f64() * 100.0) as usize;
            self.rate.store(rate, Ordering::Release);
        }
    }
    
    fn snapshot(&self) -> MigrationAccumulatedSnapshot {
        MigrationAccumulatedSnapshot {
            count: self.count.load(Ordering::Relaxed),
            migrated_items: self.migrated_items.load(Ordering::Relaxed),
            skipped_items: self.skipped_items.load(Ordering::Relaxed),
            failed_items: self.failed_items.load(Ordering::Relaxed),
            duration_sum: Duration::from_nanos(self.duration_sum.load(Ordering::Relaxed)),
            rate: self.rate.load(Ordering::Relaxed),
            failure_count: self.failure_count.load(Ordering::Relaxed),
        }
    }
    
    fn reset(&self) {
        self.count.store(0, Ordering::Relaxed);
        self.migrated_items.store(0, Ordering::Relaxed);
        self.skipped_items.store(0, Ordering::Relaxed);
        self.failed_items.store(0, Ordering::Relaxed);
        self.duration_sum.store(0, Ordering::Relaxed);
        self.rate.store(0, Ordering::Relaxed);
        self.failure_count.store(0, Ordering::Relaxed);
    }
    
    fn export_prometheus(&self) -> String {
        let snapshot = self.snapshot();
        let mut output = String::new();
        
        output.push_str("# HELP cuckoo_migration_count Total migration operations\n");
        output.push_str("# TYPE cuckoo_migration_count counter\n");
        output.push_str(&format!(
            "cuckoo_migration_count {}\n",
            snapshot.count
        ));
        
        output.push_str("# HELP cuckoo_migration_items Total migrated items\n");
        output.push_str("# TYPE cuckoo_migration_items counter\n");
        output.push_str(&format!(
            "cuckoo_migration_items {}\n",
            snapshot.migrated_items
        ));
        
        output.push_str("# HELP cuckoo_migration_skipped_items Total skipped items\n");
        output.push_str("# TYPE c极速_migration_skipped_items counter\n");
        output.push_str(&format!(
            "cuckoo_migration_skipped_items {}\n",
            snapshot.skipped_items
        ));
        
        output.push_str("# HELP cuckoo_migration_failed_items Total failed items\n");
        output.push_str("# TYPE cuckoo_migration_failed_items counter\n");
        output.push_str(&format!(
            "cuckoo_migration_failed_items {}\n",
            snapshot.failed_items
        ));
        
        let total_duration = snapshot.duration_sum.as_secs_f64();
        output.push_str("# HELP cuckoo_migration_duration_total Total migration duration (seconds)\n");
        output.push_str("# TYPE cuckoo_migration_duration_total counter\n");
        output.push_str(&format!(
            "cuckoo_migration_duration_total {:.3}\n",
            total_duration
        ));
        
        let avg_duration = if snapshot.count > 0 {
            total_duration / snapshot.count as f64
        } else {
            0.0
        };
        output.push_str("# HELP cuckoo_migration_duration_avg Average migration duration (seconds)\n");
        output.push_str("# TYPE cuckoo_migration_duration_avg gauge\n");
        output.push_str(&format!(
            "cuckoo_migration_duration_avg {:.3}\n",
            avg_duration
        ));
        
        output.push_str("# HELP cuckoo_migration_rate Migration rate (items/sec)\n");
        output.push_str("# TYPE cuckoo_migration_rate gauge\n");
        output.push_str(&format!(
            "cuckoo_migration_rate {:.2}\n",
            snapshot.rate as f64 / 100.0
        ));
        
        output.push_str("# HELP cuckoo_migration_failure_count Migration failures\n");
        output.push_str("# TYPE cuckoo_migration_failure_count counter\n");
        output.push_str(&format!(
            "cuckoo_migration_failure_count {}\n",
            snapshot.failure_count
        ));
        
        output
    }
}

/// 禁用迁移统计
#[derive(Debug, Default)]
pub struct DisabledMigrationRecorder;

impl MigrationRecorder for DisabledMigrationRecorder {
    fn start_migration(&self) {}
    fn record_migration(&self, _stats: MigrationSnapshot) {}
    fn snapshot(&self) -> MigrationAccumulatedSnapshot { MigrationAccumulatedSnapshot::default() }
    fn reset(&self) {}
    fn export_prometheus(&self) -> String { String::new() }
}