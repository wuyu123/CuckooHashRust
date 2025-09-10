//! 统计模块 - 统一管理哈希表性能指标

pub mod recorder;
pub mod operation;
pub mod memory;
pub mod migration;

use std::{sync::Arc, time::Duration};

pub use recorder::{StatsRecorder, GlobalStatsRecorder, StatsRecorderFactory,DisabledStatsRecorder};
pub use operation::{ OperationRecorder};
pub use memory::{ MemoryRecorder};
pub use migration::{MigrationStats, MigrationRecorder};

use crate::{stats::{memory::MemoryStatsSnapshot, migration::{MigrationAccumulatedSnapshot, MigrationSnapshot}}, types::OperationType};

/// 全局统计记录器
pub static GLOBAL_STATS: once_cell::sync::Lazy<Arc<dyn StatsRecorder>> = 
    once_cell::sync::Lazy::new(|| {
        Arc::new(GlobalStatsRecorder::new())
    });

/// 记录操作统计
pub fn record_operation(op_type: OperationType, duration: Duration, success: bool) {
    GLOBAL_STATS.record_operation(op_type, duration, success);
}

/// 记录内存分配
pub fn record_allocation(size: usize) {
    GLOBAL_STATS.memory_stats().record_allocation(size);
}

/// 记录内存释放
pub fn record_deallocation(size: usize) {
    GLOBAL_STATS.memory_stats().record_deallocation(size);
}

/// 记录迁移开始
pub fn record_migration_start() {
    GLOBAL_STATS.migration_stats().start_migration();
}

/// 记录迁移完成
pub fn record_migration_complete(stats: MigrationSnapshot) {
    GLOBAL_STATS.migration_stats().record_migration(stats);
}

/// 获取操作统计快照
pub fn operation_snapshot() -> operation::OperationStatsSnapshot {
    GLOBAL_STATS.operation_stats().snapshot()
}

/// 获取内存统计快照
pub fn memory_snapshot() -> MemoryStatsSnapshot {
    GLOBAL_STATS.memory_stats().snapshot()
}

/// 获取迁移统计快照
pub fn migration_snapshot() -> MigrationAccumulatedSnapshot {
    GLOBAL_STATS.migration_stats().snapshot()
}

/// 重置所有统计
pub fn reset_stats() {
    GLOBAL_STATS.reset();
}

/// 导出Prometheus格式指标
pub fn export_prometheus() -> String {
    GLOBAL_STATS.export_prometheus()
}