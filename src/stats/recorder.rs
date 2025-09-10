// src/stats/recorder.rs
//! 统计记录器接口 - 定义统一统计API

use std::{sync::Arc, time::Duration};

use crate::{
    stats::{
        memory::{MemoryRecorder, MemoryStatsSnapshot},
        migration::{MigrationRecorder, MigrationAccumulatedSnapshot},
        operation::{OperationRecorder, OperationStatsSnapshot},
    },
    types::OperationType,
};

/// 统计记录器特征
pub trait StatsRecorder: Send + Sync {
    /// 记录操作
    fn record_operation(&self, op_type: OperationType, duration: Duration, success: bool);
    /// 记录操作计数
    fn record_operation_count(&self, op_type: OperationType);
    /// 获取操作统计接口
    fn operation_stats(&self) -> &dyn OperationRecorder;
    
    /// 获取内存统计接口
    fn memory_stats(&self) -> &dyn MemoryRecorder;
    
    /// 获取迁移统计接口
    fn migration_stats(&self) -> &dyn MigrationRecorder;
    
    /// 重置所有统计
    fn reset(&self);
    
    /// 导出Prometheus格式指标
    fn export_prometheus(&self) -> String;
    
    /// 开始计时器
    fn start_timer(&self, op_type: OperationType);

    /// 获取操作持续时间
    fn get_duration(&self, op_type: OperationType) -> Option<Duration>;

    /// 获取操作统计快照
    fn operation_stats_snapshot(&self) -> OperationStatsSnapshot;
    
    /// 获取内存统计快照
    fn memory_stats_snapshot(&self) -> MemoryStatsSnapshot;
    
    /// 获取迁移统计快照
    fn migration_stats_snapshot(&self) -> MigrationAccumulatedSnapshot;
}

/// 默认统计记录器实现
#[derive(Default)]
pub struct DefaultStatsRecorder {
    operation: super::operation::AtomicOperationStats,
    memory: super::memory::AtomicMemoryStats,
    migration: super::migration::MigrationStats,
}

impl StatsRecorder for DefaultStatsRecorder {
    fn record_operation(&self, op_type: OperationType, duration: Duration, success: bool) {
        self.operation.record(op_type, duration, success);
    }
    fn record_operation_count(&self, op_type: OperationType) {
        self.operation.record_count(op_type);
    }
    fn operation_stats(&self) -> &dyn OperationRecorder {
        &self.operation
    }
    
    fn memory_stats(&self) -> &dyn MemoryRecorder {
        &self.memory
    }
    
    fn migration_stats(&self) -> &dyn MigrationRecorder {
        &self.migration
    }
    
    fn reset(&self) {
        self.operation.reset();
        self.memory.reset();
        self.migration.reset();
    }
    
    fn export_prometheus(&self) -> String {
        let mut output = String::new();
        output.push_str(&self.operation.export_prometheus());
        output.push_str(&self.memory.export_prometheus());
        output.push_str(&self.migration.export_prometheus());
        output
    }
    
    fn start_timer(&self, op_type: OperationType) {
        self.operation.start_timer(op_type);
    }
    
    fn get_duration(&self, op_type: OperationType) -> Option<Duration> {
        self.operation.get_duration(op_type)
    }
    
    fn operation_stats_snapshot(&self) -> OperationStatsSnapshot {
        self.operation.snapshot()
    }
    
    fn memory_stats_snapshot(&self) -> MemoryStatsSnapshot {
        self.memory.snapshot()
    }
    
    fn migration_stats_snapshot(&self) -> MigrationAccumulatedSnapshot {
        self.migration.snapshot()
    }
}

/// 全局统计记录器实现
#[derive(Default)]
pub struct GlobalStatsRecorder {
    operation: super::operation::AtomicOperationStats,
    memory: super::memory::AtomicMemoryStats,
    migration: super::migration::MigrationStats,
}

impl GlobalStatsRecorder {
    pub fn new() -> Self {
        Self::default()
    }
}

impl StatsRecorder for GlobalStatsRecorder {
    fn record_operation(&self, op_type: OperationType, duration: Duration, success: bool) {
        self.operation.record(op_type, duration, success);
    }
    fn record_operation_count(&self, op_type: OperationType) {
        self.operation.record_count(op_type);
    }
    fn operation_stats(&self) -> &dyn OperationRecorder {
        &self.operation
    }
    
    fn memory_stats(&self) -> &dyn MemoryRecorder {
        &self.memory
    }
    
    fn migration_stats(&self) -> &dyn MigrationRecorder {
        &self.migration
    }
    
    fn reset(&self) {
        self.operation.reset();
        self.memory.reset();
        self.migration.reset();
    }
    
    fn export_prometheus(&self) -> String {
        let mut output = String::new();
        output.push_str(&self.operation.export_prometheus());
        output.push_str(&self.memory.export_prometheus());
        output.push_str(&self.migration.export_prometheus());
        output
    }

    fn start_timer(&self, op_type: OperationType) {
        self.operation.start_timer(op_type);
    }
    
    fn get_duration(&self, op_type: OperationType) -> Option<Duration> {
        self.operation.get_duration(op_type)
    }
    
    fn operation_stats_snapshot(&self) -> OperationStatsSnapshot {
        self.operation.snapshot()
    }
    
    fn memory_stats_snapshot(&self) -> MemoryStatsSnapshot {
        self.memory.snapshot()
    }
    
    fn migration_stats_snapshot(&self) -> MigrationAccumulatedSnapshot {
        self.migration.snapshot()
    }
}

/// 健康状态
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum HealthStatus {
    Healthy,
    Unhealthy,
}

impl HealthStatus {
    /// 检查是否健康
    pub fn is_healthy(&self) -> bool {
        matches!(self, HealthStatus::Healthy)
    }
}

/// 统计记录器工厂
pub struct StatsRecorderFactory;

impl StatsRecorderFactory {
    /// 创建默认记录器
    pub fn create_default() -> Arc<dyn StatsRecorder> {
        Arc::new(GlobalStatsRecorder::new())
    }
    
    /// 创建禁用统计的记录器
    pub fn create_disabled() -> Arc<dyn StatsRecorder> {
        Arc::new(DisabledStatsRecorder)
    }
    
    /// 创建带自定义配置的记录器
    pub fn create_custom(
        operation: impl OperationRecorder + 'static,
        memory: impl MemoryRecorder + 'static,
        migration: impl MigrationRecorder + 'static,
    ) -> Arc<dyn StatsRecorder> {
        Arc::new(CustomStatsRecorder {
            operation: Box::new(operation),
            memory: Box::new(memory),
            migration: Box::new(migration),
        })
    }
}

/// 禁用统计的记录器
pub struct DisabledStatsRecorder;

impl StatsRecorder for DisabledStatsRecorder {
    fn record_operation(&self, _op_type: OperationType, _duration: Duration, _success: bool) {}
    fn record_operation_count(&self, op_type: OperationType) {
        
    }
    fn operation_stats(&self) -> &dyn OperationRecorder { &super::operation::DisabledOperationRecorder }
    fn memory_stats(&self) -> &dyn MemoryRecorder { &super::memory::DisabledMemoryRecorder }
    fn migration_stats(&self) -> &dyn MigrationRecorder { &super::migration::DisabledMigrationRecorder }
    fn reset(&self) {}
    fn export_prometheus(&self) -> String { String::new() }
    
    fn start_timer(&self, _op_type: OperationType) {}
    
    fn get_duration(&self, _op_type: OperationType) -> Option<Duration> { None }
    
    fn operation_stats_snapshot(&self) -> OperationStatsSnapshot {
        OperationStatsSnapshot::default()
    }
    
    fn memory_stats_snapshot(&self) -> MemoryStatsSnapshot {
        MemoryStatsSnapshot::default()
    }
    
    fn migration_stats_snapshot(&self) -> MigrationAccumulatedSnapshot {
        MigrationAccumulatedSnapshot::default()
    }
}

/// 自定义统计记录器
struct CustomStatsRecorder {
    operation: Box<dyn OperationRecorder>,
    memory: Box<dyn MemoryRecorder>,
    migration: Box<dyn MigrationRecorder>,
}

impl StatsRecorder for CustomStatsRecorder {
    fn record_operation(&self, op_type: OperationType, duration: Duration, success: bool) {
        self.operation.record(op_type, duration, success);
    }
    fn record_operation_count(&self, op_type: OperationType) {
        self.operation.record_count(op_type);
    }
    fn operation_stats(&self) -> &dyn OperationRecorder {
        self.operation.as_ref()
    }
    
    fn memory_stats(&self) -> &dyn MemoryRecorder {
        self.memory.as_ref()
    }
    
    fn migration_stats(&self) -> &dyn MigrationRecorder {
        self.migration.as_ref()
    }
    
    fn reset(&self) {
        self.operation.reset();
        self.memory.reset();
        self.migration.reset();
    }
    
    fn export_prometheus(&self) -> String {
        let mut output = String::new();
        output.push_str(&self.operation.export_prometheus());
        output.push_str(&self.memory.export_prometheus());
        output.push_str(&self.migration.export_prometheus());
        output
    }
    
    fn start_timer(&self, op_type: OperationType) {
        self.operation.start_timer(op_type);
    }
    
    fn get_duration(&self, op_type: OperationType) -> Option<Duration> {
        self.operation.get_duration(op_type)
    }
    
    fn operation_stats_snapshot(&self) -> OperationStatsSnapshot {
        self.operation.snapshot()
    }
    
    fn memory_stats_snapshot(&self) -> MemoryStatsSnapshot {
        self.memory.snapshot()
    }
    
    fn migration_stats_snapshot(&self) -> MigrationAccumulatedSnapshot {
        self.migration.snapshot()
    }
}