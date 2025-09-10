// src/stats/operation.rs
//! 操作统计 - 跟踪哈希表操作性能

use crate::{
    error::CuckooError, types::OperationType,
};
use std::{
    collections::HashMap,
    sync::{
        atomic::{AtomicU64, Ordering},
        RwLock,
    },
    time::{Duration, Instant},
};

/// 操作统计接口
pub trait OperationRecorder: Send + Sync {
    /// 记录操作
    fn record(&self, op_type: OperationType, duration: Duration, success: bool);
    fn record_count(&self, op_type: OperationType);
    /// 获取操作统计快照
    fn snapshot(&self) -> OperationStatsSnapshot;
    
    /// 重置统计
    fn reset(&self);
    
    /// 导出Prometheus格式指标
    fn export_prometheus(&self) -> String;

    /// 开始计时器
    fn start_timer(&self, op_type: OperationType);
    
    /// 获取操作持续时间
    fn get_duration(&self, op_type: OperationType) -> Option<Duration>;
}

/// 操作统计快照
#[derive(Debug, Default, Clone)]
pub struct OperationStatsSnapshot {
    pub insert_count: u64,
    pub get_count: u64,
    pub remove_count: u64,
    pub update_count: u64,
    pub kick_count: u64,
    pub migration_count: u64,
    pub resize_count: u64,
    pub read_count: u64,
    pub write_count: u64,
    pub statistics_count: u64,
    pub collision_count: u64,
    pub total_duration: u64, // 纳秒
    pub iterate_count: u64,
}

/// 原子操作统计
#[derive(Debug, Default)]
pub struct AtomicOperationStats {
    insert_count: AtomicU64,
    get_count: AtomicU64,
    remove_count: AtomicU64,
    update_count: AtomicU64,
    kick_count: AtomicU64,
    migration_count: AtomicU64,
    resize_count: AtomicU64,
    read_count: AtomicU64,
    write_count: AtomicU64,
    statistics_count: AtomicU64,
    collision_count: AtomicU64,
    total_duration: AtomicU64, // 纳秒
    iterate_count: AtomicU64,
    timers: RwLock<HashMap<OperationType, Instant>>,
}

impl AtomicOperationStats {
    /// 创建新统计
    pub fn new() -> Self {
        Self::default()
    }
    
    /// 开始计时器
    pub fn start_timer(&self, op_type: OperationType) {
        let mut timers = self.timers.write().unwrap();
        timers.insert(op_type, Instant::now());
    }
    
    /// 获取操作持续时间
    pub fn get_duration(&self, op_type: OperationType) -> Option<Duration> {
        let timers = self.timers.read().unwrap();
        timers.get(&op_type).map(|start| start.elapsed())
    }
}

impl OperationRecorder for AtomicOperationStats {
    fn record(&self, op_type: OperationType, duration: Duration, success: bool) {
        let nanos = duration.as_nanos() as u64;
        
        // 更新操作计数
        match op_type {
            OperationType::Insert => self.insert_count.fetch_add(1, Ordering::Relaxed),
            OperationType::Get => self.get_count.fetch_add(1, Ordering::Relaxed),
            OperationType::Remove => self.remove_count.fetch_add(1, Ordering::Relaxed),
            OperationType::Update => self.update_count.fetch_add(1, Ordering::Relaxed),
            OperationType::Kick => self.kick_count.fetch_add(1, Ordering::Relaxed),
            OperationType::Migration => self.migration_count.fetch_add(1, Ordering::Relaxed),
            OperationType::Resize => self.resize_count.fetch_add(1, Ordering::Relaxed),
            OperationType::Read => self.read_count.fetch_add(1, Ordering::Relaxed),
            OperationType::Write => self.write_count.fetch_add(1, Ordering::Relaxed),
            OperationType::Statistics => self.statistics_count.fetch_add(1, Ordering::Relaxed),
            OperationType::Iterate => self.iterate_count.fetch_add(1, Ordering::Relaxed),
        };
        
        // 更新总耗时
        self.total_duration.fetch_add(nanos, Ordering::Relaxed);
        
        // 如果操作失败，增加碰撞计数
        if !success {
            self.collision_count.fetch_add(1, Ordering::Relaxed);
        }
    }
    fn record_count(&self, op_type: OperationType) {
        match op_type {
            OperationType::Insert => self.insert_count.fetch_add(1, Ordering::Relaxed),
            OperationType::Get => self.get_count.fetch_add(1, Ordering::Relaxed),
            OperationType::Remove => self.remove_count.fetch_add(1, Ordering::Relaxed),
            OperationType::Update => self.update_count.fetch_add(1, Ordering::Relaxed),
            OperationType::Kick => self.kick_count.fetch_add(1, Ordering::Relaxed),
            OperationType::Migration => self.migration_count.fetch_add(1, Ordering::Relaxed),
            OperationType::Resize => self.resize_count.fetch_add(1, Ordering::Relaxed),
            OperationType::Read => self.read_count.fetch_add(1, Ordering::Relaxed),
            OperationType::Write => self.write_count.fetch_add(1, Ordering::Relaxed),
            OperationType::Statistics => self.statistics_count.fetch_add(1, Ordering::Relaxed),
            OperationType::Iterate => self.iterate_count.fetch_add(1, Ordering::Relaxed),
            _ => return,
        };
    }
    fn snapshot(&self) -> OperationStatsSnapshot {
        OperationStatsSnapshot {
            insert_count: self.insert_count.load(Ordering::Relaxed),
            get_count: self.get_count.load(Ordering::Relaxed),
            remove_count: self.remove_count.load(Ordering::Relaxed),
            update_count: self.update_count.load(Ordering::Relaxed),
            kick_count: self.kick_count.load(Ordering::Relaxed),
            migration_count: self.migration_count.load(Ordering::Relaxed),
            resize_count: self.resize_count.load(Ordering::Relaxed),
            read_count: self.read_count.load(Ordering::Relaxed),
            write_count: self.write_count.load(Ordering::Relaxed),
            statistics_count: self.statistics_count.load(Ordering::Relaxed),
            iterate_count: self.iterate_count.load(Ordering::Relaxed),
            collision_count: self.collision_count.load(Ordering::Relaxed),
            total_duration: self.total_duration.load(Ordering::Relaxed),
        }
    }
    
    fn reset(&self) {
        self.insert_count.store(0, Ordering::Relaxed);
        self.get_count.store(0, Ordering::Relaxed);
        self.remove_count.store(0, Ordering::Relaxed);
        self.update_count.store(0, Ordering::Relaxed);
        self.kick_count.store(0, Ordering::Relaxed);
        self.migration_count.store(0, Ordering::Relaxed);
        self.resize_count.store(0, Ordering::Relaxed);
        self.read_count.store(0, Ordering::Relaxed);
        self.write_count.store(0, Ordering::Relaxed);
        self.statistics_count.store(0, Ordering::Relaxed);
        self.iterate_count.store(0, Ordering::Relaxed);
        self.collision_count.store(0, Ordering::Relaxed);
        self.total_duration.store(0, Ordering::Relaxed);
    }
    
    fn export_prometheus(&self) -> String {
        let mut output = String::new();
        
        let op_types = [
            OperationType::Insert,
            OperationType::Get,
            OperationType::Remove,
            OperationType::Update,
            OperationType::Kick,
            OperationType::Migration,
            OperationType::Resize,
            OperationType::Read,
            OperationType::Write,
            OperationType::Statistics,
            OperationType::Iterate,
        ];
        
        for op in op_types {
            let count = match op {
                OperationType::Insert => self.insert_count.load(Ordering::Relaxed),
                OperationType::Get => self.get_count.load(Ordering::Relaxed),
                OperationType::Remove => self.remove_count.load(Ordering::Relaxed),
                OperationType::Update => self.update_count.load(Ordering::Relaxed),
                OperationType::Kick => self.kick_count.load(Ordering::Relaxed),
                OperationType::Migration => self.migration_count.load(Ordering::Relaxed),
                OperationType::Resize => self.resize_count.load(Ordering::Relaxed),
                OperationType::Read => self.read_count.load(Ordering::Relaxed),
                OperationType::Write => self.write_count.load(Ordering::Relaxed),
                OperationType::Statistics => self.statistics_count.load(Ordering::Relaxed),
                OperationType::Iterate => self.iterate_count.load(Ordering::Relaxed),
            };
            
            output.push_str(&format!(
                "# HELP cuckoo_operation_{}_count Total {} operations\n",
                op.as_str(), op.as_str()
            ));
            output.push_str(&format!(
                "# TYPE cuckoo_operation_{}_count counter\n",
                op.as_str()
            ));
            output.push_str(&format!(
                "cuckoo_operation_{}_count {}\n",
                op.as_str(), count
            ));
        }
        
        // 添加总持续时间和碰撞计数
        output.push_str("# HELP cuckoo_operation_total_duration Total operation duration (ns)\n");
        output.push_str("# TYPE cuckoo_operation_total_duration counter\n");
        output.push_str(&format!(
            "cuckoo_operation_total_duration {}\n",
            self.total_duration.load(Ordering::Relaxed)
        ));
        
        output.push_str("# HELP cuckoo_operation_collision_count Total collision count\n");
        output.push_str("# TYPE cuckoo_operation_collision_count counter\n");
        output.push_str(&format!(
            "cuckoo_operation_collision_count {}\n",
            self.collision_count.load(Ordering::Relaxed)
        ));
        
        output
    }
    
    fn start_timer(&self, op_type: OperationType) {
        self.start_timer(op_type);
    }
    
    fn get_duration(&self, op_type: OperationType) -> Option<Duration> {
        self.get_duration(op_type)
    }
}

/// 禁用操作统计实现
#[derive(Default)]
pub struct DisabledOperationRecorder;

impl OperationRecorder for DisabledOperationRecorder {
    fn record(&self, _op_type: OperationType, _duration: Duration, _success: bool) {}
    fn snapshot(&self) -> OperationStatsSnapshot { OperationStatsSnapshot::default() }
    fn reset(&self) {}
    fn export_prometheus(&self) -> String { String::new() }
    fn start_timer(&self, _op_type: OperationType) {}
    fn get_duration(&self, _op_type: OperationType) -> Option<Duration> { None }
    
    fn record_count(&self, op_type: OperationType) {
       
    }
}