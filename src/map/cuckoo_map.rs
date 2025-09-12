//! Cuckoo哈希表核心实现

use crate::{
    error::CuckooError, 
    hash::HashStrategy,
    map::bucket::{Bucket, BUCKET_SIZE}, 
    memory::{DefaultMemoryAllocator, SlotHandle}, 
    simd::{SimdSearcher, SimdStrategy}, 
    stats::recorder::{GlobalStatsRecorder, StatsRecorder}, 
    types::{Fingerprint, Key, OperationType, QueueType, SlotStateFlags, Value, GLOBAL_THREAD_POOL}, 
    version::{VersionGuard, VersionTracker}, MemoryAllocator
};
use std::{
    collections::VecDeque, fmt, sync::{
        atomic::{AtomicBool, AtomicUsize, Ordering}, mpsc, Arc, Mutex, RwLock
    }, thread, time::{Duration, Instant}
};
use rand;
use rayon::{prelude::*, ThreadPoolBuilder};
use twox_hash::xxh3::hash64;

/// 哈希表配置
#[derive(Clone, Debug)]
pub struct CuckooMapConfig {
    //初始桶数量（桶的数量）
    pub initial_capacity: usize,
    pub max_load_factor: f32,
    pub max_kick_depth: usize,
    pub slot_count_per_bucket: usize,
    pub migration_batch_size: usize,
    pub migration_parallelism: usize,
    pub migration_lock_timeout_ms: u64,
}

impl Default for CuckooMapConfig {
    fn default() -> Self {
        Self {
            initial_capacity: 1024,
            max_load_factor: 0.95,
            max_kick_depth: 32,
            slot_count_per_bucket: 4,
            migration_batch_size: 64,
            migration_parallelism: 4,
            migration_lock_timeout_ms: 100,
        }
    }
}

/// 哈希表统计信息
#[derive(Debug, Default)]
pub struct CuckooMapStats {
    pub size: usize,
    pub capacity: usize,
    pub load_factor: f32,
    pub insert_count: u64,
    pub get_count: u64,
    pub remove_count: u64,
    pub kick_count: u64,
    pub migration_count: u64,
    pub collision_rate: f32,
    pub migration_progress: f32,
}

/// 迁移计划
#[derive(Debug, Clone)]
pub struct MigrationPlan {
    pub expand_factor: Option<f32>,
    pub shrink_factor: Option<f32>,
    pub target_capacity: Option<usize>,
}

impl MigrationPlan {
    /// 创建扩容计划
    pub fn expand(factor: f32) -> Self {
        Self {
            expand_factor: Some(factor),
            shrink_factor: None,
            target_capacity: None,
        }
    }
    
    /// 创建缩容计划
    pub fn shrink(factor: f32) -> Self {
        Self {
            expand_factor: None,
            shrink_factor: Some(factor),
            target_capacity: None,
        }
    }
    
    /// 创建指定容量计划
    pub fn to_capacity(capacity: usize) -> Self {
        Self {
            expand_factor: None,
            shrink_factor: None,
            target_capacity: Some(capacity),
        }
    }
}

/// 迁移状态
#[derive(Clone)]
enum MigrationState<K: Key, V:Value> {
    NotStarted,
    Migrating(Arc<RwLock<Vec<Arc<Bucket<K, V>>>>>), // 新桶数组
    Completed(Arc<RwLock<Vec<Arc<Bucket<K, V>>>>>), // 迁移完成但未切换
    Switched, // 迁移完成且已切换
}
impl<K: Key, V: Value> MigrationState<K, V> {
    fn should_use_new_buckets(&self) -> bool {
        matches!(self, Self::Switched)
    }
    pub fn can_transition_to(&self, new_state: &Self) -> bool {
        match (self, new_state) {
            (Self::NotStarted, Self::Migrating(_)) => true,
            (Self::Migrating(_), Self::Completed(_)) => true,
            (Self::Completed(_), Self::Switched) => true,
            (Self::Switched, Self::NotStarted) => true,
            _ => false,
        }
    }
    
    pub fn get_bucket_array(&self) -> Option<&Arc<RwLock<Vec<Arc<Bucket<K, V>>>>>> {
        match self {
            Self::Migrating(buckets) | 
            Self::Completed(buckets) => Some(buckets), 
            Self::Switched | Self::NotStarted => None,
        }
    }
    
    pub fn is_migrating(&self) -> bool {
        matches!(self, Self::Migrating(_))
    }
    
    pub fn is_completed(&self) -> bool {
        matches!(self, Self::Completed(_))
    }
    
    pub fn is_switched(&self) -> bool {
        matches!(self, Self::Switched)
    }
    
    pub fn is_active(&self) -> bool {
        !matches!(self, Self::NotStarted)
    }
}
// 为 MigrationState 实现 Debug
impl<K: Key + Clone + Send + Sync, V: Value + Clone + Send + Sync> fmt::Debug for MigrationState<K,V> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            MigrationState::NotStarted => write!(f, "NotStarted"),
            MigrationState::Migrating(new_buckets) => {
                if let Ok(buckets) = new_buckets.read() {
                    write!(f, "Migrating(bucket_count={})", buckets.len())
                } else {
                    write!(f, "Migrating(locked)")
                }
            }
            MigrationState::Completed(new_buckets) => {
                if let Ok(buckets) = new_buckets.read() {
                    write!(f, "Completed(bucket_count={})", buckets.len())
                } else {
                    write!(f, "Completed(locked)")
                }
            },
            MigrationState::Switched => write!(f, "Switched"),
        }
    }

    
}



/// Cuckoo哈希表
pub struct CuckooMap<K: Key + Clone + Send + Sync, V: Value + Clone + Send + Sync> {
    // 当前活跃表
    current: Arc<RwLock<Vec<Arc<Bucket<K, V>>>>>,
    
    // 迁移状态
    migration_state: Arc<RwLock<MigrationState<K,V>>>,
    
    // 配置
    config: Arc<CuckooMapConfig>,
    
    // 哈希策略
    hasher: Arc<RwLock<Box<dyn HashStrategy + Send + Sync>>>,
    
    // 内存池
    memory_pool: Arc<dyn MemoryAllocator>,
    
    // SIMD 搜索器
    simd_searcher: Arc<dyn SimdSearcher + Send + Sync>,
    
    // 统计记录器
    stats_recorder: Arc<dyn StatsRecorder>,
    
    // 版本追踪器
    version: Arc<VersionTracker>,
    
    // 当前大小
    size: Arc<AtomicUsize>,
    
    // 迁移进度
    migration_progress: Arc<AtomicUsize>,
    // 迁移完成标志
    migration_completed: Arc<AtomicBool>,
     /// 迁移过程中的新数据写入失败重放队列
    insert_queue: Arc<Mutex<VecDeque<(K, V)>>>,
    //迁移失败重放
    migrate_queue: Arc<Mutex<VecDeque<(K, V)>>>,
    /// 队列处理标志
    queue_processing: Arc<AtomicBool>,
}


impl<K: Key+ Clone+ Send + Sync, V: Value+Clone+Send+Sync> CuckooMap<K, V> {
   

    fn start_migration(&self, plan: MigrationPlan) -> Result<(), CuckooError> {
        let thread_id = thread::current().id();
        
        // 检查当前状态
        {
            let migration_guard = self.migration_state.read().unwrap();
            if !matches!(&*migration_guard, MigrationState::NotStarted) {
                log_error!("worker {:?} Cannot start migration: invalid state {:?}", 
                        thread_id, *migration_guard);
                return Err(CuckooError::MigrationFailure);
            }
        }
        
        // 计算新容量
        let new_bucket_count = if let Some(factor) = plan.expand_factor {
            (self.current.read().unwrap().len() as f32 * factor) as usize
        } else if let Some(factor) = plan.shrink_factor {
            (self.current.read().unwrap().len() as f32 * factor) as usize
        } else if let Some(capacity) = plan.target_capacity {
            capacity
        } else {
            return Err(CuckooError::MigrationNotNeeded);
        };
        
      
        // 创建新桶数组
    
        let new_buckets_arc = Arc::new(RwLock::new(
            (0..new_bucket_count)
                .map(|_| Arc::new(Bucket::new(self.version.clone())))
                .collect()
        ));
        // 更新迁移状态
        {
            let mut migration_lock = self.migration_state.write().unwrap();
            
            // 验证状态转换合法性
            if !matches!(&*migration_lock, MigrationState::NotStarted) {
                log_error!("worker {:?} Illegal state transition during migration start", thread_id);
                return Err(CuckooError::MigrationFailure);
            }
            
            // 更新状态
            *migration_lock = MigrationState::Migrating(new_buckets_arc.clone());
            
            // 添加内存屏障
            std::sync::atomic::fence(Ordering::SeqCst);
            
            log_info!("worker {:?} Migration started with new capacity {}", thread_id, new_bucket_count);
        }
        
        // 设置迁移进度
        self.migration_progress.store(0, Ordering::Release);
        self.migration_completed.store(false, Ordering::Release);
        self.queue_processing.store(true, Ordering::Release);
        // 启动后台迁移线程
        let bucket_count = {
            let current = self.current.read().unwrap();
            current.len()
        };
          log_info!("worker {:?} Migrating from {} to {} buckets", thread_id,bucket_count, new_bucket_count);
        self.start_migration_thread(new_buckets_arc, bucket_count,new_bucket_count);
        
        
        Ok(())
    }

    fn complete_data_migration(&self) -> Result<(), CuckooError> {
        let thread_id = thread::current().id();
        
        // 获取当前迁移状态
        let buckets = {
            let migration_guard = self.migration_state.read().unwrap();
            match &*migration_guard {
                MigrationState::Migrating(buckets) => buckets.clone(),
                _ => {
                    log_error!("worker {:?} Cannot complete data migration: invalid state", thread_id);
                    return Err(CuckooError::MigrationStateConflict);
                }
            }
        };
        
        // 验证所有数据已迁移
       
        let bucket_count_old = {
            let current = self.current.read().unwrap();
            current.len()
        };
        
        if self.migration_progress.load(Ordering::Acquire) < bucket_count_old {
            log_warn!("worker {:?} Migration incomplete: {}/{} buckets migrated", 
                    thread_id, 
                    self.migration_progress.load(Ordering::Acquire),
                    bucket_count_old);
            return Err(CuckooError::MigrationIncomplete);
        }
        
        // 更新迁移状态
        {
            let mut migration_lock = self.migration_state.write().unwrap();
            
            // 验证状态转换合法性
            if !matches!(&*migration_lock, MigrationState::Migrating(_)) {
                log_error!("worker {:?} Illegal state transition during migration completion", thread_id);
                return Err(CuckooError::MigrationStateConflict);
            }
            
            // 更新状态
            *migration_lock = MigrationState::Completed(buckets.clone());
            
            // 添加内存屏障
            std::sync::atomic::fence(Ordering::SeqCst);
            
            log_info!("worker {:?} Data migration completed", thread_id);
        }
        
        Ok(())
    }

    fn complete_migration(&self) -> Result<(), CuckooError> {
        let thread_id = thread::current().id();
        log_info!("worker {:?} complete_migration start ", thread_id);
                
        // 获取新桶数组的长度
        let new_bucket_count = {
            let current_guard = self.current.read().unwrap();
            current_guard.len()
        };
        log_info!("worker {:?} Hasher capacity start to updated  {}", thread_id, new_bucket_count);
         self.log_queue_status();
        // 更新哈希策略
       {
            let start = Instant::now();
            let timeout = Duration::from_millis(100); // 100ms 超时
            
            while start.elapsed() < timeout {
                if let Ok(mut hasher_lock) = self.hasher.try_write() {
                    hasher_lock.update_capacity(new_bucket_count);
                   // log_info!("worker {:?} Hasher capacity updated to {}", thread_id, new_bucket_count);
                    break;
                }
                thread::yield_now(); // 让出 CPU
            }
            
            if start.elapsed() >= timeout {
                log_error!("worker {:?} Failed to update hasher capacity within timeout", thread_id);
                return Err(CuckooError::LockTimeout);
            }
        }
        log_info!("worker {:?} Hasher capacity updated to {}", thread_id, new_bucket_count);
        // 更新版本号
        self.version.increment();
        log_info!("worker {:?} Version incremented after migration", thread_id);
         // 设置迁移完成标志
        self.migration_completed.store(true, Ordering::Release);
        //切换到初始状态
         {
            let mut migration_lock = self.migration_state.write().unwrap();
            
            // 验证状态转换合法性
            if !matches!(&*migration_lock, MigrationState::Switched) {
                log_error!("worker {:?} Illegal state transition: expected Switched, found {:?}", thread_id, *migration_lock);
                return Err(CuckooError::MigrationFailure);
            }
            
            // 更新状态
            *migration_lock = MigrationState::NotStarted;
            
            // 添加内存屏障确保状态更新可见
            std::sync::atomic::fence(Ordering::SeqCst);
                       
        }
         log_info!("worker {:?} Migration state set to NotStarted", thread_id);
               
        // 处理插入队列
        log_info!("Before queue processing:");
        self.log_queue_status();
        
        log_info!("worker {:?} Triggering queue processing", thread_id);
        self.replay_queue();
        
        //log_info!("After queue processing:");
        //self.log_queue_status();
        
        Ok(())
    }

     fn switch_to_new_buckets(&self) -> Result<(), CuckooError> {
        let thread_id = thread::current().id();
        
        // 1. 获取迁移状态
        let buckets = {
            let migration_guard = self.migration_state.read().unwrap();
            match &*migration_guard {
                MigrationState::Completed(buckets) => buckets.clone(),
                MigrationState::Migrating(_) => {
                    log_warn!("worker {:?} Cannot switch buckets: migration still in progress", thread_id);
                    return Err(CuckooError::MigrationIncomplete);
                }
                MigrationState::Switched => {
                    log_info!("worker {:?} Buckets already switched", thread_id);
                    return Ok(());
                }
                MigrationState::NotStarted => {
                    log_error!("worker {:?} Cannot switch buckets: migration not started", thread_id);
                    return Err(CuckooError::MigrationNotNeeded);
                }
            }
        };
        
        // 2. 原子切换桶数组
        {
            // 获取新桶数组的读锁
            let new_buckets_inner = buckets.read().unwrap();
             log_info!("worker {:?} prepare switched to new bucket array", thread_id);
            // 获取当前桶数组的写锁
            let mut current_lock = self.current.write().unwrap();
            
            // 执行切换
            *current_lock = new_buckets_inner.clone();
            
            log_info!("worker {:?} Successfully switched to new bucket array", thread_id);
        }
        
        // 3. 更新迁移状态
        {
            let mut migration_lock = self.migration_state.write().unwrap();
            
            // 验证状态转换合法性
            if !matches!(&*migration_lock, MigrationState::Completed(_)) {
                log_error!("worker {:?} Illegal state transition: expected Completed, found {:?}", thread_id, *migration_lock);
                return Err(CuckooError::MigrationFailure);
            }
            
            // 更新状态
            *migration_lock = MigrationState::Switched;
            
            // 添加内存屏障确保状态更新可见
            std::sync::atomic::fence(Ordering::SeqCst);
            
            log_info!("worker {:?} Migration state set to Switched", thread_id);
        }
        
        Ok(())
    }
}

impl<K: Key + Clone + Send + Sync, V: Value + Clone + Send + Sync>  CuckooMap<K, V> {

    fn replay_queue(&self){
        println!("Starting parallel queue processing");
        let start_time = std::time::Instant::now();
        // 创建通道用于收集结果
        let (tx, rx) = mpsc::channel();
         // 并行处理迭代器
        GLOBAL_THREAD_POOL.scope(|s| {
            for &item in QueueType::all() {
                 let task_tx = tx.clone();
                
                 s.spawn(move |_| {
                     // 记录任务开始
                    let task_start = Instant::now();
                    log_debug!("Processing {} started", item.name());
                     let result = self.process_queue(item);
                // 记录任务完成
                    let duration = task_start.elapsed();
                    log_debug!("Processing {} completed in {:?}", item.name(), duration);

                    // 发送结果（包括队列类型和处理结果）
                    task_tx.send((item, result, duration)).unwrap();
            });
            }
        });

        // 关闭发送端（确保接收端知道没有更多数据）
        drop(tx);
        
        // 收集所有结果
        let mut results = Vec::new();
        for (queue_type, result, task_duration) in rx.iter() {
            results.push((queue_type, result, task_duration));
        }
         self.queue_processing.store(false, Ordering::Release);
        // 分析结果
        let total_duration = start_time.elapsed();
     //   let success_count = results.iter().filter(|(_, r, _)| r.is_ok()).count();
     //   let failure_count = results.len() - success_count;
        
        println!("All queues processed in {:?}", total_duration);
       // println!("Success: {}, Failure: {}", success_count, failure_count);
        
        // 打印详细结果
        
    }
    /// 通用队列处理函数
    fn process_queue(
        &self,
        queue_type: QueueType,
    ) {
        log_info!("Processing {} start", queue_type.name());
        
        const MAX_BATCH_SIZE: usize = 400;
        const MAX_RETRIES: usize = 5;
        const RETRY_DELAY_BASE: u64 = 50;
        
        let mut iteration = 0;
        let thread_id = thread::current().id();
        
        // 获取队列引用
        let queue = match queue_type {
            QueueType::InsertQueue => &self.insert_queue,
            QueueType::MigrateQueue => &self.migrate_queue,
        };
        
        // 处理队列直到为空或条件不满足
        while self.queue_size(queue) > 0  && self.is_migration_fully_completed(){
            iteration += 1;
            log_info!("worker {:?} Processing {} iteration {}",  thread_id, queue_type.name(), iteration);
            
            // 获取批处理数据
            let batch = self.get_next_batch(queue, MAX_BATCH_SIZE);
            
            if batch.is_empty() {
                log_info!("No items to process in {}", queue_type.name());
                break;
            }
            
            log_info!("Processing batch of {} items from {}", 
                     batch.len(), queue_type.name());
            
            // 处理批处理
            self.process_batch(
                queue, 
                batch, 
                MAX_RETRIES, 
                RETRY_DELAY_BASE,
                queue_type==QueueType::MigrateQueue
            );
            
            // 短暂休眠，让出CPU
            thread::sleep(Duration::from_millis(10));
        }
        
        log_info!("{} processing completed after {} iterations, size={}",
                 queue_type.name(), iteration, self.queue_size(queue));
    }
    
    /// 获取队列大小
    fn queue_size(&self, queue: &Arc<Mutex<VecDeque<(K, V)>>>) -> usize {
        queue.lock().unwrap().len()
    }
    
     /// 获取队列大小
    pub fn migrate_queue_size(&self) -> usize {
        self.migrate_queue.lock().unwrap().len()
    }
    pub fn insert_queue_size(&self) -> usize {
        self.insert_queue.lock().unwrap().len()
    }
    /// 检查队列是否正在处理
    pub fn is_queue_processing(&self) -> bool {
        self.queue_processing.load(Ordering::Acquire)
    }
    
    /// 记录队列状态
    pub fn log_queue_status(&self) {
        log_info!("Insert queue status: migrate_queue_size={},insert_queue_size={}, processing={},migration status={}",   self.migrate_queue_size(), self.insert_queue_size(),self.is_queue_processing(),self.is_migration_fully_completed());
    }

    /// 获取下一批处理数据
    fn get_next_batch(
        &self,
        queue: &Arc<Mutex<VecDeque<(K, V)>>>,
        max_batch_size: usize,
    ) -> Vec<(K, V)> {
        let mut queue_lock = queue.lock().unwrap();
        
        if queue_lock.is_empty() {
            return Vec::new();
        }
        
        // 每次最多处理 max_batch_size 个元素
        let batch_size = max_batch_size.min(queue_lock.len());
        queue_lock.drain(..batch_size).collect()
    }
    
    /// 处理批处理数据
    fn process_batch(
        &self,
        queue: &Arc<Mutex<VecDeque<(K, V)>>>,
        batch: Vec<(K, V)>,
        max_retries: usize,
        retry_delay_base: u64,
        migrate:bool
    ) {
        let mut failed_items = Vec::new();
        
        for (key, value) in batch {
            // 尝试插入
            let result = self.try_insert_with_retry(
              key.clone(),
              value.clone(),
                max_retries,
               retry_delay_base,
                migrate
            );
            
            if result.is_err() {
                failed_items.push((key, value));
            }
        }
        
        // 将失败项放回队列
        if !failed_items.is_empty() {
            log_info!("Returning {} failed items to {}", 
                     failed_items.len(), 
                     match queue {
                         _ if Arc::ptr_eq(queue, &self.insert_queue) => "insert_queue",
                         _ => "migrate_queue",
                     });
            
            queue.lock().unwrap().extend(failed_items);
        }
    }
    
    
}


impl<K: Key + Clone + Send + Sync, V: Value + Clone + Send + Sync> CuckooMap<K, V> {
    
    /// 获取桶的数量
    pub fn bucket_count(&self) -> usize {
        self.current.read().unwrap().len()
    }
    
    /// 获取总槽位数（容量）
    pub fn slot_capacity(&self) -> usize {
        self.bucket_count() * self.config.slot_count_per_bucket
    }
    /// 导出Prometheus格式指标
    pub fn export_prometheus(&self) -> String {
        self.stats_recorder.export_prometheus()
    }
    
    /// 创建新哈希表
    pub fn new(
        config: CuckooMapConfig,
        hasher:  Box<dyn HashStrategy + Send + Sync>,
        memory_pool: Arc<dyn MemoryAllocator>,
        simd_searcher: Arc<dyn SimdSearcher>,
        stats_recorder: Arc<dyn StatsRecorder>,
        version: Arc<VersionTracker>,
    ) -> Self {
        let buckets = (0..config.initial_capacity)
            .map(|_| Arc::new(Bucket::new(version.clone())))
            .collect();
        
        Self {
            current: Arc::new(RwLock::new(buckets)),
            migration_state: Arc::new(RwLock::new(MigrationState::NotStarted)),
            config: Arc::new(config),
            hasher:Arc::new(RwLock::new(hasher)),
            memory_pool,
            simd_searcher,
            stats_recorder,
            version,
            size:Arc::new(AtomicUsize::new(0)) ,
            migration_progress: Arc::new(AtomicUsize::new(0)),
            // 迁移完成标志
            migration_completed: AtomicBool::new(true).into(),
            migrate_queue: Arc::new(Mutex::new(VecDeque::new())),
            insert_queue:Arc::new(Mutex::new(VecDeque::new())),
            queue_processing: Arc::new(AtomicBool::new(false)),
        }
    }
    
    /// 克隆哈希表
    pub fn clone(&self) -> Self {
              
        Self {
            current: Arc::clone(&self.current),
            migration_state: Arc::clone(&self.migration_state),
            config: Arc::clone(&self.config),
            hasher: Arc::clone(&self.hasher),
            memory_pool: Arc::clone(&self.memory_pool),
            simd_searcher: Arc::clone(&self.simd_searcher),
            stats_recorder: Arc::clone(&self.stats_recorder),
            version: Arc::clone(&self.version),
            size: Arc::clone(&self.size),   
            migration_progress:  Arc::clone(&self.migration_progress),
            migration_completed:Arc::clone(&self.migration_completed),
            migrate_queue:  Arc::clone(&self.migrate_queue),  
            insert_queue:Arc::clone(&self.insert_queue),
            queue_processing: Arc::clone(&self.queue_processing),
        }
    }

    /// 插入键值对
    pub fn insert(&self, key: K, value: V,) -> Result<(), CuckooError> {
        let thread_id = thread::current().id();
       //  log_info!("worker {:?} insert tag {} ,key={}", thread_id,tag,key);
        
        const MAX_RETRIES: usize = 10;
        let mut retries = 0;
        
        // 创建key和value的克隆，用于重试
        let  current_key = key;
        let  current_value = value;
        
        loop {
            let result = {
                let _timer = self.stats_recorder.start_timer(OperationType::Insert);
                let version_guard = self.version.begin_operation(OperationType::Insert);
                 // 1. 使用内存屏障确保状态可见性
                std::sync::atomic::fence(Ordering::Acquire);
                // 1. 快速检查原子标志（无锁）
                if self.migration_completed.load(Ordering::Acquire) {
                  //   log_info!("worker {:?} insert when no Migrating 1", thread_id);
                     let result= self._insert(current_key.clone(), current_value.clone(), &version_guard,false);
                    if result.is_ok(){
                        self.stats_recorder.record_operation_count(OperationType::Insert);
                    }
                 //   let tt=self.get(&current_key);
                 //log_info!("worker {:?} insert when no Migrating 2,get ={:?}", thread_id,tt);
                   return  result
                }
                // 获取迁移状态
                let migration_state = {
                    let guard = self.migration_state.read().unwrap();
                    guard.clone()
                };
                
                // 根据迁移状态路由插入操作
                match migration_state {
                    MigrationState::NotStarted | MigrationState::Switched => {
                       // log_info!("worker {:?} insert start", thread_id);
                    let result= self._insert(current_key.clone(), current_value.clone(), &version_guard,false);
                    if result.is_ok(){
                        self.stats_recorder.record_operation_count(OperationType::Insert);
                    }
                    result
                },
                    MigrationState::Migrating(ref new_buckets) => {
                        log_info!("worker {:?} insert_queue start when Migrating", thread_id);
                       
                      if let Err(e) =  self.insert_to_new_table(&current_key, &current_value, new_buckets, new_buckets.read().unwrap().len(),&version_guard,false){
                        log_warn!("Failed to insert key to new table: {:?}, will retry later", e);
                        // 收集请求加入重试队列队列
                        self.insert_queue.lock().unwrap().push_back((current_key.clone(), current_value.clone()));
                        }
                         // log_debug!("worker {:?} insert start when Migrating", thread_id);
                   
                    Ok(())
                     
                    }
                    MigrationState::Completed(_) => {
                        // 尝试完成迁移
                        log_debug!("worker {:?} insert to Completed", thread_id);
                        if let Err(e) = self.complete_migration() {
                            log_error!("Failed to complete migration: {:?}", e);
                            return Err(CuckooError::MigrationInProgress);
                        }
                        // 返回错误让上层重试
                        return Err(CuckooError::MigrationInProgress);
                    }
                   
                }
            };
            
            match result {
                Ok(()) => {
                    // 记录操作计数
                   // self.stats_recorder.record_operation_count(OperationType::Insert);
                    return Ok(());
                }
                Err(CuckooError::MigrationInProgress) if retries < MAX_RETRIES => {
                    retries += 1;
                    log_debug!("worker {:?} insert retry {} when Migrating", thread_id, retries);
                    thread::sleep(Duration::from_millis(100));
                    // 检查迁移是否已完成
                if self.is_migration_fully_completed() {
                    // 尝试完成迁移
                    if let Err(e) = self.complete_migration() {
                        log_error!("Failed to complete migration: {:?}", e);
                    }
                }
                }
                Err(CuckooError::CycleDetected) if retries < MAX_RETRIES => {
                retries += 1;
                log_debug!("Kick cycle detected, retrying insert (attempt {})", retries);
                thread::sleep(Duration::from_millis(100));
    }
                Err(e) => return Err(e),
            }
        }
    }
    
    fn insert_to_new_table(
    &self,
    key: &K,
    value: &V,
    new_buckets: &Arc<RwLock<Vec<Arc<Bucket<K, V>>>>>,
    new_bucket_count:usize,
    version_guard: &VersionGuard,
    migrate:bool
) -> Result<(), CuckooError> {
    const MAX_KICK_DEPTH: usize = 500; // 最大踢出深度
    
    let thread_id = thread::current().id();
    log_debug!("worker {:?} inserting to new table: key={:?}", 
        thread_id, String::from_utf8_lossy(key.as_bytes()));
     // 获取哈希策略
   

     let (mut h1, mut h2,mut fp) ={
                let hasher = self.hasher.read().unwrap();
                let fp1 = hasher.fingerprint(key);
                let (h1, h2) =hasher.locate_buckets_with_capacity(key,new_bucket_count);
                (h1,h2,fp1)
        } ;
    let mut current_key = key.clone();
    let mut current_value = value.clone();
    let mut current_fp = fp;
    let mut depth = 0;

    loop {
        // 检查版本一致性
       // version_guard.check()?;
        
        // 获取新桶列表的读锁
        let buckets = new_buckets.read().unwrap();
        
        // 尝试直接插入第一个桶
        if let Some(bucket) = buckets.get(h1) {
            if let Some(slot_idx) = bucket.find_empty_slot() {
                return self.commit_insert_in_bucket(
                    bucket, slot_idx, &current_key, &current_value, current_fp, version_guard,migrate
                );
            }
        }
        
        // 尝试直接插入第二个桶
        if let Some(bucket) = buckets.get(h2) {
            if let Some(slot_idx) = bucket.find_empty_slot() {
                return self.commit_insert_in_bucket(
                    bucket, slot_idx, &current_key, &current_value, current_fp, version_guard,migrate
                );
            }
        }

        // 达到最大深度时终止
        if depth >= MAX_KICK_DEPTH {
            log_warn!("worker {:?} cuckoo kick in new table reached max depth {}", 
                thread_id, MAX_KICK_DEPTH);
            return Err(CuckooError::CycleDetected);
        }

        // 随机选择一个桶进行踢出
        let target_bucket_idx = if rand::random() { h1 } else { h2 };
        let target_bucket = match buckets.get(target_bucket_idx) {
            Some(b) => b.clone(),
            None => {
                log_error!("worker {:?} invalid bucket index: {}", thread_id, target_bucket_idx);
                return Err(CuckooError::BucketNotFound);
            }
        };

        // 随机选择槽位
        let slot_idx = rand::random::<usize>() % BUCKET_SIZE;
       
        // 获取目标槽位
        let slot = &target_bucket.slots()[slot_idx];
        
        // 尝试锁定槽位
        let current_fp_in_slot = slot.load_fingerprint();
        let guard = match slot.try_lock(current_fp_in_slot, version_guard) {
            Ok(guard) => guard,
            Err(CuckooError::VersionConflict) | Err(CuckooError::LockContention{..}) => {
                // 版本冲突或锁争用，重试当前踢出步骤
                continue;
            }
            Err(e) => return Err(e),
        };
        
        // 再次检查版本一致性
      //  version_guard.check()?;
        
        // 如果槽位为空，则直接使用
        if !guard.is_occupied() {
            // 写入当前元素
            slot.store(current_key.as_bytes(), current_value.as_bytes());
            guard.commit(current_fp);
            return Ok(());
        }
        
        // 保存被踢出的元素
        let old_key_bytes = guard.load_key().ok_or(CuckooError::ValueNotFound)?;
        let old_value_bytes = guard.load_value().ok_or(CuckooError::ValueNotFound)?;
        let old_key = K::from_bytes(&old_key_bytes).ok_or(CuckooError::KeyDeserialization)?;
        let old_value = V::from_bytes(&old_value_bytes).ok_or(CuckooError::KeyDeserialization)?;
        let old_fp = current_fp_in_slot;
        
        // 写入当前元素到槽位
        slot.store(current_key.as_bytes(), current_value.as_bytes());
        // 提交更改，使用当前元素的指纹
        guard.commit(current_fp);
        
        // 准备被踢出元素用于下一轮插入
        current_key = old_key;
        current_value = old_value;
        current_fp = old_fp;
        
        // 计算被踢出元素的新位置
         let hasher = self.hasher.read().unwrap();
       let (new_h1, new_h2) = {
            hasher.locate_buckets_with_capacity(&current_key,new_bucket_count)
       };
       h1=new_h1;
       h2=new_h2;
        depth += 1;

        log_debug!("worker {:?} kicked element at [{}, {}], depth={}, new key={:?}",
            thread_id, target_bucket_idx, slot_idx, depth, 
            String::from_utf8_lossy(current_key.as_bytes()));
    }
}


/// 在指定桶中提交插入
fn commit_insert_in_bucket(
    &self,
    bucket: &Bucket<K, V>,
    slot_idx: usize,
    key: &K,
    value: &V,
    fp: Fingerprint,
    version_guard: &VersionGuard,
    migrate:bool
) -> Result<(), CuckooError> {
    const MAX_RETRIES: usize = 3;
    let mut retries = 0;
    
    loop {
        let slot = &bucket.slots()[slot_idx];
        
        // 获取槽位的当前状态和指纹
        let current_fp = slot.load_fingerprint();
        let is_occupied = slot.is_occupied();
        
        // 尝试锁定槽位
        match slot.try_lock(current_fp, version_guard) {
            Ok(guard) => {
                // 检查槽位状态是否变化
                if is_occupied {
                    // 如果是占用状态，检查键是否匹配
                    if let Some(existing_key_bytes) = guard.load_key() {
                        // 比较键的字节表示
                        if existing_key_bytes != key.as_bytes() {
                            // 键不匹配，返回错误
                            return Err(CuckooError::SlotOccupied);
                        }
                    } else {
                        // 槽位状态变化，重试
                        if retries < MAX_RETRIES {
                            retries += 1;
                            continue;
                        }
                        return Err(CuckooError::SlotOccupied);
                    }
                }
                
                // 存储数据
                slot.store(key.as_bytes(), value.as_bytes());
                
                // 提交更改
                guard.commit(fp);
                // 如果槽位是空的，增加大小
                    if !is_occupied && !migrate{
                        self.size.fetch_add(1, Ordering::Release);
                    }
                return Ok(());
            }
            Err(CuckooError::VersionConflict) if retries < MAX_RETRIES => {
                // 版本冲突，重试
                retries += 1;
            }
            Err(e) => return Err(e),
        }
    }
}
    
    /// 内部插入实现
    fn _insert(
        &self,
        key: K,
        value: V,
        version_guard: &VersionGuard,
        migrate:bool
    ) -> Result<(), CuckooError> {
        
        let (h1, h2,fp) ={
                let hasher = self.hasher.read().unwrap();
                let fp1 = hasher.fingerprint(&key);
                let (h1, h2) =hasher.locate_buckets(&key);
                (h1,h2,fp1)
        } ;
         let thread_id = thread::current().id();
      //  log_info!("worker {:?} inserting key={:?}, buckets=[{}, {}]", thread_id, String::from_utf8_lossy(key.as_bytes()), h1, h2);

    // 1. 尝试在第一个桶直接插入
    if let Some(slot_idx) = self.try_direct_insert(h1, &key, &value, fp) {
      //  log_info!("worker {:?} found slot in bucket1: {}", thread_id, slot_idx);
        
        match self.commit_insert(h1, key.clone(), value.clone(), slot_idx, version_guard,migrate) {
            Ok(()) => {
               // log_info!("worker {:?} inserted in bucket1", thread_id);
                return Ok(());
            }
            Err(CuckooError::SlotOccupied) => {
                // 槽位已被占用，记录并继续尝试第二个桶
                log_info!("worker {:?} bucket1 slot occupied, trying bucket2", thread_id);
              //  self.stats_recorder.record_collision();
            }
            Err(e) => {
                // 其他错误直接返回
              //  log_warn!("worker {:?} bucket1 insert failed: {:?}", thread_id, e);
                return Err(e);
            }
        }
    } else {
      //  log_info!("worker {:?} no available slot in bucket1", thread_id);
    }

    // 2. 尝试在第二个桶直接插入
    if let Some(slot_idx) = self.try_direct_insert(h2, &key, &value, fp) {
       // log_info!("worker {:?} found slot in bucket2: {}", thread_id, slot_idx);
        
        match self.commit_insert(h2, key.clone(), value.clone(), slot_idx, version_guard,migrate) {
            Ok(()) => {
               // log_info!("worker {:?} inserted in bucket2", thread_id);
                return Ok(());
            }
            Err(CuckooError::SlotOccupied) => {
                // 槽位已被占用，记录并进入踢出逻辑
                log_info!("worker {:?} bucket2 slot occupied, starting kick", thread_id);
             //   self.stats_recorder.record_collision();
            }
            Err(e) => {
                // 其他错误直接返回
                log_warn!("worker {:?} bucket2 insert failed: {:?}", thread_id, e);
                return Err(e);
            }
        }
    } else {
       // log_info!("worker {:?} no available slot in bucket2", thread_id);
    }

    // 3. 两个桶都无法直接插入，进入踢出逻辑
    //log_info!("worker {:?} starting cuckoo kick:no available slot", thread_id);
        
        self.cuckoo_kick(h1, h2, key, value, fp, version_guard)
    }

    /// 尝试直接插入
    fn try_direct_insert(
        &self,
        bucket_idx: usize,
        key: &K,
        value: &V,
        fp: Fingerprint,
    ) -> Option<usize> {
        let buckets ={
                self.current.read().unwrap()
        } ;
        let bucket = buckets.get(bucket_idx)?;
        
        // 使用适当的版本守卫
        let version_guard = self.version.begin_operation(OperationType::Insert);
         let thread_id = thread::current().id();
        
        
        // 检查桶中是否有相同键
        if let Some(idx) = bucket.find_slot_with_key(key, &version_guard) {
            // 键已存在，返回槽位索引进行更新
            log_info!("Worker {:?} found existing key at bucket={}, slot={}", thread_id, bucket_idx, idx);
            return Some(idx);
        }
       // log_warn!("Worker {:?} try_direct_insert not find_slot_with_key in bucket={}, bucket_load {:.2} ", thread_id,bucket_idx,self.bucket_load(bucket_idx));
        
    // 检查空槽位
        // 1. 首先尝试查找空槽位（效率更高）
    if let Some(slot_idx) = bucket.find_empty_slot() {
      //  log_info!("Worker {:?} try_direct_insert found empty slot at bucket={}, slot={},bucket count={},load factor={}", thread_id, bucket_idx, slot_idx,self.bucket_count(),self.load_factor());
        return Some(slot_idx);
    }
   // log_info!("Worker {:?} no empty slot in bucket={}, searching for key, bucket_load {:.2} ,bucket info  ", thread_id, bucket_idx,self.bucket_load(bucket_idx));
  //  bucket.print_summary();
    None
        
    }
    /// 计算桶的负载因子
fn bucket_load(&self, bucket_idx: usize) -> f32 {
     let buckets ={
                self.current.read().unwrap()
        } ;
    if let Some(bucket) = buckets.get(bucket_idx) {
        let occupied = bucket.slots().iter().filter(|s| !s.is_empty()).count();
        return occupied as f32 / BUCKET_SIZE as f32;
    }
    0.0
}
/// 查找部分匹配的槽位（基于指纹）
    pub fn find_partial_match_slot(
        &self,
        bucket: &Bucket<K, V>, 
        fp: Fingerprint,
        version_guard: &VersionGuard,
    ) -> Option<usize> {
        // 使用 SIMD 加速搜索匹配指纹的槽位
        let slot_indices = bucket.find_slots(fp, self.simd_searcher.as_ref());
        
        for slot_idx in slot_indices {
            let slot = bucket.get_slot(slot_idx);
            
            // 检查槽位是否被占用
            if slot.is_empty() {
                continue;
            }
            
            // 尝试锁定槽位
            let current_fp = slot.load_fingerprint();
            if slot.try_lock(current_fp, version_guard).is_ok() {
                return Some(slot_idx);
            }
        }
        
        None
    }
    /// 提交插入操作
    fn commit_insert(
        &self,
        bucket_idx: usize,
        key: K,
        value: V,
        slot_idx: usize,
        version_guard: &VersionGuard,
        migrate:bool
    ) -> Result<(), CuckooError> {
        const MAX_RETRIES: usize = 3;
        let mut retries = 0;
        let thread_id = thread::current().id();
        
        loop {
            
            let bucket = {
                let buckets = self.current.read().unwrap();
                buckets.get(bucket_idx).ok_or(CuckooError::InvalidBucket)?.clone()
            };
            let slot = &bucket.slots()[slot_idx];
            
            // 获取槽位的当前状态和指纹
            let current_fp = slot.load_fingerprint();
            let is_occupied = slot.is_occupied();
            
            // 尝试锁定槽位
            match slot.try_lock(current_fp, version_guard) {
                Ok(guard) => {
                    // 检查槽位状态是否变化
                    if is_occupied {
                        // 如果是占用状态，检查键是否匹配
                        if let Some(existing_key_bytes) = guard.load_key() {
                            // 比较键的字节表示
                            if existing_key_bytes != key.as_bytes() {
                                // 键不匹配，返回错误让上层处理
                                return Err(CuckooError::SlotOccupied);
                            }
                        } else {
                            // 槽位状态变化，重试
                            if retries < MAX_RETRIES {
                                retries += 1;
                                continue;
                            }
                            return Err(CuckooError::SlotOccupied);
                        }
                    }
                    
                    // 存储数据
                    slot.store(key.as_bytes(), value.as_bytes());
                    
                    // 提交更改
                    let fp={
                        let hasher = self.hasher.read().unwrap();
                        hasher.fingerprint(&key)
                    }; 
                    guard.commit(fp);
                    
                    // 如果槽位是空的，增加大小
                    if !is_occupied && !migrate{
                        self.size.fetch_add(1, Ordering::Release);
                    }
                    
                    log_debug!(
                        "worker {:?} commit_insert: bucket={}, slot={}, key={:?}, occupied={}, size={}", 
                       thread_id, bucket_idx, slot_idx, String::from_utf8_lossy(key.as_bytes()), 
                        is_occupied, self.size.load(Ordering::Acquire)
                    );
                    
                    return Ok(());
                }
                Err(CuckooError::VersionConflict) if retries < MAX_RETRIES => {
                    // 版本冲突，重试
                    retries += 1;
                }
                Err(e) => return Err(e),
            }
        }
    }

    /// Cuckoo踢出操作
        fn cuckoo_kick(
        &self,
        h1: usize,
        h2: usize,
        key: K,
        value: V,
        fp: Fingerprint,
        version_guard: &VersionGuard,
    ) -> Result<(), CuckooError> {
        let thread_id = thread::current().id();
        log_debug!("worker {:?} cuckoo_kick start: key={:?}, h1={}, h2={}", thread_id,
            String::from_utf8_lossy(key.as_bytes()), h1, h2);

        let mut path = VecDeque::with_capacity(self.config.max_kick_depth);
        let mut current_key = key;
        let mut current_value = value;
        let mut current_fp = fp;
        let mut current_bucket_idx =  if rand::random() { h1 } else { h2 };
        let mut cycle_retry_count = 0;
        const MAX_CYCLE_RETRIES: usize = 3;

        for depth in 0..self.config.max_kick_depth {
            // 1. 找到当前桶中可踢出的槽位
            let slot_idx = self.find_kick_candidate(current_bucket_idx, current_fp)?;
            let key_hash =  hash64(current_key.as_bytes());
            path.push_back((current_bucket_idx, slot_idx, key_hash));
         //   log_info!("[{:?}] kicking for key={:?},from  :current_bucket_idx={},slot_idx={},key_hash={}", thread_id, current_key,current_bucket_idx,slot_idx,key_hash);
            // 2. 取出旧数据
        
            let bucket = {
                    let buckets = self.current.read().unwrap();
                    buckets.get(current_bucket_idx).ok_or(CuckooError::InvalidBucket)?.clone()
                };
           // bucket.print_summary();
            let slot = &bucket.slots()[slot_idx];
        
            // 使用槽位当前指纹锁定
            let slot_fp = slot.load_fingerprint();
            let guard = slot.try_lock(slot_fp, version_guard)?;
            
            let old_key_bytes = guard.load_key().ok_or(CuckooError::MigrationFailure)?;
            let old_value_bytes = guard.load_value().ok_or(CuckooError::MigrationFailure)?;
            
            // 3. 写入新数据
            slot.store(
                current_key.as_bytes(), 
                current_value.as_bytes()
            );
         //   log_info!("[{:?}] kicking into   slot: key={:?},current_bucket_idx={}", thread_id, current_key,current_bucket_idx);
            // 提交更改，使用新数据的指纹
            guard.commit(current_fp);
          //  bucket.print_summary();
            // 4. 反序列化旧数据
            current_key = K::from_bytes(&old_key_bytes).ok_or(CuckooError::KeyDeserialization)?;
            current_value = V::from_bytes(&old_value_bytes).ok_or(CuckooError::ValueDeserialization)?;
            current_fp=slot_fp;
       //     log_info!("[{:?}] Kicking outof slot: key={:?} from current_bucket_idx={}", thread_id, current_key,current_bucket_idx);
            
            // 5. 计算旧数据的新位置
            
            let (new_h1, new_h2) ={
                    let hasher = self.hasher.read().unwrap();
                  //  let fp = hasher.fingerprint(&current_key);
                    let (h1, h2) =hasher.locate_buckets(&current_key);
                    (h1,h2)
            } ;
            
            // 6. 选择下一个桶 (非来源桶)
            current_bucket_idx = if new_h1 == current_bucket_idx {
                new_h2
            } else {
                new_h1
            };
          
        //    log_info!("worker {:?} Kicking outof slot: key={:?} to new current_bucket_idx={}", thread_id, current_key.clone(), current_bucket_idx);
            // 7. 检查循环 - 只比较桶索引和指纹
            let current_key_hash = hash64(current_key.as_bytes());
            if path.iter().any(|(bucket_idx, _, path_hash)| {
                *bucket_idx == current_bucket_idx && *path_hash == current_key_hash
            }) {
                log_info!("Worker {:?} Cuckoo kick cycle detected at depth {}", thread_id, depth);
                
                    // 检查是否已经有迁移在进行
                    if self.is_migration_in_progress() {
                        // 如果迁移已经在进行，等待而不是触发新迁移
                        log_error!("worker {:?} is_migration_in_progress", thread_id);
                        return Err(CuckooError::MigrationInProgress);
                    }

                    // 尝试随机踢出打破循环
                    if cycle_retry_count < MAX_CYCLE_RETRIES {
                        cycle_retry_count += 1;
                        log_info!("worker {:?} Attempting to break cycle (attempt {}/{})", 
                            thread_id, cycle_retry_count, MAX_CYCLE_RETRIES);
                        
                        // 随机选择另一个桶进行踢出
                        current_bucket_idx = if rand::random() { new_h1 } else { new_h2 };
                        continue;
                    }
                    // 多次尝试后仍然循环，触发扩容
                    log_info!("Worker {:?} Failed to break cycle after {} attempts, triggering resize,load_factor {} ", 
                        thread_id, MAX_CYCLE_RETRIES,self.load_factor());
                    // 触发扩容

                    let expand_factor = self.calculate_expand_factor();
                    let migrate_result =self.migrate(MigrationPlan::expand(expand_factor));
                // log_warn!("Worker {:?} retry insert   after  migrate ", thread_id);
                    // 处理迁移结果
                    let result = match migrate_result {
                        Ok(_) => {
                            log_info!("Worker {:?} migration completed, success", thread_id);
                                return self.insert(current_key, current_value);
                        }
                        Err(e) => {
                            log_error!("Worker {:?} migration failed: {:?}", thread_id, e);
                        
                        }
                    };
                                    
                }
            //  log_warn!("Worker {:?} start try_direct_insert depth :{}", thread_id, depth);
                // 8. 尝试直接插入
                if let Some(slot_idx) = self.try_direct_insert(current_bucket_idx, &current_key, &current_value, current_fp) {
                    //log_warn!("Worker {:?} success try_direct_insert key={:?} depth :{}", thread_id, current_key.clone(),depth);
                    return self.commit_insert(current_bucket_idx, current_key, current_value, slot_idx, version_guard,false);
                }
               // log_warn!("Worker {:?} failed to  try_direct_insert key={:?}  depth :{}", thread_id,current_key.clone(), depth);
            }
            
            log_warn!("cuckoo_kick max depth reached: key={:?}",  String::from_utf8_lossy(current_key.as_bytes()));
            // 踢出路径超过最大深度，触发扩容
            
            let expand_factor = self.calculate_expand_factor();
            let migrate_result =self.migrate(MigrationPlan::expand(expand_factor));
                log_warn!("Worker {:?} retry insert   after  migrate ", thread_id);
                // 处理迁移结果
                let result = match migrate_result {
                    Ok(_) => {
                        log_warn!("Worker {:?} migration completed, retrying insert", thread_id);
                        return self.insert(current_key, current_value,);
                    }
                    Err(e) => {
                        log_error!("Worker {:?} migration failed: {:?}", thread_id, e);
                    
                    }
                };
            //self.insert_queue.lock().unwrap().push_back((current_key.clone(), current_value.clone()));
            Ok(())
    }

    /// 查找踢出候选槽位
    fn find_kick_candidate(
        &self,
        bucket_idx: usize,
        fp: Fingerprint,
    ) -> Result<usize, CuckooError> {
      
          let bucket = {
                let buckets = self.current.read().unwrap();
                buckets.get(bucket_idx).ok_or(CuckooError::InvalidBucket)?.clone()
            };
        // 优先选择相同指纹的槽位
        for i in 0..BUCKET_SIZE {
            if bucket.slots()[i].load_fingerprint() == fp {
                return Ok(i);
            }
        }
        
        // 随机选择一个非空槽位
                
        let mut candidates = Vec::new();
        for i in 0..BUCKET_SIZE {
            if !bucket.get_slot(i).is_empty() {
                candidates.push(i);
            }
        }

        if candidates.is_empty() {
            return Err(CuckooError::NoKickCandidate);
        }

        let idx = rand::random::<usize>() % candidates.len();
        Ok(candidates[idx])

    }

    /// 获取键值对
    pub fn get(&self, key: &K) -> Option<V> {
        let _timer = self.stats_recorder.start_timer(OperationType::Get);
        let version_guard = self.version.begin_operation(OperationType::Get);
        
        // 根据迁移状态路由查询操作
        let state = self.migration_state.read().unwrap();
        match *state {
            MigrationState::NotStarted => self.search_in_current(key, &version_guard),
            MigrationState::Migrating(ref new_buckets) => {
                // 先查新表
                if let Some(value) = self.search_in_new_table(key, new_buckets, &version_guard) {
                    return Some(value);
                }
                // 再查旧表
                self.search_in_current(key, &version_guard)
            }
            MigrationState::Completed(ref new_buckets) => {
                // 先查新表
                if let Some(value) = self.search_in_new_table(key, new_buckets, &version_guard) {
                    return Some(value);
                }
                // 再查旧表
                self.search_in_current(key, &version_guard)
            }
            MigrationState::Switched => self.search_in_current(key, &version_guard),
        }
    }
    
    /// 在当前表中搜索
    fn search_in_current(
        &self,
        key: &K,
        version_guard: &VersionGuard,
    ) -> Option<V> {

         let hasher = self.hasher.read().unwrap();
        let (h1, h2) = hasher.locate_buckets(key);
        
        if let Some(bucket) = self.get_bucket(h1) {
            if let Some(value) = self.search_bucket(&bucket, key, version_guard) {
                return Some(value);
            }
        }
        
        if let Some(bucket) = self.get_bucket(h2) {
            if let Some(value) = self.search_bucket(&bucket, key, version_guard) {
                return Some(value);
            }
        }
        
        None
    }
    
    /// 在新表中搜索
    fn search_in_new_table(
        &self,
        key: &K,
        new_buckets: &Arc<RwLock<Vec<Arc<Bucket<K, V>>>>>,
        version_guard: &VersionGuard,
    ) -> Option<V> {
        
          // 获取新桶数组的长度
        let new_bucket_count = {
            let buckets = new_buckets.read().unwrap();
            buckets.len()
        };
        let (h1, h2) ={
             let hasher = self.hasher.read().unwrap();
            hasher.locate_buckets_with_capacity(key,new_bucket_count)
        } ;
        let buckets = new_buckets.read().unwrap();
        
        if let Some(bucket) = buckets.get(h1) {
            if let Some(value) = self.search_bucket(&bucket, key, version_guard) {
                return Some(value);
            }
        }
        
        if let Some(bucket) = buckets.get(h2) {
            if let Some(value) = self.search_bucket(&bucket, key, version_guard) {
                return Some(value);
            }
        }
        
        None
    }

    /// 在桶中搜索键值对
    fn search_bucket(
        &self, 
        bucket: &Bucket<K, V>, 
        key: &K,
        version_guard: &VersionGuard,
    ) -> Option<V> {
        
        let fp = {
                 let hasher = self.hasher.read().unwrap();
                 hasher.fingerprint(key)
        };
        
        // 获取所有匹配指纹的槽位索引
        let slot_indices = bucket.find_slots(fp, self.simd_searcher.as_ref());
        
        for slot_idx in slot_indices {
            if let Some(value) = bucket.get(slot_idx, key, version_guard) {
                return Some(value);
            }
        }
        
        None
    }

    /// 删除键值对
    pub fn remove(&self, key: &K) -> Option<V> {
        let _timer = self.stats_recorder.start_timer(OperationType::Remove);
        let version_guard = self.version.begin_operation(OperationType::Remove);
        
        // 根据迁移状态路由删除操作
        let state = self.migration_state.read().unwrap();
        match *state {
            MigrationState::NotStarted => self.remove_in_current(key, &version_guard),
            MigrationState::Migrating(ref new_buckets) => {
                // 双删：同时删除当前表和新表
                let result = self.remove_in_current(key, &version_guard);
                if result.is_some() {
                    self.remove_from_new_table(key, new_buckets, &version_guard);
                }
                result
            }
            MigrationState::Completed(ref new_buckets) => {
                // 双删：同时删除当前表和新表
                let result = self.remove_in_current(key, &version_guard);
                if result.is_some() {
                    self.remove_from_new_table(key, new_buckets, &version_guard);
                }
                result
            }
            MigrationState::Switched => self.remove_in_current(key, &version_guard),
        }
    }
    
    /// 在当前表中删除
    fn remove_in_current(
        &self,
        key: &K,
        version_guard: &VersionGuard,
    ) -> Option<V> {
         
        let (h1, h2) = {
                let hasher = self.hasher.read().unwrap();
                hasher.locate_buckets(key)
        };
        
        if let Some(bucket) = self.get_bucket(h1) {
            if let Some(value) = self.search_and_remove(&bucket, key, version_guard) {
                self.size.fetch_sub(1, Ordering::Release);
                return Some(value);
            }
        }
        
        if let Some(bucket) = self.get_bucket(h2) {
            if let Some(value) = self.search_and_remove(&bucket, key, version_guard) {
                self.size.fetch_sub(1, Ordering::Release);
                return Some(value);
            }
        }
        
        None
    }
    
    /// 在新表中删除
    fn remove_from_new_table(
        &self,
        key: &K,
        new_buckets: &Arc<RwLock<Vec<Arc<Bucket<K, V>>>>>,
        version_guard: &VersionGuard,
    ) -> Option<V> {
        let new_bucket_count = {
            let buckets = new_buckets.read().unwrap();
            buckets.len()
        };
        let (h1, h2) ={
            let hasher=self.hasher.read().unwrap();
         hasher.locate_buckets_with_capacity(key,new_bucket_count)
        };
        
        let buckets = new_buckets.read().unwrap();
        
        if let Some(bucket) = buckets.get(h1) {
            if let Some(value) = self.search_and_remove(&bucket, key, version_guard) {
                return Some(value);
            }
        }
        
        if let Some(bucket) = buckets.get(h2) {
            if let Some(value) = self.search_and_remove(&bucket, key, version_guard) {
                return Some(value);
            }
        }
        
        None
    }

    /// 搜索并删除键值对
    fn search_and_remove(
        &self, 
        bucket: &Bucket<K, V>, 
        key: &K,
        version_guard: &VersionGuard,
    ) -> Option<V> {
         let fp ={

             let hasher = self.hasher.read().unwrap();
            hasher.fingerprint(key)
        };
        
        
        if let Some(slot_idx) = bucket.find_slot(fp, &*self.simd_searcher) {
            return bucket.remove(slot_idx, key, version_guard);
        }
        
        None
    }

    /// 获取负载因子
    pub fn load_factor(&self) -> f32 {
        let size = self.size.load(Ordering::Acquire) as f32;
        let bucket_count = self.current.read().unwrap().len();
        let capacity = bucket_count * self.config.slot_count_per_bucket;
        size / capacity as f32
    }

    pub fn testtt(&self,key: &K){
        let (h1, h2) = {
                let hasher = self.hasher.read().unwrap();
                hasher.locate_buckets(key)
        };
       
        println!("Hashes: h1={}, h2={}", h1, h2);
        
        let bucket1 = self.get_bucket(h1).unwrap();
        bucket1.print_summary();
        let bucket2 = self.get_bucket(h2).unwrap();
        bucket2.print_summary();
        
    }
    /// 计算动态扩容因子
fn calculate_expand_factor(&self) -> f32 {
    // 获取当前负载因子
    let current_load = self.load_factor();
    
    // 获取队列中的待插入项数量
    let queue_size = self.migrate_queue_size() as f32+self.insert_queue_size() as f32;
    
    // 计算总元素数（当前 + 队列）
    let total_items = self.size.load(Ordering::Relaxed) as f32 + queue_size;
    
    // 计算所需的最小容量（基于目标负载因子）
    let target_load = self.config.max_load_factor;
    let min_required_capacity = total_items / target_load;
    
    // 获取当前容量
    let current_capacity = self.slot_capacity() as f32;
    
    // 计算扩容因子（至少为2.0）
    let expand_factor = (min_required_capacity / current_capacity).max(2.0);
    
    // 添加安全边界
    expand_factor * 1.2
}

    /// 获取指定桶
    pub fn get_bucket(&self, index: usize) -> Option<Arc<Bucket<K, V>>> {
        self.current.read().unwrap().get(index).cloned()
    }

    /// 获取统计信息
    pub fn stats(&self) -> CuckooMapStats {
        let op_stats = self.stats_recorder.operation_stats().snapshot();
        // 计算碰撞率
        let collision_rate = if op_stats.insert_count > 0 {
            op_stats.collision_count as f32 / op_stats.insert_count as f32
        } else {
            0.0
        };
        
        // 迁移进度
        let migration_progress = self.migration_progress();
        
        CuckooMapStats {
            size: self.size.load(Ordering::Acquire),
            capacity: self.bucket_count(),
            load_factor: self.load_factor(),
            insert_count: op_stats.insert_count,
            get_count: op_stats.get_count,
            remove_count: op_stats.remove_count,
            kick_count: op_stats.kick_count,
            migration_count: op_stats.migration_count,
            collision_rate,
            migration_progress,
        }
    }
    
    /// 创建迭代器
    pub fn iter(&self) -> CuckooMapIter<K, V> {
        CuckooMapIter {
            map: self,
            bucket_index: 0,
            slot_index: 0,
        }
    }
    
    /// 启动迁移
    pub fn migrate(&self, plan: MigrationPlan) -> Result<(), CuckooError> {
          // 1. 开始迁移
        self.start_migration(plan);
         // 2. 等待迁移完成
        self.wait_for_migration_completion();
        
        // 3. 完成数据迁移
        self.complete_data_migration()?;
        
        // 验证数据完整性
        // self.validate_migration();
        // 4. 切换到新桶
        self.switch_to_new_buckets()?;
         let thread_id = thread::current().id();
         log_info!("worker {:?} Migration start complete_migration", thread_id);
        // 5. 完成整个迁移
        self.complete_migration()?;
                
        Ok(())
    }
    
     fn wait_for_migration_completion(&self) {
        let total = {
            let current = self.current.read().unwrap();
            current.len()
        };
       
        let start = Instant::now();
        
        while self.migration_progress.load(Ordering::Acquire) < total {
            if start.elapsed() > Duration::from_secs(60) {
                log_error!("Migration timeout after 60 seconds");
                break;
            }
            thread::sleep(Duration::from_millis(100));
        }
    }


    /// 启动迁移线程
    fn start_migration_thread(
        &self, 
        new_buckets: Arc<RwLock<Vec<Arc<Bucket<K, V>>>>>, 
        bucket_count: usize,
        new_bucket_count:usize,
    ) {
        let this = self.clone();
        let new_buckets_clone = new_buckets.clone(); // 克隆新桶数组
        // 创建自定义线程池
        let pool = ThreadPoolBuilder::new()
        .num_threads(num_cpus::get().min(2).max(8)) // 限制最大线程数
        .build()
        .unwrap();
        // 使用Rayon的并行迭代器
        pool.install(|| {
            (0..bucket_count).into_par_iter().for_each(|bucket_idx| {
                // 迁移逻辑
                if let Err(e) = self.migrate_bucket(
                    bucket_idx,
                    &new_buckets_clone,
                    new_buckets.read().unwrap().len(),
                ) {
                    log_error!("Failed to migrate bucket {}: {:?}", bucket_idx, e);
                }
            });
        });
    
    }


    fn validate_migration(&self, new_buckets: &Arc<RwLock<Vec<Arc<Bucket<K, V>>>>>) {
        let old_size = self.size.load(Ordering::Acquire);
        let new_size = {
            let buckets = new_buckets.read().unwrap();
            let mut size = 0;
            for bucket in buckets.iter() {
                size += bucket.count_occupied_slots();
            }
            size
    };
    
    if old_size != new_size {
        log_error!("Migration data loss: old_size={}, new_size={}", old_size, new_size);
    } else {
        log_info!("Migration data verified: size={}", old_size);
    }
}
    /// 迁移单个桶
    fn migrate_bucket(
        &self,
        bucket_idx: usize,
        new_buckets: &Arc<RwLock<Vec<Arc<Bucket<K, V>>>>>,
        new_bucket_count:usize
    ) -> Result<(), CuckooError> {
        let version_guard = self.version.begin_operation(OperationType::Migration);
        
        // 获取源桶
        let source_bucket = {
            let buckets = self.current.read().unwrap();
            buckets.get(bucket_idx).ok_or(CuckooError::InvalidBucket)?.clone()
        };
        
        // 锁定源桶
        let bucket_guard = source_bucket.try_lock()?;
        // 收集迁移失败的键值对
        
         let mut failed_keys: Vec<(K, V)> = Vec::new();
        // 迁移每个槽位
        for slot in bucket_guard.slots() {
            if slot.is_empty() {
                continue;
            }
            
            // 获取槽位锁
            let fp = slot.load_fingerprint();
            let slot_guard = slot.try_lock(fp, &version_guard)?;
            
            // 加载键值
            let key_bytes = slot_guard.load_key().ok_or(CuckooError::MigrationFailure)?;
            let value_bytes = slot_guard.load_value().ok_or(CuckooError::MigrationFailure)?;
            
            // 重建键值对象
            let key = K::from_bytes(&key_bytes).ok_or(CuckooError::KeyDeserialization)?;
            let value = V::from_bytes(&value_bytes).ok_or(CuckooError::ValueDeserialization)?;
            
            // 插入到新表
           if let Err(e) = self.insert_to_new_table(&key, &value, new_buckets,new_bucket_count, &version_guard,true) {
            log_warn!("Failed to insert key to new table: {:?}, will retry later", e);
            // 收集请求加入重试队列队列
             failed_keys.push((key, value));
            }
        }
        // 将失败的键值对加入重试队列
        let failed_count = failed_keys.len();
        if !failed_keys.is_empty() {
            let mut queue = self.migrate_queue.lock().unwrap();
            for (key, value) in failed_keys {
                queue.push_back((key, value));
            }
        }
        // 更新进度 - 整个桶处理完成后更新一次
        self.migration_progress.fetch_add(1, Ordering::Release);
        let thread_id = thread::current().id();
       // log_info!("worker {:?} Migrated bucket {} ({} items failed)", thread_id, bucket_idx, failed_count);

        Ok(())
    }

    /// 强制触发迁移
    pub fn force_migrate(&self, plan: MigrationPlan) -> Result<(), CuckooError> {
        let bucket_count = {
            let current = self.current.read().unwrap();
            current.len()
        };
        let new_capacity = if let Some(factor) = plan.expand_factor {
            (self.current.read().unwrap().len() as f32 * factor) as usize
        } else if let Some(factor) = plan.shrink_factor {
            (self.current.read().unwrap().len() as f32 * factor) as usize
        } else if let Some(capacity) = plan.target_capacity {
            capacity
        } else {
            return Err(CuckooError::MigrationNotNeeded);
        };
        
        log_info!("Forcing migration from {} to {} buckets", 
            bucket_count, new_capacity);
        
        // 创建新桶数组
        let new_buckets = Arc::new(RwLock::new(
            (0..new_capacity)
                .map(|_| Arc::new(Bucket::new(self.version.clone())))
                .collect()
        ));
        
        // 设置迁移状态为迁移中
        {
            let mut state = self.migration_state.write().unwrap();
            *state = MigrationState::Migrating(Arc::clone(&new_buckets));
        }
        
        // 启动后台迁移线程
        self.start_migration_thread(new_buckets, bucket_count,new_capacity);
        
        Ok(())
    }

   
    /// 检查迁移是否完成但未切换
pub fn is_migration_completed(&self) -> bool {
    matches!(*self.migration_state.read().unwrap(), MigrationState::Completed(_))
}
    /// 检查迁移是否完全完成
    pub fn is_migration_fully_completed(&self) -> bool {
        self.migration_completed.load(Ordering::SeqCst)
    }
    
    /// 获取迁移进度
    pub fn migration_progress(&self) -> f32 {

      //  log_info!("migration_progress state:{},{}",self.is_migration_fully_completed(), self.is_migration_in_progress());
         // 如果迁移完成但未切换，返回100%
        if self.is_migration_completed() {
            return 1.0;
        }
        if self.is_migration_fully_completed() {          
            return 1.0; // 迁移完成，返回100%
        }
      
        // 如果迁移未开始，返回0%
        if !self.is_migration_in_progress() {
            return 0.0;
        }
        
        // 获取迁移状态
    let state = self.migration_state.read().unwrap();
    
    // 获取总桶数（需要迁移的桶数量）和已迁移桶数
    let (total_buckets, migrated) = match &*state {
        MigrationState::Migrating(_) | MigrationState::Completed(_) => {
            // 总桶数是旧桶数组的大小
            let total = self.current.read().unwrap().len();
            let migrated = self.migration_progress.load(Ordering::SeqCst);
            (total, migrated)
        }
        _ => (0, 0),
    };
    // 计算进度
    if total_buckets > 0 {
        let progress = migrated as f32 / total_buckets as f32;
        // 确保进度在0-1之间
        progress.min(1.0).max(0.0)
    } else {
        0.0
    }
    }

    /// 检查迁移是否进行中
    pub fn is_migration_in_progress(&self) -> bool {
        matches!(*self.migration_state.read().unwrap(), 
            MigrationState::Migrating(_) | MigrationState::Completed(_))
    }
    

    /// 处理插入队列
//     fn process_insert_queue(&self) {
//             log_info!("Queue processing start process_insert_queue");
//         const MAX_BATCH_SIZE: usize = 400;
//         const MAX_RETRIES: usize = 5;
//         const RETRY_DELAY_BASE: u64 = 50;
        
//         let mut iteration = 0;
//          let thread_id = thread::current().id();
//         // 处理队列直到为空或迁移开始
//         while self.queue_size() > 0 && self.is_migration_fully_completed() {
//             iteration += 1;
//             log_info!("worker {:?} Processing iteration {}",thread_id, iteration);
            
//             // 获取批处理数据
//             let batch = self.get_next_batch(MAX_BATCH_SIZE);
            
//             if batch.is_empty() {
//                 log_info!("No items to process, exiting");
//                 break;
//             }
            
//             log_info!("Processing batch of {} queued inserts", batch.len());
            
//             // 处理批处理
//             self.process_batch(batch, MAX_RETRIES, RETRY_DELAY_BASE);
            
//             // 短暂休眠，让出CPU
//             thread::sleep(Duration::from_millis(10));
//         }
        
//         log_info!("Queue processing completed after {} iterations,queue size ={},migration status={}", iteration,self.queue_size(),self.is_migration_fully_completed());
// }

/// 获取下一批处理数据
// fn get_next_batch(&self, max_batch_size: usize) -> Vec<(K, V)> {
//     let mut queue = self.migrate_queue.lock().unwrap();
    
//     if queue.is_empty() {
//         return Vec::new();
//     }
    
//     // 每次最多处理 max_batch_size 个元素
//     let batch_size = max_batch_size.min(queue.len());
//     queue.drain(..batch_size).collect()
// }

/// 处理批处理数据
// fn process_batch(&self, batch: Vec<(K, V)>, max_retries: usize, retry_delay_base: u64) {
//     let mut failed_items = Vec::new();
    
//     for (key, value) in batch {
//         // 检查迁移状态
//         if self.is_migration_in_progress() {
//             log_info!("Migration started during batch processing, aborting");
//             failed_items.push((key, value));
//             break;
//         }
        
//         // 尝试插入
//         let result = self.try_insert_with_retry(
//             key.clone(), 
//             value.clone(), 
//             max_retries, 
//             retry_delay_base
//         );
        
//         if let Err(_) = result {
//             failed_items.push((key, value));
//         }
//     }
    
//     // 将失败项放回队列
//     if !failed_items.is_empty() {
//         log_info!("Returning {} failed items to queue", failed_items.len());
//         self.migrate_queue.lock().unwrap().extend(failed_items);
//     }
// }

/// 尝试插入（带重试）
fn try_insert_with_retry(
    &self,
    key: K,
    value: V,
    max_retries: usize,
    retry_delay_base: u64,
    migrate:bool
) -> Result<(), CuckooError> {
    let mut retries = 0;
    
    while retries < max_retries {
        // 检查迁移状态
        if self.is_migration_in_progress() {
            log_info!("Migration started during insert attempt, aborting");
            return Err(CuckooError::MigrationInProgress);
        }
         let version_guard: VersionGuard = self.version.begin_operation(OperationType::Insert);
        match self._insert(key.clone(), value.clone(),&version_guard,migrate) {
            Ok(()) => {
                let thread_id = thread::current().id();
              //  log_info!("Insert succeeded from queue worker {:?}", thread_id);
                return Ok(());
            }
            Err(CuckooError::MigrationInProgress) => {
                log_info!("Insert triggered migration, aborting");
                return Err(CuckooError::MigrationInProgress);
            }
            Err(e) => {
                log_warn!("Failed to process queued insert: {:?}, retry {}", e, retries);
                retries += 1;
                // 指数退避策略
                thread::sleep(Duration::from_millis(retry_delay_base * retries as u64));
            }
        }
    }
    
    log_error!("Failed to process queued insert after {} retries: key={:?}", 
        max_retries, String::from_utf8_lossy(key.as_bytes()));
    
    Err(CuckooError::InsertFailed)
}

    

    // 等待所有操作完成
    // pub fn wait_for_completion(&self) {
    //     const MAX_WAIT_TIME: Duration = Duration::from_secs(40);
    //     const POLL_INTERVAL: Duration = Duration::from_millis(100);
        
    //     let start = Instant::now();
        
    //     while start.elapsed() < MAX_WAIT_TIME {
    //          let queue_size = self.queue_size();
    //     let processing = self.is_queue_processing();
    //     let migration_state = format!("{:?}", *self.migration_state.read().unwrap());
        
    //     log_info!("wait_for_completion: size={}, processing={}, migration_state={}", queue_size, processing, migration_state);
        
    //     if queue_size == 0 && !processing {
    //         let state = self.migration_state.read().unwrap();
    //         // 检查迁移状态是否为Switched或Idle
    //         if matches!(*state, MigrationState::Switched) || matches!(*state, MigrationState::NotStarted) {
    //             log_info!("wait_for_completion: All operations completed successfully");
    //             return;
    //         }
    //         }
    //         log_info!("wait_for_completion sleep size={}, processing={},migration status={}",   self.queue_size(), self.is_queue_processing(),self.is_migration_fully_completed());
    //         thread::sleep(POLL_INTERVAL);
    //     }
        
    //     panic!("Timeout waiting for map operations to complete,left size {},processing={} migration status={},total item:{}",self.queue_size(), self.queue_processing.load(Ordering::Acquire),self.is_migration_fully_completed(),self.size.load(Ordering::Relaxed));
    // }

}

/// 哈希表迭代器
pub struct CuckooMapIter<'a, K: Key +Clone+ Send + Sync, V: Value +Clone+ Send + Sync> {
    map: &'a CuckooMap<K, V>,
    bucket_index: usize,
    slot_index: usize,
}

impl<'a, K: Key +Clone+ Send + Sync, V: Value +Clone+ Send + Sync> Iterator for CuckooMapIter<'a, K, V> {
    type Item = (K, V);
    
    fn next(&mut self) -> Option<Self::Item> {
        loop {
            // 获取桶列表的读锁
            let buckets = self.map.current.read().unwrap();
            
            if self.bucket_index >= buckets.len() {
                return None;
            }
            
            // 获取当前桶
            let bucket = buckets.get(self.bucket_index).unwrap();
            
            if self.slot_index < bucket.slots().len() {
                let slot = &bucket.slots()[self.slot_index];
                self.slot_index += 1;
                
                if !slot.is_empty() {
                    // 创建版本守卫以确保一致性
                    let version_guard = self.map.version.begin_operation(OperationType::Iterate);
                    let fp = slot.load_fingerprint();
                    
                    // 锁定槽位
                    if let Ok(guard) = slot.try_lock(fp, &version_guard) {
                        // 加载键值对
                        if let (Some(key_bytes), Some(value_bytes)) = (guard.load_key(), guard.load_value()) {
                            if let (Some(key), Some(value)) = (K::from_bytes(&key_bytes), V::from_bytes(&value_bytes)) {
                                return Some((key, value));
                            }
                        }
                    }
                }
            } else {
                // 当前桶遍历完毕，转到下一个桶
                self.bucket_index += 1;
                self.slot_index = 0;
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{
       
        hash::strategy::HashAlgorithm, memory::PoolAllocator, simd::SimdSearcherType, stats::recorder::GlobalStatsRecorder, types::ByteKey, version::VersionTracker, DoubleHashStrategy
    };
    use std::{sync::Arc, thread, time::Duration};
    use log::{LevelFilter, info, debug, warn, error};
    use env_logger::{Builder, Env};
    use std::io::Write;
    
    // 初始化日志记录器
    fn init_logger() {
        let env = Env::default()
            .filter_or("RUST_LOG", "debug")
            .write_style_or("RUST_LOG_STYLE", "always");
        
        Builder::from_env(env)
            .format(|buf, record| {
                writeln!(
                    buf,
                    "[{}] {}:{} - {}",
                    record.level(),
                    record.file().unwrap_or("unknown"),
                    record.line().unwrap_or(0),
                    record.args()
                )
            })
            .init();
    }
    
    fn create_test_map() -> CuckooMap<ByteKey, Vec<u8>> {
        let config = CuckooMapConfig::default();
        let hasher = Box::new(DoubleHashStrategy::new(AtomicUsize::new(config.initial_capacity),HashAlgorithm::XxHash));
        let memory_pool = Arc::new(PoolAllocator::new(64, 1024));
        let simd_strategy = SimdStrategy::new(SimdSearcherType::Scalar);
        let simd_searcher = simd_strategy.get_searcher();
        let stats_recorder = Arc::new(GlobalStatsRecorder::new());
        let version = Arc::new(VersionTracker::new());

        CuckooMap::new(
            config,
            hasher,
            memory_pool,
            simd_searcher,
            stats_recorder,
            version,
        )
    }

    // #[test]
    // fn test_insert_and_get() {
    //     let map = create_test_map();
    //     let key = ByteKey(b"test_key".to_vec());
    //     let value = b"test_value".to_vec();

    //     // 插入键值对
    //     map.insert(key.clone(), value.clone()).unwrap();

    //     // 查询键值对
    //     let retrieved = map.get(&key).unwrap();
    //     assert_eq!(retrieved, value);
    // }

//     #[test]
//     fn test_insert_duplicate() {
//         let map = create_test_map();
//         let key = ByteKey(b"test_key".to_vec());
//         let value1 = b"test_value1".to_vec();
//         let value2 = b"test_value2".to_vec();
        
//         // 第一次插入
//         map.insert(key.clone(), value1.clone()).unwrap();
//         assert_eq!(map.size.load(Ordering::Relaxed), 1);
        
//         // 第二次插入相同键，应覆盖
//         map.insert(key.clone(), value2.clone()).unwrap();
        
//         // 大小应保持不变
//         assert_eq!(map.size.load(Ordering::Relaxed), 1);
        
//         // 查询应返回新值
//         let retrieved = map.get(&key).unwrap();
//         assert_eq!(retrieved, value2);
//     }

//     #[test]
//     fn test_get_nonexistent() {
//         let map = create_test_map();
//         let key = ByteKey(b"test_key".to_vec());

//         // 查询不存在的键
//         assert!(map.get(&key).is_none());
//     }

//     #[test]
//     fn test_remove() {
//         let map = create_test_map();
//         let key = ByteKey(b"test_key".to_vec());
//         let value = b"test_value".to_vec();

//         // 插入
//         map.insert(key.clone(), value.clone()).unwrap();

//         // 删除
//         let removed_value = map.remove(&key).unwrap();
//         assert_eq!(removed_value, value);

//         // 再次查询应返回None
//         assert!(map.get(&key).is_none());
//     }

//     #[test]
//     fn test_remove_nonexistent() {
//         let map = create_test_map();
//         let key = ByteKey(b"test_key".to_vec());

//         // 删除不存在的键
//         assert!(map.remove(&key).is_none());
//     }

//     #[test]
//     fn test_cuckoo_kick() {
//         init_logger();
//         let map = create_test_map();
//         let config = CuckooMapConfig {
//             initial_capacity: 2, // 小容量以强制触发踢出
//             max_load_factor: 0.5,
//             max_kick_depth: 32,
//             slot_count_per_bucket: 4,
//             migration_batch_size: 64,
//             migration_parallelism: 4,
//             migration_lock_timeout_ms: 100,
//         };
        
//        let hasher = Box::new(DoubleHashStrategy::new(AtomicUsize::new(config.initial_capacity),HashAlgorithm::AHash));
//         let memory_pool = Arc::new(PoolAllocator::new(64, 1024));
//         let simd_strategy = SimdStrategy::new(SimdSearcherType::Scalar);
//         let simd_searcher = simd_strategy.get_searcher();
//         let stats_recorder = Arc::new(GlobalStatsRecorder::new());
//         let version = Arc::new(VersionTracker::new());
        
//         let map = CuckooMap::new(
//             config,
//             hasher,
//             memory_pool,
//             simd_searcher,
//             stats_recorder,
//             version,
//         );
        
//         // 填充桶以触发踢出
//         for i in 0..10 {
//             let key = ByteKey(format!("key_{}", i).into_bytes());
//             let value = format!("value_{}", i).into_bytes();
//             map.insert(key, value).unwrap();
//         }
        
//         // 验证所有键都存在
//         for i in 0..10 {
//             let key = ByteKey(format!("key_{}", i).into_bytes());
//             assert!(map.get(&key).is_some());
//         }
//     }

//     #[test]
// fn test_migration() {
//     init_logger();
//     let map = create_test_map();
//     let initial_capacity = map.bucket_count();
    
//     // 在迁移前插入数据
//     let mut keys = vec![];
//     for i in 0..100 {
//         let key = ByteKey(format!("key_{}", i).into_bytes());
//         let value = format!("value_{}", i).into_bytes();
//         map.insert(key.clone(), value.clone()).unwrap();
//         keys.push(key);
//     }
    
//     // 强制触发迁移
//     map.force_migrate(MigrationPlan::expand(2.0)).unwrap();
    
//      // 等待迁移完成
//     while !map.is_migration_fully_completed() {
//         thread::sleep(Duration::from_millis(10));
//     }
    
//     // 验证容量增加
//     //assert!(map.capacity() > initial_capacity);
    
//     // 验证迁移前插入的数据仍然存在
//     for key in &keys {
//         assert!(map.get(key).is_some());
//     }
    
//     // 验证迁移后可以插入新数据
//     let new_key = ByteKey(b"new_key".to_vec());
//     let new_value = b"new_value".to_vec();
//     map.insert(new_key.clone(), new_value.clone()).unwrap();
//     assert_eq!(map.get(&new_key).unwrap(), new_value);
// }

//     #[test]
//     fn test_iter() {
//         let map = create_test_map();
//         let items = vec![
//             (ByteKey(b"key1".to_vec()), b"value1".to_vec()),
//             (ByteKey(b"key2".to_vec()), b"value2".to_vec()),
//             (ByteKey(b"key3".to_vec()), b"value3".to_vec()),
//         ];
        
//         // 插入所有项
//         for (key, value) in &items {
//             map.insert(key.clone(), value.clone()).unwrap();
//         }
        
//         // 收集迭代器结果
//         let mut iter_results = map.iter().collect::<Vec<_>>();
//         iter_results.sort_by_key(|(k, _)| k.clone());
        
//         // 验证所有项都存在
//         assert_eq!(iter_results.len(), items.len());
//         for (i, (key, value)) in items.iter().enumerate() {
//             assert_eq!(&iter_results[i].0, key);
//             assert_eq!(&iter_results[i].1, value);
//         }
//     }

//     #[test]
//     fn test_stats() {
//         let map = create_test_map();
        
//         // 初始统计
//         let initial_stats = map.stats();
//         assert_eq!(initial_stats.size, 0);
//         assert_eq!(initial_stats.load_factor, 0.0);
        
//         // 插入项
//         for i in 0..10 {
//             let key = ByteKey(format!("key_{}", i).into_bytes());
//             let value = format!("value_{}", i).into_bytes();
//             map.insert(key, value).unwrap();
//         }
        
//         // 获取统计
//         let stats = map.stats();
//         assert_eq!(stats.size, 10);
//         assert!(stats.load_factor > 0.0);
//         assert!(stats.insert_count > 0);
//     }

//     #[test]
//     fn test_concurrent_insert() {
//         init_logger();
//         let map = Arc::new(create_test_map());
//         let mut handles = vec![];
        
//         // 创建10个线程并发插入
//         for i in 0..10 {
//             let map_clone = Arc::clone(&map);
//             let handle = std::thread::spawn(move || {
//                 for j in 0..100 {
//                     let key_str = format!("thread_{}_key_{}", i, j);
//                     let key = ByteKey(key_str.clone().into_bytes());
//                     let value = format!("value_{}_{}", i, j).into_bytes();
//                     debug!("Thread {} inserting: {}", i, key_str);
//                     map_clone.insert(key, value).unwrap();
//                 }
//             });
//             handles.push(handle);
//         }
        
//         // 等待所有线程完成
//         for handle in handles {
//             handle.join().unwrap();
//         }
        
//         // 验证所有项都存在
//         let mut missing_keys = Vec::new();
//         for i in 0..10 {
//             for j in 0..100 {
//                 let key_str = format!("thread_{}_key_{}", i, j);
//                 let key_bytes = key_str.clone().into_bytes();
//                 let key = ByteKey(key_bytes);
//                 if map.get(&key).is_none() {
//                     println!("Missing key: {}", key_str);
//                     missing_keys.push(key_str);
//                 }
//             }
//         }
        
//         if !missing_keys.is_empty() {
//             panic!("Missing keys: {:?}", missing_keys);
//         }
//     }

//     #[test]
//     fn test_load_factor() {
//         let map = create_test_map();
//         let initial_load_factor = map.load_factor();
//         assert_eq!(initial_load_factor, 0.0);
        
//         // 插入项
//         for i in 0..10 {
//             let key = ByteKey(format!("key_{}", i).into_bytes());
//             let value = format!("value_{}", i).into_bytes();
//             map.insert(key, value).unwrap();
//         }
        
//         // 验证负载因子增加
//         let load_factor = map.load_factor();
//         assert!(load_factor > 0.0);
        
//         // 删除项
//         for i in 0..5 {
//             let key = ByteKey(format!("key_{}", i).into_bytes());
//             map.remove(&key).unwrap();
//         }
        
//         // 验证负载因子减少
//         let new_load_factor = map.load_factor();
//         assert!(new_load_factor < load_factor);
//     }
}