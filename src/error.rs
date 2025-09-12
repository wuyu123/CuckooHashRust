//! 统一错误处理 - 所有可能错误类型和恢复逻辑

/// Cuckoo哈希表可能发生的错误
#[derive(Debug, thiserror::Error)]
pub enum CuckooError {
    #[error("表已满，无法插入新条目 (容量: {capacity}, 当前大小: {size}, 负载因子: {load_factor:.2})")]
    TableFull {
        capacity: usize,
        size: usize,
        load_factor: f32,
    },
    
    #[error("键已存在: {key}")]
    KeyAlreadyExists {
        key: String,
    },
    
    #[error("键不存在: {key}")]
    KeyNotFound {
        key: String,
    },
    
    #[error("并发修改检测到版本冲突")]
    VersionConflict ,
    
    #[error("踢出路径超过最大深度限制")]
    KickPathExceeded,
    
    #[error("检测到循环踢出路径")]
    CycleDetected ,
    
    #[error("内存分配失败 (大小: {size}, 对齐: {align})")]
    AllocationFailed {
        size: usize,
        align: usize,
    },
    
    #[error("迁移操作正在进行中")]
    MigrationInProgress,
    
    #[error("迁移未完成: 已迁移")]
    MigrationIncomplete ,

    #[error("无需迁移")]
    MigrationNotNeeded,

    #[error("哈希冲突无法解决")]
    HashConflict,
    
    #[error("锁争用超时 (操作: {operation})")]
    LockContention {
        operation: String,
    },
    
    #[error("操作超时: {operation}")]
    Timeout {
        operation: String,
    },
    
    #[error("表已标记为删除")]
    TableDeleted,
    
    #[error("无效配置: {reason}")]
    InvalidConfig {
        reason: String,
    },
    
    #[error("未知错误: {message}")]
    Unknown {
        message: String,
    },
    
    // 新增错误变体
    #[error("迁移失败")]
    MigrationFailure ,

    #[error("踢出路径循环")]
    KickPathCycle,
    #[error("无有效踢出候选")]
    NoKickCandidate,

    #[error("无效桶索引")]
    InvalidBucket,
    
    #[error("键反序列化失败")]
    KeyDeserialization,
    
    #[error("值反序列化失败")]
    ValueDeserialization,
    
    #[error("值未找到")]
    ValueNotFound,
   
    #[error("槽位已占")]
    SlotOccupied,

    #[error("迁移超时")]
    MigrationTimeout,

    #[error("找不到桶")]
    BucketNotFound,
    #[error("写入失败")]
    InsertFailed,
    #[error("迁移状态冲突")]
    MigrationStateConflict,
    #[error("锁超时")]
    LockTimeout,
}

impl CuckooError {
    /// 获取错误恢复建议
    pub fn recovery_suggestion(&self) -> Option<&'static str> {
        match self {
            Self::TableFull { .. } => Some("尝试扩容表或减小负载因子"),
            Self::KeyAlreadyExists { .. } => Some("检查是否需要更新操作"),
            Self::KeyNotFound { .. } => Some("确认键值是否存在"),
            Self::VersionConflict { .. } => Some("使用新版本重试操作"),
            Self::KickPathExceeded { .. } => Some("增加最大踢出深度或扩容表"),
            Self::CycleDetected { .. } => Some("尝试使用不同的哈希函数"),
            Self::AllocationFailed { .. } => Some("检查系统内存或减小表大小"),
            Self::MigrationInProgress => Some("等待迁移完成后重试"),
            Self::MigrationIncomplete { .. } => Some("重试迁移或检查系统状态"),
            Self::MigrationNotNeeded => Some("无需迁移"),
            Self::HashConflict => Some("增加表容量或使用更强的哈希函数"),
            Self::LockContention { .. } => Some("减少并发或优化锁定策略"),
            Self::Timeout { .. } => Some("优化操作或增加超时时间"),
            Self::TableDeleted => Some("表已被销毁，无法操作"),
            Self::InvalidConfig { .. } => Some("检查配置参数"),
            Self::Unknown { .. } => Some("检查日志获取详细信息"),
            
            // 新增错误的恢复建议
            Self::MigrationFailure { .. } => Some("检查迁移日志并重试"),
            Self::KickPathCycle { .. } => Some("踢出路径循环"),
            Self::NoKickCandidate { .. } => Some("无有效踢出候选"),
            Self::InvalidBucket => Some("验证桶索引是否有效"),
            Self::KeyDeserialization => Some("检查键序列化格式"),
            Self::ValueDeserialization => Some("检查值序列化格式"),
            Self::ValueNotFound => Some("确认值是否存在"),
            Self::SlotOccupied=>Some("槽位已被使用"),
            Self::MigrationTimeout=>Some("迁移超时"),
            Self::BucketNotFound=>Some("找不到桶"),
            Self::InsertFailed=>Some("写入失败"),
             Self::MigrationStateConflict=>Some("迁移状态冲突"),
             Self::LockTimeout=>Some("锁超时"),
            
        }
    }
    
    /// 判断错误是否可恢复
    pub fn is_recoverable(&self) -> bool {
        !matches!(
            self,
            Self::TableDeleted | Self::AllocationFailed { .. }
        )
    }
    
    /// 判断错误是否由并发冲突引起
    pub fn is_concurrency_error(&self) -> bool {
        matches!(
            self,
            Self::VersionConflict { .. } | Self::LockContention { .. }
        )
    }
    
    /// 是否需要操作重试
    pub fn should_retry(&self) -> bool {
        self.is_concurrency_error() || matches!(
            self,
            Self::MigrationFailure { .. } | 
            Self::MigrationIncomplete { .. } |
            Self::InvalidBucket  // 添加这一行
        )
    }
    
   
}
