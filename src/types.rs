//! 核心类型定义 - 共享类型和接口

use core::fmt;
use std::{
    any::Any, borrow::Cow, fmt::{Debug, Display}, hash::{Hash, Hasher}, sync::atomic::{AtomicU32, AtomicU64, Ordering}, time::{Duration, SystemTime}
};

/// 键特征 - 支持动态类型
pub trait Key: Debug + Display + Send + Sync + 'static {
    /// 获取键的字节表示
    fn as_bytes(&self) -> &[u8];
    
    /// 克隆键（对象安全方法）
    fn clone_key(&self) -> Box<dyn Key>;
    
    /// 比较键
    fn eq_key(&self, other: &dyn Key) -> bool;
    
    /// 计算哈希值
    fn hash_key(&self) -> u64;
    
    /// 将键转换为 `Any` 类型，以便向下转型
    fn as_any(&self) -> &dyn Any;
    
    /// 从字节重建键
    fn from_bytes(bytes: &[u8]) -> Option<Self> where Self: Sized;
    
    // 添加可读字符串表示方法
    fn as_str(&self) -> Cow<'_, str> {
        std::str::from_utf8(self.as_bytes())
            .map(Cow::Borrowed)
            .unwrap_or_else(|_| {
                Cow::Owned(format!("<invalid_utf8: {:?}>", self.as_bytes()))
            })
    }
}

/// 值类型 - 要求可克隆
pub trait Value: Clone + Debug + 'static {
     /// 获取值的字节表示
    fn as_bytes(&self) -> &[u8];
    
    /// 从字节重建值
    fn from_bytes(bytes: &[u8]) -> Option<Self>;
}

// 为Vec<u8>实现Value特征
impl Value for Vec<u8> {
    fn as_bytes(&self) -> &[u8] {
        self.as_slice()
    }
    
    fn from_bytes(bytes: &[u8]) -> Option<Self> {
        Some(bytes.to_vec())
    }
}

/// 桶内槽位ID
#[derive(Clone, Copy, PartialEq, Eq, Hash)]
pub struct SlotId(pub usize);

/// 指纹类型 - 10位无符号整数
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[repr(transparent)]
pub struct Fingerprint(u16);

impl Fingerprint {
    /// 创建新指纹 (只取低10位)
    pub const fn new(value: u16) -> Self {
        Self(value & 0x3FF) // 0x3FF = 1023 (10位掩码)
    }
    
    /// 返回零指纹（表示空槽位）
    pub const fn zero() -> Self {
        Self(0)
    }
    
    /// 从哈希值创建指纹
    pub fn from_hash(hash: u64) -> Self {
        // 使用哈希的低10位
        Self((hash & 0x3FF) as u16)
    }
    
    /// 获取指纹值
    pub const fn as_u16(&self) -> u16 {
        self.0
    }
    
    /// 转换为 i16 (用于SIMD指令)
    pub fn as_i16(self) -> i16 {
        self.0 as i16
    }
    
    /// 获取原始值的原始指针
    pub fn as_ptr(&self) -> *const u16 {
        &self.0 as *const u16
    }
    
    /// 检查是否为零（空槽位）
    pub const fn is_zero(&self) -> bool {
        self.0 == 0
    }
    
    /// 转换到SIMD向量
    #[cfg(feature = "simd")]
    pub fn to_simd(&self) -> u16x4 {
        u16x4::splat(self.0)
    }
}

impl fmt::Display for Fingerprint {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{:03X}", self.0)
    }
}

#[repr(transparent)]
#[derive(Debug)]
pub struct AtomicSlotState(AtomicU32);

impl AtomicSlotState {
    const FINGERPRINT_MASK: u32 = 0x0000_03FF; // 位0-9 (10位)
    const VERSION_MASK: u32 = 0x3FFF_C000;     // 位10-25 (16位)
    const STATE_MASK: u32 = 0xFC00_0000;       // 位26-31 (6位)

    /// 打包状态为32位整数
fn pack_state(state: SlotState) -> u32 {
    (state.fingerprint.0 as u32) |           // 位0-9
    ((state.version as u32) << 10) |         // 位10-25
    ((state.state_flags.bits as u32) << 26)   // 位26-31
}
    
    /// 从32位整数解包状态
fn unpack_state(packed: u32) -> SlotState {
    // 提取指纹 (位0-9)
    let fingerprint = Fingerprint::new((packed & Self::FINGERPRINT_MASK) as u16);
    
    // 提取版本 (位10-25)
    let version = ((packed >> 10) & 0xFFFF) as u16; // 右移10位后取16位
    
    // 提取状态标志 (位26-31)
    let state_flags = SlotStateFlags {
        bits: ((packed >> 26) & 0x3F) as u8 // 右移26位后取6位
    };
    
    SlotState {
        fingerprint,
        version,
        state_flags,
    }
}
    /// 创建新的原子槽位状态
    pub fn new(initial: SlotState) -> Self {
        let packed = Self::pack_state(initial);
        AtomicSlotState(AtomicU32::new(packed))
    }

    /// 获取当前状态
    pub fn load(&self, order: Ordering) -> SlotState {
        let packed = self.0.load(order);
        Self::unpack_state(packed)
    }

    /// 比较并交换状态
    pub fn compare_exchange(
        &self,
        current: SlotState,
        new: SlotState,
        success: Ordering,
        failure: Ordering,
    ) -> Result<(), SlotState> {
        let current_packed = Self::pack_state(current);
        let new_packed = Self::pack_state(new);
        
        match self.0.compare_exchange(current_packed, new_packed, success, failure) {
            Ok(_) => Ok(()),
            Err(actual) => Err(Self::unpack_state(actual)),
        }
    }
    
    /// 存储状态
    pub fn store(&self, state: SlotState, order: Ordering) {
        let packed = Self::pack_state(state);
        self.0.store(packed, order);
    }
    
    
}

/// 槽位状态
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct SlotState {
    pub fingerprint: Fingerprint,
    pub version: u16,
    pub state_flags: SlotStateFlags,
}

impl SlotState {
    /// 创建空状态
    pub const fn empty() -> Self {
        Self {
            fingerprint: Fingerprint::zero(),
            version: 0,
            state_flags: SlotStateFlags::empty(),
        }
    }
    
    /// 检查是否为空槽位
    pub fn is_empty(&self) -> bool {
        self.state_flags.is_empty()
    }
    
    /// 检查是否被占用
    pub fn is_occupied(&self) -> bool {
        self.state_flags.is_occupied()
    }
    
    /// 检查是否被删除
    pub fn is_deleted(&self) -> bool {
        self.state_flags.is_deleted()
    }
    
    /// 检查是否在迁移中
    pub fn is_migrating(&self) -> bool {
        self.state_flags.is_migrating()
    }
    
    /// 检查是否被锁定
    pub fn is_locked(&self) -> bool {
        self.state_flags.is_locked()
    }
}

/// 槽位状态标志 (6位)
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct SlotStateFlags {
    /// 状态位标志
    pub bits: u8,
}

impl SlotStateFlags {
    /// 空槽位 (所有标志位为0)
    pub const EMPTY: u8 = 0b0000_0000;
    
    /// 已占用槽位
    pub const OCCUPIED: u8 = 0b0000_0001;
    
    /// 已删除槽位
    pub const DELETED: u8 = 0b0000_0010;
    
    /// 迁移中槽位
    pub const MIGRATING: u8 = 0b0000_0100;
    
    /// 锁定槽位
    pub const LOCKED: u8 = 0b0000_1000;
    
    /// 访问热度标志 (高频访问)
    pub const ACCESS_HOT: u8 = 0b0001_0000;
    
    /// 需要持久化标志
    pub const NEEDS_PERSIST: u8 = 0b0010_0000;
    
    /// 创建空标志
    pub const fn empty() -> Self {
        Self { bits: Self::EMPTY }
    }
    
    /// 检查是否包含指定标志
    pub fn contains(&self, flags: u8) -> bool {
        (self.bits & flags) == flags
    }
    
    /// 添加标志
    pub fn insert(&mut self, flags: u8) {
        self.bits |= flags;
    }
    
    /// 移除标志
    pub fn remove(&mut self, flags: u8) {
        self.bits &= !flags;
    }
    
    /// 检查是否为空
    pub fn is_empty(&self) -> bool {
        self.bits == Self::EMPTY
    }
    
    /// 检查是否被占用
    pub fn is_occupied(&self) -> bool {
        self.contains(Self::OCCUPIED)
    }
    
    /// 检查是否被删除
    pub fn is_deleted(&self) -> bool {
        self.contains(Self::DELETED)
    }
    
    /// 检查是否在迁移中
    pub fn is_migrating(&self) -> bool {
        self.contains(Self::MIGRATING)
    }
    
    /// 检查是否锁定
    pub fn is_locked(&self) -> bool {
        self.contains(Self::LOCKED)
    }
    
    /// 检查是否为高频访问
    pub fn is_access_hot(&self) -> bool {
        self.contains(Self::ACCESS_HOT)
    }
    
    /// 检查是否需要持久化
    pub fn needs_persist(&self) -> bool {
        self.contains(Self::NEEDS_PERSIST)
    }
    
    /// 添加标志并返回新实例
    pub fn with_flag(&self, flag: u8) -> Self {
        Self {
            bits: self.bits | flag,
        }
    }
    
    /// 移除标志并返回新实例
    pub fn without_flag(&self, flag: u8) -> Self {
        Self {
            bits: self.bits & !flag,
        }
    }

     /// 将状态标志转换为可读字符串
    pub fn to_string(&self) -> String {
        if self.is_empty() {
            return "Empty".to_string();
        }
        
        let mut states = Vec::new();
        
        // 基本状态（互斥）
        if self.is_occupied() {
            states.push("Occupied");
        } else if self.is_deleted() {
            states.push("Deleted");
        }
        
        // 附加状态（可组合）
        if self.is_migrating() {
            states.push("Migrating");
        }
        
        if self.is_locked() {
            states.push("Locked");
        }
        
        if self.is_access_hot() {
            states.push("Hot");
        }
        
        if self.needs_persist() {
            states.push("NeedsPersist");
        }
        
        // 如果没有识别到任何状态，显示原始位值
        if states.is_empty() {
            return format!("Unknown(0b{:08b})", self.bits);
        }
        
        states.join(" | ")
    }
}

/// 内存池句柄
#[derive(Debug)]
pub struct PoolHandle(usize);

/// 桶快照
pub struct BucketSnapshot<K, V> {
    pub entries: Vec<Option<(K, V)>>,
    pub versions: Vec<u16>,
}

/// 全局配置
#[derive(Debug, Clone)]
pub struct GlobalConfig {
    pub initial_capacity: usize,
    pub max_load_factor: f32,
    pub max_kick_depth: usize,
    pub slot_count_per_bucket: usize,
    pub fingerprint_bits: u8,    // 指纹位数 (10)
    pub version_bits: u8,        // 版本号位数 (16)
    pub state_flags_bits: u8,    // 状态标志位数 (6)
}

impl Default for GlobalConfig {
    fn default() -> Self {
        Self {
            initial_capacity: 1024,
            max_load_factor: 0.95,
            max_kick_depth: 32,
            slot_count_per_bucket: 4,
            fingerprint_bits: 10,
            version_bits: 16,
            state_flags_bits: 6,
        }
    }
}

/// 操作保护器 - 用于检测并发修改
#[derive(Debug, Clone)]
pub struct OperationGuard {
    version: u64,
    start_time: SystemTime,
}

impl OperationGuard {
    pub fn new(version: u64) -> Self {
        Self {
            version,
            start_time: SystemTime::now(),
        }
    }
    
    pub fn version(&self) -> u64 {
        self.version
    }
    
    pub fn elapsed(&self) -> Duration {
        self.start_time.elapsed()
            .unwrap_or_else(|_| Duration::from_secs(0))
    }
}

/// 迁移配置
#[derive(Debug, Clone)]
pub struct MigrationConfig {
    pub batch_size: usize,
    pub parallelism: usize,
    pub enable_parallel: bool,
}

/// 原子操作统计
#[derive(Debug)]
pub struct AtomicOpStats {
    count: AtomicU64,
    latency_sum: AtomicU64,
}

impl AtomicOpStats {
    pub fn record(&self, duration: Duration) {
        self.count.fetch_add(1, Ordering::Relaxed);
        self.latency_sum.fetch_add(
            duration.as_nanos() as u64,
            Ordering::Relaxed
        );
    }
}

// 为内置类型实现Key特性
impl Key for String {
    fn as_bytes(&self) -> &[u8] {
        self.as_bytes()
    }
    
    fn from_bytes(bytes: &[u8]) -> Option<Self> {
        String::from_utf8(bytes.to_vec()).ok()
    }
    
    fn clone_key(&self) -> Box<dyn Key> {
        Box::new(self.clone())
    }
    
    fn eq_key(&self, other: &dyn Key) -> bool {
        if let Some(other_str) = other.as_any().downcast_ref::<String>() {
            self == other_str
        } else {
            false
        }
    }
    
    fn hash_key(&self) -> u64 {
        let mut hasher = std::collections::hash_map::DefaultHasher::new();
        self.as_bytes().hash(&mut hasher);
        hasher.finish()
    }
    
    fn as_any(&self) -> &dyn Any {
        self
    }
}

impl Key for u64 {
    fn as_bytes(&self) -> &[u8] {
        unsafe {
            std::slice::from_raw_parts(
                self as *const u64 as *const u8,
                std::mem::size_of::<u64>()
            )
        }
    }
    
    fn from_bytes(bytes: &[u8]) -> Option<Self> {
        if bytes.len() == std::mem::size_of::<u64>() {
            let mut arr = [0u8; 8];
            arr.copy_from_slice(bytes);
            Some(u64::from_ne_bytes(arr))
        } else {
            None
        }
    }
    
    fn clone_key(&self) -> Box<dyn Key> {
        Box::new(*self)
    }
    
    fn eq_key(&self, other: &dyn Key) -> bool {
        if let Some(other_num) = other.as_any().downcast_ref::<u64>() {
            self == other_num
        } else {
            false
        }
    }
    
    fn hash_key(&self) -> u64 {
        use std::hash::{Hash, Hasher};
        let mut hasher = std::collections::hash_map::DefaultHasher::new();
        self.hash(&mut hasher);
        hasher.finish()
    }
    
    fn as_any(&self) -> &dyn Any {
        self
    }
}

/// 字节键包装类型
#[derive(Clone, PartialEq, Eq, Hash)]
pub struct ByteKey(pub Vec<u8>);

impl ByteKey {
    /// 创建新字节键
    pub fn new(bytes: Vec<u8>) -> Self {
        Self(bytes)
    }
    
    /// 获取内部字节
    pub fn into_inner(self) -> Vec<u8> {
        self.0
    }
}

impl Debug for ByteKey {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "ByteKey(")?;
        for byte in &self.0 {
            write!(f, "{:02X}", byte)?;
        }
        write!(f, ")")
    }
}

impl Display for ByteKey {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        for byte in &self.0 {
            write!(f, "{:02X}", byte)?;
        }
        Ok(())
    }
}

impl Key for ByteKey {
    fn as_bytes(&self) -> &[u8] {
        &self.0
    }
    
    fn from_bytes(bytes: &[u8]) -> Option<Self> {
        Some(ByteKey(bytes.to_vec()))
    }
    
    fn eq_key(&self, other: &dyn Key) -> bool {
        if let Some(other_key) = other.as_any().downcast_ref::<ByteKey>() {
            self == other_key
        } else {
            false
        }
    }
    
    fn hash_key(&self) -> u64 {
        let mut hasher = std::collections::hash_map::DefaultHasher::new();
        self.0.hash(&mut hasher);
        hasher.finish()
    }
    
    fn as_any(&self) -> &dyn std::any::Any {
        self
    }
    
    fn clone_key(&self) -> Box<dyn Key> {
        Box::new(self.clone())
    }
}

impl PartialOrd for ByteKey {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        Some(self.cmp(other))
    }
}

impl Ord for ByteKey {
    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        self.0.cmp(&other.0)
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum OperationType {
    /// 插入操作
    Insert,
    /// 获取操作
    Get,
    /// 删除操作
    Remove,
    /// 更新操作
    Update,
    /// 踢出操作
    Kick,
    /// 迁移操作
    Migration,
    /// 调整大小操作
    Resize,
    /// 读取操作
    Read,
    /// 写入操作
    Write,
    /// 统计操作
    Statistics,
    /// 迭代操作
    Iterate,
}

impl OperationType {
    /// 判断是否为读操作
    pub fn is_read(&self) -> bool {
        matches!(
            self,
            OperationType::Get | OperationType::Read | OperationType::Statistics
        )
    }
    
    /// 判断是否为写操作
    pub fn is_write(&self) -> bool {
        matches!(
            self,
            OperationType::Insert | OperationType::Remove | OperationType::Update | 
            OperationType::Kick | OperationType::Migration | OperationType::Resize | 
            OperationType::Write
        )
    }
    
    /// 转换为字符串表示
    pub fn as_str(&self) -> &'static str {
        match self {
            OperationType::Insert => "insert",
            OperationType::Get => "get",
            OperationType::Remove => "remove",
            OperationType::Update => "update",
            OperationType::Kick => "kick",
            OperationType::Migration => "migration",
            OperationType::Resize => "resize",
            OperationType::Read => "read",
            OperationType::Write => "write",
            OperationType::Statistics => "statistics",
            OperationType::Iterate => "iterate",
        }
    }
}

// 单元测试
#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::atomic::Ordering;

   #[test]
fn test_slot_state_packing() {
    let state = SlotState {
        fingerprint: Fingerprint::new(0x2AB), // 0x2AB = 683
        version: 0x1234, // 4660
        state_flags: SlotStateFlags { 
            bits: SlotStateFlags::LOCKED | SlotStateFlags::OCCUPIED // 9
        },
    };
    
    // 手动计算打包值
    let packed = AtomicSlotState::pack_state(state);
    println!("Packed state: 0x{:X}", packed);
    
    // 预期打包值: 
    // 指纹 (0-9位): 0x2AB
    // 版本 (10-25位): 0x1234 << 10 = 0x48D000
    // 状态 (26-31位): 9 << 26 = 0x2400_0000
    // 总和: 0x2AB + 0x48D000 + 0x2400_0000 = 0x2448_D2AB
    assert_eq!(packed, 0x2448_D2AB, "打包状态值不正确");
    
    let atomic = AtomicSlotState::new(state);
    let loaded = atomic.load(Ordering::Acquire);
    
    // 验证解包后的值
    assert_eq!(loaded.fingerprint.as_u16(), 0x2AB, "指纹不匹配");
    assert_eq!(loaded.version, 0x1234, "版本号不匹配");
    assert!(loaded.state_flags.contains(SlotStateFlags::LOCKED), "锁定标志未设置");
    assert!(loaded.state_flags.contains(SlotStateFlags::OCCUPIED), "占用标志未设置");
}

    #[test]
    fn test_atomic_state_update() {
        let atomic = AtomicSlotState::new(SlotState {
            fingerprint: Fingerprint::zero(),
            version: 0,
            state_flags: SlotStateFlags::empty(),
        });
        
        let new_state = SlotState {
            fingerprint: Fingerprint::new(0x3CD), // 0x3CD = 973
            version: 1,
            state_flags: SlotStateFlags { bits: SlotStateFlags::OCCUPIED },
        };
        
        // 首次更新
        atomic.compare_exchange(
            SlotState { 
                fingerprint: Fingerprint::zero(), 
                version: 0, 
                state_flags: SlotStateFlags::empty() 
            },
            new_state,
            Ordering::Release,
            Ordering::Relaxed
        ).unwrap();
        
        let loaded = atomic.load(Ordering::Acquire);
        assert_eq!(loaded.fingerprint.as_u16(), 0x3CD);
        assert_eq!(loaded.version, 1);
        assert!(loaded.state_flags.contains(SlotStateFlags::OCCUPIED));
    }
    
    #[test]
    fn test_fingerprint_truncation() {
        // 测试指纹截断 (只保留低10位)
        let fp = Fingerprint::new(0xFFFF); // 0xFFFF & 0x3FF = 0x3FF
        assert_eq!(fp.as_u16(), 0x3FF);
        
        let fp = Fingerprint::new(0x1234); // 0x1234 & 0x3FF = 0x234
        assert_eq!(fp.as_u16(), 0x234);
    }
    
    #[test]
    fn test_state_flags() {
        let mut flags = SlotStateFlags::empty();
        assert!(flags.is_empty());
        
        flags.insert(SlotStateFlags::OCCUPIED);
        assert!(flags.is_occupied());
        assert!(!flags.is_deleted());
        
        flags.insert(SlotStateFlags::DELETED);
        assert!(flags.is_deleted());
        
        flags.remove(SlotStateFlags::OCCUPIED);
        assert!(!flags.is_occupied());
        assert!(flags.is_deleted());
        
        flags.insert(SlotStateFlags::ACCESS_HOT);
        assert!(flags.is_access_hot());
        
        // 测试 with_flag 和 without_flag
        let new_flags = flags.with_flag(SlotStateFlags::LOCKED);
        assert!(new_flags.is_locked());
        
        let without_deleted = new_flags.without_flag(SlotStateFlags::DELETED);
        assert!(!without_deleted.is_deleted());
    }
}