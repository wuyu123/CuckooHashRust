// src/memory/slot.rs
//! 槽位管理 - 实现原子槽位状态和内存访问

use crate::{
    error::CuckooError,
    types::{AtomicSlotState, SlotState, SlotStateFlags, Fingerprint},
    version::VersionGuard,
};
use std::{
    ptr::NonNull,
    sync::atomic::{AtomicPtr, AtomicUsize, Ordering},
};

/// 槽位元数据 - 描述槽位位置和状态
#[derive(Debug, Clone, Copy)]
pub struct SlotMetadata {
    pub ptr: NonNull<Slot>,
    pub index: usize,
    pub chunk_id: usize,
}

impl SlotMetadata {
    /// 创建新元数据
    pub fn new(ptr: NonNull<Slot>, index: usize, chunk_id: usize) -> Self {
        Self { ptr, index, chunk_id }
    }
}

/// 槽位内存句柄 - 提供对槽位内存的安全访问
#[derive(Debug)]
pub struct SlotHandle {
    metadata: SlotMetadata,
    _marker: std::marker::PhantomData<*mut ()>, // 确保!Send和!Sync
}

impl SlotHandle {
    /// 创建新槽位句柄
    pub fn new(metadata: SlotMetadata) -> Self {
        Self {
            metadata,
            _marker: std::marker::PhantomData,
        }
    }
    
    /// 获取元数据
    pub fn metadata(&self) -> SlotMetadata {
        self.metadata
    }
    
    /// 获取槽位指针
    pub fn as_ptr(&self) -> *mut Slot {
        self.metadata.ptr.as_ptr()
    }
}

/// 槽位结构 - 管理单个键值对的内存
#[repr(align(64))] // 缓存行对齐
pub struct Slot {
    state: AtomicSlotState,
    key: AtomicPtr<u8>,
    value: AtomicPtr<u8>,
    key_size: AtomicUsize,
    value_size: AtomicUsize,
}

impl Slot {
    /// 创建新槽位
    pub fn new() -> Self {
        Self {
            state: AtomicSlotState::new(SlotState::empty()),
            key: AtomicPtr::new(std::ptr::null_mut()),
            value: AtomicPtr::new(std::ptr::null_mut()),
            key_size: AtomicUsize::new(0),
            value_size: AtomicUsize::new(0),
        }
    }
    
    /// 尝试锁定槽位
    pub fn try_lock(&self, expected_fp: Fingerprint, guard: &VersionGuard) -> Result<SlotGuard, CuckooError> {
        let current = self.state.load(Ordering::Acquire);
        
        // 检查槽位是否为空（指纹为0是特殊值）
        let is_empty = current.fingerprint == Fingerprint::zero();
        
        // 检查指纹匹配或槽位为空
        if (!is_empty && current.fingerprint != expected_fp) 
            || current.state_flags.contains(SlotStateFlags::LOCKED) 
        {
            return Err(CuckooError::LockContention{operation: "try_lock".to_string()});
        }
        
        // 尝试设置锁定位
        let new_state = SlotState {
            fingerprint: current.fingerprint,
            version: guard.version() as u16,
            state_flags: current.state_flags.with_flag(SlotStateFlags::LOCKED),
        };
        
        match self.state.compare_exchange(
            current,
            new_state,
            Ordering::AcqRel,
            Ordering::Acquire
        ) {
            Ok(_) => Ok(SlotGuard::new(self, new_state)),
            Err(updated) => {
                if updated.fingerprint != expected_fp && updated.fingerprint != Fingerprint::zero() {
                    Err(CuckooError::HashConflict)
                } else {
                    Err(CuckooError::LockContention{operation: "try_lock".to_string()})
                }
            }
        }
    }
    
    /// 原子加载键值
    pub fn load(&self) -> (Vec<u8>, Vec<u8>) {
        // 使用原子操作确保内存可见性
        let key_ptr = self.key.load(Ordering::Acquire);
        let key_size = self.key_size.load(Ordering::Acquire);
        let value_ptr = self.value.load(Ordering::Acquire);
        let value_size = self.value_size.load(Ordering::Acquire);
        
        let key = if !key_ptr.is_null() && key_size > 0 {
            unsafe {
                // 创建数据的拷贝而不是接管所有权
                std::slice::from_raw_parts(key_ptr, key_size).to_vec()
            }
        } else {
            Vec::new()
        };
        
        let value = if !value_ptr.is_null() && value_size > 0 {
            unsafe {
                // 创建数据的拷贝而不是接管所有权
                std::slice::from_raw_parts(value_ptr, value_size).to_vec()
            }
        } else {
            Vec::new()
        };
        
        (key, value)
    }
    
    /// 原子存储键值
    pub fn store(&self, key: &[u8], value: &[u8]) {
        // 确保在锁定状态下调用
        let state = self.state.load(Ordering::Relaxed);
        debug_assert!(state.state_flags.contains(SlotStateFlags::LOCKED));
        
        // 释放旧内存
        self.free_key();
        self.free_value();
        
        // 分配新内存
        let key_ptr = if !key.is_empty() {
            let vec = key.to_vec();
            let ptr = vec.as_ptr() as *mut u8;
            std::mem::forget(vec);
            ptr
        } else {
            std::ptr::null_mut()
        };
        
        let value_ptr = if !value.is_empty() {
            let vec = value.to_vec();
            let ptr = vec.as_ptr() as *mut u8;
            std::mem::forget(vec);
            ptr
        } else {
            std::ptr::null_mut()
        };
        
        // 原子存储
        self.key.store(key_ptr, Ordering::Relaxed);
        self.key_size.store(key.len(), Ordering::Relaxed);
        self.value.store(value_ptr, Ordering::Relaxed);
        self.value_size.store(value.len(), Ordering::Relaxed);

        // 标记为已占用
        let mut current = self.state.load(Ordering::Acquire);
        loop {
            let new_state = SlotState {
                state_flags: current.state_flags.with_flag(SlotStateFlags::OCCUPIED),
                ..current
            };
            
            match self.state.compare_exchange(
                current,
                new_state,
                Ordering::AcqRel,
                Ordering::Acquire
            ) {
                Ok(_) => break,
                Err(updated) => current = updated,
            }
        }
    }
    
    /// 获取当前指纹
    pub fn load_fingerprint(&self) -> Fingerprint {
        // 使用更轻量的内存顺序
        self.state.load(Ordering::Acquire).fingerprint
    }
    
    /// 检查是否为空
    pub fn is_empty(&self) -> bool {
        // 使用更轻量的内存顺序
        let state = self.state.load(Ordering::Relaxed);
        // 槽位为空当且仅当未被占用且未锁定
        !state.state_flags.contains(SlotStateFlags::OCCUPIED) && !state.state_flags.contains(SlotStateFlags::LOCKED)
    }

    pub fn state(&self) -> SlotState {
        self.state.load(Ordering::Acquire)
    }

    pub fn state_flags(&self) -> SlotStateFlags {
        self.state.load(Ordering::Acquire).state_flags
    }
    
    /// 标记为迁移中
    pub fn mark_migrating(&self) {
        let mut current = self.state.load(Ordering::Acquire);
        loop {
            let new_state = SlotState {
                state_flags: current.state_flags.with_flag(SlotStateFlags::MIGRATING),
                ..current
            };
            
            match self.state.compare_exchange(
                current,
                new_state,
                Ordering::AcqRel,
                Ordering::Acquire
            ) {
                Ok(_) => break,
                Err(updated) => current = updated,
            }
        }
    }
    
    /// 清空槽位
    pub fn clear(&self) {
        self.free_key();
        self.free_value();
        // 清除状态：设置为空且未锁定
        self.state.store(SlotState::empty(), Ordering::Release);
    }
    
    /// 释放键内存
    fn free_key(&self) {
        let key_ptr = self.key.swap(std::ptr::null_mut(), Ordering::Acquire);
        if !key_ptr.is_null() {
            let size = self.key_size.load(Ordering::Acquire);
            unsafe {
                // 正确释放内存
                let _ = Vec::from_raw_parts(key_ptr, size, size);
            }
        }
    }
    
    /// 释放值内存
    fn free_value(&self) {
        let value_ptr = self.value.swap(std::ptr::null_mut(), Ordering::Acquire);
        if !value_ptr.is_null() {
            let size = self.value_size.load(Ordering::Acquire);
            unsafe {
                // 正确释放内存
                let _ = Vec::from_raw_parts(value_ptr, size, size);
            }
        }
    }

    pub fn is_occupied(&self) -> bool {
        let state = self.state.load(Ordering::Acquire);
        state.state_flags.contains(SlotStateFlags::OCCUPIED)
    }

    /// 打印槽位的简洁信息
    pub fn print_summary(&self, slot_index: usize) {
        let state = self.state.load(Ordering::Acquire);
        let state_flags = state.state_flags;
                
        let key_summary = self.key_summary();
        
        println!(
            "Slot {:02}: [{}] FP: {:?}, {}",
            slot_index,
            state_flags.to_string(),
            state.fingerprint,
            key_summary
        );
    }

    /// 格式化数据为可读形式
    fn format_data(&self, data: &[u8], prefix: &str) -> String {
        // 尝试解释为字符串
        if let Ok(data_str) = std::str::from_utf8(data) {
            if data_str.chars().all(|c| c.is_ascii() && !c.is_control()) {
                return if data_str.len() > 16 {
                    format!("{}: \"{}...\" ({} bytes)", prefix, &data_str[..16], data.len())
                } else {
                    format!("{}: \"{}\"", prefix, data_str)
                };
            }
        }
        
        // 尝试解释为数字
        if data.len() == 4 {
            let num = u32::from_be_bytes([data[0], data[1], data[2], data[3]]);
            return format!("{}: {}", prefix, num);
        } else if data.len() == 8 {
            let num = u64::from_be_bytes([
                data[0], data[1], data[2], data[3],
                data[4], data[5], data[6], data[7]
            ]);
            return format!("{}: {}", prefix, num);
        }
        
        // 默认显示为十六进制
        if data.len() > 8 {
            format!("{}: 0x{}... ({} bytes)", prefix, hex::encode(&data[..8]), data.len())
        } else {
            format!("{}: 0x{}", prefix, hex::encode(data))
        }
    }

    /// 获取键的摘要
    fn key_summary(&self) -> String {
        let key_ptr = self.key.load(Ordering::Acquire);
        let key_size = self.key_size.load(Ordering::Acquire);
        
        if !key_ptr.is_null() && key_size > 0 {
            unsafe {
                let key_slice = std::slice::from_raw_parts(key_ptr, key_size);
                self.format_data(key_slice, "Key")
            }
        } else {
            "Key: <empty>".to_string()
        }
    }
}

/// 槽位保护结构 - 确保状态改变后正确释放
pub struct SlotGuard<'a> {
    slot: &'a Slot,
    state: SlotState,
}

impl<'a> SlotGuard<'a> {
    /// 创建新保护器
    pub fn new(slot: &'a Slot, state: SlotState) -> Self {
        Self { slot, state }
    }
    
    /// 检查键是否匹配
    pub fn key_matches(&self, key: &[u8]) -> bool {
        let key_ptr = self.slot.key.load(Ordering::Relaxed);
        let key_size = self.slot.key_size.load(Ordering::Relaxed);
        
        // 如果键指针为空，那么只有当键长度为0时才匹配
        if key_ptr.is_null() {
            return key.is_empty();
        }
        
        if key_size != key.len() {
            return false;
        }
        
        unsafe {
            let stored_key = std::slice::from_raw_parts(key_ptr, key_size);
            stored_key == key
        }
    }
    
    /// 加载键
    pub fn load_key(&self) -> Option<Vec<u8>> {
        let key_ptr = self.slot.key.load(Ordering::Relaxed);
        let key_size = self.slot.key_size.load(Ordering::Relaxed);
        
        if key_ptr.is_null() || key_size == 0 {
            return None;
        }
        
        unsafe {
            // 创建数据的拷贝而不是接管所有权
            let slice = std::slice::from_raw_parts(key_ptr, key_size);
            Some(slice.to_vec())
        }
    }
    
    /// 加载值
    pub fn load_value(&self) -> Option<Vec<u8>> {
        let value_ptr = self.slot.value.load(Ordering::Relaxed);
        let value_size = self.slot.value_size.load(Ordering::Relaxed);
        
        if value_ptr.is_null() || value_size == 0 {
            return None;
        }
        
        unsafe {
            // 创建数据的拷贝而不是接管所有权
            let slice = std::slice::from_raw_parts(value_ptr, value_size);
            Some(slice.to_vec())
        }
    }
    
    /// 提交更改并更新指纹
    pub fn commit(self, new_fp: Fingerprint) {
        // 确保在提交前数据已写入
        std::sync::atomic::fence(Ordering::Release);
        
        // 创建新状态：更新指纹并清除锁定状态
        let mut state_flags = self.state.state_flags;
        state_flags.remove(SlotStateFlags::LOCKED);
        
        // 确保占用标志被设置
        state_flags = state_flags.with_flag(SlotStateFlags::OCCUPIED);

        let new_state = SlotState {
            fingerprint: new_fp,
            version: self.state.version,
            state_flags,
        };
        
        // 原子存储新状态
        self.slot.state.store(new_state, Ordering::Release);
        
        // 确保状态更新对其他线程可见
        std::sync::atomic::fence(Ordering::SeqCst);
        std::mem::forget(self);
    }

    /// 检查槽位是否被占用（锁定前的状态）
    pub fn is_occupied(&self) -> bool {
        self.state.state_flags.contains(SlotStateFlags::OCCUPIED)
    }
}

impl Drop for SlotGuard<'_> {
    fn drop(&mut self) {
        if !std::thread::panicking() {
            // 清除锁定状态
            let mut new_state = self.state;
            new_state.state_flags.remove(SlotStateFlags::LOCKED);
            self.slot.state.store(new_state, Ordering::Release);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{types::OperationType, version::VersionTracker};
    use std::sync::Arc;

    /// 创建测试用版本守卫
    fn test_version_guard() -> VersionGuard {
        let tracker = Arc::new(VersionTracker::new());
        tracker.begin_operation(OperationType::Get)
    }

    #[test]
    fn test_slot_creation() {
        let slot = Slot::new();
        assert!(slot.is_empty());
        assert_eq!(slot.load_fingerprint(), Fingerprint::zero());
    }

    #[test]
    fn test_slot_lock() {
        let slot = Slot::new();
        let fp = Fingerprint::new(42);
        let version_guard = test_version_guard();
        
        // 成功锁定（空槽位）
        let guard = slot.try_lock(fp, &version_guard).expect("锁定失败");
        assert!(guard.key_matches(&[]));
        
        // 尝试再次锁定应失败
        assert!(slot.try_lock(fp, &version_guard).is_err());
        
        // 释放后应可再次锁定
        drop(guard);
        assert!(slot.try_lock(fp, &version_guard).is_ok());
    }

    #[test]
    fn test_slot_store_and_load() {
        let slot = Slot::new();
        let version_guard = test_version_guard();
        let fp = Fingerprint::new(42);
        
        // 锁定槽位
        let guard = slot.try_lock(fp, &version_guard).expect("锁定失败");
        
        // 存储键值对
        slot.store(b"key", b"value");
        
        // 验证键值
        assert!(guard.key_matches(b"key"));
        assert_eq!(guard.load_key(), Some(b"key".to_vec()));
        assert_eq!(guard.load_value(), Some(b"value".to_vec()));
        
        // 提交更改并更新指纹
        guard.commit(fp);
        
        // 验证指纹更新
        assert_eq!(slot.load_fingerprint(), fp);
    }

    #[test]
    fn test_slot_mark_migrating() {
        let slot = Slot::new();
        
        // 初始状态应为空
        let state = slot.state.load(Ordering::Relaxed);
        assert!(!state.state_flags.contains(SlotStateFlags::MIGRATING));
        
        // 标记为迁移中
        slot.mark_migrating();
        let state = slot.state.load(Ordering::Relaxed);
        assert!(state.state_flags.contains(SlotStateFlags::MIGRATING));
    }

    #[test]
    fn test_slot_clear() {
        let slot = Slot::new();
        let version_guard = test_version_guard();
        let fp = Fingerprint::new(42);
        
        // 填充槽位
        let guard = slot.try_lock(fp, &version_guard).expect("锁定失败");
        slot.store(b"key", b"value");
        guard.commit(fp);
        
        // 验证非空
        assert!(!slot.is_empty());
        
        // 清空槽位
        slot.clear();
        
        // 验证为空
        assert!(slot.is_empty());
        assert_eq!(slot.load_fingerprint(), Fingerprint::zero());
    }

    #[test]
    fn test_slot_guard_commit() {
        let slot = Slot::new();
        let version_guard = test_version_guard();
        let fp = Fingerprint::new(42);
        
        // 获取保护器
        let guard = slot.try_lock(fp, &version_guard).expect("锁定失败");
        
        // 提交更改
        guard.commit(fp);
        
        // 验证锁定状态已清除
        assert!(slot.try_lock(fp, &version_guard).is_ok());
    }

    #[test]
    fn test_slot_guard_drop() {
        let slot = Slot::new();
        let version_guard = test_version_guard();
        let fp = Fingerprint::new(42);
        
        {
            // 获取保护器
            let _guard = slot.try_lock(fp, &version_guard).expect("锁定失败");
            
            // 尝试再次锁定应失败
            assert!(slot.try_lock(fp, &version_guard).is_err());
        }
        
        // 保护器释放后应可再次锁定
        assert!(slot.try_lock(fp, &version_guard).is_ok());
    }

    #[test]
    fn test_key_matching() {
        let slot = Slot::new();
        let version_guard = test_version_guard();
        let fp = Fingerprint::new(42);
        
        // 锁定槽位
        let guard = slot.try_lock(fp, &version_guard).expect("锁定失败");
        
        // 存储键值对
        slot.store(b"test_key", b"value");
        
        // 验证键匹配
        assert!(guard.key_matches(b"test_key"));
        assert!(!guard.key_matches(b"wrong_key"));
        
        // 提交更改
        guard.commit(fp);
    }

    #[test]
    fn test_slot_load() {
        let slot = Slot::new();
        let version_guard = test_version_guard();
        let fp = Fingerprint::new(42);
        
        // 锁定槽位
        let guard = slot.try_lock(fp, &version_guard).expect("锁定失败");
        
        // 存储键值对
        slot.store(b"key", b"value");
        
        // 验证加载
        let (key, value) = slot.load();
        assert_eq!(key, b"key");
        assert_eq!(value, b"value");
        
        // 提交更改
        guard.commit(fp);
    }

    #[test]
    fn test_slot_free_memory() {
        let slot = Slot::new();
        let version_guard = test_version_guard();
        let fp = Fingerprint::new(42);
        
        // 锁定槽位
        let guard = slot.try_lock(fp, &version_guard).expect("锁定失败");
        
        // 分配内存
        slot.store(b"key", b"value");
        
        // 验证内存分配
        assert!(!slot.key.load(Ordering::Relaxed).is_null());
        assert!(!slot.value.load(Ordering::Relaxed).is_null());
        
        // 释放内存
        slot.free_key();
        slot.free_value();
        
        // 验证内存释放
        assert!(slot.key.load(Ordering::Relaxed).is_null());
        assert!(slot.value.load(Ordering::Relaxed).is_null());
        
        // 提交更改
        guard.commit(fp);
    }

    #[test]
    fn test_slot_replacement() {
        let slot = Slot::new();
        let version_guard = test_version_guard();
        let fp = Fingerprint::new(42);
        
        // 锁定槽位
        let guard = slot.try_lock(fp, &version_guard).expect("锁定失败");
        
        // 第一次存储
        slot.store(b"key1", b"value1");
        assert!(guard.key_matches(b"key1"));
        
        // 第二次存储（替换）
        slot.store(b"key2", b"value2");
        assert!(guard.key_matches(b"key2"));
        assert!(!guard.key_matches(b"key1"));
        
        // 提交更改
        guard.commit(fp);
    }

    #[test]
    fn test_slot_empty_key_value() {
        let slot = Slot::new();
        let version_guard = test_version_guard();
        let fp = Fingerprint::new(42);
        
        // 锁定槽位
        let guard = slot.try_lock(fp, &version_guard).expect("锁定失败");
        
        // 存储空键值
        slot.store(&[], &[]);
        
        // 验证键值
        assert!(guard.key_matches(&[]));
        assert_eq!(guard.load_key(), None);
        assert_eq!(guard.load_value(), None);
        
        // 提交更改
        guard.commit(fp);
    }
    
    #[test]
    fn test_large_fingerprint() {
        let slot = Slot::new();
        let version_guard = test_version_guard();
        let fp = Fingerprint::new(0x3FF); // 最大10位指纹
        
        // 锁定槽位
        let guard = slot.try_lock(fp, &version_guard).expect("锁定失败");
        
        // 存储键值对
        slot.store(b"key", b"value");
        
        // 提交更改
        guard.commit(fp);
        
        // 验证指纹更新
        assert_eq!(slot.load_fingerprint(), fp);
    }
}