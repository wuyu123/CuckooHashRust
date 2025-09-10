// src/map/bucket.rs
//! 桶实现 - 管理一组槽位和并发访问

use crate::{
    error::CuckooError,
    memory::Slot,
    simd::SimdSearcher,
    types::{Fingerprint, Key, SlotStateFlags, Value},
    version::{VersionGuard, VersionTracker},
};
use std::{
    fmt, marker::PhantomData, sync::{
        atomic::{AtomicU32, Ordering}, 
        Arc
    }
};

pub const BUCKET_SIZE: usize = 4;

/// 桶状态标志
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BucketStateFlags {
    /// 锁定状态
    Locked = 0b00000001,
    /// 迁移中
    Migrating = 0b00000010,
    /// 已满
    Full = 0b00000100,
}

impl BucketStateFlags {
    /// 检查是否包含标志
    pub fn contains(self, flag: BucketStateFlags) -> bool {
        (self as u8) & (flag as u8) != 0
    }
    
    /// 添加标志
    pub fn with_flag(self, flag: BucketStateFlags) -> Self {
        let value = (self as u8) | (flag as u8);
        unsafe { std::mem::transmute(value) }
    }
    
    /// 移除标志
    pub fn without_flag(self, flag: BucketStateFlags) -> Self {
        let value = (self as u8) & !(flag as u8);
        unsafe { std::mem::transmute(value) }
    }
}

#[repr(align(64))]
pub struct Bucket<K: Key, V: Value> {
    slots: Vec<Arc<Slot>>,
    state: AtomicU32,
    version: Arc<VersionTracker>,
    _key_marker: PhantomData<K>,
    _value_marker: PhantomData<V>,
}
impl<K: Key, V: Value> fmt::Debug for Bucket<K, V> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "Bucket(slots: {})", self.slots.len())
    }
}
impl<K: Key, V: Value> Bucket<K, V> {
    const LOCK_MASK: u32 = BucketStateFlags::Locked as u32;
    const MIGRATING_MASK: u32 = BucketStateFlags::Migrating as u32;

    pub fn new(version: Arc<VersionTracker>) -> Self {
        let slots = (0..BUCKET_SIZE)
            .map(|_| Arc::new(Slot::new()))
            .collect();
        
        Self {
            slots,
            state: AtomicU32::new(0),
            version,
            _key_marker: PhantomData,
            _value_marker: PhantomData,
        }
    }

    pub fn try_lock(&self) -> Result<BucketGuard<'_, K, V>, CuckooError> {
        let state = self.state.load(Ordering::Acquire);
        if state & Self::LOCK_MASK != 0 {
            return Err(CuckooError::LockContention {
                operation: "bucket_lock".into(),
            });
        }

        match self.state.compare_exchange(
            state,
            state | Self::LOCK_MASK,
            Ordering::AcqRel,
            Ordering::Acquire,
        ) {
            Ok(_) => Ok(BucketGuard { bucket: self }),
            Err(_) => Err(CuckooError::LockContention {
                operation: "bucket_lock".into(),
            }),
        }
    }

    /// 返回桶中所有槽位
    pub fn slots(&self) -> &[Arc<Slot>] {
        &self.slots
    }

    pub fn get_slot(&self,slot_idx: usize) -> &Arc<Slot> {
        &self.slots[slot_idx]
    }

    pub fn find_slot(&self, fp: Fingerprint, searcher: &dyn SimdSearcher) -> Option<usize> {
        let fingerprints = [
            self.slots[0].load_fingerprint(),
            self.slots[1].load_fingerprint(),
            self.slots[2].load_fingerprint(),
            self.slots[3].load_fingerprint(),
        ];

        unsafe { searcher.search(&fingerprints, fp) }
    }

    pub fn find_slots(&self, fp: Fingerprint, searcher: &dyn SimdSearcher) -> Vec<usize> {
        let fingerprints = [
            self.slots[0].load_fingerprint(),
            self.slots[1].load_fingerprint(),
            self.slots[2].load_fingerprint(),
            self.slots[3].load_fingerprint(),
        ];

        unsafe { searcher.search_all(&fingerprints, fp) }
    }

    pub fn find_slot_with_key(&self, key: &K, version_guard: &VersionGuard) -> Option<usize> {
        let key_bytes = key.as_bytes();
        let fp = self.version.fingerprint_generator().generate(key_bytes);
        
        for (i, slot) in self.slots.iter().enumerate() {
            if slot.load_fingerprint() == fp {
                let guard = match slot.try_lock(fp, version_guard) {
                    Ok(guard) => guard,
                    Err(_) => continue,
                };
                
                // 检查键是否匹配
                if let Some(slot_key) = guard.load_key() {
                    if slot_key == key_bytes {
                        return Some(i);
                    }
                }
            }
        }
        
        None
    }

    pub fn find_empty_slot(&self) -> Option<usize> {
        for (i, slot) in self.slots.iter().enumerate() {
            if slot.is_empty() {
                return Some(i);
            }
        }
        None
    }


    pub fn insert(&self, index: usize, key: K, value: V, version_guard: &VersionGuard) -> Result<(), CuckooError> {
    let key_bytes = key.as_bytes();
    let value_bytes = value.as_bytes();
    let slot = &self.slots[index];
    
    //println!(    "Inserting key: {:?}, value: {:?} at index: {}",    key_bytes,    value_bytes,    index );
    // 生成指纹并锁定槽位
    let fp = self.version.fingerprint_generator().generate(key_bytes);
    let  guard = slot.try_lock(fp, version_guard)?;
    
    // 存储数据
    slot.store(key_bytes, value_bytes);
    guard.commit(fp);
    
    let (loaded_key, loaded_value) = slot.load();
    //println!(        "Immediately after insert: key={:?}, value={:?}",        loaded_key,        loaded_value    );
    Ok(())
    }

    pub fn get(&self, slot_idx: usize, key: &K, version_guard: &VersionGuard) -> Option<V> {

        
        let slot = &self.slots[slot_idx];
        let fp = slot.load_fingerprint();
        
        let guard = match slot.try_lock(fp, version_guard) {
            Ok(guard) => guard,
            Err(_) => return None,
        };
        // 使用 key_matches 方法检查键是否匹配
    if guard.key_matches(key.as_bytes()) {
        // 加载值
        let value_bytes = guard.load_value()?;
        V::from_bytes(&value_bytes)
    } else {
        None
    }
    }

    pub fn remove(&self, index: usize, key: &K, version_guard: &VersionGuard) -> Option<V> {
        let key_bytes = key.as_bytes();
        let slot = &self.slots[index];
        let fp = slot.load_fingerprint();
        
        let guard = match slot.try_lock(fp, version_guard) {
            Ok(guard) => guard,
            Err(_) => return None,
        };
        
        if guard.key_matches(key_bytes) {
            let value_bytes = guard.load_value()?;
            let value = V::from_bytes(&value_bytes)?;
            slot.clear();
            Some(value)
        } else {
            None
        }
    }

    pub fn mark_migrating(&self) {
        self.state.fetch_or(Self::MIGRATING_MASK, Ordering::SeqCst);
    }

    pub fn unmark_migrating(&self) {
        self.state.fetch_and(!Self::MIGRATING_MASK, Ordering::SeqCst);
    }

    /// 检查是否包含指定状态标志
    pub fn has_state_flag(&self, flag: BucketStateFlags) -> bool {
        let state = self.state.load(Ordering::Relaxed);
        (state & (flag as u32)) != 0
    }
    
    /// 获取当前状态值
    pub fn state_value(&self) -> u32 {
        self.state.load(Ordering::Relaxed)
    }

    /// 测试辅助方法：直接存储键值到指定槽位
    #[cfg(test)]
    pub fn store_direct(&self, slot_idx: usize, key: &K, value: &V) {
        let slot = &self.slots[slot_idx];
        slot.store(key.as_bytes(), value.as_bytes());
    }

    #[cfg(test)]
    pub fn insert_for_test(
        &self, 
        slot_idx: usize, 
        key: &K, 
        value: &V,
        version_guard: &VersionGuard
    ) -> Result<(), CuckooError> {
        let slot = &self.slots[slot_idx];
        
        // 生成指纹并锁定槽位
        let fp = self.version.fingerprint_generator().generate(key.as_bytes());
        let guard = slot.try_lock(fp, version_guard)?;
        
        // 存储数据
        slot.store(key.as_bytes(), value.as_bytes());
        guard.commit(fp);
        
        Ok(())
    }

    #[cfg(test)]
    pub fn fill_for_test(&self, version_guard: &VersionGuard) -> Result<(), CuckooError> {
        for slot_idx in 0..BUCKET_SIZE {

            // 创建符合泛型约束的键值
            let key = K::from_bytes(&format!("key_{}", slot_idx).into_bytes())
                .expect("Failed to create key");
            let value = V::from_bytes(&format!("value_{}", slot_idx).into_bytes())
                .expect("Failed to create value");
            
            self.insert_for_test(slot_idx, &key, &value, version_guard)?;
            
        }
        Ok(())
    }

    pub fn count_occupied_slots(&self) -> usize {
        self.slots.iter().filter(|slot| slot.is_occupied()).count()
    }

    /// 打印桶的简洁摘要
    pub fn print_summary(&self) {
        println!("Bucket Summary:");
        for (i, slot) in self.slots.iter().enumerate() {
            slot.print_summary(i);
        }
        println!("Occupancy: {}/{}", self.count_occupied_slots(), BUCKET_SIZE);
    }
    
    
}

/// 桶保护器 - 确保桶状态改变后正确释放
pub struct BucketGuard<'a, K: Key, V: Value> {
    bucket: &'a Bucket<K, V>,
}

impl<'a, K: Key, V: Value> BucketGuard<'a, K, V> {
    /// 创建新保护器
    pub fn new(bucket: &'a Bucket<K, V>) -> Self {
        Self { bucket }
    }
    
    /// 提交更改
    pub fn commit(self) {
        // 清除锁定状态
        self.bucket.state.fetch_and(!Bucket::<K, V>::LOCK_MASK, Ordering::Release);
    }

    /// 获取槽位列表
    pub fn slots(&self) -> &[Arc<Slot>] {
        self.bucket.slots()
    }
}

impl<'a, K: Key, V: Value> Drop for BucketGuard<'a, K, V> {
    fn drop(&mut self) {
        if !std::thread::panicking() {
            // 清除锁定状态
           self.bucket.state.fetch_and(!Bucket::<K, V>::LOCK_MASK, Ordering::Release);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{
        hash::DefaultFingerprintGenerator,
        simd::Avx2Searcher,
        types::{ByteKey, Fingerprint, OperationType},
        version::VersionTracker,
    };
    use std::sync::Arc;
    use std::thread;

    // 测试用键值对
    fn test_items() -> Vec<(ByteKey, Vec<u8>)> {
        vec![
            (ByteKey(b"key1".to_vec()), b"value1".to_vec()),
            (ByteKey(b"key2".to_vec()), b"value2".to_vec()),
            (ByteKey(b"key3".to_vec()), b"value3".to_vec()),
            (ByteKey(b"key4".to_vec()), b"value4".to_vec()),
        ]
    }

    // 创建测试桶
    fn test_bucket() -> Bucket<ByteKey, Vec<u8>> {
        let version = Arc::new(VersionTracker::new());
        Bucket::new(version)
    }

    #[test]
    fn test_new_bucket() {
        let bucket = test_bucket();
        assert_eq!(bucket.slots().len(), BUCKET_SIZE);
        assert!(bucket.state.load(Ordering::Relaxed) == 0);
    }

    #[test]
    fn test_try_lock() {
        let bucket = test_bucket();
        
        // 第一次锁定应成功
        let guard = bucket.try_lock().expect("锁定失败");
        
        // 第二次锁定应失败
        assert!(bucket.try_lock().is_err());
        
        // 释放后应可再次锁定
        drop(guard);
        assert!(bucket.try_lock().is_ok());
    }

    #[test]
    fn test_find_slot() {
        let bucket = test_bucket();
        let searcher = Avx2Searcher;
        let fp = Fingerprint::new(123);
        
        // 初始应为空
        assert!(bucket.find_slot(fp, &searcher).is_none());
        
        // 添加指纹后应能找到
     //   bucket.slots()[0].update_fingerprint(fp);
     //   assert_eq!(bucket.find_slot(fp, &searcher), Some(0));
    }

    #[test]
fn test_find_slot_with_key() {
    let bucket = test_bucket();
    let version_guard = bucket.version.begin_operation(OperationType::Get);
    let key = ByteKey(b"test_key".to_vec());
    
    // 初始应为空
    assert!(bucket.find_slot_with_key(&key, &version_guard).is_none());
    
    // 使用insert方法添加键值对
    let insert_guard = bucket.version.begin_operation(OperationType::Insert);
    bucket.insert(0, key.clone(), b"value".to_vec(), &insert_guard).unwrap();
    
    // 现在应该能找到
    assert_eq!(bucket.find_slot_with_key(&key, &version_guard), Some(0));
}

   #[test]
fn test_find_empty_slot() {
    let bucket = test_bucket();
    let version_guard = bucket.version.begin_operation(OperationType::Insert);
    
    // 逐个查找并填充空槽位
    for expected_index in 0..BUCKET_SIZE {
        match bucket.find_empty_slot() {
            Some(idx) => {
                // 确保找到的索引是预期的
                assert_eq!(
                    idx, expected_index,
                    "Expected empty slot at index {}, but found at index {}",
                    expected_index, idx
                );
                
                // 填充该槽位
                let dummy_key = ByteKey(vec![expected_index as u8; 4]);
                let dummy_value = vec![expected_index as u8; 4];
                bucket.insert(idx, dummy_key, dummy_value, &version_guard).unwrap();
            }
            None => panic!("Expected to find empty slot at index {}", expected_index),
        }
    }
    
    // 所有槽位填充后应找不到空位
    assert!(
        bucket.find_empty_slot().is_none(),
        "Bucket should be full but found empty slot"
    );
}

   #[test]
fn test_insert_and_get() {
    use std::collections::HashMap;
    
    let bucket = test_bucket();
    let items = test_items();
    let version_guard = bucket.version.begin_operation(OperationType::Insert);
    
    // 存储键到索引的映射
    let mut key_to_index = HashMap::new();
    
    println!("Starting test_insert_and_get");
    println!("Initial bucket state:");
    print_bucket_state(&bucket);
    
    // 插入所有项
    for (key, value) in items.iter() {
        // 查找空槽位
        let idx = bucket.find_empty_slot()
            .expect("Should find empty slot");
        
        println!("Inserting key: {:?}, value: {:?} at index: {}", key, value, idx);
        
        // 记录映射
        key_to_index.insert(key.clone(), idx);
        
        // 插入数据
        bucket.insert(idx, key.clone(), value.clone(), &version_guard).unwrap();
        
        // 打印插入后的槽位状态
        println!("After insertion at index {}:", idx);
        print_slot_state(&bucket.slots()[idx]);
    }
    
    println!("All items inserted. Bucket state:");
    print_bucket_state(&bucket);
    
    // 验证所有项存在
    let read_guard = bucket.version.begin_operation(OperationType::Get);
    for (key, value) in items.iter() {
        let idx = key_to_index.get(key).expect("Key not found");
        
        println!("Getting key: {:?} at index: {}", key, idx);
        
        let result = bucket.get(*idx, key, &read_guard);
        println!("Result: {:?}, Expected: {:?}", result, Some(value.clone()));
        
        assert_eq!(
            result,
            Some(value.clone()),
            "Failed for key {:?} at index {}",
            key, idx
        );
    }
    
    println!("Test passed successfully");
}

// 辅助函数：打印槽位状态
fn print_slot_state(slot: &Arc<Slot>) {
    let state = slot.state();
    let fingerprint = state.fingerprint;
    let flags = state.state_flags;
    
    let (key, value) = slot.load();
    
    println!(
        "Slot: fingerprint={:?}, flags={:?}, key={:?}, value={:?}, is_empty={}",
        fingerprint,
        flags,
        key,
        value,
        slot.is_empty()
    );
}

// 辅助函数：打印桶状态
fn print_bucket_state<K: Key, V: Value>(bucket: &Bucket<K, V>) {
    println!("Bucket state: {:?}", bucket.state.load(Ordering::Relaxed));
    for (i, slot) in bucket.slots().iter().enumerate() {
        print_slot_state(slot);
    }
}

    #[test]
    fn test_remove() {
        let bucket = test_bucket();
        let version_guard = bucket.version.begin_operation(OperationType::Get);
        let items = test_items();
        
        // 插入所有项
        let version_guard = bucket.version.begin_operation(OperationType::Insert);
        for (i, (key, value)) in items.iter().enumerate() {
            
bucket.insert(i, key.clone(), value.clone(), &version_guard).unwrap();
           // bucket.insert(i, key.clone(), value.clone());
        }
        
        // 删除第一项
        let (key1, value1) = &items[0];
        assert_eq!(bucket.remove(0, key1, &version_guard), Some(value1.clone()));
        
        // 验证已删除
        assert!(bucket.get(0, key1, &version_guard).is_none());
        
        // 验证其他项仍在
        for (i, (key, value)) in items.iter().enumerate().skip(1) {
            assert_eq!(bucket.get(i, key, &version_guard), Some(value.clone()));
        }
    }

    #[test]
    fn test_mark_migrating() {
        let bucket = test_bucket();
        
        // 初始状态应为0
        assert_eq!(bucket.state.load(Ordering::Relaxed), 0);
        
        // 标记迁移中
        bucket.mark_migrating();
        assert!(bucket.state.load(Ordering::Relaxed) & Bucket::<ByteKey, Vec<u8>>::MIGRATING_MASK != 0);
        
        // 取消标记
        bucket.unmark_migrating();
        assert_eq!(bucket.state.load(Ordering::Relaxed) & Bucket::<ByteKey, Vec<u8>>::MIGRATING_MASK, 0);
    }

   #[test]
fn test_concurrent_access() {
    let bucket = Arc::new(test_bucket());
    let mut handles = vec![];
    
    for i in 0..BUCKET_SIZE {
        let bucket_clone = Arc::clone(&bucket);
        let key = ByteKey(format!("key_{}", i).into_bytes());
        let value = format!("value_{}", i).into_bytes();
        
        handles.push(thread::spawn(move || {
            let version_guard = bucket_clone.version.begin_operation(OperationType::Insert);
            bucket_clone.insert(i, key, value, &version_guard).unwrap();
        }));
    }
    
    for handle in handles {
        handle.join().unwrap();
    }
    
    // 验证所有槽位已填充
    let read_guard = bucket.version.begin_operation(OperationType::Get);
    for i in 0..BUCKET_SIZE {
        let key = ByteKey(format!("key_{}", i).into_bytes());
        assert!(bucket.get(i, &key, &read_guard).is_some());
    }
}

    #[test]
    fn test_bucket_guard() {
        let bucket = test_bucket();
        
        // 获取保护器
        let guard = bucket.try_lock().expect("锁定失败");
        
        // 验证可以访问槽位
        assert_eq!(guard.slots().len(), BUCKET_SIZE);
        
        // 提交更改
        guard.commit();
        
        // 验证锁定状态已清除
        assert!(bucket.try_lock().is_ok());
    }

    #[test]
    fn test_bucket_guard_drop() {
        let bucket = test_bucket();
        
        {
            // 获取保护器
            let _guard = bucket.try_lock().expect("锁定失败");
            
            // 尝试再次锁定应失败
            assert!(bucket.try_lock().is_err());
        }
        
        // 保护器释放后应可再次锁定
        assert!(bucket.try_lock().is_ok());
    }

    #[test]
    fn test_fingerprint_consistency() {
        let bucket = test_bucket();
        let key = ByteKey(b"consistent_key".to_vec());
        let fp = bucket.version.fingerprint_generator().generate(Key::as_bytes(&key));
        
        // 多次生成应相同
        for _ in 0..10 {
            assert_eq!(bucket.version.fingerprint_generator().generate(Key::as_bytes(&key)), fp);
        }
    }

    #[test]
    fn test_full_bucket() {
        let bucket = test_bucket();
        let items = test_items();
        
        // 填充所有槽位
         let version_guard = bucket.version.begin_operation(OperationType::Insert);
        for (i, (key, value)) in items.iter().enumerate() {
           
bucket.insert(i, key.clone(), value.clone(), &version_guard).unwrap();
           // bucket.insert(i, key.clone(), value.clone());
        }
        
        // 验证桶已满
        assert!(bucket.find_empty_slot().is_none());
        
        // 尝试添加新项应失败
        let new_key = ByteKey(b"new_key".to_vec());
        assert!(bucket.find_slot_with_key(&new_key, &bucket.version.begin_operation(OperationType::Get)).is_none());
    }
}