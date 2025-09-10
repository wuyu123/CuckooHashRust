// src/memory/pool.rs
//! 内存池实现 - 高效管理槽位内存

use crate::{
    error::CuckooError,
    memory::{
        allocator::{AllocationStats, AllocatorFactory, MemoryAllocator, MemoryStats},
        slot::{Slot, SlotMetadata},
        slot_allocator::{SlotAllocator, SlotStats}, DefaultMemoryAllocator,
    }, 
    SlotHandle,
};
use crossbeam_epoch::{self, Atomic, Guard, Owned};
use std::{
    alloc::Layout, 
    mem, 
    ptr::NonNull, 
    sync::atomic::{AtomicUsize, AtomicU64, Ordering}, 
    sync::Arc
};

/// 内存池结构
pub struct MemoryPool {
    allocator: Arc<dyn MemoryAllocator>,
    free_list: Atomic<SlotMetadataNode>,
    stats: SlotStatsInternal,
}

/// 内存池内部统计
#[derive(Debug, Default)]
struct SlotStatsInternal {
    total_slots: AtomicUsize,
    current_used: AtomicUsize,
    peak_used: AtomicUsize,
    allocation_count: AtomicU64,
    deallocation_count: AtomicU64,
}

impl MemoryPool {
    /// 默认块大小（槽位数）
    const DEFAULT_CHUNK_SIZE: usize = 64; // 4KB块 (64槽位 * 64字节)
    
    /// 创建新内存池
    pub fn new() -> Self {
        Self {
            allocator: AllocatorFactory::default(),
            free_list: Atomic::null(),
            stats: SlotStatsInternal::default(),
        }
    }
    
    /// 使用自定义分配器创建内存池
    pub fn with_allocator(allocator: Arc<dyn MemoryAllocator>) -> Self {
        Self {
            allocator,
            free_list: Atomic::null(),
            stats: SlotStatsInternal::default(),
        }
    }
    
    /// 获取底层分配器统计信息
    pub fn allocator_stats(&self) -> MemoryStats {
        self.allocator.stats()
    }
    
    /// 扩展内存池
    fn expand_pool(&self) -> Result<(), CuckooError> {
        let layout = Layout::array::<Slot>(Self::DEFAULT_CHUNK_SIZE)
            .map_err(|_| CuckooError::AllocationFailed {
                size: mem::size_of::<Slot>() * Self::DEFAULT_CHUNK_SIZE,
                align: mem::align_of::<Slot>(),
            })?;
        
        // 分配新块
        let ptr = unsafe { self.allocator.allocate(layout)? };
        
        // 初始化槽位元数据
        for i in 0..Self::DEFAULT_CHUNK_SIZE {
            let slot_ptr = unsafe { ptr.as_ptr().add(i * mem::size_of::<Slot>()) as *mut Slot };
            let metadata = SlotMetadata::new(
                NonNull::new(slot_ptr).unwrap(),
                i,
                ptr.as_ptr() as usize,
            );
            
            // 添加到空闲链表
            self.add_to_free_list(metadata);
        }
        
        // 更新统计
        self.stats.total_slots.fetch_add(
            Self::DEFAULT_CHUNK_SIZE, 
            Ordering::Relaxed
        );
        
        Ok(())
    }
    
    /// 添加到空闲链表
    fn add_to_free_list(&self, metadata: SlotMetadata) {
        let new_node = Owned::new(SlotMetadataNode {
            metadata,
            next: Atomic::null(),
        });
        
        let guard = crossbeam_epoch::pin();
        let mut current = self.free_list.load(Ordering::Relaxed, &guard);
        
        loop {
            let mut new_node = Owned::new(SlotMetadataNode {
                metadata,
                next: Atomic::null(),
            });
            
            new_node.next.store(current, Ordering::Relaxed);
            
            match self.free_list.compare_exchange(
                current,
                new_node.into_shared(&guard),
                Ordering::Release,
                Ordering::Relaxed,
                &guard
            ) {
                Ok(_) => break,
                Err(e) => current = e.current,
            }
        }
    }
    
    /// 尝试从空闲链表弹出槽位
    fn try_pop_free(&self, guard: &Guard) -> Option<SlotHandle> {
        let mut head = self.free_list.load(Ordering::Acquire, guard);
        while let Some(node) = unsafe { head.as_ref() } {
            let next = node.next.load(Ordering::Relaxed, guard);
            
            match self.free_list.compare_exchange(
                head,
                next,
                Ordering::Release,
                Ordering::Relaxed,
                guard
            ) {
                Ok(_) => {
                    // 找到空闲槽位
                    let metadata = node.metadata;
                    unsafe { guard.defer_destroy(head) };
                    
                    // 更新使用统计
                    let current_used = self.stats.current_used.fetch_add(1, Ordering::Relaxed) + 1;
                    let peak_used = self.stats.peak_used.load(Ordering::Relaxed);
                    
                    // 更新峰值
                    if current_used > peak_used {
                        self.stats.peak_used.store(current_used, Ordering::Relaxed);
                    }
                    
                    self.stats.allocation_count.fetch_add(1, Ordering::Relaxed);
                    
                    return Some(SlotHandle::new(metadata));
                }
                Err(e) => head = e.current,
            }
        }
        
        None
    }
}

impl SlotAllocator for MemoryPool {
    fn allocate_slot(&self) -> Result<SlotHandle, CuckooError> {
        let guard = crossbeam_epoch::pin();
        
        // 尝试从空闲链表获取
        if let Some(handle) = self.try_pop_free(&guard) {
            return Ok(handle);
        }
        
        // 空闲链表为空，扩展内存池
        self.expand_pool()?;
        
        // 再次尝试
        self.try_pop_free(&guard)
            .ok_or(CuckooError::AllocationFailed {
                size: mem::size_of::<Slot>(),
                align: mem::align_of::<Slot>(),
            })
    }
    
    fn deallocate_slot(&self, handle: SlotHandle) {
        let metadata = handle.metadata();
        let new_node = Owned::new(SlotMetadataNode {
            metadata,
            next: Atomic::null(),
        });
        
        let guard = crossbeam_epoch::pin();
        let mut current = self.free_list.load(Ordering::Relaxed, &guard);
        
        loop {
            let mut new_node = Owned::new(SlotMetadataNode {
                metadata,
                next: Atomic::null(),
            });
            
            new_node.next.store(current, Ordering::Relaxed);
            
            match self.free_list.compare_exchange(
                current,
                new_node.into_shared(&guard),
                Ordering::Release,
                Ordering::Relaxed,
                &guard
            ) {
                Ok(_) => {
                    self.stats.current_used.fetch_sub(1, Ordering::Relaxed);
                    self.stats.deallocation_count.fetch_add(1, Ordering::Relaxed);
                    return;
                }
                Err(e) => current = e.current,
            }
        }
    }
    
    fn slot_stats(&self) -> SlotStats {
        SlotStats {
            total_slots: self.stats.total_slots.load(Ordering::Relaxed),
            current_used: self.stats.current_used.load(Ordering::Relaxed),
            peak_used: self.stats.peak_used.load(Ordering::Relaxed),
            allocation_count: self.stats.allocation_count.load(Ordering::Relaxed),
            deallocation_count: self.stats.deallocation_count.load(Ordering::Relaxed),
        }
    }
    
    /// 紧急回收 - 移除此功能以避免内存损坏
    fn emergency_reclaim(&self) {
        // 空实现 - 不再执行任何操作
    }
}

/// 空闲链表节点
struct SlotMetadataNode {
    metadata: SlotMetadata,
    next: Atomic<SlotMetadataNode>,
}

// 确保 SlotMetadataNode 是线程安全的
unsafe impl Send for SlotMetadataNode {}
unsafe impl Sync for SlotMetadataNode {}

// src/memory/pool.rs
// ... 其他代码保持不变 ...

#[cfg(test)]
mod tests {
    use super::*;
    use crate::memory::allocator::SystemAllocator;
    use crossbeam_epoch as epoch;
    use std::sync::Arc;

    /// 创建测试用内存池
    fn test_pool() -> MemoryPool {
        MemoryPool::new()
    }

    #[test]
    fn test_pool_creation() {
        let pool = test_pool();
        let stats = pool.slot_stats();
        
        // 初始状态
        assert_eq!(stats.total_slots, 0);
        assert_eq!(stats.current_used, 0);
        assert_eq!(stats.peak_used, 0);
        assert_eq!(stats.allocation_count, 0);
        assert_eq!(stats.deallocation_count, 0);
    }

    #[test]
    fn test_allocate_slot() {
        let pool = test_pool();
        
        // 第一次分配应扩展池
        let handle = pool.allocate_slot().expect("分配失败");
        let stats = pool.slot_stats();
        
        // 验证统计
        assert_eq!(stats.total_slots, MemoryPool::DEFAULT_CHUNK_SIZE);
        assert_eq!(stats.current_used, 1);
        assert_eq!(stats.peak_used, 1);
        assert_eq!(stats.allocation_count, 1);
        assert_eq!(stats.deallocation_count, 0);
        
        // 验证槽位元数据
        let metadata = handle.metadata();
        assert!(!metadata.ptr.as_ptr().is_null());
        assert!(metadata.index < MemoryPool::DEFAULT_CHUNK_SIZE);
    }

    #[test]
    fn test_deallocate_slot() {
        let pool = test_pool();
        
        // 分配槽位
        let handle = pool.allocate_slot().expect("分配失败");
        let stats_before = pool.slot_stats();
        
        // 释放槽位
        pool.deallocate_slot(handle);
        let stats_after = pool.slot_stats();
        
        // 验证统计
        assert_eq!(stats_after.total_slots, MemoryPool::DEFAULT_CHUNK_SIZE);
        assert_eq!(stats_after.current_used, stats_before.current_used - 1);
        assert_eq!(stats_after.peak_used, stats_before.peak_used);
        assert_eq!(stats_after.allocation_count, stats_before.allocation_count);
        assert_eq!(stats_after.deallocation_count, stats_before.deallocation_count + 1);
    }

    #[test]
    fn test_pool_expansion() {
        let pool = test_pool();
        let chunk_size = MemoryPool::DEFAULT_CHUNK_SIZE;
        
        // 分配超过默认块大小的槽位
        let mut handles = Vec::new();
        for _ in 0..chunk_size * 2 {
            handles.push(pool.allocate_slot().expect("分配失败"));
        }
        
        let stats = pool.slot_stats();
        
        // 验证池已扩展
        assert_eq!(stats.total_slots, chunk_size * 2);
        assert_eq!(stats.current_used, chunk_size * 2);
        assert_eq!(stats.peak_used, chunk_size * 2);
        assert_eq!(stats.allocation_count, (chunk_size * 2)as u64 );
        
        // 释放所有槽位
        for handle in handles {
            pool.deallocate_slot(handle);
        }
        
        let stats_after = pool.slot_stats();
        assert_eq!(stats_after.current_used, 0);
        assert_eq!(stats_after.deallocation_count, (chunk_size * 2)as u64);
    }

    #[test]
    fn test_stats_accuracy() {
        let pool = test_pool();
        let chunk_size = MemoryPool::DEFAULT_CHUNK_SIZE;
        
        // 分配和释放槽位
        let handle1 = pool.allocate_slot().expect("分配失败");
        let handle2 = pool.allocate_slot().expect("分配失败");
        pool.deallocate_slot(handle1);
        let handle3 = pool.allocate_slot().expect("分配失败");
        
        let stats = pool.slot_stats();
        
        // 验证统计
        assert_eq!(stats.total_slots, chunk_size);
        assert_eq!(stats.current_used, 2);
        assert_eq!(stats.peak_used, 2);
        assert_eq!(stats.allocation_count, 3);
        assert_eq!(stats.deallocation_count, 1);
        
        // 释放剩余槽位
        pool.deallocate_slot(handle2);
        pool.deallocate_slot(handle3);
    }

    #[test]
    fn test_custom_allocator() {
        let custom_alloc = Arc::new(SystemAllocator);
        let pool = MemoryPool::with_allocator(custom_alloc.clone());
        
        // 分配槽位
        let _handle = pool.allocate_slot().expect("分配失败");
        
        // 验证使用自定义分配器
        assert!(pool.allocator.as_any().is::<SystemAllocator>());
    }

    #[test]
    fn test_concurrent_allocation() {
        let pool = Arc::new(test_pool());
        let mut handles = Vec::new();
        
        // 创建多个线程并发分配
        for _ in 0..10 {
            let pool_clone = Arc::clone(&pool);
            handles.push(std::thread::spawn(move || {
                for _ in 0..100 {
                    let handle = pool_clone.allocate_slot().expect("分配失败");
                    pool_clone.deallocate_slot(handle);
                }
            }));
        }
        
        // 等待所有线程完成
        for handle in handles {
            handle.join().unwrap();
        }
        
        // 验证最终状态
        let stats = pool.slot_stats();
        assert_eq!(stats.current_used, 0);
        assert_eq!(stats.allocation_count, 1000);
        assert_eq!(stats.deallocation_count, 1000);
    }

    #[test]
    fn test_free_list_management() {
        let pool = test_pool();
        let guard = epoch::pin();
        
        // 初始空闲链表应为空
        assert!(pool.free_list.load(Ordering::Relaxed, &guard).is_null());
        
        // 分配槽位
        let handle = pool.allocate_slot().expect("分配失败");
        
        // 释放槽位
        pool.deallocate_slot(handle);
        
        // 验证空闲链表不为空
        assert!(!pool.free_list.load(Ordering::Relaxed, &guard).is_null());
    }

    #[test]
    fn test_add_to_free_list() {
        let pool = test_pool();
        let guard = epoch::pin();
        
        // 创建测试元数据
        let dummy_ptr = NonNull::dangling();
        let metadata = SlotMetadata::new(dummy_ptr, 0, 0);
        
        // 添加到空闲链表
        pool.add_to_free_list(metadata);
        
        // 验证空闲链表不为空
        assert!(!pool.free_list.load(Ordering::Relaxed, &guard).is_null());
    }

    #[test]
    fn test_try_pop_free() {
        let pool = test_pool();
        let guard = epoch::pin();
        
        // 初始应为空
        assert!(pool.try_pop_free(&guard).is_none());
        
        // 添加测试元数据
        let dummy_ptr = NonNull::dangling();
        let metadata = SlotMetadata::new(dummy_ptr, 0, 0);
        pool.add_to_free_list(metadata);
        
        // 现在应能弹出
        assert!(pool.try_pop_free(&guard).is_some());
        
        // 再次应为空
        assert!(pool.try_pop_free(&guard).is_none());
    }
}