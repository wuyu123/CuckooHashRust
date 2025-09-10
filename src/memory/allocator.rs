// src/memory/allocator.rs
//! 内存分配器接口 - 支持自定义分配策略

use crate::error::CuckooError;
use crossbeam::queue::SegQueue;
use std::{
    alloc::{GlobalAlloc, Layout, System},
    any::Any,
    ptr::NonNull,
    sync::{
        atomic::{AtomicU64, AtomicUsize, Ordering}, 
        Arc
    },
};

/// 内存统计信息
#[derive(Debug, Default)]
pub struct MemoryStats {
    pub total_allocated: u64,
    pub current_used: u64,
    pub peak_used: u64,
    pub allocation_count: u64,
    pub deallocation_count: u64,
    pub pool_utilization: f64, // 池利用率百分比 (0.0-100.0)
}

/// 内存分配器特征
pub trait MemoryAllocator: Send + Sync {
    /// 分配指定大小的内存
    unsafe fn allocate(&self, layout: Layout) -> Result<NonNull<u8>, CuckooError>;
    
    /// 释放已分配的内存
    unsafe fn deallocate(&self, ptr: NonNull<u8>, layout: Layout);
    
    /// 获取内存统计信息
    fn stats(&self) -> MemoryStats;
    
    /// 作为Any类型，支持向下转型
    fn as_any(&self) -> &dyn Any;
}

/// 分配器扩展方法 - 提供高级功能
pub trait AllocatorExt: MemoryAllocator {
    /// 分配并初始化内存
    fn allocate_and_init<T>(&self, value: T) -> Result<*mut T, CuckooError> {
        let layout = Layout::new::<T>();
        unsafe {
            let ptr = self.allocate(layout)?;
            if ptr.as_ptr().is_null() {
                return Err(CuckooError::AllocationFailed {
                    size: layout.size(),
                    align: layout.align(),
                });
            }
            ptr.as_ptr().cast::<T>().write(value);
            Ok(ptr.as_ptr().cast::<T>())
        }
    }
    
    /// 释放并析构内存
    unsafe fn deallocate_and_drop<T>(&self, ptr: *mut T) {
        if !ptr.is_null() {
            ptr.drop_in_place();
            self.deallocate(NonNull::new(ptr.cast()).unwrap(), Layout::new::<T>());
        }
    }
}

// 为所有实现MemoryAllocator的类型提供AllocatorExt
impl<T: MemoryAllocator> AllocatorExt for T {}

/// 系统分配器实现
pub struct SystemAllocator;

impl MemoryAllocator for SystemAllocator {
    unsafe fn allocate(&self, layout: Layout) -> Result<NonNull<u8>, CuckooError> {
        let ptr = System.alloc(layout);
        if ptr.is_null() {
            Err(CuckooError::AllocationFailed {
                size: layout.size(),
                align: layout.align(),
            })
        } else {
            Ok(NonNull::new(ptr).unwrap())
        }
    }
    
    unsafe fn deallocate(&self, ptr: NonNull<u8>, layout: Layout) {
        System.dealloc(ptr.as_ptr(), layout);
    }
    
    fn stats(&self) -> MemoryStats {
        MemoryStats::default()
    }
    
    fn as_any(&self) -> &dyn Any {
        self
    }
}

/// 默认分配器（带统计）
pub struct DefaultMemoryAllocator {
    inner: SystemAllocator,
    stats: AllocationStats,
}

impl DefaultMemoryAllocator {
    pub fn new() -> Self {
        Self {
            inner: SystemAllocator,
            stats: AllocationStats::default(),
        }
    }
    
    /// 获取统计信息快照
    pub fn allocation_stats_snapshot(&self) -> AllocationStatsSnapshot {
        self.stats.snapshot()
    }
}

impl MemoryAllocator for DefaultMemoryAllocator {
    unsafe fn allocate(&self, layout: Layout) -> Result<NonNull<u8>, CuckooError> {
        let ptr = self.inner.allocate(layout)?;
        self.stats.record_allocation(layout.size());
        Ok(ptr)
    }
    
    unsafe fn deallocate(&self, ptr: NonNull<u8>, layout: Layout) {
        self.inner.deallocate(ptr, layout);
        self.stats.record_deallocation(layout.size());
    }
    
    fn stats(&self) -> MemoryStats {
        MemoryStats {
            total_allocated: self.stats.total_allocated_bytes() as u64,
            current_used: self.stats.current_used_bytes() as u64,
            peak_used: self.stats.peak_used_bytes() as u64,
            allocation_count: self.stats.allocation_count() as u64,
            deallocation_count: self.stats.deallocation_count() as u64,
            pool_utilization: 0.0,
        }
    }
    
    fn as_any(&self) -> &dyn Any {
        self
    }
}

/// 池化内存分配器（高性能实现）
pub struct PoolAllocator  {
    block_size: usize,
    block_layout: Layout,
    pool: SegQueue<*mut u8>,
    stats: AllocationStats,
    total_blocks: AtomicUsize,
}

// 确保线程安全
unsafe impl Send for PoolAllocator  {}
unsafe impl Sync for PoolAllocator  {}

impl PoolAllocator  {
    pub fn new(block_size: usize, initial_capacity: usize) -> Self {
        let layout = Layout::from_size_align(block_size, 64)
            .expect("Invalid block layout");
        
        let pool = SegQueue::new();
        let stats = AllocationStats::default();
        let total_blocks = AtomicUsize::new(0);
        
        // 预分配内存池
        for _ in 0..initial_capacity {
            unsafe {
                let ptr = System.alloc(layout);
                if !ptr.is_null() {
                    pool.push(ptr);
                    // 记录分配但不增加当前使用量
                    stats.record_background_allocation(block_size);
                    total_blocks.fetch_add(1, Ordering::Relaxed);
                }
            }
        }
        
        Self {
            block_size,
            block_layout: layout,
            pool,
            stats,
            total_blocks,
        }
    }
}

impl MemoryAllocator for PoolAllocator  {
    unsafe fn allocate(&self, layout: Layout) -> Result<NonNull<u8>, CuckooError> {
        // 检查大小和对齐
        if layout.size() > self.block_size || layout.align() > self.block_layout.align() {
            return Err(CuckooError::AllocationFailed {
                size: layout.size(),
                align: layout.align(),
            });
        }
        
        // 尝试从队列获取内存块
        if let Some(ptr) = self.pool.pop() {
           // 从池中获取时记录重用
            self.stats.record_reuse(layout.size());
            return Ok(NonNull::new_unchecked(ptr));
        }
        
        // 队列为空，分配新块
        let ptr = System.alloc(self.block_layout);
        if ptr.is_null() {
            Err(CuckooError::AllocationFailed {
                size: self.block_layout.size(),
                align: self.block_layout.align(),
            })
        } else {
            self.stats.record_allocation(self.block_layout.size());
            self.total_blocks.fetch_add(1, Ordering::Relaxed);
            Ok(NonNull::new_unchecked(ptr))
        }
    }
    
    unsafe fn deallocate(&self, ptr: NonNull<u8>, layout: Layout) {
        if layout.size() <= self.block_size && layout.align() <= self.block_layout.align() {
            // 将内存块放回队列
            self.pool.push(ptr.as_ptr());
            // 记录释放操作
            self.stats.record_deallocation(layout.size())
        } else {
            // 直接释放
            System.dealloc(ptr.as_ptr(), layout);
            self.stats.record_deallocation(layout.size());
        }
    }
    
    fn stats(&self) -> MemoryStats {
        let free_blocks = self.pool.len();
        let total_blocks = self.total_blocks.load(Ordering::Relaxed);
        let pool_utilization = if total_blocks > 0 {
            (free_blocks as f64 / total_blocks as f64) * 100.0
        } else {
            0.0
        };
        
        MemoryStats {
            total_allocated: self.stats.total_allocated_bytes() as u64,
            current_used: self.stats.current_used_bytes() as u64,
            peak_used: self.stats.peak_used_bytes() as u64,
            allocation_count: self.stats.allocation_count() as u64,
            deallocation_count: self.stats.deallocation_count() as u64,
            pool_utilization,
        }
    }
    
    fn as_any(&self) -> &dyn Any {
        self
    }
}

/// 线程安全的分配统计
#[derive(Debug, Default)]
pub struct AllocationStats {
    total_allocated: AtomicU64,
    current_used: AtomicU64,
    peak_used: AtomicU64,
    allocation_count: AtomicU64,
    deallocation_count: AtomicU64,
}

impl AllocationStats {
    /// 记录分配操作
    pub fn record_allocation(&self, size: usize) {
        let size = size as u64;
        self.allocation_count.fetch_add(1, Ordering::Relaxed);
        let prev = self.current_used.fetch_add(size, Ordering::Relaxed);
        let new_used = prev + size;
        
        // 更新峰值
        let mut current_peak = self.peak_used.load(Ordering::Relaxed);
        while new_used > current_peak {
            match self.peak_used.compare_exchange_weak(
                current_peak, 
                new_used, 
                Ordering::Relaxed, 
                Ordering::Relaxed
            ) {
                Ok(_) => break,
                Err(p) => current_peak = p,
            }
        }
        
        // 更新总分配量
        self.total_allocated.fetch_add(size, Ordering::Relaxed);
    }

    /// 记录重用操作（不增加总分配量）
    pub fn record_reuse(&self, size: usize) {
        let size = size as u64;
        self.allocation_count.fetch_add(1, Ordering::Relaxed);
        let prev = self.current_used.fetch_add(size, Ordering::Relaxed);
        let new_used = prev + size;
        
        // 更新峰值
        let mut current_peak = self.peak_used.load(Ordering::Relaxed);
        while new_used > current_peak {
            match self.peak_used.compare_exchange_weak(
                current_peak, 
                new_used, 
                Ordering::Relaxed, 
                Ordering::Relaxed
            ) {
                Ok(_) => break,
                Err(p) => current_peak = p,
            }
        }
    }
    /// 记录后台分配（不增加当前使用量）
    pub fn record_background_allocation(&self, size: usize) {
        let size = size as u64;
        self.allocation_count.fetch_add(1, Ordering::Relaxed);
        self.total_allocated.fetch_add(size, Ordering::Relaxed);
    }
    /// 记录释放操作
    pub fn record_deallocation(&self, size: usize) {
        let size = size as u64;
        self.deallocation_count.fetch_add(1, Ordering::Relaxed);
        self.current_used.fetch_sub(size, Ordering::Relaxed);
    }
    
    /// 获取总分配字节数
    pub fn total_allocated_bytes(&self) -> u64 {
        self.total_allocated.load(Ordering::Relaxed)
    }
    
    /// 获取当前使用字节数
    pub fn current_used_bytes(&self) -> u64 {
        self.current_used.load(Ordering::Relaxed)
    }
    
    /// 获取峰值使用字节数
    pub fn peak_used_bytes(&self) -> u64 {
        self.peak_used.load(Ordering::Relaxed)
    }
    
    /// 获取分配次数
    pub fn allocation_count(&self) -> u64 {
        self.allocation_count.load(Ordering::Relaxed)
    }
    
    /// 获取释放次数
    pub fn deallocation_count(&self) -> u64 {
        self.deallocation_count.load(Ordering::Relaxed)
    }
    
    /// 获取统计快照
    pub fn snapshot(&self) -> AllocationStatsSnapshot {
        AllocationStatsSnapshot {
            total_allocated: self.total_allocated_bytes(),
            current_used: self.current_used_bytes(),
            peak_used: self.peak_used_bytes(),
            allocation_count: self.allocation_count(),
            deallocation_count: self.deallocation_count(),
        }
    }
}

/// 分配统计快照
#[derive(Debug, Clone)]
pub struct AllocationStatsSnapshot {
    pub total_allocated: u64,
    pub current_used: u64,
    pub peak_used: u64,
    pub allocation_count: u64,
    pub deallocation_count: u64,
}

/// 分配器工厂
pub struct AllocatorFactory;

impl AllocatorFactory {
    /// 创建系统分配器
    pub fn system() -> Arc<dyn MemoryAllocator> {
        Arc::new(SystemAllocator)
    }
    
    /// 创建默认分配器（带统计）
    pub fn default() -> Arc<dyn MemoryAllocator> {
        Arc::new(DefaultMemoryAllocator::new())
    }
    
    /// 创建池化分配器
    pub fn pooled(block_size: usize, pool_size: usize) -> Arc<dyn MemoryAllocator> {
        Arc::new(PoolAllocator ::new(block_size, pool_size))
    }
    
    /// 创建自定义分配器
    pub fn custom<T: MemoryAllocator + 'static>(allocator: T) -> Arc<dyn MemoryAllocator> {
        Arc::new(allocator)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::alloc::Layout;
    use std::sync::Arc;

    #[test]
    fn test_system_allocator() {
        let allocator = SystemAllocator;
        let layout = Layout::from_size_align(64, 8).unwrap();
        
        unsafe {
            // 分配内存
            let ptr = allocator.allocate(layout).expect("分配失败");
            assert!(!ptr.as_ptr().is_null());
            
            // 写入数据
            ptr.as_ptr().write_bytes(0xAB, 64);
            
            // 释放内存
            allocator.deallocate(ptr, layout);
        }
    }

    #[test]
    fn test_default_allocator_stats() {
        let allocator = DefaultMemoryAllocator::new();
        let layout = Layout::from_size_align(128, 8).unwrap();
        
        unsafe {
            // 分配内存
            let ptr = allocator.allocate(layout).expect("分配失败");
            
            // 检查统计
            let stats = allocator.stats();
            assert_eq!(stats.total_allocated, 128);
            assert_eq!(stats.current_used, 128);
            assert_eq!(stats.peak_used, 128);
            assert_eq!(stats.allocation_count, 1);
            assert_eq!(stats.deallocation_count, 0);
            
            // 释放内存
            allocator.deallocate(ptr, layout);
            
            // 检查统计
            let stats = allocator.stats();
            assert_eq!(stats.total_allocated, 128);
            assert_eq!(stats.current_used, 0);
            assert_eq!(stats.peak_used, 128);
            assert_eq!(stats.allocation_count, 1);
            assert_eq!(stats.deallocation_count, 1);
        }
    }

    #[test]
fn test_pool_allocator_reuse() {
    let block_size = 64;
    let pool_size = 4;
    let allocator = PoolAllocator::new(block_size, pool_size);
    
    // 初始统计
    let stats = allocator.stats();
    assert_eq!(stats.total_allocated, block_size as u64 * pool_size as u64);
    assert_eq!(stats.current_used, 0);
    assert_eq!(stats.pool_utilization, 100.0); // 初始所有块都在池中
    
    let layout = Layout::from_size_align(block_size, 8).unwrap();
    
    unsafe {
        // 分配内存
        let ptr1 = allocator.allocate(layout).expect("分配失败");
        let ptr2 = allocator.allocate(layout).expect("分配失败");
        
        // 检查统计
        let stats = allocator.stats();
        assert_eq!(stats.total_allocated, block_size as u64 * pool_size as u64);
        assert_eq!(stats.current_used, block_size as u64 * 2);
        assert_eq!(stats.pool_utilization, 50.0); // 2个在用，2个在池中
        
        // 释放内存
        allocator.deallocate(ptr1, layout);
        allocator.deallocate(ptr2, layout);
        
        // 检查统计
        let stats = allocator.stats();
        assert_eq!(stats.total_allocated, block_size as u64 * pool_size as u64);
        assert_eq!(stats.current_used, 0);
        assert_eq!(stats.pool_utilization, 100.0); // 所有块都回到池中
    }
}

    #[test]
    fn test_pool_allocator_expansion() {
        let block_size = 64;
        let initial_pool_size = 2;
        let allocator = PoolAllocator::new(block_size, initial_pool_size);
        
        let layout = Layout::from_size_align(block_size, 8).unwrap();
        
        unsafe {
            // 分配3个块（超过初始池大小）
            let ptr1 = allocator.allocate(layout).expect("分配失败");
            let ptr2 = allocator.allocate(layout).expect("分配失败");
            let ptr3 = allocator.allocate(layout).expect("分配失败");
            
            // 检查统计
            let stats = allocator.stats();
            assert_eq!(stats.total_allocated, block_size as u64 * 3);
            assert_eq!(stats.current_used, block_size as u64 * 3);
            assert_eq!(stats.pool_utilization, 0.0); // 所有块都在使用中
            
            // 释放内存
            allocator.deallocate(ptr1, layout);
            allocator.deallocate(ptr2, layout);
            allocator.deallocate(ptr3, layout);
            
            // 检查统计
            let stats = allocator.stats();
            assert_eq!(stats.total_allocated, block_size as u64 * 3);
            assert_eq!(stats.current_used, 0);
            assert_eq!(stats.pool_utilization, 100.0); // 所有块都回到池中
        }
    }

    #[test]
    fn test_pool_allocator_large_allocation() {
        let block_size = 64;
        let pool_size = 4;
        let allocator = PoolAllocator::new(block_size, pool_size);
        
        // 尝试分配大于块大小的内存
        let large_layout = Layout::from_size_align(block_size * 2, 8).unwrap();
        
        unsafe {
            match allocator.allocate(large_layout) {
                Err(CuckooError::AllocationFailed { size, align }) => {
                    assert_eq!(size, block_size * 2);
                    assert_eq!(align, 8);
                }
                _ => panic!("预期分配失败"),
            }
        }
    }

    #[test]
    fn test_allocator_factory() {
        // 创建系统分配器
        let system_alloc = AllocatorFactory::system();
        assert!(system_alloc.as_any().is::<SystemAllocator>());
        
        // 创建默认分配器
        let default_alloc = AllocatorFactory::default();
        assert!(default_alloc.as_any().is::<DefaultMemoryAllocator>());
        
        // 创建池化分配器
        let pool_alloc = AllocatorFactory::pooled(64, 4);
        assert!(pool_alloc.as_any().is::<PoolAllocator>());
        
        // 创建自定义分配器
        let custom_alloc = AllocatorFactory::custom(SystemAllocator);
        assert!(custom_alloc.as_any().is::<SystemAllocator>());
    }

    #[test]
    fn test_allocator_ext() {
        let allocator = DefaultMemoryAllocator::new();
        
        // 分配并初始化内存
        let ptr = allocator.allocate_and_init(42).expect("分配失败");
        unsafe {
            assert_eq!(*ptr, 42);
            
            // 释放并析构内存
            allocator.deallocate_and_drop(ptr);
        }
    }

    #[test]
    fn test_allocation_stats() {
        let stats = AllocationStats::default();
        
        // 记录分配
        stats.record_allocation(128);
        assert_eq!(stats.total_allocated_bytes(), 128);
        assert_eq!(stats.current_used_bytes(), 128);
        assert_eq!(stats.peak_used_bytes(), 128);
        assert_eq!(stats.allocation_count(), 1);
        
        // 记录另一个分配
        stats.record_allocation(64);
        assert_eq!(stats.total_allocated_bytes(), 192);
        assert_eq!(stats.current_used_bytes(), 192);
        assert_eq!(stats.peak_used_bytes(), 192);
        assert_eq!(stats.allocation_count(), 2);
        
        // 记录释放
        stats.record_deallocation(64);
        assert_eq!(stats.total_allocated_bytes(), 192);
        assert_eq!(stats.current_used_bytes(), 128);
        assert_eq!(stats.peak_used_bytes(), 192);
        assert_eq!(stats.deallocation_count(), 1);
        
        // 快照测试
        let snapshot = stats.snapshot();
        assert_eq!(snapshot.total_allocated, 192);
        assert_eq!(snapshot.current_used, 128);
        assert_eq!(snapshot.peak_used, 192);
        assert_eq!(snapshot.allocation_count, 2);
        assert_eq!(snapshot.deallocation_count, 1);
    }
}