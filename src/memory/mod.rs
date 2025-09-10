//! 内存管理模块 - 统一管理哈希表内存分配

pub mod allocator;
pub mod slot_allocator;
pub mod pool;
pub mod slot;

pub use allocator::{MemoryAllocator, DefaultMemoryAllocator, PoolAllocator , AllocatorFactory, AllocationStats};
pub use slot_allocator::{SlotAllocator, SlotStats};
pub use pool::MemoryPool;
pub use slot::{Slot, SlotHandle, SlotMetadata};
use once_cell::sync::Lazy;

use crate::memory::allocator::MemoryStats;

/// 全局内存池实例
pub static GLOBAL_MEMORY_POOL: Lazy<MemoryPool> = 
    Lazy::new(MemoryPool::new);

/// 分配新槽位
pub fn allocate_slot() -> Result<SlotHandle, crate::error::CuckooError> {
    GLOBAL_MEMORY_POOL.allocate_slot()
}

/// 释放槽位
pub fn deallocate_slot(handle: SlotHandle) {
    GLOBAL_MEMORY_POOL.deallocate_slot(handle)
}

/// 获取槽位统计信息
pub fn slot_stats() -> SlotStats {
    GLOBAL_MEMORY_POOL.slot_stats()
}

/// 获取底层分配器统计信息
pub fn allocator_stats() -> MemoryStats {
    GLOBAL_MEMORY_POOL.allocator_stats()
}

/// 紧急内存回收
pub fn emergency_reclaim() {
    GLOBAL_MEMORY_POOL.emergency_reclaim()
}