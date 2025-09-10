// src/memory/slot_allocator.rs
//! 槽位分配器接口 - 专用槽位管理

use crate::{
    error::CuckooError,
    memory::{ slot::SlotHandle, SlotMetadata},
};

/// 槽位统计信息
#[derive(Debug, Clone, Default)]
pub struct SlotStats {
    pub total_slots: usize,
    pub current_used: usize,
    pub peak_used: usize,
    pub allocation_count: u64,
    pub deallocation_count: u64,
}

/// 槽位分配器特征
pub trait SlotAllocator: Send + Sync {
    /// 分配槽位
    fn allocate_slot(&self) -> Result<SlotHandle, CuckooError>;
    
    /// 释放槽位
    fn deallocate_slot(&self, handle: SlotHandle);
    
    /// 获取槽位统计信息
    fn slot_stats(&self) -> SlotStats;
    
    /// 执行紧急内存回收
    fn emergency_reclaim(&self);

    //fn find_chunk_for_metadata(&self, metadata: &SlotMetadata) -> Option<&ChunkInfo> ;
}