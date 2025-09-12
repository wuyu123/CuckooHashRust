//! Cuckoo哈希表集成测试

use cuckoo_hashtable::hash::strategy::{ HashAlgorithm};
use cuckoo_hashtable::{log_info, DoubleHashStrategy};
use cuckoo_hashtable::{
    map::cuckoo_map::MigrationPlan,
    types::ByteKey,
    CuckooError, CuckooMap, 
};
use rand::{Rng, SeedableRng};
use rand::rngs::StdRng;
use std::sync::atomic::AtomicUsize;
use std::sync::Arc;
use std::thread;
use std::time::Duration;
use test_log::test;

const SEED: u64 = 42;
const ITEM_COUNT: usize = 100_000;
const KEY_SIZE: usize = 16;
const VALUE_SIZE: usize = 64;

/// 生成随机键值对
fn generate_items(count: usize) -> Vec<(ByteKey, Vec<u8>)> {
    let mut rng = StdRng::seed_from_u64(SEED);
    (0..count)
        .map(|_| {
            let mut key = vec![0u8; KEY_SIZE];
            let mut value = vec![0u8; VALUE_SIZE];
            rng.fill(&mut key[..]);
            rng.fill(&mut value[..]);
            (ByteKey(key), value)
        })
        .collect()
}

/// 创建测试用哈希表
fn create_test_map() -> CuckooMap<ByteKey, Vec<u8>> {
    let config = cuckoo_hashtable::map::cuckoo_map::CuckooMapConfig {
        initial_capacity: 2048,
        max_load_factor: 0.90,
        max_kick_depth: 500,
        slot_count_per_bucket: 4,
        migration_batch_size: 64,
        migration_parallelism: 4,
        migration_lock_timeout_ms: 100,
    };
    
    let hasher = Box::new(DoubleHashStrategy::new(AtomicUsize::new(config.initial_capacity),HashAlgorithm::XxHash));
    let memory_pool = Arc::new(cuckoo_hashtable::memory::PoolAllocator::new(64, 1024));
    let simd_searcher = Arc::new(cuckoo_hashtable::simd::ScalarSearcher);
    let stats_recorder = Arc::new(cuckoo_hashtable::stats::recorder::GlobalStatsRecorder::new());
    let version = Arc::new(cuckoo_hashtable::version::VersionTracker::new());
    
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
// fn test_basic_functionality() {
//     let map = create_test_map();
    
//     // 插入
//     let key1 = ByteKey(b"key1".to_vec());
//     let value1 = b"value1".to_vec();
//     assert!(map.insert(key1.clone(), value1.clone()).is_ok());
    
//     // 查询
//     assert_eq!(map.get(&key1), Some(value1.clone()));
    
//     // 更新
//     let value2 = b"value2".to_vec();
//     assert!(map.insert(key1.clone(), value2.clone()).is_ok());
//     assert_eq!(map.get(&key1), Some(value2.clone()));
    
//     // 删除
//     assert_eq!(map.remove(&key1), Some(value2));
//     assert_eq!(map.get(&key1), None);
// }

#[test]
fn test_high_load() {
    let start_time = std::time::Instant::now();
    let items = generate_items(ITEM_COUNT);
    let map = create_test_map();
    
    // 插入所有项
   
    for (key, value) in &items {
       
    assert!(map.insert(key.clone(), value.clone()).is_ok());
       
    }
    let total_duration = start_time.elapsed();
    println!("All inserts processed in {:?}", total_duration);
    // 验证统计信息
    let stats = map.stats();
    log_info!("load_factor {} ,size={} bucket count={}",map.load_factor(),stats.size,stats.capacity);
    // 验证所有项存在
    for (index,(key, value)) in  items.iter().enumerate() {
       // map.testtt(key);
        assert_eq!(map.get(key), Some(value.clone()), "Assertion failed at index {} for key {:?}",index, key);
    }
    
    
    assert_eq!(stats.size, ITEM_COUNT);
    //assert!(stats.load_factor > 0.8);
}

// #[test]
// fn test_concurrent_access() {
//     let items = generate_items(ITEM_COUNT);
//     let keys: Vec<ByteKey> = items.iter().map(|(k, _)| k.clone()).collect();
//     let map = Arc::new(create_test_map());
    
//     // 并发插入
//     let mut handles = vec![];
//     for chunk in items.chunks(ITEM_COUNT / 10) {
//         let map_clone = Arc::clone(&map);
//         let chunk = chunk.to_vec();
//         handles.push(thread::spawn(move || {
//             for (key, value) in chunk {
//                 map_clone.insert(key, value).unwrap();
//             }
//         }));
//     }
    
//     for handle in handles {
//         handle.join().unwrap();
//     }
    
//     // 并发读取
//     let mut handles = vec![];
//     for chunk in keys.chunks(ITEM_COUNT / 10) {
//         let map_clone = Arc::clone(&map);
//         let chunk = chunk.to_vec();
//         handles.push(thread::spawn(move || {
//             for key in chunk {
//                 assert!(map_clone.get(&key).is_some());
//             }
//         }));
//     }
    
//     for handle in handles {
//         handle.join().unwrap();
//     }
    
//     // 验证最终状态
//     assert_eq!(map.stats().size, ITEM_COUNT);
// }

// #[test]
// fn test_migration() {
//     let items = generate_items(ITEM_COUNT);
//     let map = create_test_map();
    
//     // 插入一半数据
//     for (key, value) in items.iter().take(ITEM_COUNT / 2) {
//         map.insert(key.clone(), value.clone()).unwrap();
//     }
    
//     let initial_capacity = map.bucket_count();
    
//     // 触发迁移（扩容2倍）
//     map.migrate(MigrationPlan::expand(2.0)).unwrap();
    
//     // 等待迁移完成
//     while !map.is_migration_fully_completed() {
//         thread::sleep(Duration::from_millis(10));
//     }
    
//     // 验证迁移后容量
//     assert!(map.bucket_count() > initial_capacity);
    
//     // 验证迁移后数据完整性
//     for (key, value) in items.iter().take(ITEM_COUNT / 2) {
//         assert_eq!(map.get(key), Some(value.clone()));
//     }
    
//     // 插入剩余数据
//     for (key, value) in items.iter().skip(ITEM_COUNT / 2) {
//         map.insert(key.clone(), value.clone()).unwrap();
//     }
    
//     // 验证所有数据
//     for (key, value) in &items {
//         assert_eq!(map.get(key), Some(value.clone()));
//     }
// }

// #[test]
// fn test_stats_and_monitoring() {
//     let items = generate_items(ITEM_COUNT);
//     let map = create_test_map();
    
//     // 插入数据
//     for (key, value) in &items {
//         map.insert(key.clone(), value.clone()).unwrap();
//     }
    
//     // 获取统计信息
//     let stats = map.stats();
    
//     // 验证基本统计
//     assert_eq!(stats.size, ITEM_COUNT);
//     assert!(stats.load_factor > 0.8);
//     assert!(stats.insert_count > 0);
//     assert!(stats.get_count == 0); // 尚未执行查询
    
//     // 执行查询
//     for (key, _) in &items {
//         map.get(key);
//     }
    
//     // 验证查询统计
//     let stats = map.stats();
//     assert!(stats.get_count >= ITEM_COUNT as u64);
    
//     // 生成Prometheus指标
//     let metrics = map.export_prometheus();
//     assert!(metrics.contains("cuckoo_operation_insert_count"));
//     assert!(metrics.contains("cuckoo_memory_current_used"));
// }

// #[test]
// fn test_error_handling() {
//     let map = create_test_map();
    
//     // 创建大键值对（超过默认槽大小）
//     let large_key = ByteKey(vec![0u8; 1024]);
//     let large_value = vec![0u8; 4096];
    
//     // 测试插入失败
//     match map.insert(large_key.clone(), large_value.clone()) {
//         Err(CuckooError::SlotOccupied) => {} // 预期错误
//         _ => panic!("Expected slot occupied error"),
//     }
    
//     // 测试迁移冲突
//     map.migrate(MigrationPlan::expand(2.0)).unwrap();
//     match map.migrate(MigrationPlan::expand(2.0)) {
//         Err(CuckooError::MigrationInProgress) => {} // 预期错误
//         _ => panic!("Expected migration in progress error"),
//     }
// }

// #[test]
// fn test_batch_operations() {
//     let items = generate_items(ITEM_COUNT);
//     let keys: Vec<ByteKey> = items.iter().map(|(k, _)| k.clone()).collect();
//     let map = create_test_map();
    
//     // 批量插入
//     for (key, value) in &items {
//         map.insert(key.clone(), value.clone()).unwrap();
//     }
    
//     // 批量查询
//     for key in &keys {
//         assert!(map.get(key).is_some());
//     }
    
//     // 批量删除
//     for key in keys {
//         assert!(map.remove(&key).is_some());
//     }
    
//     assert_eq!(map.stats().size, 0);
// }

// #[test]
// fn test_persistence_simulation() {
//     let items = generate_items(ITEM_COUNT);
//     let map = create_test_map();
    
//     // 插入数据
//     for (key, value) in &items {
//         map.insert(key.clone(), value.clone()).unwrap();
//     }
    
//     // 模拟快照
//     let mut snapshot = Vec::new();
//     for (key, value) in map.iter() {
//         snapshot.push((key, value));
//     }
    
//     // 创建新表
//     let new_map = create_test_map();
    
//     // 恢复数据
//     for (key, value) in snapshot {
//         new_map.insert(key, value).unwrap();
//     }
    
//     // 验证恢复
//     for (key, value) in &items {
//         assert_eq!(new_map.get(key), Some(value.clone()));
//     }
// }

// #[test]
// fn test_long_running_operations() {
//     let map = create_test_map();
//     let mut rng = StdRng::seed_from_u64(SEED);
    
//     // 长时间运行：插入、查询、删除循环
//     for i in 0..10_000 {
//         let key = ByteKey(format!("key_{}", i).into_bytes());
//         let value = vec![i as u8; VALUE_SIZE];
        
//         // 插入
//         map.insert(key.clone(), value.clone()).unwrap();
        
//         // 查询
//         assert_eq!(map.get(&key), Some(value.clone()));
        
//         // 随机删除
//         if rng.gen_bool(0.3) {
//             assert!(map.remove(&key).is_some());
//         }
        
//         // 定期迁移
//         if i % 1000 == 0 {
//             if map.load_factor() > 0.8 {
//                 map.migrate(MigrationPlan::expand(1.5)).unwrap();
//             }
//         }
//     }
    
//     // 最终健康检查
//     let stats = map.stats();
//     assert!(stats.size > 0);
//     assert!(stats.collision_rate < 0.5);
// }