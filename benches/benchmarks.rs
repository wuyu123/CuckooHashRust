//! Cuckoo哈希表性能基准测试

use criterion::{criterion_group, criterion_main, Criterion, BenchmarkId, Throughput, PlotConfiguration};

use cuckoo_hashtable::map::cuckoo_map::MigrationPlan;
use cuckoo_hashtable::{HighPerfMap, types::ByteKey, batch_insert, batch_get};
use rand::{Rng, SeedableRng};
use rand::rngs::StdRng;
use std::time::Duration;

// 基准测试配置
const SEED: u64 = 42;
const ITEM_COUNTS: [usize; 5] = [10_000, 100_000, 1_000_000, 5_000_000, 10_000_000];
const KEY_SIZE: usize = 16; // 128位键
const VALUE_SIZE: usize = 8; // 64位值（减小值大小以减少拷贝开销）

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

/// 插入操作基准测试
fn bench_insert(c: &mut Criterion) {
    let plot_config = PlotConfiguration::default().summary_scale(criterion::AxisScale::Logarithmic);
    let mut group = c.benchmark_group("Insert");
    group.plot_config(plot_config);
    
    for &count in ITEM_COUNTS.iter() {
        let items = generate_items(count);
        
        group.throughput(Throughput::Elements(count as u64));
        group.bench_with_input(
            BenchmarkId::from_parameter(count), 
            &items,
            |b, items| {
                b.iter_batched(
                    || {
                        // 每个迭代创建新哈希表
                        HighPerfMap::<ByteKey, Vec<u8>>::new()
                    },
                    |mut map| {
                        for (key, value) in items {
                            map.inner_mut().insert(key.clone(), value.clone()).unwrap();
                        }
                    },
                    criterion::BatchSize::PerIteration,
                );
            },
        );
    }
    group.finish();
}

/// 查询操作基准测试
fn bench_get(c: &mut Criterion) {
    let plot_config = PlotConfiguration::default().summary_scale(criterion::AxisScale::Logarithmic);
    let mut group = c.benchmark_group("Get");
    group.plot_config(plot_config);
    
    for &count in ITEM_COUNTS.iter() {
        let items = generate_items(count);
        let keys: Vec<ByteKey> = items.iter().map(|(k, _)| k.clone()).collect();
        
        // 预填充哈希表
        let mut map = HighPerfMap::<ByteKey, Vec<u8>>::new();
        for (key, value) in items {
            map.inner_mut().insert(key, value).unwrap();
        }
        
        group.throughput(Throughput::Elements(count as u64));
        group.bench_with_input(
            BenchmarkId::from_parameter(count), 
            &keys,
            |b, keys| {
                b.iter(|| {
                    for key in keys {
                        criterion::black_box(map.inner().get(key));
                    }
                });
            },
        );
    }
    group.finish();
}

/// 删除操作基准测试
fn bench_remove(c: &mut Criterion) {
    let plot_config = PlotConfiguration::default().summary_scale(criterion::AxisScale::Logarithmic);
    let mut group = c.benchmark_group("Remove");
    group.plot_config(plot_config);
    
    for &count in ITEM_COUNTS.iter() {
        let items = generate_items(count);
        let keys: Vec<ByteKey> = items.iter().map(|(k, _)| k.clone()).collect();
        
        group.throughput(Throughput::Elements(count as u64));
        group.bench_with_input(
            BenchmarkId::from_parameter(count), 
            &keys,
            |b, keys| {
                b.iter_batched(
                    || {
                        // 每个迭代创建新哈希表并填充
                        let mut map = HighPerfMap::<ByteKey, Vec<u8>>::new();
                        for (key, value) in &items {
                            map.inner_mut().insert(key.clone(), value.clone()).unwrap();
                        }
                        map
                    },
                    |mut map| {
                        for key in keys {
                            criterion::black_box(map.inner_mut().remove(key));
                        }
                    },
                    criterion::BatchSize::PerIteration,
                );
            },
        );
    }
    group.finish();
}

/// 批量操作基准测试
fn bench_batch_operations(c: &mut Criterion) {
    let mut group = c.benchmark_group("Batch Operations");
    
    for &count in [10_000, 100_000, 1_000_000].iter() {
        let items = generate_items(count);
        let keys: Vec<ByteKey> = items.iter().map(|(k, _)| k.clone()).collect();
        
        // 批量插入
        group.bench_with_input(
            BenchmarkId::new("Batch Insert", count), 
            &items,
            |b, items| {
                b.iter_batched(
                    || HighPerfMap::<ByteKey, Vec<u8>>::new(),
                    |mut map| {
                        batch_insert(map.inner_mut(), items.iter().cloned());
                    },
                    criterion::BatchSize::PerIteration,
                );
            },
        );
        
        // 批量查询
        let mut map = HighPerfMap::<ByteKey, Vec<u8>>::new();
        for (key, value) in items {
            map.inner_mut().insert(key, value).unwrap();
        }
        
        group.bench_with_input(
            BenchmarkId::new("Batch Get", count), 
            &keys,
            |b, keys| {
                b.iter(|| {
                    let results = batch_get(map.inner(), keys.iter());
                    criterion::black_box(results);
                });
            },
        );
    }
    group.finish();
}

/// 并发性能测试
fn bench_concurrent(c: &mut Criterion) {
    use std::sync::Arc;
    use std::thread;
    
    let mut group = c.benchmark_group("Concurrent");
    
    for &thread_count in [1, 4, 8, 16, 32, 64].iter() {
        for &count in [100_000].iter() {
            let items = generate_items(count);
            
            group.bench_with_input(
                BenchmarkId::new("Concurrent Insert", format!("{} threads", thread_count)), 
                &(thread_count, items),
                |b, (thread_count, items)| {
                    b.iter(|| {
                        let map = Arc::new(HighPerfMap::<ByteKey, Vec<u8>>::new());
                        let mut handles = vec![];
                        
                        // 每个线程处理一部分数据
                        let chunk_size = items.len() / thread_count;
                        for chunk in items.chunks(chunk_size) {
                            let map_clone = Arc::clone(&map);
                            let chunk = chunk.to_vec();
                            handles.push(thread::spawn(move || {
                                for (key, value) in chunk {
                                    map_clone.inner().insert(key, value).unwrap();
                                }
                            }));
                        }
                        
                        for handle in handles {
                            handle.join().unwrap();
                        }
                    });
                },
            );
        }
    }
    group.finish();
}

/// 迁移性能测试
fn bench_migration(c: &mut Criterion) {
    let mut group = c.benchmark_group("Migration");
    
    for &count in [1_000_000, 5_000_000, 10_000_000].iter() {
        let items = generate_items(count);
        
        group.bench_with_input(
            BenchmarkId::from_parameter(count), 
            &items,
            |b, items| {
                b.iter_batched(
                    || {
                        // 创建初始表（50%负载）
                        let mut map = HighPerfMap::<ByteKey, Vec<u8>>::new();
                        for (key, value) in items.iter().take(count / 2) {
                            map.inner_mut().insert(key.clone(), value.clone()).unwrap();
                        }
                        map
                    },
                    |mut map| {
                        // 触发迁移（扩容2倍）
                        map.inner_mut().migrate(MigrationPlan::expand(2.0)).unwrap();
                    },
                    criterion::BatchSize::PerIteration,
                );
            },
        );
    }
    group.finish();
}

criterion_group!(
    name = benches;
    config = Criterion::default()
        .sample_size(10)
        .warm_up_time(Duration::from_secs(1))
        .measurement_time(Duration::from_secs(5))
        .noise_threshold(0.05);
    targets = 
        bench_insert, 
        bench_get, 
        bench_remove, 
        bench_batch_operations,
        bench_concurrent,
        bench_migration
);
criterion_main!(benches);