use tensor_wgpu::Tensor1;
use std::collections::{HashMap, VecDeque};

#[derive(Default, Clone)]
pub struct SolverStatisticSample<T> {
    pub name: String,
    pub value: T,
    pub iteration_count: u32,
}

pub struct StatisticsGroup {
    pub bind_group: wgpu::BindGroup,
    pub bind_group_layout: wgpu::BindGroupLayout,
    pub stats: Vec<Tensor1<i32>>,  // For now it's only the concentration of the particles
    pub logging_stats: HashMap<String, [u32; 2]>  // Idx, padding
}

impl StatisticsGroup  {
    pub fn new(
        data: Vec<Tensor1<i32>>,
        logging_stats: HashMap<String, [u32; 2]>,
        device: &wgpu::Device,
    ) -> Self {
        let (bind_group_layout, bind_group) = StatisticsGroup::prepare_gpu(&data, device);
        StatisticsGroup {
            bind_group,
            bind_group_layout,
            logging_stats,
            stats: data,
        }
    }

    fn prepare_gpu(stats: &Vec<Tensor1<i32>>, device: &wgpu::Device) -> (wgpu::BindGroupLayout, wgpu::BindGroup) {
        let mut binding_num = 0;
        let mut layout_entries = Vec::<wgpu::BindGroupLayoutEntry>::new();
        let mut entries = Vec::<wgpu::BindGroupEntry>::new();
        for stat in stats {
            layout_entries.push(
                wgpu::BindGroupLayoutEntry {
                    binding: binding_num,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer { 
                        ty: wgpu::BufferBindingType::Storage { read_only: false },
                        has_dynamic_offset: false,
                        min_binding_size: wgpu::BufferSize::new(stat.buffer_size() as _)
                    },
                    count: None,
                },
            );
            entries.push(
                wgpu::BindGroupEntry {
                    binding: binding_num,
                    resource: stat.binding_resource(),
                },
            );
            binding_num += 1;
        }
        let bind_group_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("Statistics Bind Group Layout"),
            entries: &layout_entries,
        });

        let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("Statistics Bind Group"),
            layout: &bind_group_layout,
            entries: &entries,
        });
        (bind_group_layout, bind_group)
    }

}

#[derive(Debug)]
struct StatisticContainer {
    x: VecDeque<u32>,
    y: VecDeque<f32>,  // Must be f32 for the UI
}

impl StatisticContainer {

    fn new(capacity: usize) -> Self {
        StatisticContainer { x: VecDeque::with_capacity(capacity), y: VecDeque::with_capacity(capacity) }
    }

    fn add(&mut self, x: u32, y: f32) {
        while self.x.capacity() == self.x.len() {
            self.x.pop_front();
            self.y.pop_front();
        }
        self.x.push_back(x);
        self.y.push_back(y);
    }

    #[allow(dead_code)]
    fn mean(&self) -> f32 {
        let mut sum = 0.;
        for y in &self.y {
            sum += y;
        }
        sum / self.y.len() as f32
    }

    fn last(&self) -> f32 {
        match self.y.back() {
            Some(y) => *y,
            None => 0.
        }
    }
}

fn make_all_stats(metrics_log: Vec<&str>) -> HashMap<String, StatisticContainer> {
    let mut all_stats = HashMap::new();
    for metric in metrics_log {
        all_stats.insert(metric.to_string(), StatisticContainer::new(100));
    }
    all_stats
}
