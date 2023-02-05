use tensor_wgpu::Tensor1;
use futures::Future;
use std::{pin::Pin, collections::HashMap};

pub struct PendingStatisticBuffer {
    pub copy_operation: Option<Pin<Box<dyn Future<Output = std::result::Result<(), wgpu::BufferAsyncError>>>>>,
    pub buffer: wgpu::Buffer,
    padding: i32,
    resulting_sample: SolverStatisticSample,
}

#[derive(Default, Copy, Clone)]
pub struct SolverStatisticSample {
    pub value: f32,
    pub iteration_count: i32,
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