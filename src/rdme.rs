use crate::{lattice_params::Params,preprocessor::ShaderBuilder};


const WORKGROUP_SIZE: (u32, u32, u32) = (2, 1, 1);

pub struct RDME {
    compute_pipeline: wgpu::ComputePipeline,
}

impl RDME {
    pub fn new( // Receive bind groups, build compute pipeline. Simulation is responsible for everything else
        data_group_layout: &wgpu::BindGroupLayout,
        lattice_group_layout: &wgpu::BindGroupLayout,
        device: &wgpu::Device,
    ) -> Self {
        let binding = ShaderBuilder::new("rdme.wgsl").unwrap();
        let shader_builder = binding.build();
        let compute_shader = device.create_shader_module(shader_builder);

        let compute_pipeline_layout = device.create_pipeline_layout(
            &wgpu::PipelineLayoutDescriptor {
                label: Some("RDME compute"),
                bind_group_layouts: &[data_group_layout, lattice_group_layout],
                push_constant_ranges: &[],
            }
        );

        let compute_pipeline = device.create_compute_pipeline(
            &wgpu::ComputePipelineDescriptor {
                label: Some("RDME Compute pipeline"),
                layout: Some(&compute_pipeline_layout),
                module: &compute_shader,
                entry_point: "rdme",
            }
        );

        RDME { 
            compute_pipeline: compute_pipeline,
        }
    }

    pub fn step(
        &mut self,
        data_bind_group: &wgpu::BindGroup,
        simulation_bind_group: &wgpu::BindGroup,
        command_encoder: &mut wgpu::CommandEncoder,
        params: &Params,
    ) {
        // command encoder as input
        // Compute pass
        // Set pipeline, bind group
        // Dispatch
        let xdim = params.x_res as u32 + WORKGROUP_SIZE.0 - 1;
        let xgroups = xdim / WORKGROUP_SIZE.0;
        let ydim = params.y_res as u32 + WORKGROUP_SIZE.1 - 1;
        let ygroups = ydim / WORKGROUP_SIZE.1;
        let zdim = params.z_res as u32 + WORKGROUP_SIZE.1 - 1;
        let zgroups = zdim / WORKGROUP_SIZE.1;
        
        {
            let mut cpass = command_encoder.begin_compute_pass(&wgpu::ComputePassDescriptor { label: None });
            cpass.set_pipeline(&self.compute_pipeline);
            cpass.set_bind_group(0, data_bind_group, &[]);
            cpass.set_bind_group(1, simulation_bind_group, &[]);
            cpass.dispatch_workgroups(xgroups, ygroups, zgroups);
        }

    }
}