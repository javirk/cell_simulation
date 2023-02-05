use crate::{lattice_params::Params,preprocessor::ShaderBuilder, WORKGROUP_SIZE, statistics::StatisticsGroup};


pub struct RDME {
    compute_pipeline: wgpu::ComputePipeline,
}

impl RDME {
    pub fn new( // Receive bind groups, build compute pipeline. Simulation is responsible for everything else
        bind_group_layouts: &Vec<wgpu::BindGroupLayout>,
        statistics: &StatisticsGroup,
        device: &wgpu::Device,
    ) -> Self {
        let data_bind_group_layout = &bind_group_layouts[0];
        let lattice_bind_group_layout = &bind_group_layouts[1];
        let reaction_bind_group_layout = &bind_group_layouts[2];

        let binding = ShaderBuilder::new("rdme.wgsl").unwrap();
        let shader_builder = binding.build();
        let compute_shader = device.create_shader_module(shader_builder);

        let compute_pipeline_layout = device.create_pipeline_layout(
            &wgpu::PipelineLayoutDescriptor {
                label: Some("RDME compute"),
                bind_group_layouts: &[data_bind_group_layout, lattice_bind_group_layout, reaction_bind_group_layout, &statistics.bind_group_layout],
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
        &self,
        data_bind_group: &wgpu::BindGroup,
        lattice_bind_group: &wgpu::BindGroup,
        simulation_bind_group: &wgpu::BindGroup,
        statistics_bind_group: &wgpu::BindGroup,
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
        let zdim = params.z_res as u32 + WORKGROUP_SIZE.2 - 1;
        let zgroups = zdim / WORKGROUP_SIZE.2;
        
        // Main compute pass
        {
            let mut cpass = command_encoder.begin_compute_pass(&wgpu::ComputePassDescriptor { label: None });
            cpass.set_pipeline(&self.compute_pipeline);
            cpass.set_bind_group(0, data_bind_group, &[]);
            cpass.set_bind_group(1, lattice_bind_group, &[]);
            cpass.set_bind_group(2, simulation_bind_group, &[]);
            cpass.set_bind_group(3, statistics_bind_group, &[]);
            cpass.dispatch_workgroups(xgroups, ygroups, zgroups);
        }        
    }
}