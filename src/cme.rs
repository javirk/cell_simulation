use crate::{lattice_params::Params,preprocessor::ShaderBuilder, CME_WORKGROUP_SIZE, statistics::StatisticsGroup};


pub struct CME {
    compute_pipeline: wgpu::ComputePipeline,
}

impl CME {
    pub fn new( // Receive bind groups, build compute pipeline. Simulation is responsible for everything else
        bind_group_layouts: &Vec<wgpu::BindGroupLayout>,
        statistics: &StatisticsGroup,
        device: &wgpu::Device,
    ) -> Self {
        let data_bind_group_layout = &bind_group_layouts[0];
        let lattice_bind_group_layout = &bind_group_layouts[1];
        let reaction_bind_group_layout = &bind_group_layouts[2];
        let boundary_bind_group_layout = &bind_group_layouts[3];

        let binding = ShaderBuilder::new("cme.wgsl").unwrap();
        let shader_builder = binding.build();
        let compute_shader = device.create_shader_module(shader_builder);

        let compute_pipeline_layout = device.create_pipeline_layout(
            &wgpu::PipelineLayoutDescriptor {
                label: Some("CME compute"),
                bind_group_layouts: &[data_bind_group_layout, lattice_bind_group_layout, reaction_bind_group_layout, &statistics.bind_group_layout, &boundary_bind_group_layout],
                push_constant_ranges: &[],
            }
        );

        let compute_pipeline = device.create_compute_pipeline(
            &wgpu::ComputePipelineDescriptor {
                label: Some("CME Compute pipeline"),
                layout: Some(&compute_pipeline_layout),
                module: &compute_shader,
                entry_point: "cme",
            }
        );

        CME { 
            compute_pipeline: compute_pipeline,
        }
    }

    

    pub fn step(
        &self,
        data_bind_group: &wgpu::BindGroup,
        lattice_bind_group: &wgpu::BindGroup,
        simulation_bind_group: &wgpu::BindGroup,
        statistics_bind_group: &wgpu::BindGroup,
        boundaries_bind_group: &wgpu::BindGroup,
        command_encoder: &mut wgpu::CommandEncoder,
        params: &Params,
    ) {
        // command encoder as input
        // Compute pass
        // Set pipeline, bind group
        // Dispatch
        let xdim = params.res[0] as u32 + CME_WORKGROUP_SIZE.0 - 1;
        let xgroups = xdim / CME_WORKGROUP_SIZE.0;
        let ydim = params.res[1] as u32 + CME_WORKGROUP_SIZE.1 - 1;
        let ygroups = ydim / CME_WORKGROUP_SIZE.1;
        let zdim = params.res[2] as u32 + CME_WORKGROUP_SIZE.2 - 1;
        let zgroups = zdim / CME_WORKGROUP_SIZE.2;
        
        // Main compute pass
        {
            let mut cpass = command_encoder.begin_compute_pass(&wgpu::ComputePassDescriptor { label: None });
            cpass.set_pipeline(&self.compute_pipeline);
            cpass.set_bind_group(0, data_bind_group, &[]);
            cpass.set_bind_group(1, lattice_bind_group, &[]);
            cpass.set_bind_group(2, simulation_bind_group, &[]);
            cpass.set_bind_group(3, statistics_bind_group, &[]);
            cpass.set_bind_group(4, boundaries_bind_group, &[]);
            cpass.dispatch_workgroups(xgroups, ygroups, zgroups);
        }        
    }
}