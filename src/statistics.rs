use crate::{
    texture::Texture,
    reactions_params::ReactionParams,
};

pub struct StatisticsGroup {
    pub bind_group: wgpu::BindGroup,
    pub bind_group_layout: wgpu::BindGroupLayout,
    pub textures: Vec<Texture>,  // Or maybe a dict?
}

impl StatisticsGroup {
    pub fn new(
        reactions_params: &ReactionParams,
        device: &wgpu::Device,
    ) -> Self {
        let textures = vec![
            Texture::new(&[reactions_params.raw_params.num_reactions], device)
        ];
        
        // Should be a for loop over the textures to create the entries
        let bind_group_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("Statistics Bind Group Layout"),
            entries: &[
                wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: textures[0].binding_type(wgpu::StorageTextureAccess::ReadWrite),
                    count: None,
                },
            ],
        });

        let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("Statistics Bind Group"),
            layout: &bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: textures[0].binding_resource(),
                },
            ],
        });

        StatisticsGroup {
            bind_group,
            bind_group_layout,
            textures,
        }
    }
}