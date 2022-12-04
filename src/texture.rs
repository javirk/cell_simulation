use crate::lattice_params::LatticeParams;

pub struct Texture {
    pub texture_view: wgpu::TextureView,
    format: wgpu::TextureFormat,
}

impl Texture {
    pub fn new(
        device: &wgpu::Device,
        params: &Vec<usize>,
        format: wgpu::TextureFormat,
    ) -> Self {
        let texture = device.create_texture(&wgpu::TextureDescriptor {
            label: None,
            size: wgpu::Extent3d {
                width: params[1] as u32,
                height: params[0] as u32,
                depth_or_array_layers: 1,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format,
            usage: wgpu::TextureUsages::TEXTURE_BINDING 
            | wgpu::TextureUsages::COPY_DST 
            | wgpu::TextureUsages::STORAGE_BINDING,
        });
        let texture_view = texture.create_view(&wgpu::TextureViewDescriptor::default());

        Texture {
            texture_view,
            format,
        }
    }

    pub fn binding_resource(&self) -> wgpu::BindingResource {
        wgpu::BindingResource::TextureView(&self.texture_view)
    }

    pub fn binding_type(
        &self,
        access: wgpu::StorageTextureAccess
    ) -> wgpu::BindingType {
        wgpu::BindingType::StorageTexture {
            access,
            format: self.format,
            view_dimension: wgpu::TextureViewDimension::D2,
        }
    }
}