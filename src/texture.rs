use crate::lattice_params::{Params};

pub struct Texture {
    pub texture_view: wgpu::TextureView,
    format: wgpu::TextureFormat,
}

impl Texture {
    // This is important: https://www.w3.org/TR/WGSL/#storage-texel-formats
    pub fn new(
        lattice_params: &Params,
        device: &wgpu::Device,
    ) -> Self {
        // TODO: I don't know if this is a good format. Leave it for now.
        let format = wgpu::TextureFormat::R32Float;
        let texture = device.create_texture(&wgpu::TextureDescriptor {
            label: None,
            size: wgpu::Extent3d {
                width: lattice_params.x_res,
                height: lattice_params.y_res,
                depth_or_array_layers: lattice_params.z_res,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D3,
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
            view_dimension: wgpu::TextureViewDimension::D3,
        }
    }
}