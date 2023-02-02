pub struct Texture {
    pub texture_view: wgpu::TextureView,
    format: wgpu::TextureFormat,
}

impl Texture {
    // This is important: https://www.w3.org/TR/WGSL/#storage-texel-formats
    pub fn new(
        dims: &[u32],
        device: &wgpu::Device,
    ) -> Self {

        let (size, dimension) = match dims.len() {
            1 => (wgpu::Extent3d {
                    width: dims[0],
                    height: 1,
                    depth_or_array_layers: 1,
                }, wgpu::TextureDimension::D1),
            2 => (wgpu::Extent3d {
                    width: dims[0],
                    height: dims[1],
                    depth_or_array_layers: 1,
                }, wgpu::TextureDimension::D2),
            3 => (wgpu::Extent3d {
                    width: dims[0],
                    height: dims[1],
                    depth_or_array_layers: dims[2],
                }, wgpu::TextureDimension::D3),
            _ => panic!("Invalid dimensions"),
        };


        // TODO: I don't know if this is a good format. Leave it for now.
        let format = wgpu::TextureFormat::R32Float;
        let texture = device.create_texture(&wgpu::TextureDescriptor {
            label: None,
            size: size,
            mip_level_count: 1,
            sample_count: 1,
            dimension:dimension,
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