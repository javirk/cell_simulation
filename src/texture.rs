use wgpu::util::DeviceExt;

pub struct Texture {
    pub texture_view: wgpu::TextureView,
    format: wgpu::TextureFormat,
    pub lock: Option<wgpu::Buffer>,
    view_dimension: wgpu::TextureViewDimension,
}

impl Texture {
    // This is important: https://www.w3.org/TR/WGSL/#storage-texel-formats
    pub fn new(
        dims: &[u32],
        needs_lock: bool,
        device: &wgpu::Device,
    ) -> Self {

        let (size, dimension, view_dimension) = match dims.len() {
            1 => (wgpu::Extent3d {
                    width: dims[0],
                    height: 1,
                    depth_or_array_layers: 1,
                }, wgpu::TextureDimension::D1, wgpu::TextureViewDimension::D1),
            2 => (wgpu::Extent3d {
                    width: dims[0],
                    height: dims[1],
                    depth_or_array_layers: 1,
                }, wgpu::TextureDimension::D2, wgpu::TextureViewDimension::D2),
            3 => (wgpu::Extent3d {
                    width: dims[0],
                    height: dims[1],
                    depth_or_array_layers: dims[2],
                }, wgpu::TextureDimension::D3, wgpu::TextureViewDimension::D3),
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
        let mut lock = None;
        if needs_lock {
            // Multiply all the elements of dims to get the total number of elements
            let size = dims.iter().fold(1, |acc, x| acc * x);
            let lock_data = vec![0; size as usize];
            lock = Some(device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("Lock Buffer"),
                contents: bytemuck::cast_slice(&lock_data),
                usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            }));
        }

        Texture {
            texture_view,
            format,
            view_dimension,
            lock,
        }
    }

    pub fn lock_entry(&self, binding: u32) -> wgpu::BindGroupEntry {
        wgpu::BindGroupEntry {
            binding: binding,
            resource: self.lock.as_ref().unwrap().as_entire_binding(),
        }
    }

    pub fn lock_layout_entry(&self, binding: u32) -> wgpu::BindGroupLayoutEntry {
        wgpu::BindGroupLayoutEntry {
            binding: binding,
            visibility: wgpu::ShaderStages::COMPUTE,
            ty: wgpu::BindingType::Buffer {
                ty: wgpu::BufferBindingType::Storage { read_only: false },
                has_dynamic_offset: false,
                min_binding_size: None,
            },
            count: None,
        }
    }

    pub fn layout_entry(&self, binding: u32) -> wgpu::BindGroupLayoutEntry {
        wgpu::BindGroupLayoutEntry {
            binding: binding,
            visibility: wgpu::ShaderStages::COMPUTE,
            ty: self.binding_type(wgpu::StorageTextureAccess::ReadWrite),
            count: None,
        }
    }

    pub fn bind_group_entry(&self, binding: u32) -> wgpu::BindGroupEntry {
        wgpu::BindGroupEntry {
            binding: binding,
            resource: self.binding_resource(),
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
            view_dimension: self.view_dimension,
        }
    }
}