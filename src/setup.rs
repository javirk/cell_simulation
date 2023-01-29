use winit::{
    event::WindowEvent,
    window::Window,
};

pub struct Setup {
    pub window: winit::window::Window,
    instance: wgpu::Instance,
    size: winit::dpi::PhysicalSize<u32>,
    pub surface: wgpu::Surface,
    adapter: wgpu::Adapter,
    pub device: wgpu::Device,
    pub queue: wgpu::Queue,
    pub config: wgpu::SurfaceConfiguration,
}

impl Setup {
    // Think whether this should be here in the examples or inside the library
    pub async fn new(window: Window) -> Self {
        let mut builder = winit::window::WindowBuilder::new();
        builder = builder.with_title("Example");
    
        log::info!("Initializing the surface...");
    
        let backend = wgpu::util::backend_bits_from_env().unwrap_or_else(wgpu::Backends::all);
    
        let instance = wgpu::Instance::new(backend);
    
        let (size, surface) = {
            let size = window.inner_size();
    
            let surface = unsafe { instance.create_surface(&window) };
    
            (size, surface)
        };
    
        let adapter =
            wgpu::util::initialize_adapter_from_env_or_default(&instance, backend, Some(&surface))
                .await
                .expect("No suitable GPU adapters found on the system!");
    
        {
            let adapter_info = adapter.get_info();
            println!("Using {} ({:?})", adapter_info.name, adapter_info.backend);
        }
    
        let required_features = wgpu::Features::TEXTURE_ADAPTER_SPECIFIC_FORMAT_FEATURES;
        let adapter_features = adapter.features();
        assert!(
            adapter_features.contains(required_features),
            "Adapter does not support required features for this example: {:?}",
            required_features - adapter_features
        );
    
        // Make sure we use the texture resolution limits from the adapter, so we can support images the size of the surface.
        let needed_limits = wgpu::Limits::downlevel_defaults().using_resolution(adapter.limits());
    
        let trace_dir = std::env::var("WGPU_TRACE");
        let (device, queue) = adapter
            .request_device(
                &wgpu::DeviceDescriptor {
                    label: None,
                    features: (adapter_features) | required_features,
                    limits: needed_limits,
                },
                trace_dir.ok().as_ref().map(std::path::Path::new),
            )
            .await
            .expect("Unable to find a suitable GPU adapter!");

        let config = wgpu::SurfaceConfiguration {
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT,
            format: surface.get_supported_formats(&adapter)[0],
            width: size.width,
            height: size.height,
            present_mode: wgpu::PresentMode::Fifo,
            alpha_mode: wgpu::CompositeAlphaMode::Auto,
        };
        surface.configure(&device, &config);
        
        Setup {
            window,
            instance,
            size,
            surface,
            adapter,
            device,
            queue,
            config,
        }
    }

    pub fn resize(&mut self, new_size: winit::dpi::PhysicalSize<u32>) {
        if new_size.width > 0 && new_size.height > 0 {
            self.size = new_size;
            self.config.width = new_size.width;
            self.config.height = new_size.height;
            self.surface.configure(&self.device, &self.config);
        }
    }

    pub fn input(&mut self, event: &WindowEvent) -> bool {
        false
    }

    pub fn update(&mut self) {
    }

    pub fn window(&self) -> &Window {
        &self.window
    }
     
}