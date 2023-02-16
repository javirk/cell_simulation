use std::{time::{SystemTime, UNIX_EPOCH}, collections::{HashMap, VecDeque}, sync::Arc};
use std::time::Instant;
use simulation::{Simulation, Setup, Uniform, UniformBuffer, LatticeParams, Texture};
use std::error::Error;
use csv::Writer;


struct CellSimulation {
    simulation: Simulation,
    uniform_buffer: UniformBuffer,
    all_stats: HashMap<String, StatisticContaner>
}

#[derive(Debug)]
struct StatisticContaner {
    x: VecDeque<u32>,
    y: VecDeque<f32>,  // Must be f32 for the UI

}

impl StatisticContaner {

    fn new(capacity: usize) -> Self {
        StatisticContaner { x: VecDeque::with_capacity(capacity), y: VecDeque::with_capacity(capacity) }
    }

    fn add(&mut self, x: u32, y: f32) {
        while self.x.capacity() == self.x.len() {
            self.x.pop_front();
            self.y.pop_front();
        }
        self.x.push_back(x);
        self.y.push_back(y);
    }

    fn mean(&self) -> f32 {
        let mut sum = 0.;
        for y in &self.y {
            sum += y;
        }
        sum / self.y.len() as f32
    }

    fn last(&self) -> f32 {
        match self.y.back() {
            Some(y) => *y,
            None => 0.
        }
    }

    fn to_csv(&mut self, name: &str)  -> Result<(), Box<dyn Error>> {
        // TODO: This shouldn't be here.
        let mut wtr = Writer::from_path(name)?;
        while let (Some(x), Some(y)) = (self.x.pop_front(), self.y.pop_front()) {
            wtr.write_record(&[x.to_string(), y.to_string()])?;
        }
            
        wtr.flush()?;
        Ok(())
    }
}

fn make_all_stats(metrics_log: Vec<&str>) -> HashMap<String, StatisticContaner> {
    let mut all_stats = HashMap::new();
    for metric in metrics_log {
        all_stats.insert(metric.to_string(), StatisticContaner::new(100));
    }
    all_stats
}


fn setup_system(device: &wgpu::Device) -> CellSimulation {
    let uniform = Uniform {
        itime: SystemTime::now().duration_since(UNIX_EPOCH).unwrap().as_millis() as u32,
        frame_num: 0
    };
    let uniform_buffer = UniformBuffer::new(uniform, device);

    let lattice_resolution = [32, 32, 32];
    let dimensions = [1., 1., 1.];
    let tau = 3E-3;
    let lambda = 31.25E-9;
    
    //let render_params = RenderParams::new(device, &render_param_data);
    let simulation_params = LatticeParams::new(dimensions, lattice_resolution, tau, lambda);
    let texture = Texture::new(&lattice_resolution, false, &device);
    
    let mut simulation = Simulation::new(simulation_params);

    simulation.add_region("one", vec![0.,0.,0.], vec![1.,1.,1.], 8.15E-14/6.);
    // simulation.add_region("two", vec![0.2,0.2,0.2], vec![0.8,0.8,0.8], 6.3);
    simulation.add_particle("A", "one", 1000, true);
    simulation.add_particle("B", "one", 1000, false);
    simulation.add_particle("C", "one", 0, false);

    simulation.add_reaction(vec!["A", "B"], vec!["C"], 5.82);
    simulation.add_reaction(vec!["C"], vec!["A", "B"], 0.351);

    simulation.prepare_for_gpu(&uniform_buffer, &texture, device);

    let stats_container = make_all_stats(vec!["A"]);
    
    CellSimulation {
        simulation,
        uniform_buffer: uniform_buffer,
        all_stats: stats_container
    }
}

fn step_system(
    simulation: &mut CellSimulation,
    device: &wgpu::Device,
    queue: &wgpu::Queue
) {
    let frame_num = simulation.uniform_buffer.data.frame_num;
    let mut command_encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor { label: None });

    simulation.simulation.step(frame_num, &mut command_encoder, device);

    simulation.uniform_buffer.data.frame_num += 1;
    let t = SystemTime::now().duration_since(UNIX_EPOCH).unwrap().as_micros() as u32;
    assert_ne!(t, simulation.uniform_buffer.data.itime);
    simulation.uniform_buffer.data.itime = t;

    queue.write_buffer(&simulation.uniform_buffer.buffer, 0, bytemuck::cast_slice(&[simulation.uniform_buffer.data]));
    queue.submit(Some(command_encoder.finish()));
}


pub async fn run() {
    env_logger::init();

    let state = Setup::new_nowindow().await;

    let mut last_frame_inst = Instant::now();
    let (mut frame_count, mut accum_time, mut fps, mut time) = (0, 0., 0., 0.);
    // Setup the simulation.
    let mut simulation = setup_system(&state.device);

    while time < 10. {
        step_system(&mut simulation, &state.device, &state.queue);

        {
            while let Some(sample) = simulation.simulation.stats.pop_front() {
                //println!("{}: {} {}", sample.name, time, sample.value);
                println!("FPS: {}\n", fps);
                simulation.all_stats.entry(sample.name).and_modify(|k| k.add(sample.iteration_count, sample.value as f32));
            }
            accum_time += last_frame_inst.elapsed().as_secs_f32();
            last_frame_inst = Instant::now();
            frame_count += 1;
            if frame_count == 100 {
                fps = frame_count as f32 / accum_time;
                accum_time = 0.0;
                frame_count = 0;
            }
        }

        time += simulation.simulation.lattice_params.raw.tau;
    }

    simulation.all_stats.get_mut("A").unwrap().to_csv("A.csv").unwrap();
}

fn main() {
    pollster::block_on(run());
}