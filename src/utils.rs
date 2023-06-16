use rand::{Rng, rngs::ThreadRng};

#[macro_export]
macro_rules! debug_println {
    ($($arg:tt)*) => (if ::std::cfg!(debug_assertions) { ::std::println!($($arg)*); })
}

pub fn random_direction_sphere(rng: &mut ThreadRng) -> [f32; 3] {
    let theta = 2.0 * std::f32::consts::PI * rng.gen::<f32>();
    let phi = (1.0 - 2.0 * rng.gen::<f32>()).acos();
    let x = phi.sin() * theta.cos();
    let y = phi.sin() * theta.sin();
    let z = phi.cos();
    [x, y, z]
}