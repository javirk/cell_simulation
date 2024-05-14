use rand::{Rng, rngs::ThreadRng};
use serde::Deserialize;

use crate::lattice::Lattice;

#[macro_export]
macro_rules! debug_println_old {
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


pub fn random_walk_direction(rng: &mut ThreadRng, lattice: &Lattice, site: &[u32; 3], block_length: f32, step_backwards: f32, regions_idx_buffer: &Vec<u32>) -> [f32; 3] {
    /// Generates a random direction for the random walk. Taking into account the distance to the boundary of the region.
    ///
    /// # Arguments
    ///
    /// * `rng` - A mutable reference to a `ThreadRng` object, which is used to generate random numbers.
    /// * `lattice` - A reference to a `Lattice` object, which represents the lattice in which the particle is moving.
    /// * `site` - A reference to an array of three `u32` values, which represents the current site of the particle in the lattice.
    /// * `block_length` - A `f32` value representing the length of a block in the lattice.
    /// * `step_backwards` - A `f32` value representing how many steps the algorithm will take backwards if it hits a boundary.
    /// * `regions_idx_buffer` - A reference to a `Vec` of `u32` values representing the indices of the regions in the lattice.
    ///
    /// # Returns
    ///
    /// An array of three `f32` values representing the direction in which the particle should move.
    fn build_site_check(site: &[u32; 3], direction: &[f32; 3], block_length: f32, step_backwards: f32) -> [u32; 3] {
        [
            (site[0] as f32 + direction[0] * block_length * step_backwards) as u32,
            (site[1] as f32 + direction[1] * block_length * step_backwards) as u32,
            (site[2] as f32 + direction[2] * block_length * step_backwards) as u32,
        ]
    }

    let mut direction = random_direction_sphere(rng);

    // Check the region of the block at direction * block_length * step_backwards. If it's the same region, we are fine to go, otherwise regenerate direction
    let mut site_check = build_site_check(site, &direction, block_length, step_backwards);
    let mut i_ret = 0;
    while i_ret < 100 {
        if site_check[0] >= lattice.lattice_params.res[0] || site_check[1] >= lattice.lattice_params.res[1] || site_check[2] >= lattice.lattice_params.res[2] {
            direction = random_direction_sphere(rng);
            site_check = build_site_check(site, &direction, block_length, step_backwards);
            i_ret += 1;
            continue;
        }
        let position_check = lattice.site_to_idx(site_check);
        if regions_idx_buffer.contains(&position_check) {
            break;
        } else {
            direction = random_direction_sphere(rng);
            site_check = build_site_check(site, &direction, block_length, step_backwards);
            i_ret += 1;
            continue;
        }
    }
    direction
}

pub fn split_whitespace(s: &str) -> Vec<&str> {
    let words: Vec<&str> = s.split_whitespace().collect();
    return words;
}

pub fn split_comma(s: &str) -> Vec<&str> {
    let words: Vec<&str> = s.split(",").collect();
    return words;
}

pub fn split_comma_f32(s: &str) -> Vec<f32> {
    let words: Vec<f32> = s.split(",").map(|x| x.parse().unwrap()).collect();
    return words;
}

pub fn json_value_to_array<T>(value: &serde_json::Value) -> [T; 3]
where
    T: Deserialize<'static> + Copy,
{
    let array: Vec<T> = value
        .as_array()
        .unwrap()
        .iter()
        .map(|v| v.as_str().unwrap().parse().unwrap())
        .collect();
    [array[0], array[1], array[2]]
}