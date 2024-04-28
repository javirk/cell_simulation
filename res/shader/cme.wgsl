//!include random.wgsl
//!include functions.wgsl

struct Lattice {
    lattice: array<u32>,
};

@group(0) @binding(0) var<uniform> params: LatticeParams;
@group(0) @binding(1) var<uniform> reaction_params: ReactionParams;
@group(0) @binding(2) var<uniform> unif: Uniforms;

@group(1) @binding(1) var<storage, read_write> latticeDest: Lattice;
@group(1) @binding(3) var<storage, read_write> occupancyDest: array<atomic<u32>>;

@group(2) @binding(0) var <storage, read_write> concentrations: array<i32>;
@group(2) @binding(1) var <storage, read> stoichiometry: array<i32>;
@group(2) @binding(2) var <storage, read> reactions_idx: array<i32>;
@group(2) @binding(3) var <storage, read> reaction_rates: array<f32>;
@group(4) @binding(0) var<storage> reservoirs: array<u32>;

// Statistics bindings. It would be ideal to have them together in the same binding. Is that possible?
@group(3) @binding(0) var <storage, read_write> concentrations_stat: array<atomic<i32>>;


@compute @workgroup_size(2, 1, 1)  // Why repeat it? Is it necessary to add it here?
fn cme(@builtin(global_invocation_id) global_id: vec3<u32>) {
    // Solve CME with Gillespie algorithm
    let idx_occupancy: i32 = get_index_occupancy(global_id, params);
    let idx_concentration: i32 = idx_occupancy * i32(reaction_params.num_species + 1u);

    concentrations[idx_concentration] = 1; // I make it 1 so that it doesn't interfere later for the propensity
    // If reactions_idx[i] == 0 --> concentration[0] = 1 --> It doesn't limit the propensity of the reaction

    // Compute propensities
    var cumm_propensity: array<f32, MAX_REACTIONS>;
    cumm_propensity[0] = 0.;
    var total_propensity: f32 = 0.0f;
    for (var i_reaction: u32 = 1u; i_reaction <= reaction_params.num_reactions; i_reaction += 1u) { // Make sure if this is like this or with +1 somewhere
        let k: f32 = reaction_rates[i_reaction];
        let i_reaction_idx = i_reaction * 3u;
        
        var propensity: f32 = k * f32(concentrations[idx_concentration + reactions_idx[i_reaction_idx]] * 
                                      concentrations[idx_concentration + reactions_idx[i_reaction_idx + 1u]] * 
                                      concentrations[idx_concentration + reactions_idx[i_reaction_idx + 2u]]);
        total_propensity += propensity;
        cumm_propensity[i_reaction] = cumm_propensity[i_reaction - 1u] + propensity;
    }

    // Renormalize propensities 
    for (var i: u32 = 0u; i < reaction_params.num_reactions; i += 1u) {
        cumm_propensity[i] = cumm_propensity[i] / total_propensity; 
    }

    let idx_lattice = u32(get_index_lattice(global_id, params));
    // Copy the contents of the cell just in case it overflows:
    var cell_copy: array<u32, 16>;  // It should be length params.max_particles_site, but that's not allowed. TODO: Check this
    for (var i_particle: u32 = 0u; i_particle < params.max_particles_site; i_particle += 1u) {
        cell_copy[i_particle] = latticeDest.lattice[idx_lattice + i_particle];
    }

    // 1. Generate random
    // 2. Find next reaction with the propensity vector
    // 3. Perform reaction: I need to know where the different species are. That means another loop or save them in the previous one
    var state: u32 = PCG(unif.itime + global_id.x * global_id.y + global_id.z);
    var rand_number: f32 = UniformFloat(state);
    if (rand_number <= 1. - exp(-total_propensity * params.tau)) {
        // One reaction can happen in this time
        rand_number = UniformFloat(state + 1u);

        var i: u32 = 0u;  // Index of the reaction
        while (rand_number > cumm_propensity[i] & i < reaction_params.num_reactions) {
            i += 1u;
        }

        // Loop the stoichiometry matrix row and apply it to the concentrations vector. Update the lattice at the same time
        var j_lattice = 0u;
        let idx_reaction = i32(i * (reaction_params.num_species + 1u));
        var new_occupancy = i32(occupancyDest[idx_occupancy]);
        for (var idx_species = 1; idx_species <= i32(reaction_params.num_species); idx_species += 1) {
            if reservoirs[idx_lattice] == u32(idx_species) {
                // We don't update the concentrations because the particle is part of a reservoir
                continue;
            }
            
            concentrations[idx_concentration + idx_species] += stoichiometry[idx_reaction + idx_species];
            new_occupancy += stoichiometry[idx_reaction + idx_species];
            let original = atomicAdd(&concentrations_stat[idx_species], stoichiometry[idx_reaction + idx_species]); 
            // Now update the lattice:
            // Look at the concentration of the species in the site and write 
            var cc: i32 = concentrations[idx_concentration + idx_species];
            while (cc > 0) {
                latticeDest.lattice[idx_lattice + j_lattice] = u32(idx_species);
                cc -= 1;
                j_lattice += 1u;
                if (j_lattice >= params.max_particles_site) {
                    // It doesn't fit. We have to step back and not perform the reaction
                    // Reset the lattice
                    for (var i_particle: u32 = 0u; i_particle < params.max_particles_site; i_particle += 1u) {
                        latticeDest.lattice[idx_lattice + i_particle] = cell_copy[i_particle];
                    }
                    // Reset the concentrations (only up to idx_species)
                    for (var idx_species_reset = 1; idx_species_reset <= idx_species; idx_species_reset += 1) {
                        //atomicAdd(&concentrations[idx_concentration + idx_species_reset], -stoichiometry[idx_reaction + idx_species_reset]);
                        concentrations[idx_concentration + idx_species_reset] -= stoichiometry[idx_reaction + idx_species_reset];
                        atomicAdd(&concentrations_stat[idx_species_reset], -stoichiometry[idx_reaction + idx_species_reset]); 
                    }
                    // Exit the program
                    return;
                    //break;
                }
            }
        }
        // Write the new occupancy:
        atomicStore(&occupancyDest[idx_occupancy], u32(new_occupancy));
        // Fill the rest with zeros
        while (j_lattice < params.max_particles_site) {
            latticeDest.lattice[idx_lattice + j_lattice] = 0u;
            j_lattice += 1u;
        }
    } 
}