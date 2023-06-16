//!include random.wgsl
//!include functions.wgsl

struct Lattice {
    lattice: array<u32>,
};

@group(0) @binding(0) var<uniform> params: LatticeParams;
@group(0) @binding(1) var<uniform> reaction_params: ReactionParams;
@group(0) @binding(2) var<uniform> unif: Uniforms;

@group(1) @binding(1) var<storage, read_write> latticeDest: Lattice;

@group(2) @binding(0) var <storage, read_write> concentrations: array<i32>;
@group(2) @binding(1) var <storage, read> stoichiometry: array<i32>;
@group(2) @binding(2) var <storage, read> reactions_idx: array<i32>;
@group(2) @binding(3) var <storage, read> reaction_rates: array<f32>;
@group(4) @binding(0) var<storage> reservoirs: array<u32>;

// Statistics bindings. It would be ideal to have them together in the same binding. Is that possible?
@group(3) @binding(0) var <storage, read_write> concentrations_stat: array<atomic<i32>>;

@compute @workgroup_size(1, 1, 1)
fn cme(@builtin(global_invocation_id) global_id: vec3<u32>) {
    // Solve CME with Gillespie algorithm
    let idx_concentration: i32 = get_index_occupancy(global_id, params) * i32(reaction_params.num_species + 1u);

    concentrations[idx_concentration] = 1; // I make it 1 so that it doesn't interfere later for the propensity
    // If reactions_idx[i] == 0 --> concentration[0] = 1 --> It doesn't limit the propensity of the reaction

    // Compute propensities
    var propensities: array<f32, MAX_REACTIONS>;
    var cumm_propensity: array<f32, MAX_REACTIONS>;
    var total_propensity: f32 = 0.0f;
    for (var i_reaction: u32 = 1u; i_reaction <= reaction_params.num_reactions; i_reaction += 1u) { // Make sure if this is like this or with +1 somewhere
        let k: f32 = reaction_rates[i_reaction];
        let i_reaction_idx = i_reaction * 3u;
        
        let propensity: f32 = k * f32(concentrations[idx_concentration + reactions_idx[i_reaction_idx]] * 
                                      concentrations[idx_concentration + reactions_idx[i_reaction_idx + 1u]] * 
                                      concentrations[idx_concentration + reactions_idx[i_reaction_idx + 2u]]);
        propensities[i_reaction] = propensity;
        total_propensity += propensity;
        if (i_reaction > 1u) {
            cumm_propensity[i_reaction] = cumm_propensity[i_reaction - 1u] + propensity;
        } else {
            cumm_propensity[i_reaction] = propensity;
        }
    }

    // 1. Generate random
    // 2. Find next reaction with the propensity vector
    // 3. Perform reaction: I need to know where the different species are. That means another loop or save them in the previous one
    var state: u32 = PCG(unif.itime + global_id.x * global_id.y + global_id.z);
    var rand_number: f32 = UniformFloat(state);

    if (rand_number <= 1. - exp(- total_propensity * params.tau)) {
        // One reaction can happen in this time
        rand_number = UniformFloat(state + 1u);

        var i: u32 = 0u;  // Index of the reaction
        while (rand_number > cumm_propensity[i]) {
            i += 1u;
        }
        // i = 1u;

        // Loop the stoichiometry matrix row and apply it to the concentrations vector. Update the lattice at the same time
        let idx_lattice = u32(get_index_lattice(global_id, params));
        var j_lattice = 0u;
        let idx_reaction = i32(i * (reaction_params.num_species + 1u));
        for (var idx_species = 1; idx_species <= i32(reaction_params.num_species); idx_species += 1) {
            if reservoirs[idx_lattice] == u32(idx_species) {
                // We don't update the concentrations because it's part of a reservoir
                continue;
            }
            concentrations[idx_concentration + idx_species] += stoichiometry[idx_reaction + idx_species];
            atomicAdd(&concentrations_stat[idx_species], stoichiometry[idx_reaction + idx_species]);
            
            // Now update the lattice:
            // Look at the concentration of the species in the site and write 
            var cc: i32 = concentrations[idx_concentration + idx_species];
            while (cc > 0) {
                latticeDest.lattice[idx_lattice + j_lattice] = u32(idx_species);
                cc -= 1;
                j_lattice += 1u;
                if (j_lattice >= params.max_particles_site) {
                    break;
                }
            }
        }
        while (j_lattice < params.max_particles_site) {
            latticeDest.lattice[idx_lattice + j_lattice] = 0u;
            j_lattice += 1u;
        }
    } 
}