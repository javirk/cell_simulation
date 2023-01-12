//!include random.wgsl
//!include functions.wgsl

struct Uniforms {
    itime: u32,
    frame_num: u32,
}

struct Lattice {
    lattice: array<u32>,
};

struct ReactionParams {
    num_species: u32,
    num_reactions: u32
}


@group(0) @binding(0) var<uniform> params: LatticeParams;
@group(0) @binding(1) var<uniform> reaction_params: ReactionParams;
@group(0) @binding(2) var<uniform> unif: Uniforms;

@group(1) @binding(1) var<storage, read_write> latticeDest: Lattice;
@group(1) @binding(4) var texture: texture_storage_3d<r32float, read_write>;

@group(2) @binding(0) var <storage, read_write> concentrations: array<f32>;
@group(2) @binding(1) var <storage, read> stoichiometry: array<f32>;
@group(2) @binding(2) var <storage, read> reactions_idx: array<i32>;
@group(2) @binding(3) var <storage, read> reaction_rates: array<u32>;
// Have a writable concentrations vector as output. This will be used to populate the lattice later instead of the reordering.
// Information can't be read from there unless I update it in RDME. That's ugly


@compute @workgroup_size(1, 1, 1)
fn cme(@builtin(global_invocation_id) global_id: vec3<u32>) {
    // Solve CME with Gillespie algorithm
    let idx_lattice = u32(get_index_lattice(global_id, params));
    let idx_occupancy = get_index_occupancy(global_id, params);

    // Compute concentrations
    var concentrations: array<f32> = array<f32>(reaction_params.num_species + 1, 0u); // +1 for the void species
    for (var i_part: u32 = idx_lattice; i_part < idx_lattice + params.max_particles_site; i_part += 1u) {
        var particle = latticeDest.lattice[i_part];
        concentrations[particle] += 1u;  // 0s will count too, but they are not considered later
    }
    concentrations[0] = 1u;  // In fact I make it 1 so that it doesn't interfere later for the propensity
    // If reactions_idx[i] == 0 --> concentration[0] = 1 --> It doesn't limit the propensity of the reaction

    // Compute propensities
    var propensities: array<f32> = array<f32>(reaction_params.num_reactions, 0.0f);
    var cumm_propensity: array<f32> = array<f32>(reaction_params.num_reactions, 0.0f);
    var total_propensity: f32 = 0.0f;
    for (var i_reaction: u32 = 0u; i_reaction < reaction_params.num_reactions; i_reaction += 1u) { // Make sure if this is like this or with +1 somewhere
        let k: f32 = reaction_rates[i_reaction];
        let i_reaction_idx = i_reaction * 3u;
        
        let propensity: f32 = k * concentrations[reactions_idx[i_reaction_idx]] * concentrations[reactions_idx[i_reaction_idx + 1]] * concentrations[reactions_idx[i_reaction_idx + 2]];
        propensities[i_reaction] = propensity;
        total_propensity += propensity;
        if (i_reaction > 0) {
            cumm_propensity[i_reaction] = cumm_propensity[i_reaction - 1] + propensity;
        } else {
            cumm_propensity[i_reaction] = propensity;
        }
        
    }

    // 1. Generate random and calculate tau
    // 2. Find next reaction with the propensity vector
    // 3. Perform reaction: I need to know where the different species are. That means another loop or save them in the previous one
    var state: u32 = Hash_Wang(unif.itime + X * Y + Z + u32(idx_lattice));
    var rand_number: f32 = UniformFloat(state);

    if total_propensity == 0. {
        return
    }

    let tau: f32 = (1. / total_propensity) * log(1. / rand_number);
    rand_number = UniformFloat(state + 1u);

    var i: i32 = 0;  // Index of the reaction
    while (rand_number > cumm_propensity[i]) {
        i += 1;
    }

    // Loop the stoichiometry matrix row and apply it to the concentrations vector
    let idx_reaction = i * reaction_params.num_species;
    for (var idx_species = 0; idx_species < reaction_params.num_species; idx_species += 1u) {
        concentrations[idx_species] += stoichiometry[idx_reaction + idx_species];
    }


}