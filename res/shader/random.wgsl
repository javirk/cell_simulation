//-------------------------------------------------------------------------
// Hash
//-------------------------------------------------------------------------

fn Hash_Wang(_key: u32) -> u32 {
	var key: u32 = (_key ^ 61u) ^ (_key >> 16u);
	key = key + (key << 3u);
	key = key ^ (key >> 4u);
	key = key * 0x27D4EB2Du;
	key = key ^ (key >> 15u);
	return key;
}

fn PCG(state: u32) -> u32 {
	// According to this: https://www.reedbeta.com/blog/hash-functions-for-gpu-rendering/
	// Better performance and _much_ better distribution than the Wang hash
	var rng_state = state * 747796405u + 2891336453u;
	var word = ((state >> ((state >> 28u) + 4u)) ^ state) * 277803737u;
	return (word >> 22u) ^ word;
}


//-------------------------------------------------------------------------
// RNG
//-------------------------------------------------------------------------

fn UniformUintToFloat(u: u32) -> f32 {
    return f32(u) * bitcast<f32>(0x2F800000u);
}


fn UniformUint(_state: u32) -> u32{
    // Xorshift: slower than LCG better distribution for long sequences
	var state: u32 = _state ^ (_state << 13u);
	state ^= (state >> 17u);
	state ^= (state << 5u);
    
    // LCG: faster than Xorshift, but poorer distribution for long sequences
    //const uint multiplier = 1664525u;
	//const uint increment  = 1013904223u;
	//state *= multiplier;
    //state += increment;
    
    return state;
}

fn UniformFloat(state: u32) -> f32 {
    return UniformUintToFloat(UniformUint(state));
}

fn TestInclude() -> f32 {
	return 0.5f;
}