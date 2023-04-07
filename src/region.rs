pub enum RegionType {
    Rectangle { p0: [f32; 3], pf: [f32; 3] },
    Circle { center: [f32; 3], radius: u32 },
}

pub struct Rectangle {
    pub p0: [f32; 3],
    pub pf: [f32; 3],
}

pub struct Circle {
    pub center: [f32; 3],
    pub radius: u32,
}