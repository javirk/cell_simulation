/// Taken from https://github.com/elyshaffir/wgsl_preprocessor and modified slightly.

use std::{borrow, collections::HashMap};
use std::path::PathBuf;
use ex;

const INCLUDE_INSTRUCTION: &str = "//!include";

pub struct ShaderBuilder {
	/// String with the current WGSL source.
	/// It is marked public for debugging purposes.
	pub source_string: String,
	source_path: String,
}

impl ShaderBuilder {
	/// Creates a new [`ShaderBuilder`].
	///
	/// # Arguments
	/// - `source_path` - Path to the root WGSL module.
	///		All includes will be relative to the parent directory of the root WGSL module.
	/// 	Code is generated recursively with attention to `include` and `define` statements.
	/// 	See "Examples" for more details on include and macro functionality.
	pub fn new(source_path: &str) -> Result<Self, ex::io::Error> {
		let module_path = PathBuf::from(&source_path);
		let base_path = PathBuf::from("res").join("shader");
		let source_string = Self::load_shader_module(
			&base_path,
			&module_path,
		)?
		.0;
		Ok(Self {
			source_string,
			source_path: source_path.to_string(),
		})
	}

	/// Builds a [`wgpu::ShaderModuleDescriptor`] from the shader.
	/// The `label` member of the built [`wgpu::ShaderModuleDescriptor`] is the name of the shader file without the postfix.
	pub fn build(&self) -> wgpu::ShaderModuleDescriptor {
		wgpu::ShaderModuleDescriptor {
			label: Some(
				&self
					.source_path
					.rsplit(['/', '.'])
					.nth(1)
					.unwrap_or(&self.source_path),
			),
			source: wgpu::ShaderSource::Wgsl(borrow::Cow::Borrowed(&self.source_string)),
		}
	}

	fn load_shader_module(
		base_path: &PathBuf,
		module_path: &PathBuf,
	) -> Result<(String, HashMap<String, String>), ex::io::Error> {
		let module_source = ex::fs::read_to_string(base_path.join(module_path))?;
		let mut module_string = String::new();
		let mut definitions: HashMap<String, String> = HashMap::new();
		for line in module_source.lines() {
			if line.starts_with(INCLUDE_INSTRUCTION) {
				for include in line.split_whitespace().skip(1) {
					let (included_module_string, included_definitions) =
						Self::load_shader_module(base_path, &PathBuf::from(include))?;
					module_string.push_str(&included_module_string);
					definitions.extend(included_definitions);
				}
			} else {
				module_string.push_str(line);
				module_string.push('\n');
			}
		}
		definitions.iter().for_each(|(name, value)| {
			module_string = module_string.replace(name, value);
		});
		Ok((module_string, definitions))
	}
}