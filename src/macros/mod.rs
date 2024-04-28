#[macro_export]
macro_rules! debug_println {
    ($($arg:tt)*) => {
        #[cfg(feature = "debug-print")]
        {
            println!("[{}:{}] {}", file!(), line!(), format_args!($($arg)*));
        }
    }
}