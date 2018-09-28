extern crate mvp_tree;

use std::env;
use std::fs::File;
use std::io::{self, Read};

// created to reproduce panics outsize of `libFuzzer`
// https://github.com/rust-fuzz/libfuzzer-sys/issues/36
fn main() {
    let mut data = Vec::new();

    if let Some(path) = env::args().nth(1) {
        let mut file = File::open(&path).expect(&format!("failed to open file: {}", path));
        file.read_to_end(&mut data).expect("failed to read file");
    } else {
        let mut stdin = io::stdin();
        stdin.lock().read_to_end(&mut data).expect("failed to read data from stdin");
    }

    mvp_tree::fuzz_iter(&data);
}
