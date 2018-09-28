#[macro_use] extern crate honggfuzz;

extern crate mvp_tree;

fn main() {
    loop {
        fuzz!(|data: &[u8]| {
            mvp_tree::fuzz_iter(data);
        });
    }
}

