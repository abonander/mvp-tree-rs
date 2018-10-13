#![no_main]
#[macro_use] extern crate libfuzzer_sys;
extern crate mvp_tree;


fuzz_target!(|data: &[u8]| {
    mvp_tree::fuzz_knn(data);
});
