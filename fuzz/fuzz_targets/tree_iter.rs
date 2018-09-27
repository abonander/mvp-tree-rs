#![no_main]
#[macro_use] extern crate libfuzzer_sys;
extern crate mvp_tree;

use mvp_tree::MvpTree;

fuzz_target!(|data: &[u8]| {
    let mut words = data.chunks(2)
        .map(|chunk| (chunk[0] as u16 | (chunk[1] as u16) << 8))
        .collect::<Vec<_>>();

    let mut tree = MvpTree::new(|&l: &u16, &r| (l ^ r).count_ones() as u64);
    tree.extend(words.iter().cloned());
    words.sort();

    let mut words_test = tree.iter().cloned().collect::<Vec<_>>();
    words_test.sort();

    assert_eq!(words, words_test);
});
