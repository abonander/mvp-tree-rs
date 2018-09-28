#[macro_use] extern crate honggfuzz;

extern crate mvp_tree;

use mvp_tree::MvpTree;


fn main() {
    loop {
        fuzz!(|words: &[u16]| {
            let words = words.collect::<Vec<u16>>();
            let mut tree = MvpTree::new(|&l: &u16, &r| (l ^ r).count_ones() as u64);
            tree.extend(words.iter().cloned());
            words.sort();

            let mut words_test = tree.iter().cloned().collect::<Vec<_>>();
            words_test.sort();

            assert_eq!(words, words_test);
        });
    }
}

