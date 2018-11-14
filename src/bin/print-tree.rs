extern crate mvp_tree;

use mvp_tree::MvpTree;

fn main() {
    let mut tree = MvpTree::new(|l, r| if l < r { r - l } else { l - r });
    tree.extend(0u64 .. 100);
    println!("Tree structure:\n{:?}", tree);
}
