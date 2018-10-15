
use std::cmp::{self, Ordering};
use std::collections::BinaryHeap;
use std::marker::PhantomData;
use std::ptr::{self, NonNull};
use std::{mem, slice};

mod node;

use node::{Node, NODE_SIZE};

/// A n-ary tree which organizes items based on a distance metric.
///
/// Like a B-tree is a generalization of a binary tree, this is a generalization of
/// a [vantage-point (VP) tree][vp-tree-wiki] to multiple items per node. Unlike a binary search
/// tree, a vantage-point tree does not require its contents to implement any kind of ordering or
/// equality; instead, a single distance function is provided and elements are organized purely
/// based on their distances to one another.
///
/// This is useful for data with a very high dimensionality such that a regular ordering
/// is difficult to define or is simply not meaningful. The originally intended use-case of this
/// implementation is to organize, by Hamming distance (`left ^ right`, count one-bits),
/// image perceptual hash bitstrings created by [the `img_hash` crate][img_hash].
/// The equivalent k-d tree would need as many dimensions as there are bits in the hashes.
///
/// At a high level, each node has a radius chosen from the median of its distances to some number
/// of the previously inserted items. Items that fall within this radius are inserted into
/// the left child of the node while items that fall outside are inserted into the right
/// child. A multiple-vantage-point (MVP) tree has several items per node, each with its own
/// radius; an insertion or a search finds the first item in the node from the left that contains
/// the searched-for item within its radius.
///
/// Because equality is not required, this tree does not implement a map or set interface.
/// In some use-cases like the one mentioned above (the perceptual hash is only an
/// approximation of the image content; hashes can be equal while the images are quite different),
/// elements can have a 0 distance and not actually be equivalent.
///
/// The primary operation this tree is designed to support is [k-nearest-neighbor search][kNN],
/// finding the `k` elements closest to a given item in distance; the item itself may or may not
/// reside within the tree.
///
/// ### Distance Function
/// A distance function is required upon the construction of the tree, which will be used
/// to organize elements as they are added to the tree. Some academic sources for (M)VP-trees
/// describe this distance function as necessarily being a [Euclidean metric] but the author
/// of this crate is not currently sure if this is a hard requirement for the algorithm.
///
/// The properties the distance function certainly must have are as follows:
///
/// * reflexive: `dist(a, a)` must be 0
/// * deterministic: `dist(a, b)` must always return the same value for the same inputs
/// * commutative: `dist(a, b) == dist(b, a)`
/// * (unknown term): `dist(a, b) + dist(b, c) >= dist(a, c)`; the shortest distance between
/// two points (items) must be a straight line; no wormholes allowed
/// (this property may actually make the distance function into a Euclidean metric)
///
/// It is a logic error to mutate an item (via mutable references or internally mutable containers
/// like `Cell` and `RefCell` or unsafe code) in a way that changes its distances to other items in
/// the tree. Distance should also not be based on global mutable state or I/O (non-deterministic).
///
/// ### Note: Not Self-Balancing
/// The swaps and rotations that binary/B-trees perform to balance themselves are
/// not really feasible with a vantage-point tree as every node is organized
/// based on the distance to its parent. Swapping another node into a parent
/// would require recalculating the distances of all its children.
///
/// Vantage-point trees are typically designed to be constructed from a complete
/// set of data such that they can choose reasonable radii for internal nodes
/// by finding the median of the distances from one item to the rest of the dataset.
///
/// This MVP tree is designed instead to allow dynamic construction like other Rust
/// collections. Instead of calculating the median distances to large portions of the set,
/// it only chooses radii from the distances to a small set of items, typically the `~n` previously
/// inserted items. This means that if your dataset has a low entropy, e.g. items are inserted
/// in clusters where they have small distances to each other or are in some sort of ordering,
/// the resulting tree structure will be suboptimal for fast searches and insertions.
///
/// A simple example of a degenerate case is inserting items in order of their distance to each
/// other:
///
/// ```rust
/// # use mvp_tree::MvpTree;
/// let mut tree = MvpTree::new(|l, r| (l - r).abs());
/// // since the following items are all going to be outside the pivot and radius picked for the
/// // first node, the tree will be lopsided with long branches on its right side
/// tree.extend(0i32 .. 50);
/// ```
///
/// Since real-world data is expected to not sort so cleanly (e.g. Hamming distances of image
/// hashes) the tree makes no real effort to counter this. If you are concerned about degenerate
/// cases (maybe using the tree with untrusted data) then you may want to shuffle your inputs
/// somehow.
///
/// [vp-tree-wiki]: https://en.wikipedia.org/wiki/Vantage-point_tree
/// [img_hash]: https://crates.io/crates/img_hash
/// [kNN]: https://en.wikipedia.org/wiki/K-nearest_neighbors_algorithm
/// [Euclidean metric]: https://en.wikipedia.org/wiki/Euclidean_distance
pub struct MvpTree<T, Df> {
    // must be boxed so it has a stable address
    root: Option<Box<Node<T>>>,
    dist_fn: Df,
    len: usize,
    height: usize,
}

impl<T, Df> MvpTree<T, Df> where Df: Fn(&T, &T) -> u64 {
    /// Construct an empty tree with the given distance function.
    ///
    /// See the [Distance Function](#distance-function) header above for more information.
    pub fn new(dist_fn: Df) -> Self {
        MvpTree {
            root: None,
            dist_fn,
            len: 0,
            height: 0,
        }
    }

    /// The number of items in this tree.
    pub fn len(&self) -> usize {
        self.len
    }

    /// The current maximum height of the tree.
    ///
    /// This is roughly proportional to the maximum time it takes to access a node in the tree.
    pub fn height(&self) -> usize {
        self.height
    }

    /// Insert `item` into the tree based on the distance function provided at construction.
    ///
    /// See the [note about balancing](#note-not-self-balancing) above.
    pub fn insert(&mut self, mut item: T) {
        if let None = self.root {
            let mut root = Node::new_box();
            root.push_item(item);
            self.root = Some(root);
            self.height = 1;
            self.len = 1;
            return;
        }

        /// Find the node where `item` belongs and insert it there
        fn find_insert<T, Df>(node: &mut Node<T>, item: T, dist_fn: Df, depth: usize) -> usize
        where Df: Fn(&T, &T) -> u64 {
            if node.is_leaf() && !node.is_full() {
                // if the target node is a non-full leaf node, just push the item
                node.push_item(item);
                return depth;
            }

            let distances = node.get_distances(&item, &dist_fn);

            if node.is_leaf() { // && node.is_full()
                // safe because we're not modifying the tree at any other point
                if let Some((parent, NODE_SIZE)) = unsafe { node.parent_mut_and_idx() } {
                    if !parent.is_full() {
                        // we are the far right leaf child of a non-full parent
                        parent.add_child(item, &distances);
                        return depth;
                    }
                }

                // make the node internal, increasing the depth by 1
                node.make_internal(item, &distances);
                return depth + 1;
            }

            if let Some(child_idx) = node.find_parent(&distances) {
                // recurse into the appropriate child
                return find_insert(node.child_mut(child_idx), item, dist_fn, depth + 1);
            }

            find_insert(node.far_right_child_mut(), item, dist_fn, depth + 1)
        }

        let new_len = self.len.checked_add(1).expect("overflow `self.len + 1`");

        let mut node = &mut **self.root.as_mut().unwrap();
        let insert_depth = find_insert(node, item, &self.dist_fn, 0);

        // update the height if it increased
        self.height = cmp::max(self.height, insert_depth + 1);
        self.len = new_len;
    }

    /// Search the tree for the `k` closest items to the given item, based
    /// on the distance function the tree was constructed with.
    ///
    /// Neighbors are sorted in ascending order based on distance to `item`.
    ///
    /// This ignores `item` itself if it resides within the tree, as determined
    /// by referential equality (`std::ptr::eq()`, i.e. if `item` and the item in the tree
    /// have the same runtime pointer value). It is expected that the meaning of a 0 distance will
    /// vary based on context so the tree does not assume that a 0 distance means equality.
    // noinspection RsNeedlessLifetimes inspection wants us to elide `'a` here but
    // we don't want to require `item` to live as long as `'a` since it's not returned
    pub fn k_nearest<'a>(&'a self, k: usize, item: &T) -> Vec<Neighbor<'a, T>> {
        if k == 0 { return vec![]; }

        let root = match self.root {
            Some(ref root) => root,
            None => return vec![],
        };

        // don't allocate if the tree is empty
        let mut neighbors = BinaryHeap::with_capacity(k);

        let mut visitor = KnnVisitor {
            neighbors, item, k, dist_fn: &self.dist_fn,
        };

        visitor.visit(root);

        visitor.neighbors.into_sorted_vec()
    }

    /// Get an iterator that returns immutable references to items in this tree.
    ///
    /// The iteration order is unspecified but deterministic.
    ///
    /// ### Note
    /// It is a logic error to mutate an item (via mutable references or internally mutable
    /// containers like `Cell` and `RefCell` or unsafe code) in a way that changes its distances to
    /// other items in the tree.
    pub fn iter(&self) -> Iter<T> {
        Iter {
            dfs: DepthFirst::new(&self.root),
            items: None,
            _borrow: PhantomData,
        }
    }

    /// Get an iterator that returns mutable references to items in this tree.
    ///
    /// The iteration order is unspecified but deterministic.
    ///
    /// ### Note
    /// It is a logic error to mutate an item (via mutable references or internally mutable
    /// containers like `Cell` and `RefCell` or unsafe code) in a way that changes its distances to
    /// other items in the tree.
    pub fn iter_mut(&mut self) -> IterMut<T> {
        IterMut {
            dfs: DepthFirst::new(&self.root),
            items: None,
            _borrow: PhantomData,
        }
    }
}

impl<T, Df> Extend<T> for MvpTree<T, Df> where Df: Fn(&T, &T) -> u64 {
    fn extend<I: IntoIterator<Item=T>>(&mut self, iter: I) {
        for item in iter {
            self.insert(item);
        }
    }
}

/// Item returned by [`MvpTree::k_nearest()`](MvpTree::k_nearest).
#[derive(Debug)]
pub struct Neighbor<'a, T: 'a> {
    /// The distance from `item` to the item passed to `k_nearest()`.
    pub dist: u64,
    /// The item this instance concerns.
    pub item: &'a T,
    _compat: (),
}

impl<'a, T: 'a> PartialEq for Neighbor<'a, T> {
    fn eq(&self, other: &Self) -> bool {
        self.dist == other.dist
    }
}

impl<'a, T: 'a> Eq for Neighbor<'a, T> {}

impl<'a, T: 'a> PartialOrd for Neighbor<'a, T> {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

impl<'a, T: 'a> Ord for Neighbor<'a, T> {
    fn cmp(&self, other: &Self) -> Ordering {
        self.dist.cmp(&other.dist)
    }
}

impl<'a, T: 'a + PartialEq<T>> PartialEq<T> for Neighbor<'a, T> {
    fn eq(&self, other: &T) -> bool { *self.item == *other }
}

/// Immtable iterator for `MvpTree`.
///
/// The iteration order is unspecified but deterministic.
///
/// ### Note
/// It is a logic error to mutate an item (via mutable references or internally mutable
/// containers like `Cell` and `RefCell` or unsafe code) in a way that changes its distances to
/// other items in the tree.
pub struct Iter<'a, T: 'a> {
    dfs: DepthFirst<T>,
    items: Option<node::FilteredItems<'a, T>>,
    _borrow: PhantomData<&'a T>,
}

impl<'a, T: 'a> Iterator for Iter<'a, T> {
    type Item = &'a T;

    fn next(&mut self) -> Option<&'a T> {
        loop {
            if let Some((_, next)) = self.items.as_mut().and_then(|i| i.next()) {
                return Some(next);
            }

            unsafe {
                let next_node = self.dfs.next();
                self.items = Some(next_node.as_ref()?.filtered_items());
            }
        }
    }
}

/// Mutable iterator for `MvpTree`.
///
/// The iteration order is unspecified but deterministic.
///
/// ### Note
/// It is a logic error to mutate an item (via mutable references or internally mutable
/// containers like `Cell` and `RefCell` or unsafe code) in a way that changes its distances to
/// other items in the tree.
pub struct IterMut<'a, T: 'a> {
    dfs: DepthFirst<T>,
    items: Option<node::FilteredItemsMut<'a, T>>,
    _borrow: PhantomData<&'a mut T>,
}

impl<'a, T: 'a> Iterator for IterMut<'a, T> {
    type Item = &'a mut T;

    fn next(&mut self) -> Option<&'a mut T> {
        loop {
            if let Some((_, next)) = self.items.as_mut().and_then(|i| i.next()) {
                return Some(next);
            }

            unsafe {
                let next_node = self.dfs.next() as *mut Node<T>;
                self.items = Some(next_node.as_mut()?.filtered_items_mut());
            }
        }
    }
}

struct DepthFirst<T> {
    // NULLABLE
    node: *const Node<T>,
    descend: bool,
}

impl<T> DepthFirst<T> {
    fn new(root: &Option<Box<Node<T>>>) -> Self {
        DepthFirst {
            node: root.as_ref().map_or(ptr::null(), |r| &**r),
            descend: true,
        }
    }

    // NULLABLE
    unsafe fn next(&mut self) -> *const Node<T> {
        if self.node.is_null() {
            return self.node;
        }

        // `self.node` should NOT be NULL after this point
        if !self.descend {
            if let Some((parent, child_idx)) = (*self.node).parent_and_idx() {
                if child_idx + 1 < parent.len() {
                    // descend into the next sibling, excluding far right child
                    self.node = parent.child(child_idx + 1);
                    self.descend = true;
                } else if child_idx == NODE_SIZE {
                    // ascend out of the far right child
                    self.node = parent;
                } else {
                    // descend into the far right child
                    self.node = parent.far_right_child();
                    self.descend = true;
                }
            } else {
                self.node = ptr::null();
            }
        }

        if self.descend {
            while !(*self.node).is_leaf() {
                self.node = (*self.node).child(0);
            }
            self.descend = false;
        }

        self.node
    }
}

struct BreadthFirst<T> {
    node: *const Node<T>,
    depth: usize,
    max_depth: usize,
}

impl<T> BreadthFirst<T> {
    fn new(root: &Option<Box<Node<T>>>) -> Self {
        BreadthFirst {
            node: root.as_ref().map_or(ptr::null(), |r| &**r),
            depth: 0,
            max_depth: 0,
        }
    }

    // NULLABLE
    unsafe fn next(&mut self) -> *const Node<T> {
        if self.node.is_null() {
            return self.node;
        }

        while !(*self.node).is_leaf() && self.depth < self.max_depth {
            self.node = (*self.node).child(0);
            self.depth += 1;
        }

        let ret_node = self.node;

        while let Some((parent, child_idx)) = (*self.node).parent_and_idx() {
            if child_idx + 1 == (*self.node).len() {
                self.node = (*self.node).far_right_child();
                break;
            } else if child_idx < (*self.node).len() {
                self.node = (*self.node).child(child_idx + 1);
                break;
            } else {
                self.node = parent;
                self.depth -= 1;
            }
        }

        if !(*self.node).has_parent() {
            self.max_depth += 1;
        }

        ret_node
    }
}

struct KnnVisitor<'i, 'n, T: 'i + 'n, Df> {
    item: &'i T,
    dist_fn: Df,
    neighbors: BinaryHeap<Neighbor<'n, T>>,
    k: usize,
}

impl<'i, 'n, T: 'i + 'n, Df> KnnVisitor<'i, 'n, T, Df> where Df: Fn(&T, &T) -> u64 {
    /// Push a `Neighbor` onto the heap, always limiting its size to `k` items
    fn push_heap(&mut self, item: &'n T, dist: u64) {
        if ptr::eq(item, self.item) { return }

        // if the heap is full, swap the item onto it only if its distance
        // is smaller than the current maximum
        if self.neighbors.len() == self.k {
            let mut peek_mut = self.neighbors.peek_mut().expect("heap shouldn't be empty");
            if peek_mut.dist > dist {
                peek_mut.dist = dist;
                peek_mut.item = item;
            }
            // `peek_mut` sifts the item down on-drop
        } else {
            self.neighbors.push(Neighbor {
                item, dist, _compat: (),
            })
        }
    }

    fn visit(&mut self, node: &'n Node<T>) {
        let distances = node.get_distances(self.item, &self.dist_fn);

        let items_dists = node.items().iter().zip(&distances);

        if node.is_leaf() {
            for (idx, (item, &dist)) in items_dists.enumerate() {
                if !node.is_removed(idx) {
                    self.push_heap(item, dist);
                }
            }
        } else {
            for (idx, ((item, &dist), &radius)) in items_dists.zip(node.radii()).enumerate() {
                if self.should_visit(dist, radius) {
                    if !node.is_removed(idx) {
                        self.push_heap(item, dist);
                    }

                    self.visit(node.child(idx));
                }
            }

            // always visit the far right child
            self.visit(node.far_right_child());
        }
    }

    fn should_visit(&self, dist: u64, radius: u64) -> bool {
        dist <= radius || self.neighbors.peek().map_or(true, |n| {
            self.neighbors.len() < self.k || n.dist > pos_diff(dist, radius)
        })
    }
}

fn pos_diff(left: u64, right: u64) -> u64 {
    if left < right {
        right - left
    } else {
        left - right
    }
}

#[cfg(test)]
fn compare(left: &u64, right: &u64) -> u64 {
    pos_diff(*left, *right)
}

#[test]
fn empty_tree() {
    let mut tree = MvpTree::new(compare);

    assert_eq!(tree.len(), 0);
    assert_eq!(tree.height(), 0);
    assert_eq!(tree.iter().collect::<Vec<_>>(), Vec::<&u64>::new());
    assert_eq!(tree.iter_mut().collect::<Vec<_>>(), Vec::<&u64>::new());
}

#[test]
fn one_level() {
    let mut tree = MvpTree::new(compare);

    tree.extend(0 .. NODE_SIZE as u64);

    assert_eq!(tree.len(), NODE_SIZE);
    assert_eq!(tree.height(), 1);

    assert_eq!(
        tree.iter().cloned().collect::<Vec<_>>(),
        (0 .. NODE_SIZE as u64).collect::<Vec<_>>()
    );

    assert_eq!(
        tree.iter_mut().map(|x| *x).collect::<Vec<_>>(),
        (0 .. NODE_SIZE as u64).collect::<Vec<_>>()
    );
}

#[test]
fn two_levels() {
    // this test should fill exactly two levels of the tree
    let mut tree = MvpTree::new(compare);

    let vals = [
        0, 1, 2, 11, 12, 5, 6, 7, // 0 1 2 6 7 should go to left child with 5 as the pivot
        13, 22, 23, 16, 17, 18, // 11 12 13 17 18 goes left with 16 as the pivot
        24, 33, 34, 27, 28, 29, // 22 23 24 28 29 go left with 27 as pivot
        35, 44, 45, 38, 39, 40, // 33 34 35 39 40 go left with 38 as pivot
        46, 55, 56, 49, 50, 51, // 44 45 46 50 51 go left with 49 as pivot
        57, 58, 59 // 55 56 57 58 59 end up in far right child
    ];

    // use manual loop because `.extend()` might do bulk operations that can shift the distribution
    for &val in vals.iter() { tree.insert(val) }

    assert_eq!(tree.len(), 35);
    assert_eq!(tree.height(), 2);

    let root = tree.root.as_ref().unwrap();
    assert_eq!(root.items(), &[5, 16, 27, 38, 49]);
    assert_eq!(root.radii(), &[5, 5, 5, 5, 5]);

    let child_0 = root.child(0);
    assert_eq!(child_0.items(), &[0, 1, 2, 6, 7]);
    assert!(child_0.is_leaf());

    let child_1 = root.child(1);
    assert_eq!(child_1.items(), &[11, 12, 13, 17, 18]);
    assert!(child_1.is_leaf());

    let child_2 = root.child(2);
    assert_eq!(child_2.items(), &[22, 23, 24, 28, 29]);
    assert!(child_2.is_leaf());

    let child_3 = root.child(3);
    assert_eq!(child_3.items(), &[33, 34, 35, 39, 40]);
    assert!(child_3.is_leaf());

    let child_4 = root.child(4);
    assert_eq!(child_4.items(), &[44, 45, 46, 50, 51]);
    assert!(child_4.is_leaf());

    let child_5 = root.far_right_child();
    assert_eq!(child_5.items(), &[55, 56, 57, 58, 59]);
    assert!(child_5.is_leaf());
}

#[test]
fn test_dfs() {
    use std::collections::HashSet;

    let len = 35;

    let mut tree = MvpTree::new(compare);
    tree.extend(0 .. 35);

    let mut dfs = DepthFirst::new(&tree.root);
    let mut set = HashSet::new();

    loop {
        let ptr = unsafe { dfs.next() };

        if ptr.is_null() { break; }

        // not assert so we have an inner source line to break on
        if !set.insert(ptr) {
            panic!("DepthFirst returned duplicate node");
        }
    }
}

#[test]
fn test_iter() {
    use std::collections::HashSet;

    let len = 100;

    let mut tree = MvpTree::new(compare);
    tree.extend(0 .. len);
    let mut set = (0 .. len).collect::<HashSet<u64>>();

    for val in tree.iter() {
        assert!(set.remove(&val), "val returned multiple times from iterator: {}", val);
    }
}

#[test]
fn test_iter_mut() {
    use std::collections::HashSet;

    let len = 100;

    let mut tree = MvpTree::new(compare);
    tree.extend(0 .. len);
    let mut set = (0 .. len).collect::<HashSet<u64>>();

    for val in tree.iter_mut() {
        assert!(set.remove(&val), "val returned multiple times from iterator: {}", val);
    }
}

fn compare_bits(left: &u32, right: &u32) -> u64 {
    (left ^ right).count_ones() as u64
}

#[test]
fn test_knn() {
    use std::collections::HashMap;

    fn bitmask(num_bits: u32) -> u32 {
        let mut accum = 0;
        for shift in 0 .. num_bits {
            accum |= 1 << shift;
        }
        accum
    }

    let mut tree = MvpTree::new(compare_bits);

    let start_bits = 0xAAAAAAAA; // repeating 1010[...]

    tree.insert(start_bits);

    // max number of bits different
    for diff_bits in 1 ..= 4 {
        let mask = bitmask(diff_bits);

        for shift in 0 .. 8 {
            // insert new bitpatterns by xoring a shifted mask
            tree.insert(start_bits ^ (mask << shift * 4));
        }
    }

    let knn = tree.k_nearest(9, &start_bits);
    let mut knn_map: HashMap<_, _> = knn.into_iter()
        .map(|neighbor| (*neighbor.item, false)).collect();

    assert_eq!(knn_map.len(), 9);
    assert!(knn_map.contains_key(&start_bits), "kNN does not contain starting bitpattern");

    // kNN should contain all bitpatterns with 1 bit different
    let test_mask = bitmask(1);

    for shift in 0 .. 8 {
        let bitpattern = start_bits ^ (test_mask << shift * 4);
        let mut entry = knn_map.get_mut(&bitpattern)
            .expect(&format!("missing bitpattern in kNN set: {:b}", bitpattern));

        assert!(!*entry, "kNN set duplicated bit pattern: {:b}", bitpattern);
        *entry = true;
    }
}

// reinterpret fuzz input data as 16 bits to increase number of possible values
fn bytes_to_words(bytes: &[u8]) -> Vec<u16> {
    bytes.chunks(2)
        .filter(|chunk| chunk.len() == 2)
        .map(|chunk| (chunk[0] as u16 | (chunk[1] as u16) << 8))
        .collect::<Vec<_>>()
}

fn compare_u16(l: &u16, r: &u16) -> u64 {
    if l < r {
        (r - l) as u64
    } else {
        (l - r) as u64
    }
}

#[doc(hidden)]
pub fn fuzz_iter(data: &[u8]) {
    let mut words = bytes_to_words(data);

    let mut tree = MvpTree::new(compare_u16);
    tree.extend(words.iter().cloned());
    words.sort();

    let mut words_test = tree.iter().cloned().collect::<Vec<_>>();
    words_test.sort();

    assert_eq!(words, words_test);
}

#[doc(hidden)]
pub fn fuzz_knn(data: &[u8]) {
    let mut words = bytes_to_words(data);

    if words.len() < 2 { return }

    let k = words[0] as usize;
    let search_for = words[1];

    if k == 0 { return; }

    words = words.split_off(2);

    let mut tree = MvpTree::new(compare_u16);
    tree.extend(words.iter().cloned());
    words.sort();

    // if the fuzzer duplicates `search_for` in the set it should be returned
    let neighbors = tree.k_nearest(k, &search_for);

    if words.len() < k {
        assert_eq!(neighbors.len(), words.len());
    } else {
        assert_eq!(neighbors.len(), k);
    }

    // assert each neighbor is contained in the set
    for neighbor in &neighbors {
        let idx = words.binary_search(neighbor.item)
            .expect(&format!("neighbor not in original set: {:?}", neighbor));

        words.remove(idx);

        // assert each neighbor is the right distance away
        assert_eq!(compare_u16(neighbor.item, &search_for), neighbor.dist);
    }

    if neighbors.is_empty() { return }

    let farthest = neighbors.last().unwrap();

    // assert that there are no neighbors closer than the greatest distance that weren't covered
    for word in &words {
        let dist = compare_u16(farthest.item, &search_for);
        assert!(dist >= farthest.dist,
                "uncovered closer neighbor than farthest, {:?} ({:?}) < {:?}",
                word, dist, farthest);
    }
}
