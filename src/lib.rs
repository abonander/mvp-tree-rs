
use std::cmp::{self, Ordering};
use std::collections::BinaryHeap;
use std::marker::PhantomData;
use std::ptr::{self, NonNull};
use std::{mem, slice};

mod node;

use node::{Node, NODE_SIZE};

pub struct MvpTree<T, Df> {
    // must be boxed so it has a stable address
    root: Option<Box<Node<T>>>,
    dist_fn: Df,
    len: usize,
    height: usize,
}

impl<T, Df> MvpTree<T, Df> where Df: Fn(&T, &T) -> u64 {
    pub fn new(dist_fn: Df) -> Self {
        MvpTree {
            root: None,
            dist_fn,
            len: 0,
            height: 0,
        }
    }

    pub fn len(&self) -> usize {
        self.len
    }

    pub fn height(&self) -> usize {
        self.height
    }

    pub fn insert(&mut self, mut item: T) {
        self.len = self.len.checked_add(1).expect("overflow `self.len + 1`");

        if let None = self.root {
            let mut root = Node::new_box();
            root.push_item(item);
            self.root = Some(root);
            self.height = 1;
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

            if node.is_leaf() {
                // node is leaf and full
                if node.has_parent() && node.child_idx() == NODE_SIZE {
                    // we are the far right leaf child -- add a parent to our parent node
                    // safe because we're not modifying the tree at any other point
                    unsafe { node.parent_mut().add_child(item, &distances) };
                    return depth;
                } else {
                    // make the node internal, increasing the depth by 1
                    node.make_internal(item, &distances);
                    return depth + 1;
                }
            }

            if let Some(child_idx) = node.find_parent(&distances) {
                // recurse into the appropriate child
                return find_insert(node.child_mut(child_idx), item, dist_fn, depth + 1);
            }

            find_insert(node.far_right_child_mut(), item, dist_fn, depth + 1)
        }

        let mut node = &mut **self.root.as_mut().unwrap();
        let insert_depth = find_insert(node, item, &self.dist_fn, 0);

        // update the height if it increased
        self.height = cmp::max(self.height, insert_depth + 1);
    }

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

    pub fn iter(&self) -> Iter<T> {
        Iter {
            dfs: DepthFirst::new(&self.root),
            items: None,
            _borrow: PhantomData,
        }
    }

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

#[derive(Debug)]
pub struct Neighbor<'a, T: 'a> {
    pub dist: u64,
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
                if dist <= radius {
                    if !node.is_removed(idx) {
                        self.push_heap(item, dist);
                    }

                    self.visit(node.child(idx));

                    // if we haven't reached `k` neighbors or the farthest neighbor is further
                    // than `|dist - radius|`, attempt to visit the next child in line
                    let max_dist = self.neighbors.peek().map_or(0, |n| n.dist);
                    if self.neighbors.len() == self.k && max_dist < pos_diff(dist, radius) {
                        return;
                    }
                }
            }

            self.visit(node.far_right_child());
        }
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

    let len = 35;

    let mut tree = MvpTree::new(compare);
    tree.extend(0 .. len);
    let mut set = (0 .. len).collect::<HashSet<u64>>();

    for val in tree.iter() {
        assert!(set.remove(&val), "val returned multiple times from iterator: {}", val);
    }
}


#[cfg(test)]
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

#[test] #[ignore]
fn test_100() {
    let mut tree = MvpTree::new(compare);
    tree.extend(0 .. 100);
    assert_eq!(tree.len(), 100);
    assert_eq!(tree.height(), 10);
}

pub fn fuzz_iter(data: &[u8]) {
    let mut words = data.chunks(2)
        .filter(|chunk| chunk.len() == 2)
        .map(|chunk| (chunk[0] as u16 | (chunk[1] as u16) << 8))
        .collect::<Vec<_>>();

    let mut tree = MvpTree::new(|&l: &u16, &r| (l ^ r).count_ones() as u64);
    tree.extend(words.iter().cloned());
    words.sort();

    let mut words_test = tree.iter().cloned().collect::<Vec<_>>();
    words_test.sort();

    assert_eq!(words, words_test);
}
