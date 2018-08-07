
use std::cmp::{self, Ordering};
use std::collections::BinaryHeap;
use std::marker::PhantomData;
use std::ptr;

mod node;

use node::{NODE_SIZE, Node, CustomBox};

pub struct MvpTree<T, Df> {
    // must be boxed so it has a stable address
    root: Option<CustomBox<Node<T>>>,
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
            self.root = Some(Node::new_box(item));
            self.height = 1;
            return;
        }

        let mut node = &mut **self.root.as_mut().unwrap();
        let mut depth = 0;

        loop {
            assert!(node.len() > 0, "empty node");

            if node.is_leaf() && !node.is_full() {
                // if the target node is a non-full leaf node, just push the item
                node.push_item(item);
                break;
            }

            let distances = node.get_distances(&item, &self.dist_fn);

            if let Some(parent_idx) = node.find_parent(&distances) {
                node = node.child_mut(parent_idx);
                depth += 1;
                continue;
            }

            // if the node children is full, recurse into far right child
            if node.parents_full() {
                node = node.child_mut(NODE_SIZE + 1);
                depth += 1;
                continue;
            }

            depth += 1;

            if let Some(item_) = node.add_parent(item, &distances) {
                item = item_;
                continue;
            } else {
                break;
            }
        }

        // update the height if it increased
        self.height = cmp::max(self.height, depth);
    }

    pub fn k_nearest(&self, k: usize, item: &T) -> BinaryHeap<Neighbor<T>> {
        // don't allocate if the tree is empty
        let mut heap = BinaryHeap::new();

        if k == 0 { return heap; }

        let mut node = if let Some(ref root) = self.root {
            &**root
        } else {
            return heap;
        };

        heap.reserve_exact(k);

        loop {
            // Push a `Neighbor` onto the heap, always limiting its size to `k` items
            let mut push_heap = |item, dist| {
                // if the heap is full, swap the item onto it only if its distance
                // is smaller than the current maximum
                if heap.len() == k {
                    let mut peek_mut = heap.peek_mut().expect("heap shouldn't be empty");
                    if peek_mut.dist > dist {
                        peek_mut.dist = dist;
                        peek_mut.item = item;
                    }
                    // `peek_mut` sifts the item down on-drop
                } else {
                    heap.push(Neighbor {
                        item, dist, _compat: (),
                    })
                }
            };

            let distances = node.get_distances(item, self.dist_fn);
            if let Some(idx) = node.find_parent(&distances) {
                push_heap(&node.items()[idx], distances[idx]);
                node = node.left(idx);
                continue;
            }

            if node.is_full() {
                let idx = NODE_SIZE as usize - 1;
                // recurse into the far right child
                push_heap(&node.items()[idx], distances[idx]);
                node = node.far_right();
            } else {
                //
                for (item, &dist) in node.items().iter().zip(&distances) {
                    push_heap(item.item, dist);
                }
                break;
            }
        }

        // FIXME: I'm pretty sure we're supposed to hit other nodes if the heap isn't full yet

        heap
    }

    pub fn iter(&self) -> Iter<T> {
        Iter {
            dfs: DepthFirst::new(&self.root),
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
        self.dist.cmp(other.dist)
    }
}

pub struct Iter<'a, T: 'a> {
    dfs: DepthFirst<T>,
    _borrow: PhantomData<&'a T>,
}

impl<'a, T: 'a> Iterator for Iter<'a, T> {
    type Item = &'a T;

    fn next(&mut self) -> Option<&'a T> {
        unsafe {
            self.dfs.next().as_ref()
        }
    }
}

pub struct IterMut<'a, T: 'a> {
    dfs: DepthFirst<T>,
    _borrow: PhantomData<&'a mut T>,
}

impl<'a, T: 'a> Iterator for IterMut<'a, T> {
    type Item = &'a mut T;

    fn next(&mut self) -> Option<&'a mut T> {
        unsafe {
            // cast from `*const` to `*mut` is safe because we hold the borrow to the tree
            (self.dfs.next() as *mut T).as_mut()
        }
    }
}

struct DepthFirst<T> {
    // NULLABLE
    node: *const Node<T>,
    descend: bool,
}

impl<T> DepthFirst<T> {
    fn new(root: &Option<CustomBox<Node<T>>>) -> Self {
        DepthFirst {
            node: root.as_ref().map(|r| r),
            descend: true,
        }
    }

    // NULLABLE
    unsafe fn next(&mut self) -> *const Node<T> {
        if self.node.is_null() {
            return self.node;
        }

        if !self.descend {
            while let Some((parent, child_idx)) = (*self.node).parent_and_idx() {
                if child_idx + 1 < parent.children_len() {
                    self.node = parent.child(child_idx + 1);
                    self.descend = true;
                    break;
                }

                self.node = parent;
            }
        }

        if self.descend {
            while !self.node.is_leaf() {
                self.node = self.node.child(0);
            }
            self.descend = false;
        }

        self.node
    }
}
