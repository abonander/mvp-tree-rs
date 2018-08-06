
use std::cmp::Ordering;
use std::collections::BinaryHeap;
use std::ptr;

mod arrayvec;
use arrayvec::{ArrayVec, StableVec};

// cardinality of the tree
const NODE_SIZE: usize = 6;

pub struct MvpTree<T, Df> {
    // must be boxed so it has a stable address
    root: Option<Box<Node<T>>>,
    dist_fn: Df,
}

impl<T, Df> MvpTree<T, Df> where Df: Fn(&T, &T) -> u64 {
    pub fn new(dist_fn: Df) -> Self {
        MvpTree {
            root: None,
            dist_fn,
        }
    }

    pub fn insert(&mut self, mut item: T) {
        if let None = self.root {
            self.root = Some(Box::new(Node::new(item, unsafe { ptr::null() }, 0)));
            return;
        }

        let mut node = self.root.as_mut().unwrap();

        loop {
            assert!(node.len() > 0, "empty node");

            if node.is_leaf() && !node.is_full() {
                // if the target node is a non-full leaf node, just push the item
                node.items.push(item);
                return;
            }

            let distances = match node.get_distances(&item, &self.dist_fn) {
                Ok(distances) => distances,
                Err((parent_idx, _)) => {
                    // recurse into the left child of the found parent
                    node = &mut node.children[parent_idx];
                    continue;
                },
            };

            // if the node children is full, recurse into far right child
            if node.children.len() == NODE_SIZE + 1 {
                node = &mut node.children[NODE_SIZE + 1];
                continue;
            }

            let mut dists_clone = distances.clone();
            dists_clone.sort_unstable();
            let median_dist = dists_clone[distances.len() / 2];

            // we should have only left children if the node children list isn't full
            assert!(node.items.len() > node.children.len());

            let mut left = ArrayVec::new();
            let mut right = ArrayVec::new();

            // drain all items without a child and partition them
            for (node_item, dist) in node.items.drain_tail(node.children.len())
                // use the already-calculated distances
                .zip(distances.iter().skip(node.children.len()))
            {
                assert!(node_item.radius.is_none(), "node had a radius but no children");
                if dist <= median_dist {
                    left.push(node_item);
                } else {
                    right.push(node_item);
                }
            }

            let parent_idx = node.items.len();
            // push a new node item along with its left child nodes
            node.items.push(NodeItem { radius: Some(median_dist), item, removed: false, });
            node.children.push(Node::with_items(left, self, parent_idx));

            // if this is the last node, go ahead and push the right child
            if node.items.len() == NODE_SIZE {
                node.children.push(Node::with_items(right, self, parent_idx));
                return;
            }

            // otherwise, return the rest of the external nodes to the node list without children
            if node.items.len() + right.len() == NODE_SIZE + 1 {
                // but if we don't have room for all the external nodes,
                // pop one and retry inserting it
                item = right.pop().unwrap();
                node.items.extend(right);
                continue;
            } else {
                node.items.extend(right);
                return;
            }
        }
    }

    pub fn k_nearest(&self, k: usize, item: &T) -> BinaryHeap<Neighbor<T>> {
        // don't allocate if the tree is empty
        let mut heap = BinaryHeap::new();

        if k == 0 { return heap; }

        let mut node = if let Some(ref root) = self.root {
            root
        } else {
            return heap;
        };

        heap.reserve_exact(k);

        loop {
            /// Push a `Neighbor` onto the heap, always limiting its size to `k` items
            let mut push_heap = |item, dist| {
                // if the heap is full, swap the item onto it only if its distance
                // is smaller than the current maximum
                if heap.len() == k {
                    let mut peek_mut = heap.peak_mut().expect("heap shouldn't be empty");
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

            match node.get_distances(item, self.dist_fn) {
                Err((idx, dist)) => {
                    // we're within radius, recurse into the left child
                    push_heap(&node.items[idx].item, dist);
                    node = &node.children[idx];
                },
                Ok(distances) => if node.is_full() {
                    // recurse into the far right child
                    push_heap(&node.items[NODE_SIZE - 1].item, distances[NODE_SIZE - 1]);
                    node = &node.children[NODE_SIZE];
                } else {
                    //
                    for (item, dist) in node.items.iter().zip(distances) {
                        push_heap(item.item, dist);
                    }
                    break;
                },
            }
        }

        // FIXME: I'm pretty sure we're supposed to hit other nodes if the heap isn't full yet

        heap
    }

    pub fn iter(&self) -> Iter<T> {}
}

struct NodeItem<T> {
    radius: Option<u64>,
    item: T,
    removed: bool,
}

struct Node<T> {
    // never empty
    items: ArrayVec<NodeItem<T>>,
    children: StableVec<Node<T>>,
    parent: *const Node<T>,
    parent_idx: u8,
}

impl<T> Node<T> {
    fn new(item: T, parent: *const Node<T>, parent_idx: usize) -> Self {
        Self::with_items(
            Some(NodeItem { item, radius: None, removed: false }).into_iter().collect(),
            parent,
            parent_idx
        )
    }

    fn with_items(items: ArrayVec<NodeItem<T>>, parent: *const Node<T>, parent_idx: usize) -> Self {
        Node {
            items, children: Vec::new(), parent, parent_idx: parent_idx as u8,
        }
    }

    fn len(&self) -> usize {
        self.items.len()
    }

    fn is_full(&self) -> bool {
        self.items.len() == NODE_SIZE
    }

    fn is_leaf(&self) -> bool {
        self.children.is_empty()
    }

    /// Get the distances to the items in this node,
    /// or the index of the node that should become the given item's parent
    fn get_distances<Df>(&self, item: &T, dist_fn: Df) -> Result<ArrayVec<u64>, (usize, u64)>
    where Df: Fn(&T, &T) -> u64 {
        node.items.iter().enumerate()
            .map(|(idx, node_item)| {
                let dist = self.dist_fn(&node_item.item, item);

                if let Some(radius) = node_item.radius {
                    if dist <= radius {
                        return Err((idx, dist));
                    }
                }

                Ok(dist)
            })
            .collect::<Result<ArrayVec<u64>, usize>>()
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
