
mod arrayvec;
use arrayvec::ArrayVec;

// cardinality of the tree
const NODE_SIZE: usize = 6;

pub struct MvpTree<T, Df> {
    root: OptNode<T>,
    dist_fn: Df,
}

impl MvpTree<T, Df> where Df: Fn(&T, &T) -> u64 {
    pub fn new(dist_fn: Df) -> Self {
        MvpTree {
            root: None,
            dist_fn,
        }
    }

    pub fn insert(&mut self, mut item: T) {
        if let None = self.root {
            self.root = Some(Box::new(Node::new(item)));
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
                Err(parent_idx) => {
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

            // push a new node item along with its left child nodes
            node.items.push(NodeItem { radius: Some(median_dist), item });
            node.children.push(Node::with_items(left));

            // if this is the last node, go ahead and push the right child
            if node.items.len() == NODE_SIZE {
                node.children.push(Node::with_items(right));
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
}

struct NodeItem<T> {
    radius: Option<u64>,
    item: T,
}

struct Node<T> {
    // never empty
    items: ArrayVec<NodeItem<T>>,
    children: Vec<Node<T>>,
}

impl<T> Node<T> {
    fn new(item: T) -> Self {
        Self::with_items(Some(item).into_iter().collect())
    }

    fn with_items(items: ArrayVec<T>) -> Self {
        Node {
            items, children: Vec::new(),
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
    fn get_distances<Df>(&self, item: &T, dist_fn: Df) -> Result<ArrayVec<u64>, usize>
    where Df: Fn(&T, &T) -> u64 {
        node.items.iter().enumerate()
            .map(|(idx, node_item)| {
                let dist = self.dist_fn(&node_item.item, item);

                if let Some(radius) = node_item.radius {
                    if dist <= radius {
                        return Err(idx);
                    }
                }

                Ok(dist)
            })
            .collect::<Result<ArrayVec<u64>, usize>>()
    }
}


