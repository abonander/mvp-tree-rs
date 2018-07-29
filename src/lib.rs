
mod arrayvec;
use arrayvec::ArrayVec;

// cardinality of the tree
const NODE_SIZE: usize = 6;

type OptNode<T> = Option<Box<Node<T>>>;

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

    pub fn insert(&mut self, item: T) {
        if let None = self.root {
            self.root = Some(Box::new(Node::new(item)));
            return;
        }

        if let Some(new_root) = try_split_node(&mut self.root, &self.dist_fn) {
            self.root = new_root;
        }

        let mut parent = self.root.as_mut().unwrap();

        loop {

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
    children: NodeChildren<T>,
}

impl<T> Node<T> {
    fn new(item: T) -> Self {
        Node {
            items: Some(item).into_iter().collect(),
            children: Default::default(),
        }
    }

    fn len(&self) -> usize {
        self.items.len()
    }
}

#[derive(Default)]
struct NodeChildren<T>([OptNode<T>; NODE_SIZE + 1]);

impl<T> NodeChildren<T> {
    fn left(&self, i: usize) -> &OptNode<T> {
        &self.0[i]
    }

    fn left_mut(&mut self, i: usize) -> &mut OptNode<T> {
        &mut self.0[i]
    }

    fn right(&self, i: usize) -> &OptNode<T> {
        &self.0[i + 1]
    }

    fn right_mut(&mut self, i: usize) -> &mut OptNode<T> {
        &mut self.0[i + 1]
    }

    fn inserted(&mut self, i: usize) {
        assert!(self.0[NODE_SIZE].is_none(), "attempting to insert into a full NodeChildren");
        self.0[i..].rotate_right(1);
    }
}

