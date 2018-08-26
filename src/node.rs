use std::alloc::{self, Layout};
use std::mem::{self, ManuallyDrop};
use std::ops::{Deref, DerefMut};
use std::ptr::{self, NonNull};
use std::sync::atomic::{AtomicUsize, Ordering};
use std::iter::Enumerate;
use std::slice;

// cardinality of the tree
// the `u8` type saves memory as our values never exceed NODE_SIZE + 1
pub const NODE_SIZE: usize = 5;
pub const CHILD_SIZE: usize = NODE_SIZE + 1;

pub type NodeArray<T> = [T; NODE_SIZE];
pub type ChildArray<T> = [*mut Node<T>; CHILD_SIZE];
pub type Distances = NodeArray<u64>;

// FIXME: replace with a smaller atomic type when stable
type AtomicBitSet = AtomicUsize;

pub struct Node<T> {
    /// valid up to `len`
    items: ManuallyDrop<NodeArray<T>>,
    len: u8,
    /// valid up to `len`
    removed: AtomicBitSet,
    is_leaf: bool,
    /// if `!is_leaf`, `0 .. len` and `NODE_SIZE` indices are initialized
    children: ChildArray<T>,
    /// valid up to `len` if `!is_leaf`
    radii: Distances,
    parent: *const Node<T>,
    /// the index of this node in its parent, if applicable
    child_idx: u8,
}

impl<T> Node<T> {
    /// `Box` is necessary for stable addressing so parent pointers work as expected
    /// An item is required so we never have empty nodes
    pub fn new_box(item: T) -> Box<Self> {
        // FIXME: use `MaybeUninit` when stable to avoid copying garbage data
        let mut items = ManuallyDrop::new(unsafe { mem::zeroed() });

        unsafe {
            ptr::write(&mut items[0], item);
        }

        Box::new(Node {
            items,
            len: 1,
            removed: Default::default(),
            is_leaf: true,
            children: [ptr::null_mut(); CHILD_SIZE],
            radii: [0; NODE_SIZE],
            parent: ptr::null_mut(),
            child_idx: 0,
        })
    }

    fn new_child(item: T, parent: *const Self, child_idx: u8) -> *mut Self {
        let mut node = Self::new_box(item);
        node.parent = parent;
        node.child_idx = child_idx;
        Box::into_raw(node)
    }

    /// For sanity, only allows pushing to a leaf node
    pub fn push_item(&mut self, item: T) {
        assert!(self.is_leaf(), "pushing item to internal node");
        unsafe { self.unsafe_push_item(item) }
    }

    /// `self.children[..self.len()]` must be valid before this is called
    unsafe fn unsafe_push_item(&mut self, item: T) {
        let len = self.len();
        assert!(len < NODE_SIZE, "pushing to a full node");
        // not actually unsafe; destination is valid though uninitialized
        ptr::write(&mut self.items[len], item);
        self.len += 1;
    }

    /// Repartition the leaf elements of this full node given the new item
    /// and distances from it to the current elements of the node (given by `get_distances`)
    #[must_use="ejected item from full node that needs to be reinserted"]
    pub fn make_internal(&mut self, mut item: T, distances: &Distances) {
        assert!(self.is_leaf(), "attempting to make internal node internal again");
        assert_eq!(self.len(), NODE_SIZE, "trying to make non-full leaf node internal");

        self.len = 0;
        
        let radius = get_median(distances);

        self.children[0] = Self::new_child(item, self, 0);
        self.children[NODE_SIZE] = Self::new_child(item, self, NODE_SIZE as u8);
        
        for i in 0 .. NODE_SIZE {
            // safe because we only read each item once
            let item = unsafe { ptr::read(&self.items[i]) };
            
            if distances[i] <= radius {
                self.child_mut(0).push_item(item);
            } else {
                self.far_right_child_mut().push_item(item);
            }
        }

        self.radii[0] = radius;
        self.push_item(item);
        self.is_leaf = false;
    }

    /// Add a new child to this node, partitioning items from the far right child
    /// using `child_distances` and assuming `item` as the pivot
    pub fn add_child(&mut self, item: T, child_distances: &Distances) {
        assert!(!self.is_leaf(), "attempting to add child to leaf node; use make_internal()");
        assert!(self.far_right_child().is_full(),
                "attempting to add child with non-full right child");
        assert!(self.len() < NODE_SIZE, "attempting to add child to full node");

        let radius = get_median(child_distances);
        let len = self.len as usize;

        self.radii[len] = radius;
        self.children[len] = Self::new_child(item, self, self.len);

        unsafe {
            // both `self.children[len]` and `self.children[NODE_SIZE]` are valid
            (*self.children[NODE_SIZE])
                .drain_less_than(radius, child_distances, &mut *self.children[len]);

            // we just initialized `self.children[len]`
            self.unsafe_push_item(item);
        }
    }
    
    fn drain_less_than(&mut self, radius: u64, distances: &Distances, drain_to: &mut Self) {
        assert_eq!(self.len(), NODE_SIZE, "attempting to drain from non-full node");
        assert_eq!(drain_to.len(), 0, "attempting to drain to non-empty node");

        self.len = 0;

        for i in 0 .. NODE_SIZE {
            let item = unsafe { ptr::read(&self.items[i]) };

            if distances[i] <= radius {
                drain_to.push_item(item)
            } else {
                self.push_item(item);
            }
        }
    }

    pub fn len(&self) -> usize {
        self.len as usize
    }

    pub fn is_full(&self) -> bool {
        self.len as usize == NODE_SIZE
    }

    pub fn is_leaf(&self) -> bool {
        self.is_leaf
    }

    pub fn items(&self) -> Items<T> {
        Items {
            items: self.items[..self.len].iter().enumerate(),
            removed: &self.removed,
        }
    }

    pub fn items_mut(&mut self) -> ItemsMut<T> {
        ItemsMut {
            items: self.items[..self.len].iter_mut().enumerate(),
            removed: &self.removed,
        }
    }

    pub fn child(&self, idx: usize) -> &Node<T> {
        assert!(!self.is_leaf(), "attempting to get child of leaf node");
        assert!(idx < self.len(), "idx out of bounds {} ({})", idx, self.len());
        unsafe { self.children[idx].as_ref().unwrap() }
    }

    pub fn child_mut(&mut self, idx: usize) -> &mut Node<T> {
        assert!(!self.is_leaf(), "attempting to get child of leaf node");
        assert!(idx < self.len(), "idx out of bounds {} ({})", idx, self.len());
        unsafe { self.children[idx].as_mut().unwrap() }
    }
    
    pub fn far_right_child(&self) -> &Node<T> {
        assert!(!self.is_leaf(), "attempting to get far right child of leaf node");
        unsafe { self.children[NODE_SIZE].as_ref().unwrap() }
    }

    pub fn far_right_child_mut(&mut self) -> &mut Node<T> {
        assert!(!self.is_leaf(), "attempting to get far right child of leaf node");
        unsafe { self.children[NODE_SIZE].as_mut().unwrap() }
    }

    pub fn has_parent(&self) -> bool {
        !self.parent.is_null()
    }

    pub fn parent_ptr(&self) -> *const Node<T> {
        self.parent
    }

    pub fn child_idx(&self) -> usize {
        assert!(self.has_parent(), "child_idx() of root node");
        self.child_idx as usize
    }

    /// Atomically mark the item at `idx` as removed without removing it from the tree;
    /// actually removing items would require significant restructuring of the tree
    pub fn remove_item(&self, idx: usize) {
        assert!(idx < self.len(), "attempt to remove item out of bounds: {}, {}", idx, self.len);
        self.removed.fetch_or(1 << idx, Ordering::AcqRel);
    }

    /// # Safety
    /// Must ensure parent isn't being modified
    pub unsafe fn parent_and_idx(&self) -> Option<(&Node<T>, usize)> {
        if self.parent.is_null() {
            None
        } else {
            Some((&*self.parent, self.child_idx as usize))
        }
    }

    pub unsafe fn parent_mut(&mut self) -> &mut Node<T> {
        (self.parent as *mut Node<T>).as_mut().expect("getting nonexistent parent")
    }

    /// Get the distances to the items in this node
    /// Only the values in `..self.len()` are valid
    pub fn get_distances<Df>(&self, item: &T, dist_fn: Df) -> Distances
    where Df: Fn(&T, &T) -> u64 {
        let mut dists = [0u64; NODE_SIZE as usize];

        for (my_item, dist) in self.items.iter().zip(dists.iter_mut()) {
            *dist = dist_fn(item, my_item);
        }

        dists
    }

    pub fn radii(&self) -> &[u64] {
        assert!(!self.is_leaf(), "attempting to get radii of leaf node");
        &self.radii[..self.len as usize]
    }

    pub fn find_parent(&self, distances: &Distances) -> Option<usize> {
        self.radii().iter().zip(&distances[..]).position(|(rad, dist)| dist <= rad)
    }
}

impl<T> Drop for Node<T> {
    fn drop(&mut self) {
        self.parent = ptr::null();
        let len = self.len();
        self.len = 0;

        // free children first
        if !self.is_leaf {
            self.is_leaf = true;

            for &child in &self.children[..len].chain(self.children.last()) {
                unsafe {
                    drop(Box::from_raw(child));
                }
            }
        }

        for item in &mut self.items[..len] {
            unsafe {
                ptr::drop_in_place(item);
            }
        }
    }
}

fn is_removed(removed: &AtomicBitSet, idx: usize) -> bool {
    (removed.load(Ordering::Acquire) && (1 << idx)) != 0
}

pub struct Items<'a, T: 'a> {
    items: Enumerate<slice::Iter<'a, T>>,
    removed: &'a AtomicBitSet,
}

impl<'a, T: 'a> Iterator for Items<'a, T> {
    type Item = &'a T;

    fn next(&mut self) -> Option<&'a T> {
        self.items.by_ref()
            .filter(|&(idx, _)| is_removed(&self.removed, idx))
            .map(|(_, item)| item)
            .next()
    }
}

impl<'a, T: 'a> Items<'a, T> {
    pub fn is_empty(&self) -> bool {
        self.items.len() == 0
    }
}

pub struct ItemsMut<'a, T: 'a> {
    items: Enumerate<slice::IterMut<'a, T>>,
    removed: &'a AtomicBitSet,
}

impl<'a, T: 'a> ItemsMut<'a, T> {
    pub fn is_empty(&self) -> bool {
        self.items.len() == 0
    }
}

impl<'a, T: 'a> Iterator for ItemsMut<'a, T> {
    type Item = &'a mut T;

    fn next(&mut self) -> Option<&'a mut T> {
        self.items.by_ref()
            .filter(|&(idx, _)| is_removed(&self.removed, idx))
            .map(|(_, item)| item)
            .next()
    }
}

fn get_median(distances: &Distances) -> u64 {
    // get the median distance
    let mut dists_clone = *distances;
    dists_clone.sort();
    dists_clone[NODE_SIZE as usize / 2]
}

#[test]
fn test_get_median() {
    assert_eq!(get_median(&[1, 2, 3, 4, 5]), 3);
}

#[test]
fn size_asserts() {
    assert!(NODE_SIZE < u8::max_value() as usize);
    assert!(CHILD_SIZE < u8::max_value() as usize);
    assert!(NODE_SIZE % 2 != 0, "NODE_SIZE should be odd");
}

#[test]
fn test_make_internal_add_parent() {
    let mut node = Node::new_box(0);

    for i in 1 .. 5 {
        node.push_item(i);
    }

    assert_eq!(node.len(), 5);
    assert!(node.is_leaf());

    node.make_internal(5, &[5, 4, 3, 2, 1]);

    assert!(!node.is_leaf());

    assert_eq!(node.radii()[0], 3);

    assert_eq!(
        node.child(0).items(),
        &[2, 3, 4]
    );

    assert_eq!(
        node.far_right_child().items(),
        &[0, 1]
    );

    // items remain in the far right child
    for i in 6 .. 9 {
        node.far_right_child_mut().push_item(i);
    }

    assert_eq!(node.len(), 1);

    node.add_parent(9, &[9, 8, 3, 2, 1]);

    assert_eq!(node.len(), 2);
    assert_eq!(
        node.child(1).items(),
        &[6, 7, 8]
    );
    assert_eq!(
        node.far_right_child().items(),
        &[0, 1]
    )
}
