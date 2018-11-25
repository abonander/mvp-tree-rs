use std::mem::{self, ManuallyDrop};
use std::sync::atomic::{AtomicUsize, Ordering};
use std::iter::Enumerate;
use std::{fmt, ptr, slice};

// cardinality of the tree
// the `u8` type saves memory as our values never exceed NODE_SIZE + 1
pub const NODE_SIZE: usize = 5;
pub const CHILD_SIZE: usize = NODE_SIZE + 1;

pub type NodeArray<T> = [T; NODE_SIZE];
pub type ChildArray<T> = [*mut Node<T>; CHILD_SIZE];
pub type Distances = NodeArray<u64>;

// MAINTAINER NOTE: all raw pointers are to be assumed nullable unless otherwise specified.
// Raw pointer accesses should be done by `[.as_ref(), .as_mut()].unwrap()` instead of direct derefs
// to catch any errors.

pub struct Node<T> {
    /// valid up to `len`
    items: ManuallyDrop<NodeArray<T>>,
    len: u8,
    is_leaf: bool,
    /// if `!is_leaf`, `0 .. len` and `NODE_SIZE` indices are valid pointers
    children: ChildArray<T>,
    /// valid up to `len` if `!is_leaf`
    radii: Distances,
    /// NULLABLE
    parent: *const Node<T>,
    /// the index of this node in its parent, if applicable
    child_idx: u8,
}

impl<T> Node<T> {
    /// `Box` is necessary for stable addressing so parent pointers work as expected
    pub fn new_box() -> Box<Self> {
        Box::new(Node {
            // FIXME: use `MaybeUninit` when stable to avoid copying garbage data
            items: unsafe { mem::zeroed() },
            len: 0,
            is_leaf: true,
            children: [ptr::null_mut(); CHILD_SIZE],
            radii: [0; NODE_SIZE],
            parent: ptr::null_mut(),
            child_idx: 0,
        })
    }

    fn new_child(parent: *const Self, child_idx: u8) -> *mut Self {
        let mut node = Self::new_box();
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
    pub fn make_internal(&mut self, mut item: T, distances: &Distances) {
        assert!(self.is_leaf(), "attempting to make internal node internal again");
        assert_eq!(self.len(), NODE_SIZE, "trying to make non-full leaf node internal");

        self.len = 0;
        
        let radius = get_median(distances);

        self.children[0] = Self::new_child(self, 0);
        self.children[NODE_SIZE] = Self::new_child(self, NODE_SIZE as u8);
        
        for i in 0 .. NODE_SIZE {
            // safe because we only read each item once
            let item = unsafe { ptr::read(&self.items[i]) };

            if distances[i] <= radius {
                // we just initialized these indices
                unsafe { unwrap_ptr!(mut self.children[0]).push_item(item); }
            } else {
                unsafe { unwrap_ptr!(mut self.children[NODE_SIZE]).push_item(item); }
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
        assert_ne!(self.len(), NODE_SIZE, "attempting to add child to full node");

        let radius = get_median(child_distances);
        let len = self.len as usize;

        self.radii[len] = radius;
        self.children[len] = Self::new_child(self, self.len);

        unsafe {
            // both `self.children[len]` and `self.children[NODE_SIZE]` are valid
            unwrap_ptr!(mut self.children[NODE_SIZE])
                .drain_less_than(radius, child_distances, unwrap_ptr!(mut self.children[len]));

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

    pub fn items(&self) -> &[T] {
        &self.items[..self.len as usize]
    }

    pub fn items_mut(&mut self) -> &mut [T] {
        &mut self.items[..self.len as usize]
    }

    pub fn child(&self, idx: usize) -> &Node<T> {
        assert!(!self.is_leaf(), "attempting to get child of leaf node");
        assert!(idx < self.len(), "idx out of bounds {} ({})", idx, self.len());
        unsafe { unwrap_ptr!(self.children[idx]) }
    }

    pub fn child_mut(&mut self, idx: usize) -> &mut Node<T> {
        assert!(!self.is_leaf(), "attempting to get child of leaf node");
        assert!(idx < self.len(), "idx out of bounds {} ({})", idx, self.len());
        unsafe { unwrap_ptr!(mut self.children[idx]) }
    }
    
    pub fn far_right_child(&self) -> &Node<T> {
        assert!(!self.is_leaf(), "attempting to get far right child of leaf node");
        unsafe { unwrap_ptr!(self.children[NODE_SIZE]) }
    }

    pub fn far_right_child_mut(&mut self) -> &mut Node<T> {
        assert!(!self.is_leaf(), "attempting to get far right child of leaf node");
        unsafe { unwrap_ptr!(mut self.children[NODE_SIZE]) }
    }

    /// Remove `idx` from the node, returning it and its left child if this is not a leaf node,
    /// or a null pointer otherwise.
    pub fn remove(&mut self, idx: usize) -> (T, *mut Node<T>) {
        let len = self.len();
        assert!(idx < len, "idx out of bounds {} ({})", idx, len);
        // both safe because we reduce the length by 1 afterward
        let item = unsafe { shift_remove(self.items_mut(), idx) };
        let child = unsafe { shift_remove(&mut self.children[..len], idx) };
        self.len = (len - 1) as u8;
        (item, child)
    }

    pub fn has_parent(&self) -> bool {
        !self.parent.is_null()
    }

    /// NULLABLE
    pub fn parent_ptr(&self) -> *const Node<T> {
        self.parent
    }

    pub fn child_idx(&self) -> usize {
        assert!(self.has_parent(), "child_idx() of root node");
        self.child_idx as usize
    }

    /// # Safety
    /// Must ensure parent isn't being modified
    pub unsafe fn parent_and_idx(&self) -> Option<(&Node<T>, usize)> {
        self.parent.as_ref().map(|p| (p, self.child_idx as usize))
    }

    /// # Safety
    /// Must ensure parent isn't being modified
    pub unsafe fn parent_mut_and_idx(&mut self) -> Option<(&mut Node<T>, usize)> {
        let child_idx = self.child_idx as usize;
        (self.parent as *mut Node<T>).as_mut().map(|p| (p, child_idx))
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

    /// Free empty children if this is a non-leaf and then drain the contained items
    ///
    /// ### Safety
    /// The returned iterator is not bound by a lifetime, you must ensure it doesn't outlive
    /// this `Node`
    pub unsafe fn drain_items(&mut self) -> DrainItems<T> {
        let len = self.len();
        self.len = 0;
        if !self.is_leaf() {
            self.is_leaf = true;

            for idx in (0 .. len).chain(Some(NODE_SIZE)) {
                {
                    let child = unsafe { unwrap_ptr!(self.children[idx]) };
                    assert_eq!(child.len(), 0, "draining items with nonempty children");
                }

                unsafe { drop(unwrap_ptr!(self.children[idx])); }
                // as a check to ensure further access/drops will panic/segfault
                self.children[idx] = ptr::null_mut();
            }
        }

        DrainItems {
            items: unsafe { &mut self.items[..len] },
        }
    }
}

impl<T: fmt::Debug> fmt::Debug for Node<T> {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        let mut debug_struct = f.debug_struct("Node");

        debug_struct.field("items", &self.items());

        if !self.is_leaf() {
            debug_struct.field("radii", &self.radii())
                .field("children", &DebugChildren { children: &self.children, len: self.len() });
        }

        debug_struct.finish()
    }
}

struct DebugChildren<'a, T: 'a> {
    children: &'a ChildArray<T>,
    len: usize,
}

impl<'a, T: fmt::Debug> fmt::Debug for DebugChildren<'a, T> {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        let children_iter = self.children[..self.len].iter()
            .chain(self.children.last())
            // INVARIANT:
            // if we have safe access to this node then we have shared access to the children
            .map(|&c| unsafe { unwrap_ptr!(c) });

        f.debug_list().entries(children_iter).finish()
    }
}

unsafe impl<T: Send> Send for Node<T> {}
unsafe impl<T: Sync> Sync for Node<T> {}

impl<T> Drop for Node<T> {
    fn drop(&mut self) {
        self.parent = ptr::null();
        let len = self.len();
        self.len = 0;

        // free children first
        if !self.is_leaf {
            self.is_leaf = true;

            for &child in self.children[..len].iter().chain(self.children.last()) {
                unsafe {
                    drop(unwrap_ptr!(Box child));
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

pub struct DrainItems<T> {
    items: *mut [T],
}

impl<T> Iterator for DrainItems<T> {
    type Item = T;

    fn next(&mut self) -> Option<T> {
        assert!(!self.items.is_null());
        let items = unsafe { &mut *self.items };

        if items.is_empty() { return None }

        let val = unsafe { ptr::read(&items[0]) };
        self.items = &mut items[1..];
        Some(val)
    }
}

fn get_median(distances: &Distances) -> u64 {
    // get the median distance
    let mut dists_clone = *distances;
    dists_clone.sort();
    dists_clone[NODE_SIZE as usize / 2]
}

/// Remove the item at `idx`, returning it and rotating all the items after it down.
///
/// After this returns, the last item in the slice is uninitialized.
///
/// ### Safety
/// Can cause a double-drop if used on a slice of `Drop` elements
unsafe fn shift_remove<T>(slice: &mut [T], idx: usize) -> T {
    if idx + 1 == slice.len() {
        return ptr::read(&slice[idx]);
    }

    let len = slice.len() - (idx + 1);
    let val = ptr::read(&slice[idx]);
    ptr::copy(&slice[idx + 1], &mut slice[idx], len);
    val
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
    let mut node = Node::new_box();

    for i in 0 .. 5 {
        node.push_item(i);
    }

    assert_eq!(node.len(), 5);
    assert!(node.is_leaf());

    node.make_internal(5, &[5, 4, 3, 2, 1]);

    assert!(!node.is_leaf());

    assert_eq!(node.radii()[0], 3);
    assert_eq!(node.items(), &[5]);

    assert_eq!(node.child(0).items(), &[2, 3, 4]);
    assert_eq!(node.far_right_child().items(), &[0, 1]);

    // items remain in the far right child
    for i in 6 .. 9 {
        node.far_right_child_mut().push_item(i);
    }

    assert_eq!(node.len(), 1);

    node.add_child(9, &[9, 8, 3, 2, 1]);

    assert_eq!(node.len(), 2);
    assert_eq!(node.items(), &[5, 9]);

    assert_eq!(
        node.child(1).items(),
        &[6, 7, 8]
    );
    assert_eq!(
        node.far_right_child().items(),
        &[0, 1]
    )
}

#[test]
fn test_shift_remove() {
    let mut items = [0, 1, 2, 3, 4];
    assert_eq!(unsafe { shift_remove(&mut items, 0) }, 0);
    assert_eq!(items, [1, 2, 3, 4, 4]);

    let mut items = [0, 1, 2, 3, 4];
    assert_eq!(unsafe { shift_remove(&mut items, 1) }, 1);
    assert_eq!(items, [0, 2, 3, 4, 4]);

    let mut items = [0, 1, 2, 3, 4];
    assert_eq!(unsafe { shift_remove(&mut items, 2) }, 2);
    assert_eq!(items, [0, 1, 3, 4, 4]);

    let mut items = [0, 1, 2, 3, 4];
    assert_eq!(unsafe { shift_remove(&mut items, 3) }, 3);
    assert_eq!(items, [0, 1, 2, 4, 4]);

    let mut items = [0, 1, 2, 3, 4];
    assert_eq!(unsafe { shift_remove(&mut items, 4) }, 4);
    // last item is uninitialized even though it equals the same
    assert_eq!(items, [0, 1, 2, 3, 4]);
}
