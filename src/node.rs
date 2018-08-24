use std::alloc::{self, Layout};
use std::mem::ManuallyDrop;
use std::ops::{Deref, DerefMut};
use std::ptr::{self, NonNull};

// cardinality of the tree
// the `u8` type saves memory as our values never exceed NODE_SIZE + 1
pub const NODE_SIZE: usize = 6;
pub const CHILD_SIZE: usize = NODE_SIZE + 1;

pub type NodeArray<T> = [T; NODE_SIZE];
pub type ChildArray<T> = [Node<T>; CHILD_SIZE];
pub type Distances = NodeArray<u64>;

struct NodeItem<T> {
    radius: Option<u64>,
    item: T,
    removed: bool,
}

pub struct Node<T> {
    // never empty
    items: ManuallyDrop<NodeArray<T>>,
    items_len: u8,
    removed: NodeArray<bool>,
    parents_len: u8,
    // these two only valid up to `num_parents`
    radii: Distances,
    children: Option<CustomBox<ChildArray<T>>>,
    parent: *const Node<T>,
    parent_idx: u8,
}

impl<T> Node<T> {
    pub fn new_box(item: T) -> CustomBox<Self> {
        unsafe {
            let mut node = CustomBox::<Self>::alloc();
            Self::init_inplace(&mut *node).push_item(item);
            node

        }
    }

    unsafe fn init_inplace<'a>(this: *mut Self) -> &'a mut Self {
        let mut this = &mut *this;
        this.items_len = 0;
        this.removed = [false; NODE_SIZE as usize];
        this.radii = [0u64; NODE_SIZE as usize];
        this.parents_len = 0;
        this.parent = ptr::null_mut();
        this.parent_idx = 0;
        this.children = None;
        this
    }

    pub fn push_item(&mut self, item: T) {
        assert!((self.items_len as usize) < NODE_SIZE, "pushing to a full node");
        unsafe {
            ptr::write(&mut self.items[self.items_len as usize], item);
        }
        self.items_len += 1;
    }

    /// Repartition the leaf elements of this full node given the new item
    /// and distances from it to the current elements of the node (given by `get_distances`
    #[must_use="ejected item from full node that needs to be reinserted"]
    pub fn make_internal(&mut self, mut item: T, distances: &Distances) {
        assert_eq!((self.items_len as usize), NODE_SIZE, "trying to add a parent to a non-full node");
        assert!(self.is_leaf(), "attempting to add parent to non-leaf node");
        
        self.items_len = 0;
        
        let radius = get_median(distances);
        
        let mut children = self.children.get_or_insert_with(|| unsafe { CustomBox::alloc() });
        
        unsafe {
            Self::init_inplace(&mut children[0]);
            Self::init_inplace(&mut children[NODE_SIZE as usize]);
        }
        
        for i in 0 .. NODE_SIZE as usize {
            // safe because we only read each item once
            let item = unsafe { ptr::read(&self.items[i]) };
            
            if distances[i] <= radius {
                children[0].push_item(item);
            } else {
                children[NODE_SIZE as usize].push_item(item);
            }
        }

        self.items_len = 1;
        self.parents_len = 1;
        self.radii[0] = radius;
        unsafe {
            ptr::write(&mut self.items[0], item);
            
        }
    }
    
    pub fn add_parent(&mut self, item: T, child_distances: &Distances) {
        assert_eq!(self.far_right_child().items_len as usize, NODE_SIZE,
                   "attempting to add parent with non-full right child");
        assert!((self.parents_len as usize) < NODE_SIZE, "attempting to add parent to full node");
        
        self.push_item(item);

        let radius = get_median(child_distances);
        self.radii[self.parents_len as usize] = radius;

        {
            let mut left = unsafe {
                let parents_len = self.parents_len;
                // breaks the borrow so we can get both left and right children
                Self::init_inplace(self.child_mut_unchecked(parents_len))
            };
            let mut right = self.far_right_child_mut();

            right.drain_less_than(radius, child_distances, &mut left);
        }

        self.parents_len += 1;
    }
    
    fn drain_less_than(&mut self, radius: u64, distances: &Distances, drain_to: &mut Self) {
        assert_eq!(self.items_len as usize, NODE_SIZE, "attempting to drain from non-full node");
        assert_eq!(drain_to.items_len, 0, "attempting to drain to non-empty node");

        self.items_len = 0;

        for i in 0 .. NODE_SIZE as usize {
            let item = unsafe { ptr::read(&self.items[i]) };

            if distances[i] <= radius {
                drain_to.push_item(item)
            } else {
                self.push_item(item);
            }
        }
    }

    pub fn len(&self) -> usize {
        self.items_len as usize
    }

    pub fn is_full(&self) -> bool {
        self.items_len as usize == NODE_SIZE
    }

    pub fn parents_full(&self) -> bool {
        self.parents_len as usize == NODE_SIZE
    }

    /// Does *not* include the far-right child
    pub fn children_len(&self) -> usize {
        self.parents_len as usize
    }

    pub fn is_leaf(&self) -> bool {
        self.children.is_none()
    }

    pub fn items(&self) -> &[T] {
        &self.items[..self.items_len as usize]
    }

    pub fn items_mut(&mut self) -> &mut [T] {
        &mut self.items[..self.items_len as usize]
    }

    pub fn child(&self, idx: usize) -> &Node<T> {
        assert!(idx < self.children_len(), "idx out of bounds {} ({})", idx, self.children_len());
        &self.children()[idx]
    }

    pub fn child_mut(&mut self, idx: usize) -> &mut Node<T> {
        assert!(idx < self.children_len(), "idx out of bounds {} ({})", idx, self.children_len());
        &mut self.children_mut()[idx]
    }

    unsafe fn child_mut_unchecked(&mut self, idx: u8) -> &mut Node<T> {
        &mut self.children_mut()[idx as usize]
    }
    
    pub fn far_right_child(&self) -> &Node<T> {
        assert!(!self.is_leaf(), "attempting to get child of leaf node");
        &self.children()[NODE_SIZE as usize]
    }

    pub fn far_right_child_mut(&mut self) -> &mut Node<T> {
        assert!(!self.is_leaf(), "attempting to get child of leaf node");
        &mut self.children_mut()[NODE_SIZE as usize]
    }
    
    fn children(&self) -> &ChildArray<T> {
        &**self.children.as_ref().expect("node children not allocated")
    }
    
    fn children_mut(&mut self) -> &mut ChildArray<T> {
        self.children.as_mut().expect("node children not allocated")
    }

    pub fn has_parent(&self) -> bool {
        !self.parent.is_null()
    }

    pub fn parent_ptr(&self) -> *const Node<T> {
        self.parent
    }

    pub fn parent_idx(&self) -> usize {
        assert!(self.has_parent(), "parent idx of root node makes no sense");
        self.parent_idx as usize
    }

    /// # Safety
    /// Must ensure parent isn't being modified
    pub unsafe fn parent_and_idx(&self) -> Option<(&Node<T>, usize)> {
        if self.parent.is_null() {
            None
        } else {
            Some((&*self.parent, self.parent_idx as usize))
        }
    }

    pub unsafe fn parent_mut(&mut self) -> Option<&mut Node<T>> {
        (self.parent as *mut Node<T>).as_mut()
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
        &self.radii[..self.parents_len as usize]
    }

    pub fn find_parent(&self, distances: &Distances) -> Option<usize> {
        self.radii.iter().zip(&distances[..]).position(|(rad, dist)| dist <= rad)
    }
}

/// Allocates uninitialized memory and does not drop automatically
pub struct CustomBox<T>(NonNull<T>);

impl<T> CustomBox<T> {
    unsafe fn alloc() -> Self {
        let layout = Self::layout();
        let ptr = alloc::alloc(layout) as *mut T;
        CustomBox(NonNull::new(ptr).unwrap_or_else(|| alloc::handle_alloc_error(layout)))
    }

    fn layout() -> Layout {
        Layout::new::<T>()
    }

    pub fn free(mut self) {
        let layout = Self::layout();

    }
}

impl<T> Deref for CustomBox<T> {
    type Target = T;

    fn deref(&self) -> &T {
        unsafe {
            self.0.as_ref()
        }
    }
}

impl<T> DerefMut for CustomBox<T> {
    fn deref_mut(&mut self) -> &mut T {
        unsafe {
            self.0.as_mut()
        }
    }
}

fn get_median(distances: &Distances) -> u64 {
    // get the median distance
    let mut dists_clone = *distances;
    dists_clone.sort();
    dists_clone[NODE_SIZE as usize / 2]
}
