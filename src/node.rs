use std::alloc::{self, Layout};
use std::mem::ManuallyDrop;
use std::ops::{Deref, DerefMut};
use std::ptr::{self, NonNull};

// cardinality of the tree
// the `u8` type saves memory as our values never exceed NODE_SIZE + 1
pub const NODE_SIZE: u8 = 6;

pub type NodeArray<T> = [T; NODE_SIZE as usize];

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
    radii: NodeArray<u64>,
    children: Option<CustomBox<[Node<T>; NODE_SIZE as usize + 1]>>,
    parent: *const Node<T>,
    child_idx: u8,
}

impl<T> Node<T> {
    pub fn new_box(item: T) -> CustomBox<Self> {
        unsafe {
            let mut node = CustomBox::alloc();
            Self::init_inplace(node.as_mut()).push_item(item);
            node

        }
    }

    unsafe fn init_inplace<'a>(this: *mut Self) -> &'a mut Self {
        this.items_len = 0;
        this.removed = [false; NODE_SIZE];
        this.radii = [0u64; NODE_SIZE];
        this.parents_len = 0;
        this.parent = ptr::null_mut();
        this.child_idx = 0;
        this.children = None;
        &mut *this
    }

    pub fn push_item(&mut self, item: T) {
        assert!(self.items_len < NODE_SIZE, "pushing to a full node");
        unsafe {
            ptr::write(&mut self.items[self.items_len], item);
        }
        self.items_len += 1;
    }

    /// Repartition the leaf elements of this full node given the new item
    /// and distances from it to the current elements of the node (given by `get_distances`
    #[must_use="ejected item from full node that needs to be reinserted"]
    pub fn add_parent(&mut self, mut item: T, distances: &NodeArray<u64>) -> Option<T> {
        assert_eq!(self.items_len, NODE_SIZE, "trying to add a parent to a non-full node");
        assert!(self.parents_len < NODE_SIZE, "pushing an excess parent");
        debug_assert_eq!(self.find_parent(distances), None, "this item has a parent");

        // get the median distance
        let mut dists_clone = *distances;
        dists_clone.sort();
        let radius = dists_clone[NODE_SIZE / 2];

        let mut children = unsafe { self.children.get_or_insert_with(CustomBox::alloc()).as_mut() };
        let mut left = unsafe { Self::init_inplace(children[self.parents_len as usize]) };
        left.child_idx = self.parents_len;
        self.radii[self.parents_len as usize] = radius;
        self.items_len = self.parents_len;
        self.parents_len += 1;

        if self.parents_len == NODE_SIZE {
            let mut right = unsafe { Self::init_inplace(&mut children[NODE_SIZE + 1]) };
            right.child_idx = NODE_SIZE + 1;

            for i in self.items_len .. NODE_SIZE {
                let i = i as usize;
                let mut target = if distances[i] <= radius {
                    left
                } else {
                    right
                };
                unsafe {
                    ptr::copy(&self.items[i], &mut target.items[target.items_len as usize]);
                }
                target.items_len += 1;
            }
        } else {
            for i in self.items_len .. NODE_SIZE {
                let i = i as usize;
                if distances[i] <= radius {
                    unsafe {
                        ptr::copy(&self.items[i], &mut left.items[left.items_len as usize]);
                    }
                    left.items_len += 1;
                } else if self.items_len as usize != i {
                    unsafe {
                        ptr::copy(&self.items[i] as *const _, &mut self.items[self.items_len as usize]);
                    }
                    self.items_len += 1;
                }
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
        self.parents_len == NODE_SIZE
    }

    pub fn children_len(&self) -> usize {
        // if the parents list is full then we have n + 1 children
        (if self.parents_len == NODE_SIZE {
            NODE_SIZE + 1
        } else {
            // n children otherwise
            self.parents_len
        }) as usize
    }

    pub fn is_leaf(&self) -> bool {
        self.children.is_none()
    }

    pub fn items(&self) -> &[T] {
        &self.items[..self.items_len as usize]
    }

    pub fn items_mut(&mut self) -> &[T] {
        &mut self.items[..self.items_len as usize]
    }

    pub fn child(&self, idx: usize) -> &Node<T> {
        assert!(idx < self.children_len(), "idx out of bounds {} ({})", idx, self.children_len());
        &self.children.as_ref().expect("node children not allocated")[idx]
    }

    pub fn child_mut(&mut self, idx: usize) -> &mut Node<T> {
        assert!(idx < self.children_len(), "idx out of bounds {} ({})", idx, self.children_len());
        &mut self.children.as_mut().expect("node children not allocated")[idx]
    }

    /// # Safety
    /// Must ensure parent isn't being modified
    pub unsafe fn parent_and_idx(&self) -> Option<(&Node<T>, usize)> {
        if self.parent.is_null() {
            None
        } else {
            Some((&*self.parent, self.child_idx))
        }
    }

    pub unsafe fn parent_mut(&mut self) -> Option<&mut Node<T>> {
        (self.parent as *mut _).as_mut()
    }

    /// Get the distances to the items in this node
    /// Only the values in `..self.len()` are valid
    pub fn get_distances<Df>(&self, item: &T, dist_fn: Df) -> NodeArray<u64>
    where Df: Fn(&T, &T) -> u64 {
        let mut dists = [0u64; NODE_SIZE as usize];

        for (my_item, dist) in self.items.iter().zip(dists.iter_mut()) {
            *dist = dist_fn(item, my_item);
        }

        dists
    }

    pub fn find_parent(&self, distances: &NodeArray<u64>) -> Option<usize> {
        self.radii[..self.parents_len as usize].iter().zip(&distances)
            .position(|&(rad, dist)| dist <= rad)
    }
}

/// Allocates uninitialized memory and does not drop automatically
pub struct CustomBox<T>(NonNull<T>);

impl<T> CustomBox<T> {
    unsafe fn alloc() -> Self {
        let layout = Self::layout();

        let ptr = unsafe {
            alloc::alloc(layout)
        };

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
