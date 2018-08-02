//! No-bs array-vec impl

use std::alloc::{self, Layout};
use std::borrow::{Borrow, BorrowMut};
use std::ops::{Index, IndexMut};
use std::iter::FromIterator;
use std::mem::{self, ManuallyDrop};
use std::num::NonZeroUsize;
use std::ops::{Deref, DerefMut};
use std::ptr;

use NODE_SIZE;

pub struct ArrayVec<T> {
    len: usize,
    items: ManuallyDrop<[T; NODE_SIZE]>,
}

// only provide well-defined constructors since we use uninitialized data
impl<T> ArrayVec<T> {
    pub fn new() -> Self {
        ArrayVec {
            len: 0,
            items: unsafe { mem::uninitialized() },
        }
    }
}

impl<T> ArrayVec<T> {
    pub fn len(&self) -> usize {
        self.len
    }

    pub fn push(&mut self, item: T) {
        assert!(self.len < NODE_SIZE, "attempting to push into full ArrayVec");

        unsafe {
            ptr::write(self.items(self.len), item);
        }

        self.len += 1;
    }

    pub fn pop(&mut self) {
        if self.len > 0 {
            self.len -= 1;
            Some(unsafe { ptr::read(self.items(self.len)) })
        } else {
            None
        }
    }

    pub fn insert(&mut self, i: usize, item: T) {
        assert!(self.len < NODE_SIZE, "attempting to push into full ArrayVec");

        self.items_mut(i..).rotate_right(1);

        unsafe {
            ptr::write(self.items_mut(i), item);
        }

        self.len += 1; // shouldn't overflow
    }

    pub fn remove(&mut self, i: usize) -> T {
        assert!(i < self.len, "index out of bounds: {} len: {}", i, self.len);


        let item = unsafe { ptr::read(self.items_mut(i)) };
        self.items_mut(i..).rotate_left(1);

        self.len -= 1;

        item
    }

    /// Simplified `drain()` where the range is always `start..`
    pub fn drain_tail(&mut self, start: usize) -> DrainTail<T> {
        assert!(start <= self.len, "start out of bounds: {} len: {}", start, self.len);
        let len = self.len;
        self.len = start;
        DrainTail {
            idx: start,
            slice: self.items_mut(..),
        }
    }
}

impl<T: Clone> Clone for ArrayVec<T> {
    fn clone(&self) -> Self {
        self.iter().cloned().collect()
    }
}

impl<T> Deref for ArrayVec<T> {
    type Target = [T];

    fn deref(&self) -> &[T] {
        self.items(..self.len)
    }
}

impl<T> DerefMut for ArrayVec<T> {
    fn deref_mut(&mut self) -> &mut [T] {
        self.items_mut(..self.len)
    }
}

impl<T> IntoIterator for ArrayVec<T> {
    type Item = T;
    type IntoIter = IntoIter<T>;

    fn into_iter(self) -> IntoIter<T> {
        IntoIter {
            items: self.items,
            len: self.len,
            idx: 0,
        }
    }
}

impl<T> Extend<T> for ArrayVec<T> {
    fn extend<I>(&mut self, iter: I) where I: IntoIterator<Item = T> {
        for item in iter {
            self.push(item);
        }
    }
}

impl<T> FromIterator<T> for ArrayVec<T> {
    fn from_iter<I: IntoIterator<Item=T>>(iter: I) -> Self {
        let mut array = ArrayVec::new();

        for item in iter {
            // this will panic if `iter` contains more than `VEC_SIZE` items
            // but it's a private interface anyway
            array.push(item);
        }

        array
    }
}

impl<T> Drop for ArrayVec<T> {
    fn drop(&mut self) {
        for item in self {
            unsafe {
                ptr::drop_in_place(item);
            }
        }
    }
}

pub struct IntoIter<T> {
    items: ManuallyDrop<[T; NODE_SIZE]>,
    idx: usize,
    len: usize,
}

impl<T> Iterator for IntoIter<T> {
    type Item = T;

    fn next(&mut self) -> Option<T> {
        if self.idx == self.len {
            return None;
        }

        let item = unsafe { ptr::read(&self.items[self.idx]) };
        self.idx += 1;
        Some(item)
    }
}

impl<T> Drop for IntoIter<T> {
    fn drop(&mut self) {
        for _ in self {}
    }
}

pub struct DrainTail<'a, T: 'a> {
    idx: usize,
    slice: &'a mut [T],
}

impl<'a, T: 'a> Iterator for DrainTail<'a, T> {
    type Item = T;

    fn next(&mut self) -> Option<T> {
        if self.idx == self.slice.len() {
            return None;
        }

        let val = unsafe { ptr::read(&self.slice[self.idx]) };
        self.idx += 1;
        Some(val)
    }
}

impl<'a, T: 'a> Drop for DrainTail<'a, T> {
    fn drop(&mut self) {
        for _ in self {}
    }
}
