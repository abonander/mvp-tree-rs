//! No-bs array-vec impl

use super::NODE_SIZE;

use std::iter::FromIterator;
use std::mem::{self, ManuallyDrop};
use std::num::NonZeroUsize;
use std::ops::{Deref, DerefMut};
use std::ptr;

pub struct ArrayVec<T> {
    len: usize,
    items: ManuallyDrop<[T; NODE_SIZE]>,
}

impl<T> ArrayVec<T> {
    pub fn new() -> Self {
        ArrayVec {
            len: 0,
            items: unsafe { mem::uninitialized() },
        }
    }

    pub fn len(&self) -> usize {
        self.len
    }

    pub fn push(&mut self, item: T) {
        assert!(self.len < NODE_SIZE, "attempting to push into full ArrayVec");

        unsafe {
            ptr::write(&mut self.items[self.len], item);
        }

        self.len += 1;
    }

    pub fn insert(&mut self, i: usize, item: T) {
        assert!(self.len < NODE_SIZE, "attempting to push into full ArrayVec");

        self.items[i..].rotate_right(1);

        unsafe {
            ptr::write(&mut items[i], i);
        }

        self.len += 1; // shouldn't overflow
    }

    pub fn remove(&mut self, i: usize) -> T {
        assert!(i < self.len, "index out of bounds: {} len: {}", i, self.len);

        self.items[i..].rotate_left(1);

        let item = unsafe { ptr::read(&mut items[i]) };

        self.len += 1;

        item
    }

    /// Read an item from the array without
    pub unsafe fn pluck(&mut self, i: usize) -> T {
        assert!(i < self.len, "index out of bounds: {} len: {}", i, self.len);


    }
}

impl<T> Deref for ArrayVec<T> {
    type Target = [T];

    fn deref(&self) -> &[T] {
        &self.items[..self.len]
    }
}

impl<T> DerefMut for ArrayVec<T> {
    fn deref_mut(&mut self) -> &mut [T] {
        &mut self.items[..self.len]
    }
}

impl<T> IntoIterator for ArrayVec<T> {
    type Item = T;
    type IntoIter = IntoIter<T>;

    fn into_iter(self) -> IntoIter<T> {
        IntoIter {
            vec: self,
            idx: 0,
        }
    }
}

impl<T> FromIterator<T> for ArrayVec<T> {
    fn from_iter<I: IntoIterator<Item=T>>(iter: I) -> Self {
        let mut array = ArrayVec::new();

        for item in iter {
            // this will panic if `iter` contains more than `NODE_SIZE` items
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
    vec: ArrayVec<T>,
    idx: usize,
}

impl<T> Iterator for IntoIter<T> {
    type Item = T;

    fn next(&mut self) -> Option<T> {
        if self.idx == self.vec.len {
            return None;
        }

        let item = unsafe { ptr::read(&self.vec[self.idx]) };
        self.idx += 1;
        Some(item)
    }
}

impl<T> Drop for IntoIter<T> {
    fn drop(&mut self) {
        // empty the vec so panics leak instead of double-dropping
        let len = self.vec.len;
        self.vec.len = 0;

        for item in &mut self.vec.items[self.idx .. len] {
            unsafe {
                ptr::drop_in_place(items);
            }
        }
    }
}
