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

pub type StableVec<T> = ArrayVec<T, ArrayBox<T>>;

pub struct ArrayVec<T, B = [T; NODE_SIZE]> {
    len: usize,
    buf: ManuallyDrop<B>,
}

impl<T, B: Buf<T>> ArrayVec<T, B> {
    pub fn new() -> Self {
        ArrayVec {
            len: 0,
            buf: unsafe { ManuallyDrop::new(B::new_uninit()) },
        }
    }

    pub fn len(&self) -> usize {
        self.len
    }

    pub fn push(&mut self, item: T) {
        assert!(self.len < NODE_SIZE, "attempting to push into full ArrayVec");
        self.buf.write_val(self.len, item);
        self.len += 1;
    }

    pub fn pop(&mut self) -> Option<T> {
        if self.len > 0 {
            self.len -= 1;
            Some(unsafe { self.buf.read_val(self.len) })
        } else {
            None
        }
    }

    pub fn insert(&mut self, i: usize, item: T) {
        assert!(self.len < NODE_SIZE, "attempting to push into full ArrayVec");
        self.buf.write_val(i, item);
        self.len += 1; // shouldn't overflow
    }

    pub fn remove(&mut self, i: usize) -> T {
        assert!(i < self.len, "index out of bounds: {} len: {}", i, self.len);

        let item = unsafe { self.buf.read_val(i) };
        self.buf.slice_from_mut(i).rotate_left(1);

        self.len -= 1;

        item
    }

    /// Simplified `drain()` where the range is always `start..`
    pub fn drain_tail(&mut self, start: usize) -> DrainTail<T> {
        assert!(start <= self.len, "start out of bounds: {} len: {}", start, self.len);
        let len = self.len;
        self.len = start;
        DrainTail {
            idx: 0,
            slice: self.buf.slice_from_mut(start),
        }
    }
}

impl<T: Clone, B: Buf<T>> Clone for ArrayVec<T, B> {
    fn clone(&self) -> Self {
        self.iter().cloned().collect()
    }
}

impl<T, B: Buf<T>> Deref for ArrayVec<T, B> {
    type Target = [T];

    fn deref(&self) -> &[T] {
        self.buf.slice_to(self.len)
    }
}

impl<T, B: Buf<T>> DerefMut for ArrayVec<T, B> {
    fn deref_mut(&mut self) -> &mut [T] {
        self.buf.slice_to_mut(self.len)
    }
}

impl<T, B: Buf<T>> IntoIterator for ArrayVec<T, B> {
    type Item = T;
    type IntoIter = IntoIter<T, B>;

    fn into_iter(self) -> IntoIter<T, B> {
        IntoIter {
            buf: self.buf,
            len: self.len,
            idx: 0,
        }
    }
}

impl<T, B: Buf<T>> Extend<T> for ArrayVec<T, B> {
    fn extend<I>(&mut self, iter: I) where I: IntoIterator<Item = T> {
        for item in iter {
            self.push(item);
        }
    }
}

impl<T, B: Buf<T>> FromIterator<T> for ArrayVec<T, B> {
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

impl<T, B: Buf<T>> Drop for ArrayVec<T, B> {
    fn drop(&mut self) {
        for item in self {
            unsafe {
                ptr::drop_in_place(item);
            }
        }
    }
}

pub struct IntoIter<T, B> {
    buf: ManuallyDrop<B>,
    idx: usize,
    len: usize,
}

impl<T, B: Buf<T>> Iterator for IntoIter<T, B> {
    type Item = T;

    fn next(&mut self) -> Option<T> {
        if self.idx == self.len {
            return None;
        }

        let item = unsafe { self.buf.read_val(self.idx) };
        self.idx += 1;
        Some(item)
    }
}

impl<T, B: Buf<T>> Drop for IntoIter<T, B> {
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

pub trait Buf<T> {
    unsafe fn new_uninit() -> Self;

    fn capacity(&self) -> usize { NODE_SIZE }

    fn slice_to(&self, to: usize) -> &[T];

    fn slice_to_mut(&mut self, to: usize) -> &mut [T];

    fn slice_from_mut(&mut self, from: usize) -> &mut [T];

    fn write_val(&mut self, idx: usize, val: T);

    unsafe fn read_val(&self, idx: usize) -> T;

    unsafe fn free(&mut self) {}
}

impl<T> Buf<T> for [T; NODE_SIZE] {
    unsafe fn new_uninit() -> Self {
        mem::uninitialized()
    }

    fn slice_to(&self, to: usize) -> &[T] {
        &self[..to]
    }

    fn slice_to_mut(&mut self, to: usize) -> &mut [T] {
        &mut self[..to]
    }

    fn slice_from_mut(&mut self, from: usize) -> &mut [T] {
        &mut self[from..]
    }

    fn write_val(&mut self, idx: usize, val: T) {
        unsafe {
            ptr::write(&mut self[idx], val);
        }
    }

    unsafe fn read_val(&self, idx: usize) -> T {
        ptr::read(&self[idx])
    }
}

pub struct ArrayBox<T>(*mut [T; NODE_SIZE + 1]);

impl<T> Buf<T> for ArrayBox<T> {
    unsafe fn new_uninit() -> Self {
        ArrayBox(ptr::null_mut())
    }

    fn capacity(&self) -> usize {
        NODE_SIZE + 1
    }

    fn slice_to(&self, to: usize) -> &[T] {
        if self.0.is_null() {
            assert_eq!(to, 0, "slice end out of bounds");
            &[]
        } else {
            unsafe {
                &(*self.0)[..to]
            }
        }
    }

    fn slice_to_mut(&mut self, to: usize) -> &mut [T] {
        if self.0.is_null() {
            assert_eq!(to, 0, "slice end out of bounds");
            &mut []
        } else {
            unsafe {
                &mut (*self.0)[..to]
            }
        }
    }

    fn slice_from_mut(&mut self, from: usize) -> &mut [T] {
        assert!(!self.0.is_null(), "slice start out of bounds");
        unsafe {
            &mut (*self.0)[from..]
        }
    }

    fn write_val(&mut self, idx: usize, val: T) {
        if self.0.is_null() {
            self.0 = unsafe {
                alloc::alloc(Self::layout()) as *mut _
            };

            if self.0.is_null() {
                alloc::handle_alloc_error(Self::layout());
            }
        }

        unsafe {
            // using index instead of offset to get free bounds check
            ptr::write(&mut (*self.0)[idx], val)
        }
    }

    unsafe fn read_val(&self, idx: usize) -> T {
        assert!(!self.0.is_null(), "attempting to read from a null pointer");
        ptr::read(&(*self.0)[idx])
    }

    unsafe fn free(&mut self) {
        assert!(!self.0.is_null(), "double-free");
        let ptr = self.0 as *mut u8;
        self.0 = ptr::null_mut();
        alloc::dealloc(ptr, Self::layout())
    }
}

impl<T> ArrayBox<T> {
    fn layout() -> Layout {
        Layout::new::<[T; NODE_SIZE + 1]>
    }
}
