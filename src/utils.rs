pub mod scopeguard {
    pub fn guard<T, F: FnOnce(T)>(v: T, f: F) -> Guard<T, F> {
        Guard {
            v: Some(v),
            f: Some(f),
        }
    }

    pub struct Guard<T, F: FnOnce(T)> {
        v: Option<T>,
        f: Option<F>,
    }

    impl<T, F: FnOnce(T)> Drop for Guard<T, F> {
        fn drop(&mut self) {
            if let (Some(v), Some(f)) = (self.v.take(), self.f.take())
            {
                f(v);
            }
        }
    }
}
