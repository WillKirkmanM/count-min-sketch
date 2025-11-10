use std::hash::Hash;
use std::marker::PhantomData;
use ahash::RandomState;

/// A Count-Min Sketch implementation.
///
/// This probabilistic data structure is used for estimating the frequency
/// of items in a data stream.
///
/// - `T`: The type of item to be counted (must implement `Hash`).
/// - `C`: The type of the counter (e.g., u32, u64).
#[derive(Debug)]
pub struct CountMinSketch<T, C = u64> {
    /// The 2D array of counters.
    /// Dimensions are d (depth) x w (width).
    counters: Vec<Vec<C>>,
    
    /// The 'd' independent hash builders.
    /// We use RandomState from 'ahash' seeded differently for each row.
    hashers: Vec<RandomState>,
    
    /// The width of the sketch.
    width: usize,
    
    /// The depth of the sketch (number of hash functions).
    depth: usize,
    
    /// Marker to show we store items of type T
    _phantom: PhantomData<T>,
}

impl<T: Hash, C> CountMinSketch<T, C>
where
    C: Default + Copy + Ord + std::ops::AddAssign<C> + From<u8>,
{
    /// Creates a new `CountMinSketch` with a given error tolerance and confidence.
    ///
    /// The sketch is sized to guarantee that estimates are within `epsilon`
    /// of the true frequency with a probability of `1.0 - delta`.
    ///
    /// # Arguments
    ///
    /// * `epsilon`: The additive error factor (e.g., 0.001).
    ///   Smaller values create a wider sketch (more `width`).
    /// * `delta`: The probability of the estimate being wrong (e.g., 0.01 for 99% confidence).
    ///   Smaller values create a deeper sketch (more `depth`).
    ///
    /// # Panics
    ///
    /// Panics if `epsilon` or `delta` are not in the range (0.0, 1.0).
    pub fn new(epsilon: f64, delta: f64) -> Self {
        if !(0.0..=1.0).contains(&epsilon) || !(0.0..=1.0).contains(&delta) {
            panic!("Epsilon and Delta must be between 0.0 and 1.0");
        }

        // Calculate optimal width (w) and depth (d)
        // w = ceil(e / epsilon)
        let width = (std::f64::consts::E / epsilon).ceil() as usize;
        // d = ceil(ln(1 / delta))
        let depth = (1.0 / delta).ln().ceil() as usize;

        // Initialize 'd' hash builders, each with a different seed.
        // We use the row index 'i' as the primary seed to ensure
        // each RandomState is different and thus produces different hashes.
        let hashers = (0..depth)
            .map(|i| RandomState::with_seeds(i as u64, 0, 0, 0))
            .collect();

        // Initialize the counter matrix with zeros
        let counters = vec![vec![C::default(); width]; depth];

        CountMinSketch {
            counters,
            hashers,
            width,
            depth,
            _phantom: PhantomData,
        }
    }

    /// Adds (increments the count of) an item to the sketch.
    pub fn add(&mut self, item: &T) {
        for i in 0..self.depth {
            let index = self.get_index(item, i);
            // Increment by one using From<u8> to construct a C = 1
            self.counters[i][index] += C::from(1u8);
        }
    }

    /// Estimates the frequency of an item.
    ///
    /// This value is guaranteed to be **at least** the true frequency.
    /// It will never undercount.
    pub fn estimate(&self, item: &T) -> C {
        let mut min_count = C::default(); // Will be replaced by first value

        for i in 0..self.depth {
            let index = self.get_index(item, i);
            let count = self.counters[i][index];

            if i == 0 || count < min_count {
                min_count = count;
            }
        }
        min_count
    }

    /// Helper function to get the column index for a given item and hash function (row).
    fn get_index(&self, item: &T, row_index: usize) -> usize {
        let hash = self.hashers[row_index].hash_one(item);
        // Modulo the hash by the width to get the column index
        (hash % self.width as u64) as usize
    }

    /// Returns the depth (d) of the sketch.
    pub fn depth(&self) -> usize {
        self.depth
    }

    /// Returns the width (w) of the sketch.
    pub fn width(&self) -> usize {
        self.width
    }
}

// A version of 'add' specific to integer counters for cleaner incrementing.
impl<T: Hash> CountMinSketch<T, u64> {
     /// Adds (increments the count of) an item to the sketch.
    pub fn add_u64(&mut self, item: &T) {
        for i in 0..self.depth {
            let index = self.get_index(item, i);
            self.counters[i][index] += 1;
        }
    }
}

fn main() {
    // Create a sketch with:
    // - 0.1% error (epsilon = 0.001)
    // - 99.9% confidence (delta = 0.001)
    let mut sketch = CountMinSketch::<&str, u64>::new(0.001, 0.001);

    println!("Sketch created with: ");
    println!("  -> Width (w): {}", sketch.width());
    println!("  -> Depth (d): {}", sketch.depth());
    println!("---");

    // Add some items
    sketch.add_u64(&"apple");
    sketch.add_u64(&"apple");
    sketch.add_u64(&"orange");
    sketch.add_u64(&"apple");
    sketch.add_u64(&"banana");
    sketch.add_u64(&"orange");

    // Add a lot of another item to force collisions
    for _ in 0..100 {
        sketch.add_u64(&"grape");
    }

    // Estimate counts
    println!("Estimates:");
    println!("  -> 'apple':  {}", sketch.estimate(&"apple")); // Expected: 3
    println!("  -> 'orange': {}", sketch.estimate(&"orange")); // Expected: 2
    println!("  -> 'banana': {}", sketch.estimate(&"banana")); // Expected: 1
    println!("  -> 'grape':  {}", sketch.estimate(&"grape"));  // Expected: 100
    
    // This item was never added, but its estimate might be > 0
    // due to hash collisions. This is the "overcounting" nature.
    println!("  -> 'kiwi':   {}", sketch.estimate(&"kiwi"));   // Expected: 0 (but possibly > 0)
}
