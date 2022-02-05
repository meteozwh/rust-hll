pub mod bit_array;
mod data;

use siphasher::sip::SipHasher13;
use std::cmp::Ordering::{Equal, Greater, Less};
use std::hash::{Hash, Hasher};
use std::iter::repeat;

use bit_array::BitArray;
use data::{BIAS_DATA, RAW_ESTIMATE_DATA, THRESHOLD_DATA};

pub struct HyperLogLog {
    pub reader: HyperLogLogReader,
    sip: SipHasher13,
}

#[allow(dead_code)]
impl HyperLogLog {
    pub fn new_from_error(error_rate: f64, compressed: bool) -> Self {
        HyperLogLog {
            reader: HyperLogLogReader::new_from_error(error_rate, compressed),
            sip: SipHasher13::new_with_keys(rand::random(), rand::random()),
        }
    }

    pub fn new(bits: u8, compressed: bool) -> Self {
        HyperLogLog {
            reader: HyperLogLogReader::new(bits, compressed),
            sip: SipHasher13::new_with_keys(rand::random(), rand::random()),
        }
    }

    pub fn from_sequence(sequence: &[u8]) -> Option<Self> {
        HyperLogLogReader::from_sequence(sequence).map(|reader| HyperLogLog {
            reader,
            sip: SipHasher13::new_with_keys(rand::random(), rand::random()),
        })
    }

    pub fn new_from_template(hll: &HyperLogLog) -> Self {
        HyperLogLog {
            reader: HyperLogLogReader::new_from_template(&hll.reader),
            sip: hll.sip,
        }
    }

    pub fn insert<V: Hash>(&mut self, value: &V) {
        let sip = &mut self.sip.clone();
        value.hash(sip);
        let x = sip.finish();
        self.insert_by_hash_value(x);
    }

    pub fn insert_by_hash_value(&mut self, x: u64) {
        self.reader.insert_by_hash_value(x);
    }

    pub fn len(&self) -> f64 {
        self.reader.len()
    }

    pub fn is_empty(&self) -> bool {
        self.reader.is_empty()
    }

    pub fn merge(&mut self, src: &HyperLogLog) {
        let sip1 = &mut src.sip.clone();
        let sip2 = &mut self.sip.clone();
        42.hash(sip1);
        42.hash(sip2);
        if sip1.finish() != sip2.finish() {
            return;
        }

        self.reader.merge(&src.reader);
    }

    pub fn clear(&mut self) {
        self.reader.clear();
    }
}

pub struct HyperLogLogReader {
    alpha: f64,
    p: u8,
    m: usize,
    seq: BitArray,
    compressed: bool,
}

impl HyperLogLogReader {
    pub fn new_from_error(error_rate: f64, compressed: bool) -> Self {
        assert!(error_rate > 0.0 && error_rate < 1.0);
        let sr = 1.04 / error_rate;
        let mut p = f64::ln(sr * sr).ceil() as u8;
        if p < 4 {
            p = 4;
        }
        if p > 18 {
            p = 18;
        }
        let alpha = Self::get_alpha(p);
        let m = 1usize << p;
        let ele_size = if compressed { 6usize } else { 8usize };
        let mut seq = BitArray::new(ele_size);
        (0..m).for_each(|_| {
            seq.insert(0u8);
        });
        HyperLogLogReader {
            alpha,
            p,
            m,
            seq,
            compressed,
        }
    }

    pub fn new(bits: u8, compressed: bool) -> Self {
        let mut p = bits;
        if p < 4 {
            p = 4;
        }
        if p > 18 {
            p = 18;
        }
        let alpha = Self::get_alpha(p);
        let m = 1usize << p;
        let ele_size = if compressed { 6usize } else { 8usize };
        let mut seq = BitArray::new(ele_size);
        (0..m).for_each(|_| {
            seq.insert(0u8);
        });
        HyperLogLogReader {
            alpha,
            p,
            m,
            seq,
            compressed,
        }
    }

    pub fn from_sequence(sequence: &[u8]) -> Option<Self> {
        if sequence.is_empty() || sequence[0] == 0 {
            return None;
        }

        let compressed = (sequence[0] & 32u8) != 0;
        let p = sequence[0] & !32u8;
        if p < 4 {
            return None;
        }
        if p > 18 {
            return None;
        }
        let alpha = Self::get_alpha(p);
        let m = 1usize << p;
        let ele_size = if compressed { 6usize } else { 8usize };
        let ele_mask = if ele_size < 8 {
            (1u8 << ele_size) - 1
        } else {
            !0u8
        };

        let seq_size = ((m * ele_size) + 7usize) / 8usize;

        if sequence.len() < seq_size + 1 {
            return None;
        }

        let mut seq = BitArray::new(ele_size);
        (0..m).for_each(|i| {
            if !compressed {
                seq.insert(sequence[i + 1]);
            } else {
                let byte_idx: usize = (i * ele_size) / 8usize;
                let low_bit = (i * ele_size) % 8usize;
                let high_bit = low_bit + ele_size;

                if let Some(value) = sequence.get(byte_idx + 1).and_then(|low_v| {
                    let low_val = (*low_v >> low_bit) & ele_mask;
                    if high_bit <= 8usize {
                        Some(low_val)
                    } else {
                        sequence.get(byte_idx + 2).map(|high_v| {
                            if (8 - low_bit) < 8 {
                                let mask = ele_mask >> (8 - low_bit);
                                let high_val = (*high_v & mask) << (8 - low_bit);
                                high_val + low_val
                            } else {
                                low_val
                            }
                        })
                    }
                }) {
                    seq.insert(value);
                }
            }
        });
        Some(HyperLogLogReader {
            alpha,
            p,
            m,
            seq,
            compressed,
        })
    }

    pub fn new_from_template(hll: &HyperLogLogReader) -> Self {
        let ele_size = if hll.compressed { 6usize } else { 8usize };
        let mut seq = BitArray::new(ele_size);
        (0..hll.m).for_each(|_| {
            seq.insert(0u8);
        });
        HyperLogLogReader {
            alpha: hll.alpha,
            p: hll.p,
            m: hll.m,
            seq,
            compressed: hll.compressed,
        }
    }

    pub fn len(&self) -> f64 {
        let zeros = Self::vec_count_zero(&self.seq);
        if zeros > 0 {
            let count = self.m as f64 * (self.m as f64 / zeros as f64).ln();
            if count <= Self::get_threshold(self.p) {
                count
            } else {
                self.ep()
            }
        } else {
            self.ep()
        }
    }

    pub fn is_empty(&self) -> bool {
        self.len() == 0.0
    }

    pub fn insert_by_hash_value(&mut self, x: u64) {
        let j = x as usize & (self.m - 1);
        let w = x >> self.p;
        let rho = Self::get_rho(w, 64 - self.p);
        let mjr = self.seq.get(j).unwrap();
        if rho > mjr {
            self.seq.set(j, rho);
        }
    }

    pub fn merge(&mut self, src: &HyperLogLogReader) {
        if src.p != self.p || src.m != self.m {
            return;
        }

        for i in 0..self.m {
            let src_mir = src.seq.get(i).unwrap();
            let mir = self.seq.get(i).unwrap();
            if src_mir > mir {
                self.seq.set(i, src_mir);
            }
        }
    }

    pub fn clear(&mut self) {
        self.seq.clear();
    }

    fn get_threshold(p: u8) -> f64 {
        THRESHOLD_DATA[(p - 4) as usize]
    }

    fn get_alpha(p: u8) -> f64 {
        assert!((4..=18).contains(&p));
        match p {
            4 => 0.673,
            5 => 0.697,
            6 => 0.709,
            _ => 0.7213 / (1.0 + 1.079 / (1usize << (p as usize)) as f64),
        }
    }

    fn bit_length(x: u64) -> u8 {
        64u8 - (x.leading_zeros() as u8)
    }

    fn get_rho(w: u64, max_width: u8) -> u8 {
        let rho = max_width - Self::bit_length(w) + 1;
        assert!(rho > 0);
        rho
    }

    fn vec_count_zero(v: &BitArray) -> usize {
        v.count_zeros()
    }

    fn estimate_bias(error: f64, p: u8) -> f64 {
        let bias_vector = BIAS_DATA[(p - 4) as usize];
        let nearest_neighbors =
            Self::get_nearest_neighbors(error, RAW_ESTIMATE_DATA[(p - 4) as usize]);
        let sum = nearest_neighbors
            .iter()
            .fold(0.0, |acc, &neighbor| acc + bias_vector[neighbor]);
        sum / nearest_neighbors.len() as f64
    }

    fn get_nearest_neighbors(error: f64, estimate_vector: &[f64]) -> Vec<usize> {
        let ev_len = estimate_vector.len();
        let mut r: Vec<(f64, usize)> = repeat((0.0f64, 0usize)).take(ev_len).collect();
        for i in 0..ev_len {
            let dr = error - estimate_vector[i];
            r[i] = (dr * dr, i);
        }
        r.sort_by(|a, b| {
            if a < b {
                Less
            } else if a > b {
                Greater
            } else {
                Equal
            }
        });
        r.truncate(6);
        r.iter()
            .map(|&ez| {
                let (_, b) = ez;
                b
            })
            .collect()
    }

    fn ep(&self) -> f64 {
        let sum = (0..self.seq.length).fold(0.0, |acc, idx| {
            acc + 2.0f64.powi(-(self.seq.get(idx).unwrap() as i32))
        });
        let error = self.alpha * (self.m * self.m) as f64 / sum;
        if error <= (5 * self.m) as f64 {
            error - Self::estimate_bias(error, self.p)
        } else {
            error
        }
    }
}

#[cfg(test)]
mod tests {
    use super::HyperLogLog;
    use std::collections::HashSet;

    #[test]
    fn hyperloglog_test_simple() {
        let mut hll = HyperLogLog::new_from_error(0.000408, false);
        let keys = ["test1", "test2", "test3", "test2", "test2", "test2"];
        for k in &keys {
            hll.insert(k);
        }
        assert!((hll.len().round() - 3.0).abs() < std::f64::EPSILON);
        assert!(!hll.is_empty());
        hll.clear();
        assert!(hll.is_empty());
        assert!(hll.len() == 0.0);
    }

    #[test]
    fn hyperloglog_test_large() {
        let bits: Vec<u8> = vec![4, 5, 6, 7, 8, 9, 10, 14, 18];
        for bit in bits.iter() {
            println!(
                "\nBits: {}, Bucket number: {}, Array size: {}",
                bit,
                1u32 << bit,
                (((1u32 << bit) * 6u32) >> 3) + 1u32
            );
            let count: Vec<usize> = vec![
                3, 5, 10, 50, 100, 200, 500, 1000, 2000, 5000, 10000, 20000, 50000, 500000,
            ];
            let mut hll = HyperLogLog::new(*bit, true);
            let mut keys: Vec<String> = Vec::new();
            let mut set: HashSet<String> = HashSet::new();

            for c in count.iter() {
                for _ in 0..*c {
                    let key = format!("test{}", rand::random::<u32>());
                    set.insert(key.clone());
                    keys.push(key);
                }
                for k in &keys {
                    hll.insert(k);
                }
                let len = set.len();
                let hll_len = (hll.len()).round() as usize;
                let err = (((len as f64 - hll_len as f64).abs() / (len as f64)) * 10000.0f64)
                    .round()
                    / 100.0f64;
                let tab_sep2 = "\t\t";
                let tab_sep1 = if *c > 99999 { "\t" } else { tab_sep2 };
                println!(
                    "ID count: {}{}Estimated count: {}{}error: {}%",
                    len, tab_sep1, hll_len, tab_sep2, err
                );
                hll.clear();
                keys.clear();
                set.clear();
            }
        }
    }

    #[test]
    fn hyperloglog_test_merge() {
        let mut hll = HyperLogLog::new_from_error(0.00408, false);
        let keys = ["test1", "test2", "test3", "test2", "test2", "test2"];
        for k in &keys {
            hll.insert(k);
        }
        println!("======{}======", hll.len());
        assert!((hll.len().round() - 3.0).abs() < std::f64::EPSILON);

        let mut hll2 = HyperLogLog::new_from_template(&hll);
        let keys2 = ["test3", "test4", "test4", "test4", "test4", "test1"];
        for k in &keys2 {
            hll2.insert(k);
        }
        assert!((hll2.len().round() - 3.0).abs() < std::f64::EPSILON);

        hll.merge(&hll2);
        assert!((hll.len().round() - 4.0).abs() < std::f64::EPSILON);
    }
}
