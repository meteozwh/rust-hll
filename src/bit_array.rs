pub struct BitArray {
    pub vec: Vec<u8>,
    pub length: usize,
    element_size: usize,
    element_mask: u8,
}

impl BitArray {
    pub fn new(element_size: usize) -> Self {
        let ele_size = if element_size > 8 {
            8
        } else if element_size == 0 {
            1
        } else {
            element_size
        };
        let element_mask = if ele_size == 8 {
            0xff
        } else {
            (1u8 << ele_size) - 1
        };
        BitArray {
            vec: Vec::new(),
            length: 0,
            element_size: ele_size,
            element_mask,
        }
    }

    pub fn count_zeros(&self) -> usize {
        let mut count = 0;
        for i in 0..self.length {
            if self.get(i).map(|v| v == 0).unwrap_or(false) {
                count += 1;
            }
        }
        count
    }

    pub fn clear(&mut self) {
        self.vec.iter_mut().for_each(|v| *v = 0);
    }

    pub fn get(&self, index: usize) -> Option<u8> {
        if self.element_size == 8 {
            return self.vec.get(index).cloned();
        }

        let byte_idx: usize = (index * self.element_size) / 8usize;
        let low_bit = (index * self.element_size) % 8usize;
        let high_bit = low_bit + self.element_size;

        self.vec.get(byte_idx).and_then(|low_v| {
            let low_val = (*low_v >> low_bit) & self.element_mask;
            if high_bit <= 8usize {
                Some(low_val)
            } else {
                self.vec.get(byte_idx + 1).map(|high_v| {
                    if (8 - low_bit) < 8 {
                        let mask = self.element_mask >> (8 - low_bit);
                        let high_val = (*high_v & mask) << (8 - low_bit);
                        high_val + low_val
                    } else {
                        low_val
                    }
                })
            }
        })
    }

    pub fn set(&mut self, index: usize, mut val: u8) {
        if self.element_size == 8 {
            self.vec[index] = val;
            return;
        }

        val &= self.element_mask;
        let byte_idx: usize = (index * self.element_size) / 8usize;
        let low_bit = (index * self.element_size) % 8usize;
        let high_bit = low_bit + self.element_size;

        if let Some(low_v) = self.vec.get_mut(byte_idx) {
            *low_v = ((*low_v) & !(self.element_mask << low_bit)) | (val << low_bit);

            if high_bit > 8usize {
                if let Some(high_v) = self.vec.get_mut(byte_idx + 1) {
                    if (8 - low_bit) < 8 {
                        let mask = self.element_mask >> (8 - low_bit);
                        *high_v = ((*high_v) & !mask) | (val >> (8 - low_bit));
                    }
                }
            }
        }
    }

    pub fn insert(&mut self, val: u8) {
        let byte_length = ((self.length + 1) * self.element_size + 7usize) / 8usize;
        if byte_length > self.vec.len() {
            self.vec.push(0u8);
        }
        self.set(self.length, val);
        self.length += 1;
    }
}
