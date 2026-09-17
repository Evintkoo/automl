//! SIMD-optimized operations
//! 
//! Provides architecture-agnostic SIMD operations with fallbacks.

#[cfg(target_arch = "x86_64")]
use std::arch::x86_64::*;

/// SIMD-accelerated mathematical operations
pub struct SimdOps;

impl SimdOps {
    /// Sum of array elements using SIMD
    #[cfg(target_arch = "x86_64")]
    pub fn sum_f64(data: &[f64]) -> f64 {
        if !is_x86_feature_detected!("avx2") || data.len() < 4 {
            return data.iter().sum();
        }

        unsafe { Self::sum_f64_avx(data) }
    }

    #[cfg(not(target_arch = "x86_64"))]
    pub fn sum_f64(data: &[f64]) -> f64 {
        data.iter().sum()
    }

    #[cfg(target_arch = "x86_64")]
    #[target_feature(enable = "avx2")]
    unsafe fn sum_f64_avx(data: &[f64]) -> f64 {
        let chunks = data.chunks_exact(4);
        let remainder = chunks.remainder();

        let mut sum_vec = _mm256_setzero_pd();

        for chunk in chunks {
            let vec = _mm256_loadu_pd(chunk.as_ptr());
            sum_vec = _mm256_add_pd(sum_vec, vec);
        }

        // Horizontal sum
        let low = _mm256_castpd256_pd128(sum_vec);
        let high = _mm256_extractf128_pd(sum_vec, 1);
        let sum128 = _mm_add_pd(low, high);
        let high64 = _mm_unpackhi_pd(sum128, sum128);
        let sum64 = _mm_add_sd(sum128, high64);
        
        let mut result = _mm_cvtsd_f64(sum64);

        // Add remainder
        for &val in remainder {
            result += val;
        }

        result
    }

    /// Mean of array elements using SIMD
    pub fn mean_f64(data: &[f64]) -> f64 {
        if data.is_empty() {
            return 0.0;
        }
        Self::sum_f64(data) / data.len() as f64
    }

    /// Dot product using SIMD
    #[cfg(target_arch = "x86_64")]
    pub fn dot_f64(a: &[f64], b: &[f64]) -> f64 {
        assert_eq!(a.len(), b.len(), "Arrays must have same length");
        
        if !is_x86_feature_detected!("avx2") || a.len() < 4 {
            return a.iter().zip(b.iter()).map(|(x, y)| x * y).sum();
        }

        unsafe { Self::dot_f64_avx(a, b) }
    }

    #[cfg(not(target_arch = "x86_64"))]
    pub fn dot_f64(a: &[f64], b: &[f64]) -> f64 {
        a.iter().zip(b.iter()).map(|(x, y)| x * y).sum()
    }

    #[cfg(target_arch = "x86_64")]
    #[target_feature(enable = "avx2")]
    unsafe fn dot_f64_avx(a: &[f64], b: &[f64]) -> f64 {
        let chunks_a = a.chunks_exact(4);
        let chunks_b = b.chunks_exact(4);
        let remainder_a = chunks_a.remainder();
        let remainder_b = chunks_b.remainder();

        let mut sum_vec = _mm256_setzero_pd();

        for (chunk_a, chunk_b) in chunks_a.zip(chunks_b) {
            let vec_a = _mm256_loadu_pd(chunk_a.as_ptr());
            let vec_b = _mm256_loadu_pd(chunk_b.as_ptr());
            let prod = _mm256_mul_pd(vec_a, vec_b);
            sum_vec = _mm256_add_pd(sum_vec, prod);
        }

        // Horizontal sum
        let low = _mm256_castpd256_pd128(sum_vec);
        let high = _mm256_extractf128_pd(sum_vec, 1);
        let sum128 = _mm_add_pd(low, high);
        let high64 = _mm_unpackhi_pd(sum128, sum128);
        let sum64 = _mm_add_sd(sum128, high64);
        
        let mut result = _mm_cvtsd_f64(sum64);

        // Add remainder
        for (&va, &vb) in remainder_a.iter().zip(remainder_b.iter()) {
            result += va * vb;
        }

        result
    }

    /// Euclidean distance using SIMD
    pub fn euclidean_distance(a: &[f64], b: &[f64]) -> f64 {
        Self::squared_euclidean_distance(a, b).sqrt()
    }

    /// Squared Euclidean distance using SIMD
    #[cfg(target_arch = "x86_64")]
    pub fn squared_euclidean_distance(a: &[f64], b: &[f64]) -> f64 {
        assert_eq!(a.len(), b.len(), "Arrays must have same length");
        
        if !is_x86_feature_detected!("avx2") || a.len() < 4 {
            return a.iter().zip(b.iter()).map(|(x, y)| (x - y).powi(2)).sum();
        }

        unsafe { Self::squared_distance_avx(a, b) }
    }

    #[cfg(not(target_arch = "x86_64"))]
    pub fn squared_euclidean_distance(a: &[f64], b: &[f64]) -> f64 {
        a.iter().zip(b.iter()).map(|(x, y)| (x - y).powi(2)).sum()
    }

    #[cfg(target_arch = "x86_64")]
    #[target_feature(enable = "avx2")]
    unsafe fn squared_distance_avx(a: &[f64], b: &[f64]) -> f64 {
        let chunks_a = a.chunks_exact(4);
        let chunks_b = b.chunks_exact(4);
        let remainder_a = chunks_a.remainder();
        let remainder_b = chunks_b.remainder();

        let mut sum_vec = _mm256_setzero_pd();

        for (chunk_a, chunk_b) in chunks_a.zip(chunks_b) {
            let vec_a = _mm256_loadu_pd(chunk_a.as_ptr());
            let vec_b = _mm256_loadu_pd(chunk_b.as_ptr());
            let diff = _mm256_sub_pd(vec_a, vec_b);
            let sq = _mm256_mul_pd(diff, diff);
            sum_vec = _mm256_add_pd(sum_vec, sq);
        }

        // Horizontal sum
        let low = _mm256_castpd256_pd128(sum_vec);
        let high = _mm256_extractf128_pd(sum_vec, 1);
        let sum128 = _mm_add_pd(low, high);
        let high64 = _mm_unpackhi_pd(sum128, sum128);
        let sum64 = _mm_add_sd(sum128, high64);
        
        let mut result = _mm_cvtsd_f64(sum64);

        // Add remainder
        for (&va, &vb) in remainder_a.iter().zip(remainder_b.iter()) {
            result += (va - vb).powi(2);
        }

        result
    }

    /// Variance using SIMD for the squared-difference reduction
    pub fn variance_f64(data: &[f64]) -> f64 {
        if data.len() < 2 {
            return 0.0;
        }

        let mean = Self::mean_f64(data);
        // Build a temporary diff array, then use SIMD dot product (diff . diff)
        // to compute sum of squares without scalar loop
        let diffs: Vec<f64> = data.iter().map(|&x| x - mean).collect();
        let sq_diff_sum = Self::dot_f64(&diffs, &diffs);
        sq_diff_sum / data.len() as f64
    }

    /// Standard deviation using SIMD
    pub fn std_f64(data: &[f64]) -> f64 {
        Self::variance_f64(data).sqrt()
    }

    /// Element-wise addition
    #[cfg(target_arch = "x86_64")]
    pub fn add_f64(a: &[f64], b: &[f64]) -> Vec<f64> {
        assert_eq!(a.len(), b.len(), "Arrays must have same length");
        
        if !is_x86_feature_detected!("avx2") || a.len() < 4 {
            return a.iter().zip(b.iter()).map(|(x, y)| x + y).collect();
        }

        unsafe { Self::add_f64_avx(a, b) }
    }

    #[cfg(not(target_arch = "x86_64"))]
    pub fn add_f64(a: &[f64], b: &[f64]) -> Vec<f64> {
        a.iter().zip(b.iter()).map(|(x, y)| x + y).collect()
    }

    #[cfg(target_arch = "x86_64")]
    #[target_feature(enable = "avx2")]
    unsafe fn add_f64_avx(a: &[f64], b: &[f64]) -> Vec<f64> {
        let mut result = vec![0.0; a.len()];
        
        let chunks_a = a.chunks_exact(4);
        let chunks_b = b.chunks_exact(4);
        let chunks_r = result.chunks_exact_mut(4);

        for ((chunk_a, chunk_b), chunk_r) in chunks_a.zip(chunks_b).zip(chunks_r) {
            let vec_a = _mm256_loadu_pd(chunk_a.as_ptr());
            let vec_b = _mm256_loadu_pd(chunk_b.as_ptr());
            let sum = _mm256_add_pd(vec_a, vec_b);
            _mm256_storeu_pd(chunk_r.as_mut_ptr(), sum);
        }

        // Handle remainder
        let remainder_start = (a.len() / 4) * 4;
        for i in remainder_start..a.len() {
            result[i] = a[i] + b[i];
        }

        result
    }

    /// Element-wise multiplication
    #[cfg(target_arch = "x86_64")]
    pub fn mul_f64(a: &[f64], b: &[f64]) -> Vec<f64> {
        assert_eq!(a.len(), b.len(), "Arrays must have same length");
        
        if !is_x86_feature_detected!("avx2") || a.len() < 4 {
            return a.iter().zip(b.iter()).map(|(x, y)| x * y).collect();
        }

        unsafe { Self::mul_f64_avx(a, b) }
    }

    #[cfg(not(target_arch = "x86_64"))]
    pub fn mul_f64(a: &[f64], b: &[f64]) -> Vec<f64> {
        a.iter().zip(b.iter()).map(|(x, y)| x * y).collect()
    }

    #[cfg(target_arch = "x86_64")]
    #[target_feature(enable = "avx2")]
    unsafe fn mul_f64_avx(a: &[f64], b: &[f64]) -> Vec<f64> {
        let mut result = vec![0.0; a.len()];
        
        let chunks_a = a.chunks_exact(4);
        let chunks_b = b.chunks_exact(4);
        let chunks_r = result.chunks_exact_mut(4);

        for ((chunk_a, chunk_b), chunk_r) in chunks_a.zip(chunks_b).zip(chunks_r) {
            let vec_a = _mm256_loadu_pd(chunk_a.as_ptr());
            let vec_b = _mm256_loadu_pd(chunk_b.as_ptr());
            let prod = _mm256_mul_pd(vec_a, vec_b);
            _mm256_storeu_pd(chunk_r.as_mut_ptr(), prod);
        }

        // Handle remainder
        let remainder_start = (a.len() / 4) * 4;
        for i in remainder_start..a.len() {
            result[i] = a[i] * b[i];
        }

        result
    }

    /// Scalar multiplication using SIMD
    #[cfg(target_arch = "x86_64")]
    pub fn scale_f64(data: &[f64], scalar: f64) -> Vec<f64> {
        if !is_x86_feature_detected!("avx2") || data.len() < 4 {
            return data.iter().map(|&x| x * scalar).collect();
        }
        unsafe { Self::scale_f64_avx(data, scalar) }
    }

    #[cfg(not(target_arch = "x86_64"))]
    pub fn scale_f64(data: &[f64], scalar: f64) -> Vec<f64> {
        data.iter().map(|&x| x * scalar).collect()
    }

    #[cfg(target_arch = "x86_64")]
    #[target_feature(enable = "avx2")]
    unsafe fn scale_f64_avx(data: &[f64], scalar: f64) -> Vec<f64> {
        let mut result = vec![0.0; data.len()];
        let scalar_vec = _mm256_set1_pd(scalar);

        let chunks = data.chunks_exact(4);
        let chunks_r = result.chunks_exact_mut(4);

        for (chunk, chunk_r) in chunks.zip(chunks_r) {
            let vec = _mm256_loadu_pd(chunk.as_ptr());
            let prod = _mm256_mul_pd(vec, scalar_vec);
            _mm256_storeu_pd(chunk_r.as_mut_ptr(), prod);
        }

        let remainder_start = (data.len() / 4) * 4;
        for i in remainder_start..data.len() {
            result[i] = data[i] * scalar;
        }

        result
    }

    /// Find maximum value
    pub fn max_f64(data: &[f64]) -> Option<f64> {
        data.iter().cloned().reduce(f64::max)
    }

    /// Find minimum value
    pub fn min_f64(data: &[f64]) -> Option<f64> {
        data.iter().cloned().reduce(f64::min)
    }

    /// Argmax - index of maximum value
    pub fn argmax_f64(data: &[f64]) -> Option<usize> {
        data.iter()
            .enumerate()
            .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
            .map(|(idx, _)| idx)
    }

    /// Argmin - index of minimum value
    pub fn argmin_f64(data: &[f64]) -> Option<usize> {
        data.iter()
            .enumerate()
            .min_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
            .map(|(idx, _)| idx)
    }

    /// Softmax function — single allocation, in-place normalization
    pub fn softmax_f64(data: &[f64]) -> Vec<f64> {
        let max = Self::max_f64(data).unwrap_or(0.0);
        let mut result: Vec<f64> = data.iter().map(|&x| (x - max).exp()).collect();
        let sum = Self::sum_f64(&result);
        let inv_sum = 1.0 / sum;
        for v in result.iter_mut() {
            *v *= inv_sum;
        }
        result
    }

    /// Log softmax function (more numerically stable)
    pub fn log_softmax_f64(data: &[f64]) -> Vec<f64> {
        let max = Self::max_f64(data).unwrap_or(0.0);
        let log_sum_exp: f64 = data.iter().map(|&x| (x - max).exp()).sum::<f64>().ln() + max;
        data.iter().map(|&x| x - log_sum_exp).collect()
    }
}

/// f32 versions for memory efficiency
impl SimdOps {
    /// Sum of f32 array using SIMD
    #[cfg(target_arch = "x86_64")]
    pub fn sum_f32(data: &[f32]) -> f32 {
        if !is_x86_feature_detected!("avx2") || data.len() < 8 {
            return data.iter().sum();
        }

        unsafe { Self::sum_f32_avx(data) }
    }

    #[cfg(not(target_arch = "x86_64"))]
    pub fn sum_f32(data: &[f32]) -> f32 {
        data.iter().sum()
    }

    #[cfg(target_arch = "x86_64")]
    #[target_feature(enable = "avx2")]
    unsafe fn sum_f32_avx(data: &[f32]) -> f32 {
        let chunks = data.chunks_exact(8);
        let remainder = chunks.remainder();

        let mut sum_vec = _mm256_setzero_ps();

        for chunk in chunks {
            let vec = _mm256_loadu_ps(chunk.as_ptr());
            sum_vec = _mm256_add_ps(sum_vec, vec);
        }

        // Horizontal sum
        let mut result_arr = [0.0f32; 8];
        _mm256_storeu_ps(result_arr.as_mut_ptr(), sum_vec);
        let mut result: f32 = result_arr.iter().sum();

        // Add remainder
        for &val in remainder {
            result += val;
        }

        result
    }

    /// Mean of f32 array
    pub fn mean_f32(data: &[f32]) -> f32 {
        if data.is_empty() {
            return 0.0;
        }
        Self::sum_f32(data) / data.len() as f32
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_sum() {
        let data = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0];
        let sum = SimdOps::sum_f64(&data);
        assert!((sum - 55.0).abs() < 1e-10);
    }

    #[test]
    fn test_mean() {
        let data = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let mean = SimdOps::mean_f64(&data);
        assert!((mean - 3.0).abs() < 1e-10);
    }

    #[test]
    fn test_dot() {
        let a = vec![1.0, 2.0, 3.0, 4.0];
        let b = vec![1.0, 2.0, 3.0, 4.0];
        let dot = SimdOps::dot_f64(&a, &b);
        assert!((dot - 30.0).abs() < 1e-10); // 1+4+9+16
    }

    #[test]
    fn test_euclidean_distance() {
        let a = vec![0.0, 0.0, 0.0];
        let b = vec![1.0, 0.0, 0.0];
        let dist = SimdOps::euclidean_distance(&a, &b);
        assert!((dist - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_variance() {
        let data = vec![2.0, 4.0, 4.0, 4.0, 5.0, 5.0, 7.0, 9.0];
        let var = SimdOps::variance_f64(&data);
        assert!((var - 4.0).abs() < 1e-10);
    }

    #[test]
    fn test_add() {
        let a = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let b = vec![5.0, 4.0, 3.0, 2.0, 1.0];
        let result = SimdOps::add_f64(&a, &b);
        assert!(result.iter().all(|&x| (x - 6.0).abs() < 1e-10));
    }

    #[test]
    fn test_softmax() {
        let data = vec![1.0, 2.0, 3.0];
        let softmax = SimdOps::softmax_f64(&data);
        
        // Sum should be 1
        let sum: f64 = softmax.iter().sum();
        assert!((sum - 1.0).abs() < 1e-10);
        
        // Should be monotonically increasing
        assert!(softmax[0] < softmax[1]);
        assert!(softmax[1] < softmax[2]);
    }

    #[test]
    fn test_argmax() {
        let data = vec![1.0, 5.0, 3.0, 4.0, 2.0];
        assert_eq!(SimdOps::argmax_f64(&data), Some(1));
    }

    #[test]
    fn test_argmin() {
        let data = vec![3.0, 1.0, 4.0, 1.0, 5.0];
        assert_eq!(SimdOps::argmin_f64(&data), Some(1));
    }
}
