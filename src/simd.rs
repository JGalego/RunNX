//! SIMD-accelerated element-wise operations on f32 slices.
//!
//! Each function processes 8 floats per iteration using 256-bit SIMD
//! (AVX2 where the hardware supports it; `wide` falls back automatically
//! to two 128-bit SSE2 operations or scalar code on other targets).
//!
//! Callers are expected to obtain a contiguous `&mut [f32]` slice (e.g.
//! via `ndarray::ArrayD::as_slice_mut`) before calling these helpers.
//! Non-contiguous arrays should fall back to ndarray's `mapv_inplace`.

use wide::f32x8;

/// Apply ReLU (`max(0, x)`) to every element of `s` in-place.
pub(crate) fn relu(s: &mut [f32]) {
    let zero = f32x8::splat(0.0);
    let (main, tail) = split8(s);
    for chunk in main.chunks_exact_mut(8) {
        let v = f32x8::from(<[f32; 8]>::try_from(&*chunk).unwrap());
        chunk.copy_from_slice(&<[f32; 8]>::from(v.max(zero)));
    }
    for x in tail.iter_mut() {
        *x = x.max(0.0);
    }
}

/// Apply sigmoid (`1 / (1 + exp(-x))`) to every element of `s` in-place.
///
/// Inputs are clamped to `[-88, 88]` to keep `exp` within f32 range while
/// still rounding correctly to 0 or 1 at the extremes.
pub(crate) fn sigmoid(s: &mut [f32]) {
    let one = f32x8::splat(1.0);
    let lo = f32x8::splat(-88.0);
    let hi = f32x8::splat(88.0);
    let (main, tail) = split8(s);
    for chunk in main.chunks_exact_mut(8) {
        let v = f32x8::from(<[f32; 8]>::try_from(&*chunk).unwrap());
        let clamped = v.max(lo).min(hi);
        let result = one / (one + (-clamped).exp());
        chunk.copy_from_slice(&<[f32; 8]>::from(result));
    }
    for x in tail.iter_mut() {
        *x = 1.0 / (1.0 + (-x.clamp(-88.0, 88.0)).exp());
    }
}

/// Apply `exp()` to every element of `s` in-place using a polynomial
/// approximation (accurate to ≈1 ULP within the representable f32 range).
pub(crate) fn exp(s: &mut [f32]) {
    let (main, tail) = split8(s);
    for chunk in main.chunks_exact_mut(8) {
        let v = f32x8::from(<[f32; 8]>::try_from(&*chunk).unwrap());
        chunk.copy_from_slice(&<[f32; 8]>::from(v.exp()));
    }
    for x in tail.iter_mut() {
        *x = x.exp();
    }
}

/// Apply `sqrt()` to every element of `s` in-place.
///
/// Negative inputs produce NaN, consistent with `f32::sqrt()`.  Callers
/// that need to enforce non-negativity should validate before calling.
pub(crate) fn sqrt(s: &mut [f32]) {
    let (main, tail) = split8(s);
    for chunk in main.chunks_exact_mut(8) {
        let v = f32x8::from(<[f32; 8]>::try_from(&*chunk).unwrap());
        chunk.copy_from_slice(&<[f32; 8]>::from(v.sqrt()));
    }
    for x in tail.iter_mut() {
        *x = x.sqrt();
    }
}

/// Split `s` into the largest prefix whose length is a multiple of 8 and
/// the remaining tail.  The prefix is guaranteed to work with
/// `chunks_exact_mut(8)`; the tail is handled by scalar fallback.
#[inline(always)]
fn split8(s: &mut [f32]) -> (&mut [f32], &mut [f32]) {
    let n8 = (s.len() / 8) * 8;
    s.split_at_mut(n8)
}
