//! simdeez distance implementations
//!
//!

#![cfg(feature = "simdeez_f")]
use simdeez::avx2::*;
use simdeez::sse2::*;
use simdeez::*;

use super::dist::M_MIN;

pub(super) unsafe fn distance_l1_f32<S: Simd>(va: &[f32], vb: &[f32]) -> f32 {
    assert_eq!(va.len(), vb.len());
    //
    let mut dist_simd = S::setzero_ps();
    //
    let nb_simd = va.len() / S::VF32_WIDTH;
    let simd_length = nb_simd * S::VF32_WIDTH;
    let mut i = 0;
    while i < simd_length {
        let a = S::loadu_ps(&va[i]);
        let b = S::loadu_ps(&vb[i]);
        let delta = S::abs_ps(a - b);
        dist_simd += delta;
        //
        i += S::VF32_WIDTH;
    }
    let mut dist: f32 = S::horizontal_add_ps(dist_simd);
    for i in simd_length..va.len() {
        //        log::debug!("distance_l1_f32, i {:?} len {:?} nb_simd {:?} VF32_WIDTH {:?}", i, va.len(), nb_simd, S::VF32_WIDTH);
        dist += (va[i] - vb[i]).abs();
    }
    assert!(dist >= 0.);
    dist
} // end of distance_l1_f32

#[target_feature(enable = "avx2")]
pub(super) unsafe fn distance_l1_f32_avx2(va: &[f32], vb: &[f32]) -> f32 {
    distance_l1_f32::<Avx2>(va, vb)
}

//========================================================================

pub(super) unsafe fn distance_l2_f32<S: Simd>(va: &[f32], vb: &[f32]) -> f32 {
    //
    assert_eq!(va.len(), vb.len());
    //
    let nb_simd = va.len() / S::VF32_WIDTH;
    let simd_length = nb_simd * S::VF32_WIDTH;
    let mut dist_simd = S::setzero_ps();
    let mut i = 0;
    while i < simd_length {
        let a = S::loadu_ps(&va[i]);
        let b = S::loadu_ps(&vb[i]);
        let mut delta = a - b;
        delta *= delta;
        dist_simd = dist_simd + delta;
        //
        i += S::VF32_WIDTH;
    }
    let mut dist = S::horizontal_add_ps(dist_simd);
    for i in simd_length..va.len() {
        dist += (va[i] - vb[i]) * (va[i] - vb[i]);
    }
    assert!(dist >= 0.);
    dist.sqrt()
} // end of distance_l2_f32

#[target_feature(enable = "avx2")]
pub(super) unsafe fn distance_l2_f32_avx2(va: &[f32], vb: &[f32]) -> f32 {
    distance_l2_f32::<Avx2>(va, vb)
}

//======================================================================

pub(super) unsafe fn distance_dot_f32<S: Simd>(va: &[f32], vb: &[f32]) -> f32 {
    //
    assert_eq!(va.len(), vb.len());
    //
    let mut i = 0;
    let mut dot_simd = S::setzero_ps();
    let nb_simd = va.len() / S::VF32_WIDTH;
    let simd_length = nb_simd * S::VF32_WIDTH;
    while i < simd_length {
        let a = S::loadu_ps(&va[i]);
        let b = S::loadu_ps(&vb[i]);
        let delta = a * b;
        dot_simd += delta;
        //
        i += S::VF32_WIDTH;
    }
    let mut dot = S::horizontal_add_ps(dot_simd);
    for i in simd_length..va.len() {
        dot += va[i] * vb[i];
    }
    assert!(dot <= 1.000002);
    (1. - dot).max(0.)
} // end of distance_dot_f32

#[target_feature(enable = "avx2")]
pub(super) unsafe fn distance_dot_f32_avx2(va: &[f32], vb: &[f32]) -> f32 {
    distance_dot_f32::<Avx2>(va, vb)
}

#[target_feature(enable = "sse2")]
pub(super) unsafe fn distance_dot_f32_sse2(va: &[f32], vb: &[f32]) -> f32 {
    distance_dot_f32::<Sse2>(va, vb)
}

//============================================================================

pub(super) unsafe fn distance_hellinger_f32<S: Simd>(va: &[f32], vb: &[f32]) -> f32 {
    assert_eq!(va.len(), vb.len());
    let mut dist_simd = S::setzero_ps();
    //
    let mut i = 0;
    let nb_simd = va.len() / S::VF32_WIDTH;
    let simd_length = nb_simd * S::VF32_WIDTH;
    while i < simd_length {
        let a = S::loadu_ps(&va[i]);
        let b = S::loadu_ps(&vb[i]);
        let prod = a * b;
        let prod_s = S::sqrt_ps(prod);
        dist_simd += prod_s;
        //
        i += S::VF32_WIDTH;
    }
    let mut dist = S::horizontal_add_ps(dist_simd);
    for i in simd_length..va.len() {
        dist += va[i].sqrt() * vb[i].sqrt();
    }
    assert!(1. - dist >= -0.000001);
    dist = (1. - dist).max(0.).sqrt();
    dist
} // end of distance_hellinger_f32

#[target_feature(enable = "avx2")]
pub(super) unsafe fn distance_hellinger_f32_avx2(va: &[f32], vb: &[f32]) -> f32 {
    distance_hellinger_f32::<Avx2>(va, vb)
}

//=============================================================================

pub(super) unsafe fn distance_jeffreys_f32<S: Simd>(va: &[f32], vb: &[f32]) -> f32 {
    let mut dist_simd = S::setzero_ps();
    //
    let mut i = 0;
    let mut logslice = Vec::<f32>::with_capacity(S::VF32_WIDTH as usize);
    let nb_simd = va.len() / S::VF32_WIDTH;
    let simd_length = nb_simd * S::VF32_WIDTH;
    while i < simd_length {
        let a = S::loadu_ps(&va[i]);
        let b = S::loadu_ps(&vb[i]);
        let delta = a - b;
        for j in 0..S::VF32_WIDTH {
            // take care of zeros!
            logslice.push((va[i + j].max(M_MIN) / vb[i + j].max(M_MIN)).ln());
        }
        let prod_s = delta * S::loadu_ps(&logslice.as_slice()[0]);
        dist_simd += prod_s;
        logslice.clear();
        //
        i += S::VF32_WIDTH;
    }
    let mut dist = S::horizontal_add_ps(dist_simd);
    for i in simd_length..va.len() {
        if vb[i] > 0. {
            dist += (va[i] - vb[i]) * (va[i].max(M_MIN) / vb[i].max(M_MIN)).ln();
        }
    }
    dist
} // end of distance_hellinger_f32

#[target_feature(enable = "avx2")]
pub(super) unsafe fn distance_jeffreys_f32_avx2(va: &[f32], vb: &[f32]) -> f32 {
    distance_jeffreys_f32::<Avx2>(va, vb)
}

//=================================================================

#[target_feature(enable = "avx2")]
pub(super) unsafe fn distance_hamming_i32_avx2(va: &[i32], vb: &[i32]) -> f32 {
    distance_hamming_i32::<Avx2>(va, vb)
}

pub(super) unsafe fn distance_hamming_i32<S: Simd>(va: &[i32], vb: &[i32]) -> f32 {
    assert_eq!(va.len(), vb.len());
    //
    let mut dist_simd = S::setzero_epi32();
    //
    let nb_simd = va.len() / S::VI32_WIDTH;
    let simd_length = nb_simd * S::VI32_WIDTH;
    let mut i = 0;
    while i < simd_length {
        let a = S::loadu_epi32(&va[i]);
        let b = S::loadu_epi32(&vb[i]);
        let delta = S::cmpneq_epi32(a, b);
        dist_simd = S::add_epi32(dist_simd, delta);
        //
        i += S::VI32_WIDTH;
    }
    // get the sum of value in dist
    let mut simd_res: Vec<i32> = (0..S::VI32_WIDTH).into_iter().map(|_| 0).collect();
    S::storeu_epi32(&mut simd_res[0], dist_simd);
    let mut dist: i32 = simd_res.into_iter().sum();
    // Beccause simd returns 0xFFFF... when neq true and 0 else
    dist = -dist;
    // add the residue
    for i in simd_length..va.len() {
        dist = dist + if va[i] != vb[i] { 1 } else { 0 };
    }
    return dist as f32 / va.len() as f32;
} // end of distance_hamming_i32

#[allow(unused)]
#[target_feature(enable = "avx2")]
pub(super) unsafe fn distance_hamming_f64_avx2(va: &[f64], vb: &[f64]) -> f32 {
    distance_hamming_f64::<Avx2>(va, vb)
}

/// special implementation for f64 exclusively in the context of SuperMinHash algorithm
#[allow(unused)]
pub(super) unsafe fn distance_hamming_f64<S: Simd>(va: &[f64], vb: &[f64]) -> f32 {
    assert_eq!(va.len(), vb.len());
    //
    let mut dist_simd = S::setzero_epi64();
    //    log::debug!("initial simd_res : {:?}", dist_simd);
    //
    let nb_simd = va.len() / S::VF64_WIDTH;
    let simd_length = nb_simd * S::VF64_WIDTH;
    let mut i = 0;
    while i < simd_length {
        let a = S::loadu_pd(&va[i]);
        let b = S::loadu_pd(&vb[i]);
        let delta = S::cmpneq_pd(a, b);
        let delta_i = S::castpd_epi64(delta);
        //        log::debug!("delta_i : , {:?}", delta_i);
        // cast to i64 to transform the 0xFFFFF.... to -1
        dist_simd = S::add_epi64(dist_simd, delta_i);
        //
        i += S::VF64_WIDTH;
    }
    // get the sum of value in dist
    let mut simd_res: Vec<i64> = (0..S::VF64_WIDTH).into_iter().map(|_| 0).collect();
    //    log::trace!("simd_res : {:?}", dist_simd);
    S::storeu_epi64(&mut simd_res[0], dist_simd);
    // cmp_neq returns 0xFFFFFFFFFF if true and 0 else, we need to transform 0xFFFFFFF... to 1
    simd_res.iter_mut().for_each(|x| *x = -*x);
    //    log::debug!("simd_res : {:?}", simd_res);
    let mut dist: i64 = simd_res.into_iter().sum();
    // Beccause simd returns 0xFFFF... when neq true and 0 else
    // add the residue
    for i in simd_length..va.len() {
        dist = dist + if va[i] != vb[i] { 1 } else { 0 };
    }
    return (dist as f64 / va.len() as f64) as f32;
} // end of distance_hamming_f64

//=======================================================================================

#[cfg(test)]

mod tests {
    use super::*;
    use crate::dist::*;

    use rand::distributions::{Distribution, Uniform};

    fn init_log() -> u64 {
        let mut builder = env_logger::Builder::from_default_env();
        let _ = builder.is_test(true).try_init();
        println!("\n ************** initializing logger *****************\n");
        return 1;
    }

    #[test]
    fn test_avx2_hamming_i32() {
        #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
        {
            init_log();
            log::info!("running test_simd_hamming_i32 for avx2");
            //
            let size_test = 500;
            let imax = 3;
            let mut rng = rand::thread_rng();
            for i in 4..size_test {
                // generer 2 va et vb s des vecteurs<i32> de taille i  avec des valeurs entre -imax et + imax et controler les resultat
                let between = Uniform::<i32>::from(-imax..imax);
                let va: Vec<i32> = (0..i)
                    .into_iter()
                    .map(|_| between.sample(&mut rng))
                    .collect();
                let vb: Vec<i32> = (0..i)
                    .into_iter()
                    .map(|_| between.sample(&mut rng))
                    .collect();
                let simd_dist = unsafe { distance_hamming_i32::<Avx2>(&va, &vb) } as f32;

                let easy_dist: u32 = va
                    .iter()
                    .zip(vb.iter())
                    .map(|(a, b)| if a != b { 1 } else { 0 })
                    .sum();
                let easy_dist = easy_dist as f32 / va.len() as f32;
                log::debug!(
                    "test size {:?} simd  exact = {:?} {:?}",
                    i,
                    simd_dist,
                    easy_dist
                );
                if (easy_dist - simd_dist).abs() > 1.0e-5 {
                    println!(" jsimd = {:?} , easy dist = {:?}", simd_dist, easy_dist);
                    println!("va = {:?}", va);
                    println!("vb = {:?}", vb);
                    std::process::exit(1);
                }
            }
        } // cfg
    } // end of test_simdeez_hamming_i32

    #[test]
    fn test_avx2_hamming_f64() {
        #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
        {
            init_log();
            log::info!("running test_simd_hamming_f64 for avx2");
            //
            let size_test = 500;
            let fmax: f64 = 3.;
            let mut rng = rand::thread_rng();
            for i in 300..size_test {
                // generer 2 va et vb s des vecteurs<i32> de taille i  avec des valeurs entre -imax et + imax et controler les resultat
                let between = Uniform::<f64>::from(-fmax..fmax);
                let va: Vec<f64> = (0..i)
                    .into_iter()
                    .map(|_| between.sample(&mut rng))
                    .collect();
                let mut vb: Vec<f64> = (0..i)
                    .into_iter()
                    .map(|_| between.sample(&mut rng))
                    .collect();
                // reset half of vb to va
                for i in 0..i / 2 {
                    vb[i] = va[i];
                }
                let simd_dist = unsafe { distance_hamming_f64::<Avx2>(&va, &vb) } as f32;

                let j_exact = ((i / 2) as f32) / (i as f32);
                let easy_dist: u32 = va
                    .iter()
                    .zip(vb.iter())
                    .map(|(a, b)| if a != b { 1 } else { 0 })
                    .sum();
                let h_dist = DistHamming.eval(&va, &vb);
                let easy_dist = easy_dist as f32 / va.len() as f32;
                log::debug!("test size {:?} simd  = {:.3e} HammingDist {:.3e} easy : {:.3e} exact : {:.3e} ", i, simd_dist, h_dist, easy_dist, 0.5);
                if (easy_dist - simd_dist).abs() > 1.0e-5 {
                    println!(" jsimd = {:?} , jexact = {:?}", simd_dist, easy_dist);
                    log::debug!("va = {:?}", va);
                    log::debug!("vb = {:?}", vb);
                    std::process::exit(1);
                }
                if (j_exact - h_dist).abs() > 1. / i as f32 + 1.0E-5 {
                    println!(
                        " jhamming = {:?} , jexact = {:?}, j_easy : {:?}",
                        h_dist, j_exact, easy_dist
                    );
                    log::debug!("va = {:?}", va);
                    log::debug!("vb = {:?}", vb);
                    std::process::exit(1);
                }
            }
        } // cfg
    } // end of test_simd_hamming_f64
} // end of mod tests
