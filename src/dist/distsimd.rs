#![cfg(feature = "stdsimd")]

//
// std simd implementations
//
use std::simd::{f32x16, u32x16, u64x8};
use std::simd::{i32x16, i64x8};

use std::simd::cmp::SimdPartialEq;
use std::simd::num::SimdFloat;

//=====================    DistL1

pub(super) fn distance_l1_f32_simd(va: &[f32], vb: &[f32]) -> f32 {
    //
    assert_eq!(va.len(), vb.len());
    //
    let nb_lanes = 16;
    let nb_simd = va.len() / nb_lanes;
    let simd_length = nb_simd * nb_lanes;
    //
    let dist_simd = va
        .chunks_exact(nb_lanes)
        .map(f32x16::from_slice)
        .zip(vb.chunks_exact(nb_lanes).map(f32x16::from_slice))
        .map(|(a, b)| (a - b).abs())
        .sum::<f32x16>();
    //
    let mut dist = dist_simd.to_array().iter().sum::<f32>();
    // residual
    for i in simd_length..va.len() {
        dist = dist + (va[i] - vb[i]).abs();
    }
    return dist as f32;
} // end of distance_l1_f32_simd

//
//==== DistL2
//

pub(super) fn distance_l2_f32_simd(va: &[f32], vb: &[f32]) -> f32 {
    //
    assert_eq!(va.len(), vb.len());
    //
    let nb_lanes = 16;
    let nb_simd = va.len() / nb_lanes;
    let simd_length = nb_simd * nb_lanes;
    //
    let dist_simd = va
        .chunks_exact(nb_lanes)
        .map(f32x16::from_slice)
        .zip(vb.chunks_exact(nb_lanes).map(f32x16::from_slice))
        .map(|(a, b)| (a - b) * (a - b))
        .sum::<f32x16>();
    //
    let mut dist = dist_simd.to_array().iter().sum::<f32>();
    // residual
    for i in simd_length..va.len() {
        dist = dist + (va[i] - vb[i]) * (va[i] - vb[i]);
    }
    let dist = dist.sqrt();
    return dist as f32;
} // end of distance_l2_f32_simd

//
//  DistDot
//

#[allow(unused)]
pub(super) fn distance_dot_f32_simd_loop(va: &[f32], vb: &[f32]) -> f32 {
    //
    assert_eq!(va.len(), vb.len());
    //
    let mut i = 0;
    let nb_lanes = 16;
    let nb_simd = va.len() / nb_lanes;
    let simd_length = nb_simd * nb_lanes;
    let mut dist_simd = f32x16::splat(0.);
    while i < simd_length {
        let a = f32x16::from_slice(&va[i..]);
        let b = f32x16::from_slice(&vb[i..]);
        let delta = a * b;
        dist_simd += delta;
        //
        i += f32x16::LEN;
    }
    // residual
    let mut dist = dist_simd.to_array().iter().sum::<f32>();
    for i in simd_length..va.len() {
        dist = dist + va[i] * vb[i];
    }
    assert!(dist <= 1.000002);
    return (1. - dist).max(0.);
}

// iter version as fast as loop version
pub(super) fn distance_dot_f32_simd_iter(va: &[f32], vb: &[f32]) -> f32 {
    //
    assert_eq!(va.len(), vb.len());
    //
    let nb_lanes = f32x16::LEN;
    let nb_simd = va.len() / nb_lanes;
    let simd_length = nb_simd * nb_lanes;
    let dist_simd = va
        .chunks_exact(nb_lanes)
        .map(f32x16::from_slice)
        .zip(vb.chunks_exact(nb_lanes).map(f32x16::from_slice))
        .map(|(a, b)| a * b)
        .sum::<f32x16>();
    //
    let mut dist = dist_simd.to_array().iter().sum::<f32>();
    // residual
    for i in simd_length..va.len() {
        dist = dist + va[i] * vb[i];
    }
    assert!(dist <= 1.000002);
    return (1. - dist).max(0.);
}

//
//====   DistHamming
//
pub(super) fn distance_jaccard_u32_16_simd(va: &[u32], vb: &[u32]) -> f32 {
    //
    assert_eq!(va.len(), vb.len());
    //
    let nb_lanes = 16;
    let nb_simd = va.len() / nb_lanes;
    let simd_length = nb_simd * nb_lanes;
    //
    let dist_simd = va
        .chunks_exact(nb_lanes)
        .map(u32x16::from_slice)
        .zip(vb.chunks_exact(nb_lanes).map(u32x16::from_slice))
        .map(|(a, b)| a.simd_ne(b).to_int())
        .sum::<i32x16>();
    // recall a test return 0 if false -1 if true!
    let mut dist = -dist_simd.to_array().iter().sum::<i32>();
    // residual
    for i in simd_length..va.len() {
        dist = dist + if va[i] != vb[i] { 1 } else { 0 };
    }
    return dist as f32 / va.len() as f32;
} // end of distance_jaccard_u32_simd

//
pub(super) fn distance_jaccard_f32_16_simd(va: &[f32], vb: &[f32]) -> f32 {
    //
    assert_eq!(va.len(), vb.len());
    //
    let nb_lanes = 16;
    let nb_simd = va.len() / nb_lanes;
    let simd_length = nb_simd * nb_lanes;
    //
    let dist_simd = va
        .chunks_exact(nb_lanes)
        .map(f32x16::from_slice)
        .zip(vb.chunks_exact(nb_lanes).map(f32x16::from_slice))
        .map(|(a, b)| a.simd_ne(b).to_int())
        .sum::<i32x16>();
    // recall a test return 0 if false -1 if true!
    let mut dist = -dist_simd.to_array().iter().sum::<i32>();
    // residual
    for i in simd_length..va.len() {
        dist = dist + if va[i] != vb[i] { 1 } else { 0 };
    }
    return dist as f32 / va.len() as f32;
} // end of distance_jaccard_u32_simd

//
pub(super) fn distance_jaccard_u64_8_simd(va: &[u64], vb: &[u64]) -> f32 {
    //
    assert_eq!(va.len(), vb.len());
    //
    let nb_lanes = 8;
    let nb_simd = va.len() / nb_lanes;
    let simd_length = nb_simd * nb_lanes;
    //
    let dist_simd = va
        .chunks_exact(nb_lanes)
        .map(u64x8::from_slice)
        .zip(vb.chunks_exact(nb_lanes).map(u64x8::from_slice))
        .map(|(a, b)| a.simd_ne(b).to_int())
        .sum::<i64x8>();
    // recall a test return 0 if false -1 if true!
    let mut dist = -dist_simd.to_array().iter().sum::<i64>();
    // residual
    for i in simd_length..va.len() {
        dist = dist + if va[i] != vb[i] { 1 } else { 0 };
    }
    return dist as f32 / va.len() as f32;
} // end of distance_jaccard_u64_8_simd

//=======================================================================================

#[cfg(test)]

mod tests {

    use super::*;
    use rand::distributions::{Distribution, Uniform};

    fn init_log() -> u64 {
        let mut builder = env_logger::Builder::from_default_env();
        let _ = builder.is_test(true).try_init();
        println!("\n ************** initializing logger *****************\n");
        return 1;
    }

    //  to run with cargo test --features stdsimd distsimd::tests::test_simd_hamming_u32 [-- --nocapture]
    #[test]
    fn test_simd_hamming_u32() {
        init_log();
        log::info!("testing test_simd_hamming_u32 with packed_simd_2");
        //
        let size_test = 500;
        let imax = 3;
        let mut rng = rand::thread_rng();
        for i in 4..size_test {
            // generer 2 va et vb s des vecteurs<i32> de taille i  avec des valeurs entre -imax et + imax et controler les resultat
            let between = Uniform::<u32>::from(0..imax);
            let va: Vec<u32> = (0..i)
                .into_iter()
                .map(|_| between.sample(&mut rng))
                .collect();
            let vb: Vec<u32> = (0..i)
                .into_iter()
                .map(|_| between.sample(&mut rng))
                .collect();
            let simd_dist = distance_jaccard_u32_16_simd(&va, &vb);

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
                println!(" jsimd = {:?} , jexact = {:?}", simd_dist, easy_dist);
                println!("va = {:?}", va);
                println!("vb = {:?}", vb);
                std::process::exit(1);
            }
        }
    } // end of test_simd_hamming_u32

    #[test]
    fn test_simd_hamming_u64() {
        init_log();
        log::info!("testing test_simd_hamming_u32 with packed_simd_2");
        //
        let size_test = 500;
        let imax = 3;
        let mut rng = rand::thread_rng();
        for i in 4..size_test {
            // generer 2 va et vb s des vecteurs<i32> de taille i  avec des valeurs entre -imax et + imax et controler les resultat
            let between = Uniform::<u64>::from(0..imax);
            let va: Vec<u64> = (0..i)
                .into_iter()
                .map(|_| between.sample(&mut rng))
                .collect();
            let vb: Vec<u64> = (0..i)
                .into_iter()
                .map(|_| between.sample(&mut rng))
                .collect();
            let simd_dist = distance_jaccard_u64_8_simd(&va, &vb);

            let easy_dist: u64 = va
                .iter()
                .zip(vb.iter())
                .map(|(a, b)| if a != b { 1 } else { 0 })
                .sum();
            let easy_dist = easy_dist as f32 / va.len() as f32;
            println!(
                "test size {:?} simd  exact = {:?} {:?}",
                i, simd_dist, easy_dist
            );
            if (easy_dist - simd_dist).abs() > 1.0e-5 {
                println!(" jsimd = {:?} , jexact = {:?}", simd_dist, easy_dist);
                println!("va = {:?}", va);
                println!("vb = {:?}", vb);
                std::process::exit(1);
            }
        }
    } // end of test_simd_hamming_u64

    //    #[cfg(feature = "stdsimd")]
    #[test]
    fn test_simd_hamming_f32() {
        init_log();
        log::info!("testing test_simd_hamming_f32 with packed_simd_2");
        //
        let size_test = 500;
        let mut rng = rand::thread_rng();
        for i in 4..size_test {
            // generer 2 va et vb s des vecteurs<i32> de taille i  avec des valeurs entre -imax et + imax et controler les resultat
            let begin = 0.;
            let end = 1.;
            let between = Uniform::<f32>::from(begin..end);
            let va: Vec<f32> = (0..i)
                .into_iter()
                .map(|_| between.sample(&mut rng))
                .collect();
            let mut vb: Vec<f32> = (0..i)
                .into_iter()
                .map(|_| between.sample(&mut rng))
                .collect();
            // reset half of vb to va
            for i in 0..i / 2 {
                vb[i] = va[i];
            }
            let simd_dist = distance_jaccard_f32_16_simd(&va, &vb);

            let easy_dist: u64 = va
                .iter()
                .zip(vb.iter())
                .map(|(a, b)| if a != b { 1 } else { 0 })
                .sum();
            let easy_dist = easy_dist as f32 / va.len() as f32;
            println!(
                "test size {:?} simd  exact = {:?} {:?}",
                i, simd_dist, easy_dist
            );
            if (easy_dist - simd_dist).abs() > 1.0e-5 {
                println!(" jsimd = {:?} , jexact = {:?}", simd_dist, easy_dist);
                println!("va = {:?}", va);
                println!("vb = {:?}", vb);
                std::process::exit(1);
            }
        }
    } // end of test_simd_hamming_f32
}
