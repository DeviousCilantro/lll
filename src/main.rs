use rug::{Float, ops::Pow};

fn lll(b: &mut Vec<Vec<Float>>, delta: Float) {
    let mut gs: Vec<Vec<Float>> = compute_gs(b);
    loop {
        let mut flag = false;
        for k in 1..b.len() {
            for j in (0..k).rev() {
                let mu_kj = mu(&b[k], &gs[j]);
                if mu_kj.clone().abs() > Float::with_val(53, 0.5) {
                    let adjustment = b[j]
                        .iter()
                        .map(|x| x.clone() * mu_kj.clone().round())
                        .collect::<Vec<Float>>();
                    b[k] = subtract(&b[k], &adjustment);
                    gs = compute_gs(b);
                    flag = true;
                }
            }
        }
        for i in 0..(b.len() - 1) {
            if norm(&gs[i + 1]).clone() / norm(&gs[i]).clone().pow(2) < delta.clone() - mu(&gs[i + 1], &gs[i]).clone().pow(2) {
                b.swap(i, i + 1);
                flag = true;
            }
        }

        if !flag { break; }
    }
}

fn norm(vector: &[Float]) -> Float {
    let squared = multiply(vector, vector);
    squared.iter()
           .fold(Float::with_val(53, 0), |acc, x| acc + x)
           .sqrt()
}

fn mu(first: &[Float], second: &[Float]) -> Float {
    let product = multiply(first, second);
    let sum_product = product.iter().fold(Float::with_val(53, 0), |acc, x| acc + x);
    let sum_second_squared = second.iter()
                                   .map(|x| x.clone().pow(2))
                                   .fold(Float::with_val(53, 0), |acc, x| acc + x);

    sum_product / sum_second_squared
}


fn subtract(first: &[Float], second: &[Float]) -> Vec<Float> {
    first.iter()
         .zip(second.iter())
         .map(|(x, y)| x - y)
         .map(|z| Float::with_val(53, z))
         .collect()
}

fn add(first: &[Float], second: &[Float]) -> Vec<Float> {
    first.iter()
         .zip(second.iter())
         .map(|(x, y)| x + y)
         .map(|z| Float::with_val(53, z))
         .collect()
}

fn multiply(first: &[Float], second: &[Float]) -> Vec<Float> {
    first.iter()
         .zip(second.iter())
         .map(|(x, y)| x * y)
         .map(|z| Float::with_val(53, z))
         .collect()
}

fn dot_product(first: &[Float], second: &[Float]) -> Float {
    first.iter()
         .zip(second)
         .map(|(x, y)| x * y)
         .fold(Float::with_val(53, 0), |acc, x| acc + x)
}

fn orthogonality_measure(basis: &[Vec<Float>]) -> Float {
    let mut sum = Float::with_val(53, 0);
    let mut count = 0;
    for i in 0..basis.len() {
        for j in i+1..basis.len() {
            sum += dot_product(&basis[i], &basis[j]).abs();
            count += 1;
        }
    }
    sum / Float::with_val(53, count)
}

fn average_vector_length(basis: &[Vec<Float>]) -> Float {
    let sum = basis.iter().map(|vector| norm(vector)).fold(Float::with_val(53, 0), |acc, x| acc + x);
    sum / Float::with_val(53, basis.len())
}

fn compute_gs(b: &[Vec<Float>]) -> Vec<Vec<Float>> {
    let mut v = vec![vec![Float::with_val(53, 0); b[0].len()]; b.len()];
    v[0] = b[0].clone();
    for i in 1..b.len() {
        let mut sum = vec![Float::with_val(53, 0); b[0].len()];
        for j in 0..i {
            let mu_b_i_v_j = b[i].iter().zip(&v[j])
                                .map(|(x, y)| x * y)
                                .fold(Float::with_val(53, 0), |acc, x| acc + x);
            let sum_update = v[j].iter()
                                .map(|x| x * &mu_b_i_v_j)
                                .map(|z| Float::with_val(53, z))
                                .collect::<Vec<Float>>();
            sum = add(&sum, &sum_update);
        }
        v[i] = subtract(&b[i], &sum);
    }
    v
}

/*fn bnp(b: &[Vec<Float>], target: &[Float], n: usize, v: &[Vec<Float>]) -> Vec<Float> {
    if n == 0 { 
        return target.iter().cloned().collect(); 
    }

    let c_i = mu(target, &v[n - 1]);
    let adjustment = b[n - 1].iter()
                             .map(|x| x * &c_i)
                             .map(|z| Float::with_val(53, z))
                             .collect::<Vec<Float>>();
    let new_target = subtract(target, &adjustment);

    if n > 1 {
        bnp(b, &new_target, n - 1, v)
    } else {
        new_target
    }
}

fn reconstruct(basis: &[Vec<Float>], coeffs: &[Float]) -> Vec<Float> {
    let mut result = vec![Float::with_val(53, 0.0); basis[0].len()];
    for (i, coeff) in coeffs.iter().enumerate() {
        for (j, val) in basis[i].iter().enumerate() {
            result[j] += val * coeff;
        }
    }
    result
}*/

fn bkz(basis: &mut Vec<Vec<Float>>, delta: Float, block_size: usize, max_iterations: usize) {
    let mut gs = compute_gs(basis);

    let mut iteration = 0;
    while iteration < max_iterations {
        let mut flag = false;

        for k in 0..=(basis.len().saturating_sub(block_size)) {
            let original_block = basis[k..k + block_size].to_vec();
            let mut block = original_block.clone();
            lll(&mut block, delta.clone());

            // Check if any changes have been made to the block
            if block != original_block {
                flag = true;
                // Update the basis vectors with the results from LLL
                for (j, vec) in block.into_iter().enumerate() {
                    basis[k + j] = vec;
                }
                gs = compute_gs(basis);
            }
        }

        for k in 2..=basis.len() {
            for j in (1..k).rev() {
                let mu_kj = mu(&basis[k - 1], &gs[j - 1]);
                if mu_kj.clone().abs() > Float::with_val(53, 0.5) {
                    let adjustment = basis[j - 1]
                        .iter()
                        .map(|x| x.clone() * mu_kj.clone().round())
                        .collect::<Vec<Float>>();
                    basis[k - 1] = subtract(&basis[k - 1], &adjustment);
                    gs = compute_gs(basis);
                    flag = true;
                }
            }
        }

        for i in 1..basis.len() {
            if norm(&gs[i]).clone() / norm(&gs[i - 1]).clone().pow(2) < delta.clone() - mu(&gs[i], &gs[i - 1]).clone().pow(2) {
                basis.swap(i - 1, i);
                flag = true;
                gs = compute_gs(basis);
            }
        }

        if !flag {
            break;
        }
        iteration += 1;
    }
}

fn main() {
    let mut basis = vec![
        vec![Float::with_val(53, 11.0), Float::with_val(53, 3.0), Float::with_val(53, 4.0), Float::with_val(53, 4.0)],
        vec![Float::with_val(53, 2.0), Float::with_val(53, 11.0), Float::with_val(53, 5.0), Float::with_val(53, 6.0)],
        vec![Float::with_val(53, 7.0), Float::with_val(53, 8.0), Float::with_val(53, 11.0), Float::with_val(53, 9.0)],
        vec![Float::with_val(53, 10.0), Float::with_val(53, 11.0), Float::with_val(53, 12.0), Float::with_val(53, 9.0)],
    ];    
    //let gs = compute_gs(&basis);
    let mut basis1 = basis.clone();
    println!("Original basis: {:?}", basis);
    println!("\nOrthogonality: {}", orthogonality_measure(&basis));
    println!("Average Vector Length: {}", average_vector_length(&basis));
    let delta = Float::with_val(53, 0.75);
    lll(&mut basis, delta.clone());
    bkz(&mut basis1, delta.clone(), 3, 1000);
    println!("\n\nReduced basis (LLL): {:?}", basis);
    println!("\nOrthogonality: {}", orthogonality_measure(&basis));
    println!("Average Vector Length: {}", average_vector_length(&basis));
    println!("\n\nReduced basis (BKZ): {:?}", basis1);
    println!("\nOrthogonality: {}", orthogonality_measure(&basis1));
    println!("Average Vector Length: {}", average_vector_length(&basis1));
    /*let target = vec![Float::with_val(53, 2.5), Float::with_val(53, 3.5), Float::with_val(53, 4.5)];
    let bnp_result = bnp(&basis, &target, basis.len(), &gs);
    let reconstructed = reconstruct(&basis, &bnp_result);
    println!("bnp: {:?}", bnp_result);
    println!("Reconstructed vector: {:?}", reconstructed);*/
}
