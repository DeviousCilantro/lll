fn lll(b: &mut Vec<Vec<f32>>, delta: f32) {
    let mut gs: Vec<Vec<f32>> = compute_gs(b);
    loop {
        let mut flag = false;
        for k in 1..b.len() {
            for j in (0..k).rev() {
                let mu_kj = mu(&b[k], &gs[j]);
                if mu_kj.abs() > 0.5 {
                    let adjustment = b[j].iter().map(|&x| x * mu_kj.round()).collect::<Vec<f32>>();
                    b[k] = subtract(&b[k], &adjustment);
                    gs = compute_gs(b);
                    flag = true;
                }
            }
        }
        for i in 0..(b.len() - 1) {
            if (norm(&gs[i + 1]) / norm(&gs[i])).powf(2.0) < delta - mu(&gs[i + 1], &gs[i]).powf(2.0) {
                b.swap(i, i + 1);
                flag = true;
            }
        }

        if !flag { break; }
    }
}

fn norm(vector: &Vec<f32>) -> f32 {
    multiply(&vector, &vector)
        .iter()
        .sum::<f32>()
        .sqrt()
}

fn mu(first: &Vec<f32>, second: &Vec<f32>) -> f32 {
    multiply(&first, &second).iter().sum::<f32>() / 
        second
        .iter()
        .map(|x| x * x)
        .sum::<f32>()
}

fn subtract(first: &Vec<f32>, second: &Vec<f32>) -> Vec<f32> {
    first
        .iter()
        .zip(
            second
            .iter())
        .map(|(x, y)| x - y)
        .collect()
}

fn add(first: &Vec<f32>, second: &Vec<f32>) -> Vec<f32> {
    first
        .iter()
        .zip(
            second
            .iter())
        .map(|(x, y)| x + y)
        .collect()
}

fn multiply(first: &Vec<f32>, second: &Vec<f32>) -> Vec<f32> {
    first
        .iter()
        .zip(
            second
            .iter())
        .map(|(x, y)| x * y)
        .collect()
}

fn compute_gs(b: &Vec<Vec<f32>>) -> Vec<Vec<f32>> {
    let mut v: Vec<Vec<f32>> = vec![vec![0.0; b[0].len()]; b.len()];
    v[0] = b[0].clone();
    for i in 1..b.len() {
        let mut sum = vec![0.0; b[0].len()];
        for j in 0..i {
            sum = add(&sum, &v[j].iter().map(|&x| x * mu(&b[i], &v[j])).collect::<Vec<f32>>());
        }
        v[i] = subtract(&b[i], &sum);
    }
    v
}

fn bnp(b: &Vec<Vec<f32>>, target: &Vec<f32>, n: usize, v: &Vec<Vec<f32>>) -> Vec<f32> {
    if n == 0 { return target.to_vec(); }
    let c_i = mu(&target, &v[n - 1]);
    let new_target: Vec<f32> = subtract(&target, &b[n - 1].iter().map(|x| x * c_i).collect());
        if n > 1 {
        return bnp(b, &new_target, n - 1, v);
    } else {
        return new_target;
    }
}

fn main() {
    let mut basis: Vec<Vec<f32>> = vec![vec![1.0, 2.0, 3.0], vec![-1.0, 0.0, 1.0], vec![0.0, 1.0, 1.0]];
    let gs = compute_gs(&basis);
    lll(&mut basis, 0.75);
    println!("basis: {:?}", basis);
    let target: Vec<f32> = vec![2.5, 3.5, 4.5];
    println!("bnp: {:?}", bnp(&basis, &target, basis.len(), &gs));
}
