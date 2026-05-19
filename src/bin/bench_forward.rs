use std::time::Instant;

fn main() {
    time("noop", 1, || ());
}

fn time<T>(name: &str, iters: u32, mut f: impl FnMut() -> T) {
    f();
    let start = Instant::now();
    for _ in 0..iters {
        std::hint::black_box(f());
    }
    let elapsed = start.elapsed();
    println!(
        "{name}: {:.3} ms/iter over {iters} iterations",
        elapsed.as_secs_f64() * 1000.0 / f64::from(iters)
    );
}
