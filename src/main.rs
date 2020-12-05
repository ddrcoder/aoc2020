#[macro_use]
extern crate scan_fmt;

use std::collections::hash_set::HashSet;
use std::env;
use std::io;
use std::io::BufRead;

fn day1() {
    let nums: HashSet<i32> = io::stdin()
        .lock()
        .lines()
        .map(|line| line.ok().unwrap().parse().unwrap())
        .collect();
    for num in &nums {
        if nums.contains(&(2020 - num)) {
            println!("{}", num * (2020 - num));
        }
    }
}

fn day2() {
    // 2-9 c: ccccccccc
    let count = io::stdin()
        .lock()
        .lines()
        .map(|line| line.ok().unwrap())
        .filter(|s| {
            let (min, max, ch_constrained, password) =
                scan_fmt!(&s, "{}-{} {}: {}", usize, usize, char, String)
                    .ok()
                    .unwrap();
            let reps = password.chars().filter(|ch| *ch == ch_constrained).count();
            min <= reps && reps <= max
        })
        .count();
    println!("{}", count);
}

fn main() {
    let args: Vec<_> = env::args().collect();
    match args[1].parse().ok().unwrap() {
        1 => day1(),
        2 => day2(),
        _ => panic!("Pick a day!"),
    }
}
