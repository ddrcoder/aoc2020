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

fn main() {
    let args: Vec<_> = env::args().collect();
    match args[1].parse().ok().unwrap() {
        1 => day1(),
        _ => panic!("Pick a day!"),
    }
}
