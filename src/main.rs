#[macro_use]
extern crate scan_fmt;

use std::collections::hash_set::HashSet;
use std::env;
use std::io;
use std::io::BufRead;

fn sum2(nums: &HashSet<i32>, target: i32) -> Option<(i32, i32)> {
    for num in nums {
        if nums.contains(&(target - num)) {
            return Some((*num, target - num));
        }
    }
    None
}

fn read_1() -> HashSet<i32> {
    io::stdin()
        .lock()
        .lines()
        .map(|line| line.ok().unwrap().parse().unwrap())
        .collect()
}

fn day1a() {
    if let Some((a, b)) = sum2(&read_1(), 2020) {
        println!("{}", a * b);
    }
}

fn day1b() {
    let nums = read_1();
    for num in &nums {
        if let Some((a, b)) = sum2(&nums, 2020 - num) {
            println!("{}", num * a * b);
        }
    }
}

fn read_2() -> Vec<(usize, usize, char, String)> {
    io::stdin()
        .lock()
        .lines()
        .map(|line| {
            scan_fmt!(
                &line.ok().unwrap(),
                "{}-{} {}: {}",
                usize,
                usize,
                char,
                String
            )
            .ok()
            .unwrap()
        })
        .collect()
}

fn day2a() {
    let lines = read_2();
    let count = lines
        .into_iter()
        .map(|(min, max, ch_constrained, password)| {
            let reps = password.chars().filter(|ch| *ch == ch_constrained).count();
            min <= reps && reps <= max
        })
        .count();
    println!("{}", count);
}

fn day2b() {
    let lines = read_2();
    let count = lines
        .into_iter()
        .filter(|(c1, c2, ch_constrained, password)| {
            let (i1, i2) = (c1 - 1, c2 - 1);
            let reps = password
                .chars()
                .enumerate()
                .filter(|(i, ch)| (*i == i1 || *i == i2) && ch == ch_constrained)
                .count();
            reps == 1
        })
        .count();
    println!("{}", count);
}

fn day3a() {
    let count = io::stdin()
        .lock()
        .lines()
        .map(|line| line.ok().unwrap())
        .enumerate()
        .filter(|(i, line)| {
            let w = line.len();
            let j = i * 3 % w;
            let ch = line.chars().skip(j).next().unwrap();
            ch == '#'
        })
        .count();
    println!("{}", count);
}

fn day3b() {
    let strides = [(1, 1), (3, 1), (5, 1), (7, 1), (1, 2)];
    let mut trees = [0; 5];
    for (i, line) in io::stdin().lock().lines().enumerate() {
        let line = line.ok().unwrap();
        let w = line.len();
        for (route, (over, down)) in strides.iter().cloned().enumerate() {
            if i % down == 0 {
                let j = i * over / down % w;
                let ch = line.chars().skip(j).next().unwrap();
                if ch == '#' {
                    trees[route] += 1;
                }
            }
        }
    }
    println!("{:?}", trees);
    let answer = trees.iter().fold(1u64, |acc, n| acc * n);
    println!("{}", answer);
}

fn main() {
    let args: Vec<_> = env::args().collect();
    match &args[1][..] {
        "1a" => day1a(),
        "1b" => day1b(),
        "2a" => day2a(),
        "2b" => day2b(),
        "3a" => day3a(),
        "3b" => day3b(),
        _ => panic!("Pick a day!"),
    }
}
