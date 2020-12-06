#[macro_use]
extern crate scan_fmt;

use std::collections::hash_map::HashMap;
use std::collections::hash_set::HashSet;
use std::env;
use std::io;
use std::io::BufRead;
use std::iter::Iterator;

fn sum2(nums: &HashSet<i32>, target: i32) -> Option<(i32, i32)> {
    for num in nums {
        if nums.contains(&(target - num)) {
            return Some((*num, target - num));
        }
    }
    None
}

fn read_nums() -> HashSet<i32> {
    io::stdin()
        .lock()
        .lines()
        .map(|line| line.ok().unwrap().parse().unwrap())
        .collect()
}

fn day1a() {
    if let Some((a, b)) = sum2(&read_nums(), 2020) {
        println!("{}", a * b);
    }
}

fn day1b() {
    let nums = read_nums();
    for num in &nums {
        if let Some((a, b)) = sum2(&nums, 2020 - num) {
            println!("{}", num * a * b);
        }
    }
}

fn read_passwords() -> Vec<(usize, usize, char, String)> {
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
    let lines = read_passwords();
    let count = lines
        .into_iter()
        .map(|(min, max, ch_constrained, password)| {
            (min..=max).contains(&password.chars().filter(|ch| *ch == ch_constrained).count())
        })
        .count();
    println!("{}", count);
}

fn day2b() {
    let lines = read_passwords();
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

fn read_seats() -> Vec<usize> {
    io::stdin()
        .lock()
        .lines()
        .map(|line| {
            line.ok().unwrap().chars().fold(0, |v, ch| {
                v * 2
                    + match ch {
                        'F' | 'L' => 0,
                        'B' | 'R' => 1,
                        _ => panic!("Bad seat char: {}", ch),
                    }
            })
        })
        .collect()
}

fn day5a() {
    let seats = read_seats();
    let answer = *seats.iter().max().unwrap();
    println!("{}", answer);
}

fn day5b() {
    let mut seats = read_seats();
    seats.sort();
    let mut pairs = seats.iter().zip(seats.iter().skip(1));
    let answer = pairs
        .find_map(|(a, b)| if b - a > 1 { Some(a + 1) } else { None })
        .unwrap();
    println!("{}", answer);
}

fn is_valid_field(key: &str, value: &str) -> bool {
    fn good_year(value: &str, min: u16, max: u16) -> bool {
        (min..=max).contains(&value.parse().ok().unwrap())
    }
    match key {
        //byr (Birth Year) - four digits; at least 1920 and at most 2002.
        "byr" => good_year(value, 1920, 2002),
        //iyr (Issue Year) - four digits; at least 2010 and at most 2020.
        "iyr" => good_year(value, 2010, 2020),
        //eyr (Expiration Year) - four digits; at least 2020 and at most 2030.
        "eyr" => good_year(value, 2020, 2030),
        //hgt (Height) - a number followed by either cm or in:
        "hgt" => {
            //If cm, the number must be at least 150 and at most 193.
            if let Some(v) = value.strip_suffix("cm") {
                (150..=193).contains(&v.parse().ok().unwrap())
            }
            //If in, the number must be at least 59 and at most 76.
            else if let Some(v) = value.strip_suffix("in") {
                (59..=76).contains(&v.parse().ok().unwrap())
            } else {
                false
            }
        }
        //hcl (Hair Color) - a # followed by exactly six characters 0-9 or a-f.
        "hcl" => {
            if let Some(hex) = value.strip_prefix("#") {
                hex.len() == 6 && u32::from_str_radix(hex, 16).is_ok()
            } else {
                false
            }
        }
        //ecl (Eye Color) - exactly one of: amb blu brn gry grn hzl oth.
        "ecl" => match value {
            "amb" | "blu" | "brn" | "gry" | "grn" | "hzl" | "oth" => true,
            _ => false,
        },
        //pid (Passport ID) - a nine-digit number, including leading zeroes.
        "pid" => value.len() == 9 && value.parse::<u32>().is_ok(),
        //cid (Country ID) - ignored, missing or not.
        "cid" => true,
        _ => false,
    }
}

fn is_valid_passport(current: &HashMap<String, String>) -> bool {
    let count_cid = if current.contains_key("cid") { 1 } else { 0 };
    let is_valid = current.len() - count_cid == 7;
    is_valid
}

fn count_valid_groups<State: Clone, AddLine: Fn(&mut State, &str), Validate: Fn(&State) -> bool>(
    init: State,
    add_line: AddLine,
    validate: Validate,
) -> usize {
    let mut valid_count = 0;
    let mut state = init.clone();
    let mut empty = true;

    for line in io::stdin().lock().lines() {
        let line = line.ok().unwrap();
        if line.is_empty() {
            if validate(&state) {
                valid_count += 1;
            }
            state = init.clone();
            empty = true;
        } else {
            add_line(&mut state, &line);
            empty = false;
        }
    }
    if !empty && validate(&state) {
        valid_count += 1;
    }
    valid_count
}

fn day4(validate_fields: bool) {
    let answer = count_valid_groups(
        HashMap::new(),
        |current, line| {
            for pair in line.split(' ') {
                let mut entries = pair.split(':');
                let k = entries.next().unwrap();
                let v = entries.next().unwrap();
                if !validate_fields || is_valid_field(k, v) {
                    current.insert(k.to_string(), v.to_string());
                }
            }
        },
        is_valid_passport,
    );
    println!("{}", answer);
}

fn day4a() {
    day4(false);
}

fn day4b() {
    day4(true);
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
        "4a" => day4a(),
        "4b" => day4b(),
        "5a" => day5a(),
        "5b" => day5b(),
        _ => panic!("Pick a day!"),
    }
}
