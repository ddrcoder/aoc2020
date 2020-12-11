#[macro_use]
extern crate scan_fmt;

use std::collections::hash_map::HashMap;
use std::collections::hash_set::HashSet;
use std::fs::File;
use std::io::BufRead;
use std::io::BufReader;
use std::iter::Iterator;

fn lines(filename: &str) -> Vec<String> {
    BufReader::new(File::open(filename).ok().unwrap())
        .lines()
        .map(|line| line.ok().unwrap())
        .collect()
}

fn sum2(nums: &HashSet<i64>, target: i64) -> Option<(i64, i64)> {
    for num in nums {
        if nums.contains(&(target - num)) && num + num != target {
            return Some((*num, target - num));
        }
    }
    None
}

fn read_nums() -> HashSet<i64> {
    lines("day1.txt")
        .iter()
        .map(|str| str.parse().ok().unwrap())
        .collect()
}

fn day1(gold: bool) -> usize {
    if !gold {
        if let Some((a, b)) = sum2(&read_nums(), 2020) {
            return (a * b) as usize;
        }
    } else {
        let nums = read_nums();
        for num in &nums {
            if let Some((a, b)) = sum2(&nums, 2020 - num) {
                return (num * a * b) as usize;
            }
        }
    }
    panic!();
}

fn read_passwords() -> Vec<(usize, usize, char, String)> {
    lines("day2.txt")
        .iter()
        .map(|line| {
            scan_fmt!(line, "{}-{} {}: {}", usize, usize, char, String)
                .ok()
                .unwrap_or_else(|| panic!("bad password line: {}", line))
        })
        .collect()
}

fn day2(gold: bool) -> usize {
    let passwords = read_passwords().into_iter();
    if !gold {
        passwords
            .map(|(min, max, ch_constrained, password)| {
                (min..=max).contains(&password.chars().filter(|ch| *ch == ch_constrained).count())
            })
            .count()
    } else {
        passwords
            .filter(|(c1, c2, ch_constrained, password)| {
                let (i1, i2) = (c1 - 1, c2 - 1);
                let reps = password
                    .chars()
                    .enumerate()
                    .filter(|(i, ch)| (*i == i1 || *i == i2) && ch == ch_constrained)
                    .count();
                reps == 1
            })
            .count()
    }
}

fn day3(gold: bool) -> usize {
    let lines = lines("day3.txt");
    let terrain = lines.iter().enumerate();
    if !gold {
        terrain
            .filter(|(i, line)| {
                let w = line.len();
                let j = i * 3 % w;
                let ch = line.chars().skip(j).next().unwrap();
                ch == '#'
            })
            .count()
    } else {
        let strides = [(1, 1), (3, 1), (5, 1), (7, 1), (1, 2)];
        let mut trees = [0; 5];
        for (i, line) in terrain {
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
        trees.iter().fold(1, |acc, n| acc * n)
    }
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

fn sum_groups<State, MergeLine: Fn(Option<State>, &str) -> State, GetCount: Fn(&State) -> usize>(
    lines: Vec<String>,
    merge_line: MergeLine,
    get_count: GetCount,
) -> usize {
    let mut sum = 0;
    let mut maybe_state = None;

    for line in lines {
        if line.is_empty() {
            if let Some(state) = maybe_state {
                sum += get_count(&state);
                maybe_state = None;
            }
        } else {
            maybe_state = Some(merge_line(maybe_state, &line));
        }
    }
    if let Some(state) = maybe_state {
        sum += get_count(&state);
    }
    sum
}

fn day4(gold: bool) -> usize {
    let validate_fields = gold;
    sum_groups(
        lines("day4.txt"),
        |current, line| {
            let mut map = current.unwrap_or(HashMap::new());
            for pair in line.split(' ') {
                let (k, v) = scan_fmt!(pair, "{}:{}", String, String).ok().unwrap();
                if !validate_fields || is_valid_field(&k, &v) {
                    map.insert(k, v);
                }
            }
            map
        },
        |current| if is_valid_passport(current) { 1 } else { 0 },
    )
}

fn day5(gold: bool) -> usize {
    let lines = lines("day5.txt");
    let seats = lines.iter().map(|line| {
        line.chars()
            .map(|ch| match ch {
                'F' | 'L' => 0,
                'B' | 'R' => 1,
                _ => panic!("Bad seat char: {}", ch),
            })
            .fold(0, |v, bit| v * 2 + bit)
    });

    if !gold {
        seats.max()
    } else {
        let mut seats: Vec<usize> = seats.collect();
        seats.sort();
        let mut pairs = seats.iter().zip(seats.iter().skip(1));
        pairs.find_map(|(a, b)| if b - a > 1 { Some(a + 1) } else { None })
    }
    .unwrap()
}

fn day6(gold: bool) -> usize {
    let and = gold;
    sum_groups(
        lines("day6.txt"),
        |set, line| {
            let new: HashSet<char> = line.chars().collect();
            if let Some(old) = set {
                if and {
                    &old & &new
                } else {
                    &old | &new
                }
            } else {
                new
            }
        },
        |set| set.len(),
    )
}

fn can_reach(end: &str, next: &str, graph: &HashMap<String, HashMap<String, usize>>) -> bool {
    if next == end {
        true
    } else if let Some(nested) = graph.get(next) {
        nested.keys().any(|key| can_reach(end, key, graph))
    } else {
        false
    }
}

fn total_nested(graph: &HashMap<String, HashMap<String, usize>>, next: &str, n: usize) -> usize {
    n + if let Some(nested) = graph.get(next) {
        nested
            .iter()
            .map(|(k, v)| total_nested(graph, k, n * v))
            .sum()
    } else {
        0
    }
}

fn day7(gold: bool) -> usize {
    let lines = lines("day7.txt");
    let mut contains: HashMap<String, HashMap<String, usize>> = HashMap::new();
    for line in lines {
        // shiny gold bags contain 1 dark olive bag, 2 vibrant plum bags.
        let mut r = line.split(" bags contain ");
        let outer = r.next().unwrap();
        let inners = r.next().unwrap().strip_suffix(".").unwrap();
        for inner in inners.split(", ") {
            let inner = inner
                .strip_suffix("s")
                .unwrap_or(inner)
                .strip_suffix(" bag")
                .unwrap();
            if inner != "no other" {
                let cut = inner.find(' ').unwrap();
                let n = inner[0..cut].parse().ok().unwrap();
                let name = &inner[(cut + 1)..];
                contains
                    .entry(outer.to_string())
                    .or_insert_with(Default::default)
                    .insert(name.to_string(), n);
            }
        }
    }
    if !gold {
        contains
            .keys()
            .filter(|outer| can_reach("shiny gold", outer, &contains))
            .count()
    } else {
        total_nested(&contains, "shiny gold", 1) - 1
    }
}
#[derive(Clone, Debug)]
enum Op {
    Nop,
    Jmp,
    Acc,
}

enum Result {
    InfiniteLoop(i64),
    Terminate(i64),
}

fn run(code: &[(Op, i64)], fix_ip: Option<usize>) -> Result {
    let mut acc = 0;
    let mut ip = 0;
    let mut hits = vec![0; code.len()];
    let fix = fix_ip.unwrap_or(code.len());
    while ip < code.len() {
        let hits = &mut hits[ip];
        if *hits != 0 {
            return Result::InfiniteLoop(acc);
        }
        *hits += 1;
        let (op, arg) = code[ip].clone();
        let op = match op {
            Op::Nop if ip == fix => Op::Jmp,
            Op::Jmp if ip == fix => Op::Nop,
            _ => op,
        };
        ip = (ip as i64
            + match op {
                Op::Nop => 1,
                Op::Jmp => arg,
                Op::Acc => {
                    acc += arg;
                    1
                }
            }) as usize;
    }
    Result::Terminate(acc)
}

fn day8(gold: bool) -> usize {
    let code: Vec<(Op, i64)> = lines("day8.txt")
        .iter()
        .map(|line| {
            (
                match &line[0..3] {
                    "acc" => Op::Acc,
                    "jmp" => Op::Jmp,
                    // Took me 10 minutes to see that I initially wrote:
                    // "nop" => Op::Acc,
                    "nop" => Op::Nop,
                    _ => panic!(),
                },
                line[4..].parse().ok().unwrap(),
            )
        })
        .collect();
    if !gold {
        match run(&code[..], None) {
            Result::InfiniteLoop(acc) => {
                return acc as usize;
            }
            _ => {}
        }
    } else {
        for fix in 0..code.len() {
            match run(&code[..], Some(fix)) {
                Result::Terminate(acc) => return acc as usize,
                _ => {}
            }
        }
    }
    panic!();
}

fn day9(gold: bool) -> usize {
    let lines = lines("day9.txt");
    let input: Vec<i64> = lines.iter().map(|x| x.parse().ok().unwrap()).collect();
    if !gold {
        let mut hist: HashMap<i64, usize> = HashMap::new();
        let mut set = HashSet::new();

        let w = 25;
        for i in 0..input.len() {
            let n = input[i];
            if i >= w {
                if let Some((a, b)) = sum2(&set, n) {
                } else {
                    return n as usize;
                }

                let g = input[i - w];
                let r = hist.get_mut(&g).unwrap();
                *r -= 1;
                if *r == 0 {
                    set.remove(&g);
                }
            }
            *hist.entry(n).or_insert_with(Default::default) += 1;
            set.insert(n);
        }
        panic!();
    } else {
        let mut lo = 0;
        let mut hi = 0;
        let mut s = 0;
        let target = 36845998;

        loop {
            if s < target {
                s += input[hi];
                hi += 1;
            } else if s > target {
                s -= input[lo];
                lo += 1;
            } else {
                let r = &input[lo..hi];
                return (r.iter().min().unwrap() + r.iter().max().unwrap()) as usize;
            }
        }
    }
}

fn day10(gold: bool) -> usize {
    let lines = lines("day10.txt");
    let mut input: Vec<usize> = lines.iter().map(|x| x.parse().ok().unwrap()).collect();
    input.push(0);
    input.sort();
    input.push(input[input.len() - 1] + 3);
    let mut paths = vec![0; input.len()];
    if !gold {
        let mut c1 = 0;
        let mut c3 = 0;
        for i in 1..input.len() {
            match input[i] - input[i - 1] {
                1 => {
                    c1 += 1;
                }
                3 => {
                    c3 += 1;
                }
                _ => {
                    panic!();
                }
            };
        }
        c1 * c3
    } else {
        paths[0] = 1;
        for i in 1..input.len() {
            let here = input[i];
            for o in 1..=i {
                let j = i - o;
                let old = input[j];
                if here - old > 3 {
                    break;
                }
                paths[i] += paths[j];
            }
        }
        if cfg!(debug) {
            println!("{:?}", input);
            println!("{:?}", paths);
        }
        paths[paths.len() - 1]
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
enum Spot {
    Floor,
    Empty,
    Occupied,
}
fn day11(gold: bool) -> usize {
    let lines = lines("day11.txt");
    let mut map: Vec<Vec<_>> = lines
        .iter()
        .map(|line| {
            line.chars()
                .map(|ch| match ch {
                    '#' => Spot::Occupied,
                    '.' => Spot::Floor,
                    'L' => Spot::Empty,
                    _ => panic!(),
                })
                .collect()
        })
        .collect();
    let h = map.len();
    let w = map[0].len();
    loop {
        let last = map.clone();
        for i in 0..h {
            for j in 0..w {
                let mut occ = 0;
                if last[i][j] == Spot::Floor {
                    continue;
                }
                for dy in -1..=1 {
                    for dx in -1..=1 {
                        if dx == 0 && dy == 0 {
                            continue;
                        }
                        for s in 1..=(if gold { w + h } else { 1 } as i32) {
                            let ni = i as i32 + dy * s;
                            let nj = j as i32 + dx * s;
                            if ni < 0 || nj < 0 || ni >= h as i32 || nj >= w as i32 {
                                break;
                            }
                            match last[ni as usize][nj as usize] {
                                Spot::Occupied => {
                                    occ += 1;
                                    break;
                                }
                                Spot::Empty => {
                                    break;
                                }
                                _ => {}
                            };
                        }
                    }
                }
                let old = last[i][j].clone();
                map[i][j] = match old {
                    Spot::Empty if occ == 0 => Spot::Occupied,
                    Spot::Occupied if occ >= 5 => Spot::Empty,
                    _ => old,
                };
            }
        }
        if cfg!(debug) {
            println!(
                "{}",
                map.iter()
                    .map(|line| line
                        .iter()
                        .map(|s| match s {
                            Spot::Floor => '.',
                            Spot::Occupied => '#',
                            Spot::Empty => 'L',
                        })
                        .collect::<String>())
                    .fold("\n".to_string(), |mut str, line| {
                        str.push_str(&line);
                        str.push_str("\n");
                        str
                    })
            );
        }
        if last == map {
            return map
                .iter()
                .map(|line| line.iter().filter(|s| **s == Spot::Occupied).count())
                .sum();
        }
    }
}

fn main() {
    let solutions = [
        day1, day2, day3, day4, day5, day6, day7, day8, day9, day10, day11,
    ];
    for (i, solution) in solutions.iter().enumerate() {
        println!("{}: {}, {}", i + 1, solution(false), solution(true));
    }
}
