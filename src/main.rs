#[macro_use]
extern crate scan_fmt;

use std::hash::Hash;
use std::{collections::VecDeque, fmt::Debug};

use regex::Regex;
use std::collections::{hash_map::HashMap, hash_set::HashSet};
use std::fs::File;
use std::io::{BufRead, BufReader, Read};
use std::iter::{once, Iterator};

fn lines(filename: &str) -> Vec<String> {
    BufReader::new(File::open(filename).ok().unwrap())
        .lines()
        .map(|line| line.ok().unwrap())
        .collect()
}

fn content(filename: &str) -> String {
    let mut r = String::new();
    File::open(filename)
        .ok()
        .unwrap()
        .read_to_string(&mut r)
        .ok();
    r
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
                if let Some(_) = sum2(&set, n) {
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

fn day12(gold: bool) -> usize {
    fn rotate((dx, dy): (i32, i32), n: i32) -> (i32, i32) {
        match n {
            270 => (-dy, dx),
            180 => (-dx, -dy),
            90 => (dy, -dx),
            _ => panic!(),
        }
    };
    let start = ((0, 0), if gold { (10, 1) } else { (1, 0) });
    let ((x, y), _) = lines("day12.txt").iter().fold(start, |(p, d), line| {
        let (x, y) = p;
        let (dx, dy) = d;
        let n = line[1..].parse().ok().unwrap();
        match line.chars().next().unwrap() {
            'F' => ((x + n * dx, y + n * dy), d),
            'N' if gold => (p, (dx, dy + n)),
            'E' if gold => (p, (dx + n, dy)),
            'S' if gold => (p, (dx, dy - n)),
            'W' if gold => (p, (dx - n, dy)),
            'N' => ((x, y + n), d),
            'E' => ((x + n, y), d),
            'S' => ((x, y - n), d),
            'W' => ((x - n, y), d),
            'L' => (p, rotate(d, 360 - n)),
            'R' => (p, rotate(d, n)),
            _ => panic!(),
        }
    });
    (x.abs() + y.abs()) as usize
}

fn day13(gold: bool) -> usize {
    let input = lines("day13.txt");
    let start: usize = input[0].parse().ok().unwrap();
    let busses: Vec<Option<usize>> = input[1]
        .split(',')
        .map(|s| match s {
            "x" => None,
            s => s.parse().ok(),
        })
        .collect();
    if !gold {
        let bus = busses
            .into_iter()
            .filter_map(|x| x)
            .min_by_key(|b| (b - start % *b) % b)
            .unwrap();

        (bus - start % bus) * bus
    } else {
        busses
            .iter()
            .cloned()
            .enumerate()
            // Keep only the busses which are defined
            .filter_map(|(i, b)| b.map(|b| (i, b)))
            .fold((0, 1), |(time, product), (index, interval)| {
                (
                    // Find the next time which lines up by advancing int steps
                    // of the cumulative product of the prior busses' intervals,
                    // thus keeping them all on a stop.
                    (0..interval)
                        .map(|s| time + s * product)
                        .filter(|t| (t + index) % interval == 0)
                        .next()
                        .unwrap(),
                    // Every bus makes us advance in bigger steps as we
                    // accumulate a large product.
                    product * interval,
                )
            })
            .0
    }
}
fn day14(gold: bool) -> usize {
    let input = lines("day14.txt");
    let mut mem = HashMap::new();
    let (mut or, mut and, mut float) = (0u64, 0u64, 0u64);
    for line in input {
        if let Some(mask) = line.strip_prefix("mask = ") {
            let (new_or, new_and, new_float) = mask.chars().rev().enumerate().fold(
                (0, !0, 0),
                |(or, and, float), (b, ch)| match ch {
                    '1' => (or | (1 << b), and, float),
                    '0' => (or, and & !(1 << b), float),
                    'X' => (or, and, float | (1 << b)),
                    _ => (or, and, float),
                },
            );
            or = new_or;
            and = new_and;
            float = new_float;
        } else if let Some((addr, value)) = scan_fmt!(&line, "mem[{}] = {}", u64, u64).ok() {
            if gold {
                let bits = (0..64).filter(|b| 0 != ((1 << b) & float));

                for v in 0..(1 << float.count_ones()) {
                    // Generate all dense bitvectors of the length necessary.
                    let spread: u64 = bits
                        .clone()
                        .enumerate() // Effecitvely a map of src bit ->dst bit.
                        .map(|(src_bit, dst_bit)| (1 & (v >> src_bit)) << dst_bit)
                        // or or them
                        .fold(0, std::ops::BitOr::bitor);
                    let addr = (addr | or) & !float | spread;
                    let slot = mem.entry(addr).or_insert(0);
                    *slot = value;
                }
            } else {
                let slot = mem.entry(addr).or_insert(0);
                *slot = (value & and) | or;
            }
        } else {
            panic!();
        }
    }
    mem.values().sum::<u64>() as usize
}

fn day15(gold: bool) -> usize {
    let mut map = HashMap::new();
    let mut speak = 0;
    for (t, v) in content("day15.txt")
        .split(',')
        .map(|str| str.parse().ok().unwrap())
        .enumerate()
    {
        map.insert(v, t + 1);
        speak = v
    }
    for t in map.len()..(if gold { 30000000 } else { 2020 }) {
        if let Some(last_mention) = map.insert(speak, t) {
            speak = t - last_mention;
        } else {
            speak = 0;
        }
    }
    speak
}

fn day16(gold: bool) -> usize {
    fn parse_fields(txt: &str) -> Vec<(String, u16, u16, u16, u16)> {
        txt.split("\n")
            .map(|s| {
                scan_fmt!(s, "{/[^:]*/}: {}-{} or {}-{}", String, u16, u16, u16, u16)
                    .ok()
                    .unwrap_or_else(|| {
                        panic!("Bad line: {}", s);
                    })
            })
            .collect()
    }
    fn parse_ticket(txt: &str) -> Vec<u16> {
        txt.split(',').map(|s| s.parse().ok().unwrap()).collect()
    }
    fn parse_my_ticket(txt: &str) -> Vec<u16> {
        parse_ticket(txt.strip_prefix("your ticket:\n").unwrap())
    }
    fn parse_nearby_tickets(txt: &str) -> Vec<Vec<u16>> {
        txt.strip_prefix("nearby tickets:\n")
            .unwrap()
            .split('\n')
            .filter(|s| !s.is_empty())
            .map(parse_ticket)
            .collect()
    }
    fn could_be_field(v: u16, (_, lo1, hi1, lo2, hi2): &(String, u16, u16, u16, u16)) -> bool {
        (*lo1..=*hi1).contains(&v) || (*lo2..=*hi2).contains(&v)
    }

    let input = content("day16.txt");
    let mut blocks = input.split("\n\n");
    let fields = parse_fields(blocks.next().unwrap());
    let mine = parse_my_ticket(blocks.next().unwrap());
    let nearby = parse_nearby_tickets(blocks.next().unwrap());
    let is_valid_ticket = |ticket: &Vec<u16>| {
        !ticket
            .iter()
            .cloned()
            .any(|v| !fields.iter().any(|field| could_be_field(v, field)))
    };

    if !gold {
        nearby
            .iter()
            .map(|ticket| {
                let bad: usize = ticket
                    .iter()
                    .cloned()
                    .filter(|v| !fields.iter().any(|field| could_be_field(*v, field)))
                    .sum::<u16>() as usize;
                bad
            })
            .sum::<usize>()
    } else {
        let mut possible_fields: Vec<HashSet<u8>> = fields
            .iter()
            .map(|_| (0..fields.len()).map(|i| i as u8).collect())
            .collect();
        // NB: For my input "your ticket" wasn't necessary for unambiguous mapping.
        let all_tickets = once(&mine).chain(nearby.iter());
        for ticket in all_tickets.clone() {
            if !is_valid_ticket(ticket) {
                continue;
            }
            for (v, field_indices) in ticket.iter().zip(possible_fields.iter_mut()) {
                let mut to_remove = HashSet::new();
                for field_index in field_indices.iter() {
                    let field = &fields[*field_index as usize];
                    if !could_be_field(*v, field) {
                        to_remove.insert(*field_index);
                    }
                }
                if !to_remove.is_empty() {
                    *field_indices = field_indices.difference(&to_remove).cloned().collect();
                }
            }
        }

        let mut assigned = vec![None; fields.len()];
        let mut remaining = fields.len();
        while remaining > 0 {
            for (col, field_indices) in possible_fields.iter_mut().enumerate() {
                let mut to_remove = HashSet::new();
                for field_index in field_indices.iter() {
                    if assigned[*field_index as usize].is_some() {
                        to_remove.insert(*field_index);
                    }
                }
                if !to_remove.is_empty() {
                    *field_indices = field_indices.difference(&to_remove).cloned().collect();
                }
                if field_indices.len() == 1 {
                    let field_index = field_indices.iter().next().unwrap();
                    assert!(assigned[*field_index as usize].is_none());
                    assigned[*field_index as usize] = Some(col);
                    field_indices.clear();
                    remaining -= 1;
                }
            }
        }
        let assigned: Vec<_> = assigned
            .into_iter()
            .enumerate()
            .map(|(fid, src_col)| (&fields[fid], src_col.unwrap()))
            .collect();

        for (field, src_col) in assigned.iter() {
            for (tn, ticket) in all_tickets.clone().enumerate() {
                if is_valid_ticket(ticket) {
                    assert!(
                        could_be_field(ticket[*src_col], field),
                        "t#{}[{}] = {} can't be {}",
                        tn,
                        src_col,
                        ticket[*src_col],
                        field.0
                    );
                }
            }
        }
        assigned
            .iter()
            .filter(|(field, _)| field.0.starts_with("departure"))
            .map(|(_, src_col)| mine[*src_col] as usize)
            .product::<usize>() as usize
    }
}

fn day17(gold: bool) -> usize {
    let lines = lines("day17.txt");
    let mut vol = HashSet::new();
    for (y, line) in lines.iter().enumerate() {
        for (x, ch) in line.chars().enumerate() {
            if ch == '#' {
                vol.insert((x as i32, y as i32, 0 as i32, 0 as i32));
            }
        }
    }

    for _ in 0..6 {
        let mut active_neighbors = HashMap::new();
        for (x, y, z, w) in vol.iter().cloned() {
            for nw in if gold { (w - 1)..=(w + 1) } else { w..=w } {
                for nz in (z - 1)..=(z + 1) {
                    for ny in (y - 1)..=(y + 1) {
                        for nx in (x - 1)..=(x + 1) {
                            if nx == x && ny == y && nz == z && nw == w {
                                continue;
                            }
                            *active_neighbors.entry((nx, ny, nz, nw)).or_insert(0) += 1;
                        }
                    }
                }
            }
        }
        vol = active_neighbors
            .iter()
            .filter_map(|(p, n)| match n {
                3 => Some(*p),
                2 if vol.contains(&p) => Some(*p),
                _ => None,
            })
            .collect();
    }
    return vol.len();
}

#[derive(Default, Debug)]
struct ExprState {
    ops: Vec<char>,
    values: Vec<usize>,
    gold: bool,
}

impl ExprState {
    fn new(gold: bool) -> ExprState {
        ExprState {
            gold,
            ..Default::default()
        }
    }
    fn ret(&mut self) -> Option<usize> {
        while !self.ops.is_empty() {
            self.resolve_one();
        }
        self.values.pop()
    }

    fn resolve_one(&mut self) {
        assert!(self.values.len() >= 2);
        let a = self.values.pop().unwrap();
        let b = self.values.pop().unwrap();
        self.values.push(match self.ops.pop().unwrap() {
            '*' => a * b,
            '+' => a + b,
            op => panic!("{}", op),
        });
    }

    fn prec(&self, ch: char) -> u8 {
        match ch {
            '*' => 2,
            '+' => {
                if self.gold {
                    3
                } else {
                    2
                }
            }
            '(' => 0,
            _ => panic!(),
        }
    }

    fn append(&mut self, ch: char) {
        match ch {
            '0' | '1' | '2' | '3' | '4' | '5' | '6' | '7' | '8' | '9' => {
                self.values.push(ch as usize - '0' as usize);
            }
            '(' => {
                self.ops.push('(');
            }
            ')' => {
                while !self.ops.is_empty() && *self.ops.last().unwrap() != '(' {
                    if cfg!(debug) {
                        println!(")-resolve {:?}", self);
                    }
                    self.resolve_one();
                }
                assert_eq!(self.ops.pop(), Some('('));
            }
            '*' | '+' => {
                let current_prec = self.prec(ch);
                while !self.ops.is_empty() && current_prec <= self.prec(*self.ops.last().unwrap()) {
                    if cfg!(debug) {
                        println!("prec-resolve {} {:?}", ch, self);
                    }
                    self.resolve_one();
                }
                self.ops.push(ch);
            }
            _ => panic!("{}??", ch),
        }
    }
    fn eval(gold: bool, line: &str) -> usize {
        let mut state = ExprState::new(gold);
        for (i, ch) in line.chars().enumerate() {
            if ch == ' ' {
                continue;
            }
            state.append(ch);
            if cfg!(debug) {
                println!("{3:?}\n{0}\n{2:>1$}\n", line, i + 1, '^', state);
            }
        }
        state.ret().unwrap()
    }
}

fn day18(gold: bool) -> usize {
    lines("day18.txt")
        .iter()
        .map(|line| ExprState::eval(gold, line))
        .sum()
}

#[derive(Debug)]
enum Expansion {
    Char(char),
    Rules(Vec<Vec<u8>>),
}
fn day19(gold: bool) -> usize {
    let lines = content("day19.txt");
    let mut parts = lines.split("\n\n");
    let rules = parts.next().unwrap();
    let input = parts.next().unwrap();
    let rules: HashMap<u8, Expansion> = rules
        .split('\n')
        .map(|rule| {
            let mut parts = rule.split(": ");
            let name = parts.next().unwrap();
            let id: u8 = name.parse().ok().unwrap();
            let expansions = parts.next().unwrap();
            (
                id,
                if expansions.starts_with('"') {
                    Expansion::Char(expansions.chars().skip(1).next().unwrap())
                } else {
                    Expansion::Rules(
                        expansions
                            .split(" | ")
                            .map(|option| {
                                option.split(' ').map(|s| s.parse().ok().unwrap()).collect()
                            })
                            .collect(),
                    )
                },
            )
        })
        .collect();
    fn build_regex(
        mut str: String,
        rule: u8,
        gold: bool,
        rules: &HashMap<u8, Expansion>,
    ) -> String {
        match rule {
            8 if gold => {
                let mut str = build_regex(str, 42, gold, rules);
                str.push('+');
                str
            }
            11 if gold => {
                let r42 = build_regex(String::new(), 42, gold, rules);
                let r31 = build_regex(String::new(), 31, gold, rules);
                let mut nested = format!("{}{}", r42, r31);
                for _ in 0..7 {
                    nested = format!("{}({})?{}", r42, nested, r31);
                }
                str.push_str(&nested);
                str
            }
            id => match rules.get(&id).unwrap() {
                Expansion::Char(ch) => {
                    str.push(*ch);
                    str
                }
                Expansion::Rules(options) => {
                    str.push('(');
                    let mut first = true;
                    for exp in options {
                        if !first {
                            str.push('|');
                        } else {
                            first = false;
                        }
                        for id in exp {
                            str = build_regex(str, *id, gold, rules);
                        }
                    }
                    str.push(')');
                    str
                }
            },
        }
    };
    let mut r0 = build_regex("^".to_string(), 0, gold, &rules);
    r0.push('$');
    if cfg!(debug) {
        println!("{}", &r0);
    }
    let regex = Regex::new(&r0).ok().unwrap();
    input
        .split('\n')
        .filter(|line| regex.is_match(line))
        .count()
}

#[derive(Clone, Hash, PartialEq, Eq)]
struct Tile {
    chars: Vec<Vec<char>>,
}

impl Tile {
    fn new(chars: Vec<Vec<char>>) -> Tile {
        Tile { chars }
    }
    fn instance(&self, r: u8) -> Tile {
        let mut copy = self.clone();
        for y in 0..10 {
            for x in 0..10 {
                let (j, i) = match r {
                    0 => (x, y),
                    1 => (x, 9 - y),
                    2 => (9 - x, y),
                    3 => (9 - x, 9 - y),
                    4 => (y, x),
                    5 => (9 - y, x),
                    6 => (y, 9 - x),
                    7 => (9 - y, 9 - x),
                    _ => panic!(),
                };
                copy.chars[y][x] = self.chars[i][j];
            }
        }
        copy
    }
    fn left_sig(&self) -> String {
        self.chars.iter().map(|row| row.first().unwrap()).collect()
    }
    fn right_sig(&self) -> String {
        self.chars.iter().map(|row| row.last().unwrap()).collect()
    }
    fn top_sig(&self) -> String {
        self.chars.first().unwrap().iter().cloned().collect()
    }
    fn bottom_sig(&self) -> String {
        self.chars.last().unwrap().iter().cloned().collect()
    }
}

fn day20(gold: bool) -> usize {
    let input = content("day20.txt");
    let mut tiles: HashMap<_, Tile> = HashMap::new();
    for block in input.split("\n\n") {
        let mut lines = block.split("\n");
        let id = scan_fmt!(lines.next().unwrap(), "Tile {}:", usize)
            .ok()
            .unwrap();
        let chars = lines.map(|str| str.chars().collect()).take(10).collect();
        let tile = Tile::new(chars);
        for r in 0..8 {
            tiles.insert((id, r), tile.instance(r));
        }
    }
    let mut index = HashMap::new();
    for (key, tile) in tiles.iter() {
        let left = tile.left_sig();
        let top = tile.top_sig();
        for sig in [
            (None, None),
            (None, Some(top.clone())),
            (Some(left.clone()), None),
            (Some(left.clone()), Some(top.clone())),
        ]
        .iter()
        {
            let v = index.entry(sig.clone()).or_insert(vec![]);
            assert!(!v.contains(key));
            v.push(key.clone());
        }
    }
    fn render(path: &[(usize, u8)], tiles: &HashMap<(usize, u8), Tile>) -> Vec<Vec<char>> {
        let mut image = vec![];
        for gy in 0..DIM {
            if gy * DIM >= path.len() {
                break;
            }
            for y in 1..9 {
                let mut line = vec![];
                for gx in 0..DIM {
                    let index = gy * DIM + gx;
                    if index >= path.len() {
                        break;
                    }
                    let (id, r) = path[gy * DIM + gx];
                    let tile = tiles.get(&(id, r)).unwrap();
                    for x in 1..9 {
                        line.push(tile.chars[y][x]);
                    }
                }
                image.push(line);
            }
        }
        image
    }
    const DIM: usize = 12;
    //assert_eq!(tiles.len(), DIM * DIM * 8);
    fn find_tiles(
        tiles: &HashMap<(usize, u8), Tile>,
        used: &mut HashSet<usize>,
        path: &mut Vec<(usize, u8)>,
        choices: &mut Vec<Vec<(usize, u8)>>,
        index: &HashMap<(Option<String>, Option<String>), Vec<(usize, u8)>>,
    ) {
        if path.len() == DIM * DIM {
            choices.push(path.clone());
            return;
        }
        let sig_left = if path.len() % DIM != 0 {
            Some(tiles.get(&path[path.len() - 1]).unwrap().right_sig())
        } else {
            None
        };
        let sig_top = if path.len() >= DIM {
            Some(tiles.get(&path[path.len() - DIM]).unwrap().bottom_sig())
        } else {
            None
        };
        let key = (sig_left.clone(), sig_top.clone());
        if let Some(matches) = index.get(&key) {
            for choice in matches {
                if !used.insert(choice.0) {
                    continue;
                }
                path.push(choice.clone());
                find_tiles(tiles, used, path, choices, index);
                used.remove(&choice.0);
                path.pop();
            }
        }
    }
    let mut choices = vec![];
    find_tiles(
        &tiles,
        &mut HashSet::new(),
        &mut vec![],
        &mut choices,
        &index,
    );
    let monster_pattern: Vec<Vec<usize>> = [
        "                  # ",
        "#    ##    ##    ###",
        " #  #  #  #  #  #   ",
    ]
    .iter()
    .map(|line_pattern| {
        line_pattern
            .chars()
            .enumerate()
            .filter_map(|(j, ch)| if ch == '#' { Some(j) } else { None })
            .collect()
    })
    .collect();
    assert_eq!(monster_pattern[0][0], 18);
    assert_eq!(monster_pattern[1][1], 5);
    for path in choices {
        let checksum =
            path[0].0 * path[DIM - 1].0 * path[DIM * DIM - DIM].0 * path[DIM * DIM - 1].0;
        if !gold {
            return checksum;
        }
        fn monster_at(
            image: &Vec<Vec<char>>,
            i: usize,
            j: usize,
            pattern: &Vec<Vec<usize>>,
        ) -> bool {
            if i + pattern.len() > image.len() {
                return false;
            }
            for di in 0..pattern.len() {
                let pline = &pattern[di];
                for dj in pline {
                    if j + dj >= image[i + di].len() {
                        return false;
                    }
                    if image[i + di][j + dj] != '#' {
                        return false;
                    }
                }
            }
            return true;
        }
        fn mark_monster(image: &mut Vec<Vec<char>>, i: usize, j: usize, pattern: &Vec<Vec<usize>>) {
            for di in 0..pattern.len() {
                for dj in pattern[di].iter() {
                    image[i + di][j + dj] = 'O';
                }
            }
        }
        let d = 8 * DIM;
        let image = render(&path[..], &tiles);
        assert_eq!(d, image[0].len());
        let mut monster_locations = vec![];
        for i in 0..d {
            for j in 0..d {
                if monster_at(&image, i, j, &monster_pattern) {
                    monster_locations.push((i, j));
                }
            }
        }
        if monster_locations.is_empty() {
            continue;
        }
        let mut marked = image;
        for (i, j) in monster_locations.iter().cloned() {
            mark_monster(&mut marked, i, j, &monster_pattern);
        }
        let mut roughness = 0;
        for line in marked {
            for c in line {
                if c == '#' {
                    roughness += 1;
                }
                print!("{}", c);
            }
            println!();
        }
        println!("contains {} monsters.", monster_locations.len());
        return roughness;
    }
    0
}

fn day21(gold: bool) -> usize {
    if gold {
        return 0;
    }
    let lines = lines("day21.txt");
    let foods: Vec<(HashSet<String>, HashSet<String>)> = lines
        .iter()
        .map(|line| {
            let (ingredients, allergens) = scan_fmt!(
                line,
                "{/[^ ]+( [^( ]+)*/} (contains {/[^)]*/})",
                String,
                String
            )
            .ok()
            .unwrap_or_else(|| panic!("'{}'", line));
            (
                ingredients.split(' ').map(|s| s.to_string()).collect(),
                allergens.split(", ").map(|s| s.to_string()).collect(),
            )
        })
        .collect();

    let mut a2isets: HashMap<String, HashSet<String>> = HashMap::new();
    for (ingredients, allergens) in foods.iter() {
        for a in allergens {
            a2isets
                .entry(a.to_string())
                .and_modify(|iset| {
                    *iset = iset.intersection(&ingredients).cloned().collect();
                })
                .or_insert(ingredients.clone());
        }
    }
    dbg!(&a2isets);
    let mut i2a = HashMap::new();
    while !a2isets.is_empty() {
        let mut rm_a = HashSet::new();
        let mut rm_i = HashSet::new();
        for (a, iset) in a2isets.iter() {
            if iset.len() == 1 {
                let i = iset.iter().next().unwrap().clone();
                i2a.insert(i.clone(), a.clone());
                rm_i.insert(i);
                rm_a.insert(a.clone());
            }
        }
        for (_, iset) in a2isets.iter_mut() {
            for i in rm_i.iter() {
                iset.remove(i);
            }
        }
        for a in rm_a.iter() {
            a2isets.remove(a);
        }
    }
    let mut ilist: Vec<_> = i2a.iter().collect();
    ilist[..].sort_by_key(|(_, v)| *v);
    println!(
        "{}",
        ilist.iter().fold("".to_string(), |mut s, (i, _)| {
            if !s.is_empty() {
                s.push(',');
            }
            s.push_str(i);
            s
        })
    );

    foods
        .iter()
        .flat_map(|(ingredients, _)| ingredients.iter())
        .filter(|i| !i2a.contains_key(*i))
        .count()
}

fn day22(gold: bool) -> usize {
    let input = content("day22.txt");
    let decks: Vec<VecDeque<u8>> = input
        .split("\n\n")
        .map(|lines| {
            lines
                .split('\n')
                .skip(1)
                .map(|n| n.parse().ok().unwrap())
                .collect()
        })
        .collect();
    let mut decks = decks.into_iter();
    let mut d0 = decks.next().unwrap();
    let mut d1 = decks.next().unwrap();
    let win0 = if !gold {
        while !d0.is_empty() && !d1.is_empty() {
            let (c0, c1) = (d0.pop_front().unwrap(), d1.pop_front().unwrap());
            assert_ne!(c0, c1);
            if cfg!(debug) {
                println!("{} vs {} ", c0, c1);
            }
            let win0 = c0 > c1;
            if win0 {
                d0.push_back(c0);
                d0.push_back(c1);
            } else {
                d1.push_back(c1);
                d1.push_back(c0);
            }
            if cfg!(debug) {
                println!("1: {:?}\n2: {:?} ", &d0, &d1);
            }
        }
        d1.is_empty()
    } else {
        fn play(d0: &mut VecDeque<u8>, d1: &mut VecDeque<u8>, depth: usize) -> bool {
            let mut visited = HashSet::new();
            loop {
                if !visited.insert((d0.clone(), d1.clone())) {
                    if cfg!(debug) {
                        println!(
                            "{indent:>depth$}Cycle! 1 wins.",
                            indent = ' ',
                            depth = depth * 3,
                        );
                    }
                    return true;
                } else if d1.is_empty() {
                    if cfg!(debug) {
                        println!(
                            "{indent:>depth$}2 empty, 1 wins.",
                            indent = ' ',
                            depth = depth * 3,
                        );
                    }
                    return true;
                } else if d0.is_empty() {
                    if cfg!(debug) {
                        println!(
                            "{indent:>depth$}1 empty, 2 wins.",
                            indent = ' ',
                            depth = depth * 3,
                        );
                    }
                    return false;
                }
                let (c0, c1) = (d0.pop_front().unwrap(), d1.pop_front().unwrap());
                if cfg!(debug) {
                    println!(
                        "{indent:>depth$}1: {} {:?}\n{indent:>depth$}2: {} {:?}",
                        c0,
                        &d0,
                        c1,
                        &d1,
                        indent = 'B',
                        depth = depth * 3
                    );
                }
                let win0 = if d0.len() >= c0 as usize && d1.len() >= c1 as usize {
                    let mut d0 = d0.iter().take(c0 as usize).cloned().collect();
                    let mut d1 = d1.iter().take(c1 as usize).cloned().collect();
                    if cfg!(debug) {
                        println!(
                            "{indent:>depth$}Recursing...",
                            indent = ' ',
                            depth = depth * 3,
                        );
                    }
                    play(&mut d0, &mut d1, depth + 1)
                } else {
                    c0 > c1
                };
                if win0 {
                    d0.push_back(c0);
                    d0.push_back(c1);
                } else {
                    d1.push_back(c1);
                    d1.push_back(c0);
                }
                if cfg!(debug) {
                    println!(
                        "{indent:>depth$}{}1: {:?}\n{indent:>depth$}{}2: {:?}\n",
                        if win0 { '*' } else { ' ' },
                        &d0,
                        if win0 { ' ' } else { '*' },
                        &d1,
                        indent = 'A',
                        depth = depth * 3,
                    );
                }
            }
        }
        play(&mut d0, &mut d1, 0)
    };
    if cfg!(debug) {
        println!("End!\n1: {:?}\n2:{:?}", d0, d1);
    }
    let winner = if win0 { d0 } else { d1 };
    winner
        .iter()
        .rev()
        .enumerate()
        .map(|(i, v)| (i + 1) * (*v as usize))
        .sum()
}

fn day23(gold: bool) -> usize {
    let (turns, max) = if gold { (10000000, 1000000) } else { (100, 9) };
    let mut succ: Vec<u32> = vec![0; max as usize + 1];
    let s = "253149867";
    let cups = s.chars().map(|ch| ch as u32 - '0' as u32).chain(10..=max);
    let mut current = cups.clone().next().unwrap();
    for (from, to) in cups.clone().zip(cups.clone().skip(1)) {
        succ[from as usize] = to;
    }
    succ[cups.clone().rev().next().unwrap() as usize] = current;
    for turn in 1..=turns {
        let pickup1 = succ[current as usize];
        let pickup2 = succ[pickup1 as usize];
        let pickup3 = succ[pickup2 as usize];
        let next = succ[pickup3 as usize];
        let mut dest = current;
        loop {
            assert_ne!(dest, 0);
            if dest == 1 {
                dest = max;
            } else {
                dest -= 1;
            }
            if dest != pickup1 && dest != pickup2 && dest != pickup3 {
                break;
            }
        }
        succ[pickup3 as usize] = succ[dest as usize];
        succ[dest as usize] = pickup1;
        succ[current as usize] = next;
        current = next;
    }
    if gold {
        let a = succ[1] as usize;
        let b = succ[a] as usize;
        a * b
    } else {
        let (mut s, mut d) = (0, succ[1]);
        while d != 1 {
            s = s * 10 + d as usize;
            d = succ[d as usize];
        }
        s
    }
}

fn day24(gold: bool) -> usize {
    let lines = lines("day24.txt");
    let mut flipped = HashSet::new();
    for line in lines {
        let mut pos = (0, 0);
        let mut chars = line.chars();
        while let Some(ch) = chars.next() {
            match ch {
                'n' => {
                    pos = match chars.next().unwrap() {
                        'e' => (pos.0, pos.1 + 1),
                        'w' => (pos.0 - 1, pos.1 + 1),
                        _ => panic!(),
                    }
                }
                's' => {
                    pos = match chars.next().unwrap() {
                        'e' => (pos.0 + 1, pos.1 - 1),
                        'w' => (pos.0, pos.1 - 1),
                        _ => panic!(),
                    }
                }
                'e' => {
                    pos.0 += 1;
                }
                'w' => {
                    pos.0 -= 1;
                }
                _ => panic!("Unknown {}", ch),
            }
        }
        if flipped.contains(&pos) {
            flipped.remove(&pos);
        } else {
            flipped.insert(pos);
        }
    }

    let mut floor = flipped;
    if gold {
        for t in 0..100 {
            let mut neighbors: HashMap<(i32, i32), usize> = HashMap::new();
            for (x, y) in floor.iter().cloned() {
                for (dx, dy) in [(1, 0), (-1, 0), (0, 1), (-1, 1), (1, -1), (0, -1)]
                    .iter()
                    .cloned()
                {
                    let (u, v) = (x + dx, y + dy);
                    *neighbors.entry((u, v)).or_insert(0) += 1;
                }
            }
            floor = neighbors
                .iter()
                .filter_map(|(p, n)| {
                    if match n {
                        0 | 3 | 4 | 5 | 6 if floor.contains(p) => false,
                        2 if !floor.contains(p) => true,
                        _ => floor.contains(p),
                    } {
                        Some(*p)
                    } else {
                        None
                    }
                })
                .collect();
        }
    }
    floor.len()
}

fn day25(gold: bool) -> usize {
    if gold {
        return 0;
    }
    let (card_key, door_key) = scan_fmt!(&content("day25.txt"), "{}\n{}", usize, usize)
        .ok()
        .unwrap();
    //let (card_key, door_key) = (5764801, 17807724);
    let seq = |subject| {
        (1..)
            .scan(1, move |s, _| {
                *s = *s * subject % 20201227;
                Some(*s)
            })
            .enumerate()
    };
    let loop_size = |target| seq(7).find_map(|(s, v)| if v == target { Some(s) } else { None });
    let (card_loop, door_loop) = (loop_size(card_key), loop_size(door_key));
    dbg!((card_loop, door_loop));
    seq(door_key).skip(card_loop.unwrap()).next().unwrap().1
}

fn main() {
    let solutions = [
        day1, day2, day3, day4, day5, day6, day7, day8, day9, day10, day11, day12, day13, day14,
        day15, day16, day17, day18, day19, day20, day21, day22, day23, day24, day25,
    ];
    for (i, solution) in solutions.iter().enumerate().skip(19) {
        println!("{}: {}, {}", i + 1, solution(false), solution(true));
    }
}
