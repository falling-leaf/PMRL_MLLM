#!/usr/bin/env python3
"""
Script to find the top 10 cases with the largest (pmrl_gen - base_gen) differences
"""

def parse_gen_file(file_path):
    """
    Parse a gen file and return a dictionary mapping case number to value
    """
    case_values = {}
    with open(file_path, 'r') as f:
        for line in f:
            line = line.strip()
            if line.startswith('Case'):
                parts = line.split(': ')
                case_num = int(parts[0].replace('Case ', ''))
                value = float(parts[1])
                case_values[case_num] = value
    return case_values


def find_top_differences(base_file, pmrl_file, top_n=10):
    """
    Find the top N cases with the largest (pmrl_gen - base_gen) differences
    """
    base_values = parse_gen_file(base_file)
    pmrl_values = parse_gen_file(pmrl_file)
    
    # Calculate differences
    differences = []
    common_cases = set(base_values.keys()) & set(pmrl_values.keys())
    
    for case_num in common_cases:
        diff = pmrl_values[case_num] - base_values[case_num]
        differences.append((case_num, diff))
    
    # Sort by difference in descending order
    differences.sort(key=lambda x: x[1], reverse=True)
    
    # Return top N differences
    return differences[:top_n]


def main():
    base_file = '/root/autodl-tmp/results/base_gen.txt'
    pmrl_file = '/root/autodl-tmp/results/pmrl_gen.txt'
    
    top_differences = find_top_differences(base_file, pmrl_file, 10)
    
    print("Top 10 cases with the largest (pmrl_gen - base_gen) differences:")
    print("Case Number\tDifference")
    print("-" * 30)
    for case_num, diff in top_differences:
        print(f"{case_num}\t\t{diff:.6f}")


if __name__ == "__main__":
    main()