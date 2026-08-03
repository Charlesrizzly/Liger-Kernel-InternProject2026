#!/usr/bin/env python3
"""Parse LIGER_RMS_DEBUG logs and summarize compile/cache/fast-path behavior."""
import sys
import json
from collections import Counter, defaultdict


def parse_log(path):
    stats = {}
    stats['total_lines'] = 0
    stats['json_entries'] = 0
    stats['compiled_msgs'] = Counter()
    stats['reused_msgs'] = Counter()
    stats['fast_params_counter'] = Counter()
    stats['force_split_count'] = 0
    stats['ncols_counter'] = Counter()
    per_shape = defaultdict(int)

    with open(path, 'r', encoding='utf-8', errors='ignore') as f:
        for line in f:
            stats['total_lines'] += 1
            line = line.strip()
            if not line:
                continue
            # Attempt json payloads
            if line.startswith('{') or line.startswith('['):
                try:
                    obj = json.loads(line)
                    stats['json_entries'] += 1
                    if 'n_cols' in obj:
                        ncols = int(obj.get('n_cols'))
                        stats['ncols_counter'][ncols] += 1
                    if 'fast_params' in obj:
                        stats['fast_params_counter'][str(obj.get('fast_params'))] += 1
                    if obj.get('force_split'):
                        stats['force_split_count'] += 1
                    continue
                except Exception:
                    pass
            # non-json messages: compiled / reused / fwd_vec messages
            if 'Compiled kernel for key' in line or 'Compiled fwd_vec kernel' in line:
                stats['compiled_msgs'][line] += 1
            if 'Reusing compiled kernel for key' in line or 'Reusing fwd_vec kernel' in line:
                stats['reused_msgs'][line] += 1

    return stats


def print_summary(stats):
    print(f"Total lines: {stats['total_lines']}")
    print(f"JSON entries parsed: {stats['json_entries']}")
    print(f"Force-split count: {stats['force_split_count']}")
    print("\nTop n_cols seen:")
    for n, c in stats['ncols_counter'].most_common(10):
        print(f"  {n}: {c}")
    print("\nfast_params distribution:")
    for k, c in stats['fast_params_counter'].most_common():
        print(f"  {k}: {c}")
    print("\nCompiled messages (sample):")
    for k, c in stats['compiled_msgs'].most_common(5):
        print(f"  ({c}) {k}")
    print("\nReused messages (sample):")
    for k, c in stats['reused_msgs'].most_common(5):
        print(f"  ({c}) {k}")


if __name__ == '__main__':
    if len(sys.argv) != 2:
        print('Usage: parse_rms_debug.py <logfile>')
        sys.exit(2)
    stats = parse_log(sys.argv[1])
    print_summary(stats)
