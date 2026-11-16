"""Summarize published final scores without credentials or API calls."""
import argparse
from collections import defaultdict
import json
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--scope-ratio', type=float, default=1.0)
    args = parser.parse_args()
    if args.scope_ratio < 0:
        parser.error('scope ratio must be nonnegative')
    rows = json.loads((Path(__file__).parent / 'data/predictions.json').read_text())
    counts = defaultdict(lambda: {'scored': 0, 'false_negatives': 0})
    for row in rows:
        if row.get('analysis_excluded'):
            continue
        ratio = row.get('output_to_input_ratio')
        if ratio is None or ratio < args.scope_ratio:
            continue
        label = row.get('pangram', {}).get('prediction_short')
        if label not in {'AI', 'Mixed', 'Human'}:
            continue
        counts[row['run']]['scored'] += 1
        counts[row['run']]['false_negatives'] += label in {'Mixed', 'Human'}
    for values in counts.values():
        values['false_negative_rate'] = values['false_negatives'] / values['scored']
    print(json.dumps(dict(counts), indent=2))


if __name__ == '__main__':
    main()
