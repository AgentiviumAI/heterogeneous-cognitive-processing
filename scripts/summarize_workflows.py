"""Summarize released episode metrics without model calls or third-party packages."""
import json
import statistics
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def main():
    output = {}
    for path in sorted((ROOT / 'results').glob('*/metrics.jsonl')):
        rows = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
        for mode in sorted({row['configuration'] for row in rows}):
            group = [row for row in rows if row['configuration'] == mode]
            assert len(group) == 96
            assert len({row['question_id'] for row in group}) == 96
            output[f'{path.parent.name}/{mode}'] = {
                'questions': len(group),
                'completed': sum(row['status'] == 'success' for row in group),
                'completed_letter_match': sum(row['status'] == 'success' and row['correct'] is True for row in group),
                'raw_letter_match': sum(row['letter_matches_gold'] is True for row in group),
                'median_all_attempt_seconds': round(statistics.median(row['wall_clock_ms'] for row in group) / 1000, 1),
            }
    print(json.dumps(output, indent=2))


if __name__ == '__main__':
    main()
