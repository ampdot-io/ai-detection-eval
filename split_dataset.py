"""Keep each conversation's first surviving prompt, deduplicate, split evenly."""
import argparse
from collections import Counter
from contextlib import ExitStack
import hashlib
import json
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq

SPLITS = ("train", "validation", "test")
# Preserve the partition identities after the user-requested train/test swap.
ASSIGNMENT_ORDER = ("test", "validation", "train")


def assign_groups(sizes, seed=42):
    loads = dict.fromkeys(SPLITS, 0)
    assignments = {}
    order = sorted(sizes, key=lambda key: (-sizes[key], hashlib.sha256(f"{seed}:{key}".encode()).digest()))
    for key in order:
        split = min(ASSIGNMENT_ORDER, key=lambda name: loads[name])
        assignments[key] = split
        loads[split] += sizes[key]
    return assignments, loads


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=Path("data/wildchat/short_prompts"))
    parser.add_argument("--output", type=Path, default=Path("data/wildchat/splits"))
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    paths = sorted(args.input.glob("*.parquet"))
    if not paths:
        parser.error(f"No Parquet files in {args.input}")
    # One canonical occurrence per exact UTF-8 prompt, chosen in source order.
    seen = set()
    seen_conversations = set()
    sizes = Counter()
    for path in paths:
        for batch in pq.ParquetFile(path).iter_batches(columns=["prompt", "conversation_id"]):
            for prompt, key in zip(batch.column(0).to_pylist(), batch.column(1).to_pylist()):
                if key in seen_conversations:
                    continue
                seen_conversations.add(key)
                fingerprint = hashlib.sha256(prompt.encode("utf-8")).digest()
                if fingerprint not in seen:
                    seen.add(fingerprint)
                    sizes[key] += 1
    assignments, planned_counts = assign_groups(sizes, args.seed)
    if max(planned_counts.values()) - min(planned_counts.values()) > 1:
        raise RuntimeError("Conversation groups cannot be balanced evenly by this allocation")
    seen.clear()
    seen_conversations.clear()
    groups = {split: set() for split in SPLITS}
    counts, source_counts, duplicate_counts = Counter(), Counter(), Counter()
    for split in SPLITS:
        (args.output / split).mkdir(parents=True, exist_ok=True)
    for path in paths:
        source = pq.ParquetFile(path)
        with ExitStack() as stack:
            writers = {split: stack.enter_context(pq.ParquetWriter(
                args.output / split / path.name, source.schema_arrow, compression="zstd"))
                for split in SPLITS}
            for batch in source.iter_batches(batch_size=4096):
                table = pa.Table.from_batches([batch])
                prompts = table["prompt"].to_pylist()
                ids = table["conversation_id"].to_pylist()
                indices = {split: [] for split in SPLITS}
                for index, (prompt, conversation_id) in enumerate(zip(prompts, ids)):
                    source_counts["rows"] += 1
                    if conversation_id in seen_conversations:
                        source_counts["later_surviving_prompts"] += 1
                        continue
                    seen_conversations.add(conversation_id)
                    fingerprint = hashlib.sha256(prompt.encode("utf-8")).digest()
                    if fingerprint in seen:
                        duplicate_counts["rows"] += 1
                        continue
                    seen.add(fingerprint)
                    split = assignments[conversation_id]
                    indices[split].append(index)
                    groups[split].add(conversation_id)
                    counts[split] += 1
                for split in SPLITS:
                    if indices[split]:
                        writers[split].write_table(table.take(pa.array(indices[split], type=pa.int64())))
        print(f"Processed {path.name}: {sum(counts.values()):,} unique prompts so far", flush=True)
    assert sum(counts.values()) + duplicate_counts["rows"] + source_counts["later_surviving_prompts"] == source_counts["rows"]
    assert not (groups["train"] & groups["validation"] or groups["train"] & groups["test"]
                or groups["validation"] & groups["test"])
    report = {
        "input": str(args.input), "seed": args.seed,
        "source_rows": source_counts["rows"], "unique_prompts": len(seen),
        "first_surviving_prompts": len(seen_conversations),
        "later_surviving_prompts_removed": source_counts["later_surviving_prompts"],
        "duplicates_removed": duplicate_counts["rows"],
        "target_fractions": dict.fromkeys(SPLITS, 1 / 3),
        "assignment_tie_order": list(ASSIGNMENT_ORDER),
        "splits": {split: {"rows": counts[split], "conversations": len(groups[split]),
                           "fraction": counts[split] / len(seen) if seen else 0}
                   for split in SPLITS},
        "method": "Keep only the first surviving prompt per conversation in filtered source order; remove duplicate UTF-8 prompts keeping their first occurrence. Assign records in SHA-256(seed:conversation_id) order to the smallest split.",
        "notes": "One prompt per conversation. A duplicate first surviving prompt removes that conversation; no later prompt is substituted. Source remains unchanged. Near-duplicates are not removed.",
    }
    (args.output / "manifest.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report["splits"], indent=2), flush=True)


if __name__ == "__main__":
    main()
