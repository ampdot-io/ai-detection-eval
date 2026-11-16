#!/usr/bin/env python3
"""Regex broad phase, then NLTK sentence counting for one/two-sentence prompts."""
import argparse
from collections import Counter
from functools import lru_cache
import json
from pathlib import Path
import re

import nltk
from nltk.tokenize import PunktTokenizer
import pyarrow as pa
import pyarrow.parquet as pq

NLTK_DATA = Path(__file__).resolve().parent / "data" / "nltk_data"
nltk.data.path.insert(0, str(NLTK_DATA))
LANGUAGES = {name.capitalize(): name for name in (
    "czech", "danish", "dutch", "english", "estonian", "finnish", "french",
    "german", "greek", "italian", "norwegian", "polish", "portuguese",
    "russian", "slovene", "spanish", "swedish", "turkish")}
BOUNDARY = re.compile(r"[.!?]+\s+(?=[A-Z])")
AMBIGUITY = re.compile(r"\b[A-Za-z]\.|\d\.\d|\.{2,}|https?://|www\.|[\"“”«»]|```")


@lru_cache(maxsize=24)
def tokenizer(language):
    return PunktTokenizer(language)


@lru_cache(maxsize=24)
def abbreviations(language):
    values = sorted(tokenizer(language)._params.abbrev_types, key=len, reverse=True)
    return re.compile(r"(?i)(?<!\w)(?:" + "|".join(re.escape(x) for x in values) + r")\.")


def broadphase(text, language="English"):
    """Reject obvious long English prose; defer ambiguous/non-English text."""
    if not text.strip():
        return False, 0
    if language != "English" or AMBIGUITY.search(text) or abbreviations("english").search(text):
        return True, None
    # Capital-starting sentences are a conservative lower estimate: lowercase
    # and other ambiguous boundaries are left to NLTK rather than discarded.
    estimate = len(BOUNDARY.findall(text)) + 1
    return estimate <= 2, estimate


@lru_cache(maxsize=8192)
def sentence_count(text, language="English"):
    model = LANGUAGES.get(language, "english")
    # Punkt's shipped models do not recognize unspaced CJK terminators. Normalize
    # these only for counting; exported prompt text is left byte-for-byte intact.
    normalized = re.sub(r"[。！？؟]", ". ", text)
    return len(tokenizer(model).tokenize(normalized.strip()))


def classify(text, language="English"):
    candidate, estimate = broadphase(text, language)
    if not candidate:
        return False, "empty" if estimate == 0 else "regex_rejected", None, estimate
    count = sentence_count(text, language)
    return 1 <= count <= 2, "nltk_kept" if 1 <= count <= 2 else "nltk_rejected", count, estimate


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=Path("data/wildchat/prompts"))
    parser.add_argument("--output", type=Path, default=Path("data/wildchat/short_prompts"))
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    counts = Counter()
    languages = Counter()
    samples = {}
    shards = []
    for path in sorted(args.input.glob("*.parquet")):
        source = pq.ParquetFile(path)
        schema = source.schema_arrow.append(pa.field("sentence_count", pa.int16()))
        destination = args.output / path.name
        shard_counts = Counter()
        with pq.ParquetWriter(destination, schema, compression="zstd") as writer:
            for batch in source.iter_batches(batch_size=4096):
                texts = batch.column(batch.schema.get_field_index("prompt")).to_pylist()
                langs = batch.column(batch.schema.get_field_index("language")).to_pylist()
                decisions = [classify(text, lang or "English") for text, lang in zip(texts, langs)]
                mask = [d[0] for d in decisions]
                table = pa.Table.from_batches([batch]).append_column(
                    "sentence_count", pa.array([d[2] for d in decisions], type=pa.int16()))
                writer.write_table(table.filter(pa.array(mask)))
                for text, lang, decision in zip(texts, langs, decisions):
                    keep, stage, count, estimate = decision
                    shard_counts[stage] += 1
                    if lang not in LANGUAGES:
                        languages[lang or "unknown"] += 1
                    if len(samples.setdefault(stage, [])) < 3:
                        samples[stage].append({"prompt": text[:1500], "nltk_count": count,
                                               "regex_estimate": estimate, "kept": keep})
        counts.update(shard_counts)
        shards.append({"file": path.name, "input_prompts": source.metadata.num_rows,
                       "kept_prompts": pq.ParquetFile(destination).metadata.num_rows,
                       "stages": dict(shard_counts)})
        print(f"{path.name}: {shard_counts['nltk_kept']:,} kept / {source.metadata.num_rows:,}", flush=True)
    report = {"total_input": sum(counts.values()), "kept": counts["nltk_kept"],
              "stages": dict(counts), "shards": shards, "samples": samples,
              "nltk_version": nltk.__version__, "max_sentences": 2,
              "unsupported_language_counts": dict(languages),
              "method": "Conservative regex broad phase; NLTK Punkt narrow phase; keep nonempty prompts with 1 or 2 sentences.",
              "limitations": "Sentence boundaries follow Punkt, not semantic human judgments. Unsupported languages use the English model with CJK/Arabic terminator normalization. Regex prefilter handles unambiguous English capital-start boundaries; abbreviations, quotes, URLs, initials and non-English inputs defer to NLTK."}
    args.output.parent.joinpath("sentence_filter_report.json").write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n")
    print(json.dumps({"total_input": report["total_input"], "kept": report["kept"], "stages": dict(counts)}), flush=True)


if __name__ == "__main__":
    main()
