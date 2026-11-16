#!/usr/bin/env python3
"""Download pinned WildChat and export user prompts without humanization wording."""
import argparse
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from functools import lru_cache
import hashlib
import json
from pathlib import Path
import re
import time
import urllib.request

import pyarrow as pa
import pyarrow.parquet as pq

REPO = "allenai/WildChat"
# Explicit English lexical heuristics, not a semantic or multilingual classifier.
RULES = {
    "humanlike_style": r"\bhuman[\s-]?like\s+(?:style|writing|text|tone|voice|language|response|content)\b",
    "humanize": r"\bhumani[sz](?:e|ed|ing|ation)\b",
    "human_written": r"\b(?:as\s+(?:if\s+)?(?:it\s+(?:was|were)\s+)?|like\s+(?:it\s+(?:was|were)\s+)?|appear\s+|sound\s+|seem\s+)?(?:written|generated|created)\s+by\s+(?:a\s+)?(?:real\s+)?human\b|\bhuman[ -]written\b|\b(?:write|respond|sound|speak)\s+(?:more\s+)?like\s+(?:a\s+)?(?:real\s+)?human\b",
    "human_style": r"\b(?:in|with|using)\s+(?:a\s+)?(?:more\s+)?human\s+(?:style|tone|voice)\b",
    "not_ai_written": r"\b(?:not|n't|never|avoid|without)\b[^.!?\n]{0,80}\b(?:sound(?:ing)?|seem(?:ing)?|look(?:ing)?|appear(?:ing)?)\b[^.!?\n]{0,40}\b(?:ai|artificial intelligence|(?:a\s+)?(?:robot|machine|chatbot))\b|\b(?:not|n't)\s+(?:by\s+)?(?:an?\s+)?(?:ai|machine|robot)[ -](?:written|generated)\b",
    "detection_evasion": r"\b(?:bypass|evade|avoid|beat|fool|pass|undetectable|undetected)\b[^.!?\n]{0,100}\b(?:ai|gpt|chatgpt|machine)[\s-]*(?:text\s+|content\s+)?detect(?:or|ion|ed|able)s?\b|\b(?:ai|gpt|chatgpt|machine)[\s-]*(?:text\s+|content\s+)?detect(?:or|ion)s?\b[^.!?\n]{0,80}\b(?:bypass|evade|avoid|undetectable|undetected)\b",
    "perplexity_burstiness": r"\b(?:perplexity|burstiness)\b",
    "unnatural_sentences": r"\b(?:avoid(?:ance\s+of|ing)?|without|eliminat(?:e|ing))\s+(?:any\s+)?unnatural\s+(?:sentences|phrases|phrasing|language)\b",
    "non_repetitive_phrases": r"\bnon[ -]?repetitive\s+(?:phrases|phrasing|sentences)\b",
}
COMPILED = {name: re.compile(pattern, re.I) for name, pattern in RULES.items()}

# Match modifiers, including their introducers, without deleting the task itself.
EDIT_RULES = [
    ("machine_style_sentence", r"(?:Your|The)\s+(?:answers?|responses?|text|writing)\s+(?:should|must)\s+not\s+sound\s+like\s+(?:a\s+)?(?:machine|robot|ai)\b[^.!?\n]*[.!?]?", ""),
    ("artificial_errors", r"(?:To\s+make\s+[^.!?\n]{0,100}(?:natural|human)[^.!?\n]{0,100},\s*)?(?:include|introduce|add|use)\s+(?:(?:some|occasional|intentional|deliberate|minor)\s+)*(?:grammatical\s+errors|grammar\s+mistakes|typos)\s+(?:to\s+[^.!?\n]{0,100}(?:human|natural)[^.!?\n]*)[.!?]?", ""),
    ("humanlike_style", r"\b(?:(?:with|in|using)\s+(?:a\s+)?)?human[\s-]?like\s+(?:style|writing|text|tone|voice|language|responses?|content)\b", ""),
    ("human_style", r"\b(?:in|with|using)\s+(?:a\s+)?(?:more\s+)?human\s+(?:style|tone|voice)\b", ""),
    ("non_repetitive_phrases", r"\b(?:with\s+)?non[ -]?repetitive\s+(?:phrases|phrasing|sentences)\b", ""),
    ("unnatural_sentences", r"\b(?:and\s+)?(?:avoid(?:ance\s+of|ing)?|without|eliminat(?:e|ing))\s+(?:any\s+)?unnatural\s+(?:sentences|phrases|phrasing|language)\b", ""),
    ("human_written", r"\b(?:(?:make|ensure)\s+(?:that\s+)?(?:it|this|the\s+(?:text|content|response|article|essay))\s+)?(?:sounds?|looks?|reads?|seems?|appears?)\s+(?:more\s+)?(?:like\s+)?(?:it\s+(?:was|were|is)\s+)?(?:written\s+by\s+)?(?:a\s+)?(?:real\s+)?human\b", ""),
    ("human_written_modifier", r"\b(?:as\s+(?:if\s+)?(?:it\s+(?:was|were)\s+)?|like\s+(?:it\s+(?:was|were)\s+)?)written\s+by\s+(?:a\s+)?(?:real\s+)?human\b", ""),
    ("human_written_adjective", r"\bhuman[ -]written\b", ""),
    ("write_like_human", r"\b(write|respond|speak)\s+(?:more\s+)?like\s+(?:a\s+)?(?:real\s+)?human\b", r"\1"),
    ("humanize", r"\bhumani[sz]e\b(?=\s+(?:(?:this|the|my|your|following|provided|above)\s+)?(?:text|essay|article|content|response|paragraph|writing|draft|passage|answer)\b)", "rewrite"),
    ("humanized_modifier", r"\bhumani[sz]ed\b(?=\s+(?:text|essay|article|content|response|paragraph|writing|draft|passage|answer)\b)", ""),
    ("not_ai_written", r"\b(?:(?:please\s+)?(?:do\s+not|don't|avoid|without|never)\s+)(?:make\s+(?:it|this|the\s+(?:text|content|response))\s+)?(?:sound(?:ing)?|look(?:ing)?|seem(?:ing)?|appear(?:ing)?)\s+like\s+(?:an?\s+)?(?:ai|robot|machine|chatbot)(?:[ -](?:generated|written))?\b", ""),
    ("ai_detection_clause", r"\b(?:to\s+)?(?:bypass|evade|avoid|beat|fool|pass)\s+(?:(?:all|any|an?|the)\s+)?(?:ai|gpt|chatgpt|machine)[\s-]*(?:text\s+|content\s+)?detect(?:or|ion)s?\b", ""),
    ("complexity_instruction", r"\b(?:consider|use|include|incorporate|ensure|maintain)\s+[^.!?\n]{0,60}\bperplexity\b[^.!?\n]{0,60}\bburstiness\b[^.!?\n]*[.!?]?", ""),
    ("perplexity_burstiness", r"\b(?:(?:with|using|use|include|incorporate|maintain)\s+(?:(?:a\s+)?(?:high|low|good|varied|higher|greater|significant)\s+(?:degree\s+of\s+|level\s+of\s+)?)?|(?:a\s+)?(?:high|low|good|varied|higher|greater|significant)\s+(?:degree\s+of\s+|level\s+of\s+)?)(?:perplexity(?:\s+and\s+(?:(?:high|low)\s+)?burstiness)?|burstiness)\b", ""),
]
EDIT_COMPILED = [(name, re.compile(pattern, re.I), replacement)
                 for name, pattern, replacement in EDIT_RULES]


@lru_cache(maxsize=8192)
def clean_prompt(text):
    lower = " " + text.lower()
    if not any(term in lower for term in ("human", "unnatural", "repetitive", "perplex", "burstiness", " ai", "robot", "machine", "gpt", "chatbot", "typos", "grammar")):
        return text, [], []
    edits = []
    cleaned = text
    for name, pattern, replacement in EDIT_COMPILED:
        def replace(match):
            if name == "human_written_adjective":
                # Preserve factual descriptions such as "600 human-written samples".
                prefix = cleaned[:match.start()]
                sentence = re.split(r"[.!?\n]", prefix)[-1]
                if not re.search(r"\b(?:write|writes|rewrite|create|generate|produce|compose)\b", sentence, re.I):
                    return match.group()
            new = match.expand(replacement)
            edits.append({"rule": name, "removed": match.group(), "replacement": new})
            return new
        cleaned = pattern.sub(replace, cleaned)
    if edits:
        # Repair punctuation left by removed list items; preserve paragraph breaks.
        cleaned = re.sub(r"[ \t]+", " ", cleaned)
        cleaned = re.sub(r" +([,.;:!?])", r"\1", cleaned)
        cleaned = re.sub(r",(?:\s*,)+", ",", cleaned)
        cleaned = re.sub(r",\s*([.;!?])", r"\1", cleaned)
        cleaned = re.sub(r"\b(?:and|with|in|using)\s*([.;!?]|$)", r"\1", cleaned, flags=re.I)
        cleaned = re.sub(r"(?m)^\s*[,.;:]+\s*", "", cleaned)
        cleaned = cleaned.strip(" ,;\t\r\n")
    residual = matches([{"role": "user", "content": cleaned}])
    return cleaned, edits, residual


def fetch(url, target):
    if target.exists():
        return
    target.parent.mkdir(parents=True, exist_ok=True)
    partial = target.with_suffix(target.suffix + ".partial")
    for attempt in range(5):
        try:
            with urllib.request.urlopen(url, timeout=180) as response, partial.open("wb") as out:
                while chunk := response.read(4 * 1024 * 1024):
                    out.write(chunk)
            partial.replace(target)
            return
        except Exception:
            partial.unlink(missing_ok=True)
            if attempt == 4:
                raise
            time.sleep(2 ** attempt)


def matches(conversation):
    found = []
    for index, message in enumerate(conversation):
        if message.get("role") != "user":
            continue
        text = message.get("content") or ""
        for name, pattern in COMPILED.items():
            match = pattern.search(text)
            if match:
                found.append({"message_index": index, "rule": name,
                              "matched_text": match.group(),
                              "context": text[max(0, match.start()-100):match.end()+100]})
    return found


def digest(path):
    h = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(4 * 1024 * 1024):
            h.update(chunk)
    return h.hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("data/wildchat"))
    args = parser.parse_args()
    root = args.output
    root.mkdir(parents=True, exist_ok=True)
    metadata_path = root / "source_metadata.json"
    if not metadata_path.exists():
        fetch(f"https://huggingface.co/api/datasets/{REPO}", metadata_path)
    metadata = json.loads(metadata_path.read_text())
    revision = metadata["sha"]
    base = f"https://huggingface.co/datasets/{REPO}/resolve/{revision}"
    shards = sorted(s["rfilename"] for s in metadata["siblings"] if s["rfilename"].endswith(".parquet"))
    for name in ["README.md", "LICENSE.md"]:
        fetch(f"{base}/{name}", root / "upstream" / name)
    print(f"Downloading {len(shards)} shards at revision {revision}", flush=True)
    def download(name):
        target = root / "raw" / Path(name).name
        fetch(f"{base}/{name}", target)
        print(f"Downloaded {target.name}: {target.stat().st_size:,} bytes", flush=True)
        return target
    with ThreadPoolExecutor(max_workers=3) as pool:
        paths = list(pool.map(download, shards))
    clean_dir = root / "prompts"
    clean_dir.mkdir(exist_ok=True)
    counts = dict.fromkeys(RULES, 0)
    total = prompt_count = edited_count = review_count = 0
    manifest = []
    schema = pa.schema([("conversation_id", pa.string()), ("source_row", pa.int64()),
                        ("message_index", pa.int32()), ("model", pa.string()),
                        ("language", pa.string()), ("prompt", pa.string()),
                        ("edited", pa.bool_()), ("needs_review", pa.bool_())])
    row25_prompts = []
    with (root / "prompt_edits.jsonl").open("w") as audit:
        for path in paths:
            source = pq.ParquetFile(path)
            destination = clean_dir / path.name
            shard_total = shard_prompts = shard_edits = 0
            with pq.ParquetWriter(destination, schema, compression="zstd") as writer:
                for batch in source.iter_batches(batch_size=1024, columns=["conversation", "conversation_id", "model"]):
                    conversations = batch.column(batch.schema.get_field_index("conversation")).to_pylist()
                    ids = batch.column(batch.schema.get_field_index("conversation_id")).to_pylist()
                    models = batch.column(batch.schema.get_field_index("model")).to_pylist()
                    records = []
                    for conversation, conversation_id, model in zip(conversations, ids, models):
                        for index, message in enumerate(conversation):
                            if message.get("role") != "user":
                                continue
                            original = message.get("content") or ""
                            cleaned, edits, residual = clean_prompt(original)
                            needs_review = bool(residual) or (bool(original.strip()) and not cleaned.strip())
                            record = {"conversation_id": conversation_id, "source_row": total,
                                      "message_index": index, "model": model, "language": message.get("language"),
                                      "prompt": cleaned, "edited": bool(edits), "needs_review": needs_review}
                            records.append(record)
                            prompt_count += 1
                            shard_prompts += 1
                            review_count += needs_review
                            if total == 25:
                                row25_prompts.append(record)
                            if edits or residual:
                                audit.write(json.dumps({**record, "original_prompt": original,
                                                        "edits": edits, "residual_flags": residual}, ensure_ascii=False) + "\n")
                            if edits:
                                edited_count += 1
                                shard_edits += 1
                                for rule in {edit["rule"] for edit in edits}:
                                    counts[rule] = counts.get(rule, 0) + 1
                        total += 1
                        shard_total += 1
                    writer.write_table(pa.Table.from_pylist(records, schema=schema))
            kept = pq.ParquetFile(destination).metadata.num_rows
            if kept != shard_prompts:
                raise RuntimeError(f"Row count mismatch: {path.name}")
            manifest.append({"file": path.name, "download_bytes": path.stat().st_size,
                             "source_sha256": digest(path), "filtered_sha256": digest(destination),
                             "source_conversations": shard_total, "prompts": kept, "edited_prompts": shard_edits})
            print(f"Processed {path.name}: {kept:,} prompts, {shard_edits:,} edited", flush=True)
    expected = metadata.get("cardData", {}).get("dataset_info", {}).get("splits", [{}])[0].get("num_examples")
    if expected is not None and total != expected:
        raise RuntimeError(f"Expected {expected} rows, found {total}")
    report = {"dataset": REPO, "revision": revision, "created_at": datetime.now(timezone.utc).isoformat(),
              "total_conversations": total, "total_prompts": prompt_count, "edited_prompts": edited_count,
              "needs_review": review_count, "row_25_prompts": row25_prompts, "rule_prompt_counts": counts,
              "edit_rules": EDIT_RULES, "shards": manifest,
              "method": "Extract every user message; remove humanization wording using explicit English regexes. No assistant responses in prompt output. Raw download remains unmodified.",
              "limitations": "English lexical heuristics; may miss paraphrases and other languages or alter quoted/discussed examples. Residual matches and emptied prompts are marked needs_review. Review prompt_edits.jsonl before relying on a fully cleaned corpus."}
    (root / "prompt_report.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({key: report[key] for key in ["total_conversations", "total_prompts", "edited_prompts", "needs_review"]}), flush=True)


if __name__ == "__main__":
    main()
