# WildChat prose-writing detection benchmark

This repository contains the dataset, frozen evaluation prompts, detector predictions, and evaluation harness for a prose-writing benchmark using Pangram.

## Interactive report

Open [`release/pangram-miss-benchmark.html`](release/pangram-miss-benchmark.html) in a browser to inspect the benchmark. The report defaults to Gemini 4 Argon High (100 prompts). The model selector is its only interactive control. It retains rendered default views, aggregate results, and the visible document tables. Raw output bodies, raw score records, alternate-view controls, and their data have been removed. Historical tool-call totals include calls to detemplatize. Google Fonts supplies Roboto Mono and requires internet access.

## Included data

- `data/wildchat/splits/` contains 327,516 prompts in equal train, validation, and test partitions of 109,172 rows each.
- `data/eval_set.json` contains all 160 distinct frozen evaluation prompts, including cleaned and original wording.
- `data/evals/*/manifest.json` preserves sample order and generation settings for 14 runs.
- `data/predictions.json` contains 609 saved final result records and their available detector scores. Some runs share prompts and initial generations. These rows are not independent samples.
- `data/benchmark_runs.json` records planned sample counts and available final results. Missing results must not be treated as successful detections or evasion.
- `data/manual_exclusions.json` identifies the two manually excluded programming requests. The frozen evaluation set retains them for traceability.
- `data/wildchat/original_edits.jsonl` restores original wording for edited prompts without downloading the upstream conversations.

## Selection and measurement

Source data is AllenAI WildChat at revision `f66566ceaaeb619dd98ffb0f3bf3ce1f86775ac4`. Preparation extracts user messages, removes English humanization modifiers with lexical rules, and retains nonempty prompts with one or two sentences. It keeps the earliest surviving prompt per conversation, removes exact duplicate text, and partitions deterministically with seed 42. The train and test labels were exchanged during development. The split manifest preserves that history. Near-duplicates remain.

Evaluation eligibility requires the English label, a clear review flag, at least one whole-word action (`write`, `draft`, `compose`, `create`), and at least one genre (`article`, `essay`, `story`, `letter`, `speech`, `blog`, `email`). It excludes whole-word matches for `code`, `python`, `javascript`, `html`, `css`, `translate`, `translation`, and `rewrite`. Candidates are ordered by a seeded SHA-256 digest. `document` is not an eligibility keyword.

Mixed and Human predictions count as false negatives for generated outputs. The default analysis scope requires output length to be at least original prompt length, measured in trimmed Unicode characters. Scope is applied after generation and scoring. Online runs retry AI predictions up to three times with the same prompt and fresh context, stopping on Mixed or Human. Manual exclusions are applied after sampling.

The published results describe this selected prose-writing subset. They are not population estimates for all WildChat tasks. Model names and settings are the recorded historical identifiers. Availability depends on the provider. Gemini runs used a separately configured CLI, so their generation environment differs from the included Claude adapter.

## Installation

Use Python 3.11 or later.

```sh
python3 -m venv .venv
. .venv/bin/activate
pip install -r requirements.txt
python -m nltk.downloader -d data/nltk_data punkt_tab
```

The prepared dataset and published predictions can be inspected without API credentials. Summarize the included scores locally.

```sh
python summarize_public.py
```

## New evaluations

The included `writing-scaffold.mjs` runs the scaffold MCP server with its bundled library. Install Node.js 22 or later and start it in a separate terminal.

```sh
npm run mcp
```

The server listens at `http://127.0.0.1:3000/mcp`. It exposes `write_with_scaffold`, `list_scaffolds`, `get_scaffold`, and `list_references`. It also supports stdio through `node writing-scaffold.mjs`. No npm dependency installation is required for the bundled server.

The public server excludes detemplatize and its provider integration. Historical benchmark runs could and did call that tool. Their published scores therefore describe the original environment, and fresh runs with this public server evaluate a different tool configuration. `data/mcp_export.json` records source and public bundle hashes. The adapter discovers the tools exposed by the server.

Set `ANTHROPIC_API_KEY` and `PANGRAM_API_KEY`, then run from the repository root. These commands make paid API requests.

```sh
python eval_pangram.py --input data/wildchat/splits/train --output runs/initial --n 60 --seed 42 --mcp-url http://127.0.0.1:3000/mcp
python eval_online.py --offline runs/initial --output runs/online --n 60 --max-retries 3 --mcp-url http://127.0.0.1:3000/mcp
```

To regenerate the full dataset from the pinned upstream source, run `prepare_wildchat.py`, `filter_short_prompts.py`, then `split_dataset.py`. The raw download requires approximately 1.5 GB. The bundled original-wording audit supports fresh samples from the included splits.

## Publication scope and licensing

The structured data export includes detector labels and metrics and omits generated output bodies. The HTML report includes rendered default views and historical tool-call totals, with raw generated outputs removed. Private rewrite implementation, full tool traces, credentials, account records, and the separate experimental rewrite directory are excluded.

WildChat attribution and its ODC-BY database license are included in `DATA_LICENSE.md`, `CITATION.bib`, and `data/wildchat/upstream/`. The upstream authors describe de-identification and content filtering in the preserved dataset card. Prompts can still contain sensitive or offensive material. Code licensing remains for the repository owner to assign before public release.
