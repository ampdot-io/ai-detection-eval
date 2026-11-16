"""Manual exclusions apply to analysis without changing frozen sample indices."""
import json
from pathlib import Path

REGISTRY=Path(__file__).resolve().parent/'data/manual_exclusions.json'


def exclusion_map():
    if not REGISTRY.exists():return {}
    return {row['conversation_id']:row['reason'] for row in json.loads(REGISTRY.read_text())['exclusions']}


def excluded_indices(manifest):
    excluded=exclusion_map()
    return {i:excluded[row['conversation_id']] for i,row in enumerate(manifest['sample']) if row.get('conversation_id') in excluded}


def analysis_n(manifest):
    return len(manifest['sample'])-len(excluded_indices(manifest))
