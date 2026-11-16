"""Reproducible held-out evaluation of scaffold-assisted Claude writing on Pangram."""
import argparse
import asyncio
from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import re
import time
import uuid

from dotenv import load_dotenv
import httpx2 as httpx
import pyarrow.parquet as pq

from claude_mcp_local import run
from evaluation_scope import excluded_indices,analysis_n

SYSTEM = ("Complete the user's writing request using the writing-scaffold MCP. "
          "Start with write_with_scaffold, then retrieve and use a suitable scaffold. "
          "Return only the finished writing, without process notes or discussion of the scaffold.")
ACTION = re.compile(r"\b(write|draft|compose|create)\b", re.I)
GENRE = re.compile(r"\b(article|essay|story|letter|speech|blog|email)\b", re.I)
EXCLUDE = re.compile(r"\b(code|python|javascript|html|css|translate|translation|rewrite)\b", re.I)
PANGRAM = "https://text.external-api.pangram.com"


def save(path, value):
    temporary = path.with_name(path.name + '.' + uuid.uuid4().hex + '.tmp')
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n")
    temporary.replace(path)


def select_sample(root, n, seed, exclude_conversations=None, exclude_prompt_hashes=None):
    exclude_conversations=exclude_conversations or set()
    exclude_prompt_hashes=exclude_prompt_hashes or set()
    eligible = []
    for path in sorted(root.glob("*.parquet")):
        for batch in pq.ParquetFile(path).iter_batches():
            for row in batch.to_pylist():
                text = row["prompt"]
                if row['conversation_id'] in exclude_conversations or hashlib.sha256(text.encode()).hexdigest() in exclude_prompt_hashes:
                    continue
                if row["language"] == "English" and not row["needs_review"] and ACTION.search(text) and GENRE.search(text) and not EXCLUDE.search(text):
                    eligible.append(row)
    eligible.sort(key=lambda row: hashlib.sha256(f"{seed}:{row['conversation_id']}:{row['message_index']}".encode()).digest())
    if len(eligible) < n:
        raise ValueError(f"Only {len(eligible)} eligible prompts")
    sample = eligible[:n]
    assert len({row['conversation_id'] for row in sample}) == n
    return sample, len(eligible)


def wilson(successes, n):
    if not n:
        return None
    z = 1.959963984540054
    p = successes / n
    center = (p + z*z/(2*n))/(1+z*z/n)
    half = z*math.sqrt(p*(1-p)/n+z*z/(4*n*n))/(1+z*z/n)
    return [center-half, center+half]


def attach_originals(manifest):
    """Resolve the exact raw user message, before wording cleanup."""
    audit_path = Path('data/wildchat/original_edits.jsonl')
    originals = {}
    if audit_path.exists():
        for line in audit_path.read_text().splitlines():
            row = json.loads(line)
            originals[(row['source_row'], row['message_index'])] = row['original_prompt']
    for row in manifest['sample']:
        if 'original_prompt' not in row:
            key = (row['source_row'], row['message_index'])
            if row.get('edited') and key not in originals:
                raise ValueError(f'Missing original text for edited prompt {key}')
            row['original_prompt'] = originals.get(key, row['prompt'])
        row['original_prompt_characters'] = len(row['original_prompt'].strip())
    manifest['scope_rule']='All outputs are scored. Post-hoc scope: output/input trimmed Unicode character ratio must be at least the adjustable threshold (default 1.0).'


def summarize(root, manifest, scope_ratio=1.0):
    results=[]
    excluded=excluded_indices(manifest)
    generated_n=0
    for path in sorted(root.glob("item_*/result.json")):
        result=json.loads(path.read_text())
        generated_n+=1
        result['analysis_excluded']=result['index'] in excluded
        result['analysis_exclusion_reason']=excluded.get(result['index'])
        if result['analysis_excluded']:
            result['false_negative']=None
            save(path,result)
            continue
        original=manifest['sample'][result['index']]['original_prompt']
        result['original_prompt']=original
        result['original_prompt_characters']=len(original.strip())
        result['output_characters']=len(result['generated_text'].strip())
        result['output_to_input_ratio']=result['output_characters']/result['original_prompt_characters'] if result['original_prompt_characters'] else None
        result['scope_status']='out_of_scope' if result['output_characters']<scope_ratio*result['original_prompt_characters'] else 'in_scope'
        result['false_negative']=None if result['scope_status']=='out_of_scope' or not result.get('pangram') else result['pangram'].get('prediction_short','').strip().lower() in {'mixed','human'}
        save(path,result)
        results.append(result)
    in_scope_scored=[r for r in results if r['scope_status']=='in_scope' and r.get('pangram')]
    scored=[r for r in in_scope_scored if r['pangram'].get('prediction_short','').strip().lower() in {'ai','mixed','human'}]
    out_of_scope=[r for r in results if r['scope_status']=='out_of_scope']
    errors = [json.loads(p.read_text()) for p in sorted(root.glob("item_*/error.json")) if not (p.parent/'result.json').exists() and int(p.parent.name.split('_')[-1]) not in excluded]
    calls = Counter(call['name'] for r in results for call in r['tool_calls'])
    labels = Counter(r['pangram'].get('prediction_short', '').strip().lower() for r in scored)
    false_negatives = labels['mixed'] + labels['human']
    summary = {
        'planned_n': analysis_n(manifest),'sample_n':len(manifest['sample']),
        'excluded_n':len(excluded),'excluded_indices':list(excluded),'generated_n_including_excluded':generated_n,
        'completed_n': len(results), 'failed_n': len(errors),
        'out_of_scope_n':len(out_of_scope),'scored_in_scope_n':len(scored),
        'scope_ratio_threshold':scope_ratio,
        'out_of_scope_indices':[r['index'] for r in out_of_scope],
        'primary_metric': 'False-negative rate: prediction_short Mixed or Human on fully generated outputs',
        'false_negatives': false_negatives,
        'false_negative_rate': false_negatives/len(scored) if scored else None,
        'false_negative_rate_wilson95': wilson(false_negatives,len(scored)),
        'mixed_count': labels['mixed'], 'human_count': labels['human'], 'ai_count': labels['ai'],
        'unrecognized_prediction_count': len(in_scope_scored)-len(scored),
        'headlines': dict(Counter(r['pangram'].get('headline','') for r in scored)),
        'predictions': dict(Counter(r['pangram'].get('prediction_short','') for r in scored)),
        'pangram_versions': dict(Counter(r['pangram'].get('version','') for r in scored)),
        'mean_fractions': {key: sum(float(r['pangram'].get(key,0)) for r in scored)/len(scored) if scored else None for key in ['fraction_ai','fraction_ai_assisted','fraction_human']},
        'mean_word_count': sum(r['word_count'] for r in results)/len(results) if results else None,
        'items_with_scaffold_retrieval': sum(any(c['name']=='get_scaffold' and 'result' in c and not c['result'].get('isError',False) for c in r['tool_calls']) for r in results),
        'tool_call_counts':dict(calls),
        'anthropic_usage':dict(sum((Counter(r['usage']) for r in results), Counter())),
        'errors':errors,
        'limitations': ['FNR uses the requested strict definition: Mixed and Human both count as misses.',
                        'English prose-writing subset; not representative of all WildChat prompts.',
                        'No no-MCP baseline, so this does not estimate the causal effect of enabling MCP.',
                        'Scaffold-assisted text can reuse human-authored structure and phrasing.',
                        'Pangram fractions are text proportions, not probabilities.'],
    }
    summary['by_generation_model']={}
    for model in sorted({r.get('generation_model','claude-opus-5-5') for r in results}):
        subset=[r for r in scored if r.get('generation_model','claude-opus-5-5')==model]
        misses=sum(r['false_negative'] for r in subset)
        summary['by_generation_model'][model]={'completed':sum(r.get('generation_model','claude-opus-5-5')==model for r in results),
            'scored_in_scope':len(subset),'false_negatives':misses,'false_negative_rate':misses/len(subset) if subset else None}
    save(root/'summary.json',summary)
    with (root/'results.jsonl').open('w') as output:
        for result in results:
            output.write(json.dumps(result,ensure_ascii=False)+'\n')
    return summary


async def evaluate(args):
    load_dotenv()
    for key in ['ANTHROPIC_API_KEY','PANGRAM_API_KEY']:
        if not os.getenv(key):
            raise ValueError(f'{key} is missing')
    root=args.output
    root.mkdir(parents=True,exist_ok=True)
    manifest_path=root/'manifest.json'
    if manifest_path.exists():
        manifest=json.loads(manifest_path.read_text())
        if manifest['n']!=args.n or manifest['seed']!=args.seed:
            raise ValueError('Existing run has a different sample; choose another output directory')
    else:
        sample,eligible=select_sample(args.input,args.n,args.seed)
        manifest={'n':args.n,'seed':args.seed,'input':str(args.input),'eligible_count':eligible,
                  'created_at':datetime.now(timezone.utc).isoformat(),'model':args.model,'effort':'xhigh',
                  'system':SYSTEM,'max_tokens_per_turn':args.max_tokens,'max_turns':16,
                  'mcp_url':args.mcp_url,'mcp_implementation':'external server supplied by the user',
                  'pangram_endpoint':PANGRAM+'/task','sample':sample,
                  'eligibility':{'language':'English','exclude_needs_review':True,'action_regex':ACTION.pattern,'genre_regex':GENRE.pattern,'exclude_regex':EXCLUDE.pattern},
                  'primary_detection_rule':'prediction_short in {Mixed, Human} counts as a false negative'}
        save(manifest_path,manifest)
    attach_originals(manifest)
    save(manifest_path,manifest)
    if args.summarize_only:
        print(json.dumps(summarize(root,manifest,args.scope_ratio),indent=2))
        return
    semaphore=asyncio.Semaphore(args.concurrency)
    async with httpx.AsyncClient(timeout=60,headers={'x-api-key':os.environ['PANGRAM_API_KEY']}) as detector:
        async def one(index,row):
            if index in excluded_indices(manifest):return
            directory=root/f'item_{index:03d}'
            directory.mkdir(exist_ok=True)
            if (directory/'result.json').exists():
                return
            if (directory/'error.json').exists():
                prior=json.loads((directory/'error.json').read_text())
                if 'blocked by content filtering policy' in prior.get('message','') and not args.retry_blocked:
                    print(f'{index+1}/{args.n} remains unscored: provider content filter returned no output',flush=True)
                    return
            async with semaphore:
                start=time.monotonic()
                if (directory/'error.json').exists():
                    prior_error=json.loads((directory/'error.json').read_text())
                    save(directory/f'error.attempt{len(list(directory.glob("error.attempt*.json")))+1}.json',prior_error)
                (directory/'error.json').unlink(missing_ok=True)
                try:
                    trace_path=directory/'generation.json'
                    trace=json.loads(trace_path.read_text()) if trace_path.exists() else {}
                    if not trace.get('text') and trace.get('messages') and trace['messages'][-1]['stop_reason'] in {'end_turn','refusal','max_tokens','stop_sequence'}:
                        trace['text']='\n'.join(b['text'] for b in trace['messages'][-1]['content'] if b['type']=='text')
                    if not trace.get('text'):
                        if trace_path.exists():
                            attempt=len(list(directory.glob('generation.attempt*.json')))+1
                            trace_path.rename(directory/f'generation.attempt{attempt}.json')
                        trace=await run(row['prompt'],url=args.mcp_url,system=SYSTEM,max_tokens=args.max_tokens,
                                        checkpoint=lambda value:save(trace_path,value),model=args.model)
                    text=trace['text'].strip()
                    if not text:
                        raise ValueError('Empty generated text')
                    usage=Counter()
                    for message in trace['messages']:
                        usage.update({k:v for k,v in message.get('usage',{}).items() if isinstance(v,int)})
                    task_path=directory/'pangram_task.json'
                    if task_path.exists():
                        task=json.loads(task_path.read_text())
                    else:
                        response=await detector.post(PANGRAM+'/task',json={'text':text,'public_dashboard_link':False})
                        response.raise_for_status()
                        task=response.json()
                        save(task_path,task)
                    deadline=time.monotonic()+900
                    while True:
                        response=await detector.get(PANGRAM+'/task/'+task['task_id'])
                        response.raise_for_status()
                        prediction=response.json()
                        save(directory/'pangram.json',prediction)
                        if prediction.get('stage')=='STAGE_SUCCESS':
                            break
                        if prediction.get('stage')=='STAGE_FAILED':
                            raise ValueError('Pangram failed: '+prediction.get('headline','unknown'))
                        if time.monotonic()>deadline:
                            raise TimeoutError('Pangram task still pending after 15 minutes')
                        await asyncio.sleep(3)
                    result={'index':index,'prompt':row,'generated_text':text,'word_count':len(text.split()),
                            'tool_calls':trace['tool_calls'],'usage':dict(usage),'pangram':prediction,
                            'generation_stop_reason':trace['messages'][-1]['stop_reason'],
                            'generation_model':trace.get('model',args.model),
                            'max_tokens_per_turn':trace.get('max_tokens_per_turn',16384),
                            'elapsed_seconds':time.monotonic()-start}
                    save(directory/'result.json',result)
                    (directory/'error.json').unlink(missing_ok=True)
                    print(f"{index+1}/{args.n} complete: {prediction.get('prediction_short')} ({len(text.split())} words)",flush=True)
                except Exception as exc:
                    def flatten(error):
                        return '; '.join(flatten(e) for e in error.exceptions) if isinstance(error,BaseExceptionGroup) else f'{type(error).__name__}: {error}'
                    detail=flatten(exc)
                    save(directory/'error.json',{'index':index,'type':type(exc).__name__,'message':detail,'elapsed_seconds':time.monotonic()-start})
                    print(f'{index+1}/{args.n} failed: {detail}',flush=True)
        chosen=list(enumerate(manifest['sample']))
        if args.indices:
            chosen=[pair for pair in chosen if pair[0] in {int(v) for v in args.indices.split(',')}]
        chosen=chosen[args.offset:]
        if args.limit:
            chosen=chosen[:args.limit]
        await asyncio.gather(*(one(index,row) for index,row in chosen))
    summary=summarize(root,manifest,args.scope_ratio)
    print(json.dumps({k:summary[k] for k in ['completed_n','failed_n','false_negative_rate','predictions']},indent=2),flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input',type=Path,default=Path('data/wildchat/splits/train'))
    parser.add_argument('--output',type=Path,default=Path('data/evals/pangram_mcp_n60_seed42'))
    parser.add_argument('--n',type=int,default=60)
    parser.add_argument('--seed',type=int,default=42)
    parser.add_argument('--concurrency',type=int,default=6)
    parser.add_argument('--max-tokens',type=int,default=16384)
    parser.add_argument('--mcp-url',default='http://127.0.0.1:3000/mcp')
    parser.add_argument('--limit',type=int)
    parser.add_argument('--offset',type=int,default=0)
    parser.add_argument('--indices',help='Comma-separated zero-based sample indices to retry')
    parser.add_argument('--summarize-only',action='store_true')
    parser.add_argument('--scope-ratio',type=float,default=1.0)
    parser.add_argument('--model',default='claude-opus-5-5')
    parser.add_argument('--retry-blocked',action='store_true')
    asyncio.run(evaluate(parser.parse_args()))
