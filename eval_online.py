"""Online benchmark: retry an unchanged prompt when Pangram says AI."""
import argparse
import asyncio
import fcntl
from collections import Counter
import hashlib
import json
import os
from datetime import datetime,timezone
from pathlib import Path

from dotenv import load_dotenv
import httpx2 as httpx

from claude_mcp_local import run
from eval_pangram import PANGRAM, SYSTEM, save, summarize
from evaluation_scope import excluded_indices


async def retry_flagged(initial, make_attempt, max_retries=3):
    if not 0 <= max_retries <= 3:
        raise ValueError('max_retries must be between 0 and 3')
    history=[initial]
    for attempt in range(1,max_retries+1):
        label=history[-1]['pangram']['prediction_short'].strip().lower()
        if label in {'mixed','human'}:
            break
        if label!='ai':
            raise ValueError(f'Unknown Pangram label: {label}')
        history.append(await make_attempt(attempt))
    return history


async def predict(client,text,model,folder):
    folder.mkdir(parents=True,exist_ok=True)
    path=folder/f'{model}.json'
    if path.exists():
        return json.loads(path.read_text())
    task_path=folder/f'{model}.task.json'
    if task_path.exists():
        task=json.loads(task_path.read_text())
    else:
        response=await client.post(PANGRAM+'/task',json={'text':text,'model':model,'public_dashboard_link':False})
        response.raise_for_status();task=response.json();save(task_path,task)
    for _ in range(300):
        response=await client.get(PANGRAM+'/task/'+task['task_id'])
        response.raise_for_status();result=response.json()
        if result.get('stage')=='STAGE_SUCCESS':
            save(path,result);return result
        if result.get('stage')=='STAGE_FAILED':
            raise RuntimeError(result.get('headline','Pangram task failed'))
        await asyncio.sleep(3)
    raise TimeoutError('Pangram task pending after 15 minutes')


async def main(args):
    load_dotenv()
    source=json.loads((args.offline/'manifest.json').read_text())
    sample=source['sample'][:args.n]
    # Freeze and reuse initial outputs: online retries never replace the offline run.
    missing=[i for i in range(len(sample)) if not (args.offline/f'item_{i:03d}'/'result.json').exists()]
    if missing and not args.wait_for_offline:
        raise ValueError(f'Offline initial outputs are not ready at indices {missing}')
    args.output.mkdir(parents=True,exist_ok=True)
    saved_manifest=args.output/'manifest.json'
    if saved_manifest.exists():
        saved=json.loads(saved_manifest.read_text())
        if args.max_tokens is None and saved.get('max_tokens_per_turn') is not None:
            args.max_tokens=saved['max_tokens_per_turn']
        if args.model is None and saved.get('max_tokens_per_turn') is not None:
            args.model=saved['model']
    config=args.offline/'pangram_models.json'
    score_models=args.score_models or (json.loads(config.read_text())['selected'] if config.exists() else [args.retry_model])
    score_models=list(dict.fromkeys([args.retry_model,*score_models]))
    async with httpx.AsyncClient(headers={'x-api-key':os.environ['PANGRAM_API_KEY']},timeout=60) as detector:
        response=await detector.get(PANGRAM+'/models');response.raise_for_status()
        available=response.json()['models']
        if not set(score_models)<=set(available): raise ValueError(f'Choose from {available}')
        manifest={**source,'n':len(sample),'sample':sample,'variant':'online',
                  'offline_source':str(args.offline),'retry_model':args.retry_model,
                  'score_models':score_models,'max_retries':args.max_retries,'max_attempts':args.max_retries+1,
                  'retry_rule':'Fresh generation from unchanged prompt after AI; stop on Mixed or Human. Scope is applied only afterward.',
                  'max_tokens_per_turn':args.max_tokens,
                  'generation_policy':'Inherit each initial output model/token limit unless explicitly overridden.'}
        manifest_path=args.output/'manifest.json'
        if manifest_path.exists():
            existing=json.loads(manifest_path.read_text())
            for key in ['n','retry_model','max_tokens_per_turn']:
                if existing[key]!=manifest[key]: raise ValueError('Use a new output folder for different settings')
            if existing['max_retries']!=args.max_retries and not (args.extend_existing and args.max_retries>=existing['max_retries']):
                raise ValueError('Use --extend-existing to increase an existing retry budget')
            for key in ['budget_history_note','recovery_runs']:
                if key in existing:manifest[key]=existing[key]
        if args.recovery_max_tokens:
            manifest.setdefault('recovery_runs',[]).append({'at':datetime.now(timezone.utc).isoformat(),
                'max_tokens_per_turn':args.recovery_max_tokens,'indices':args.indices})
        save(manifest_path,manifest)
        save(args.output/'pangram_models.json',{'selected':score_models,'available':available})
        semaphore=asyncio.Semaphore(args.concurrency)

        async def one(index,row):
            if index in excluded_indices(manifest):return
            folder=args.output/f'item_{index:03d}';folder.mkdir(exist_ok=True)
            if args.extend_existing:
                while not (folder/'result.json').exists() and not (folder/'error.json').exists():
                    await asyncio.sleep(5)
            if (folder/'result.json').exists():
                previous=json.loads((folder/'result.json').read_text())
                if previous['pangram']['prediction_short'].strip().lower()!='ai' or previous['selected_attempt']>=args.max_retries:
                    return
            baseline_path=args.offline/f'item_{index:03d}'/'result.json'
            while not baseline_path.exists():
                if (args.offline/'generation_complete.json').exists():
                    save(folder/'error.json',{'type':'MissingInitialOutput','message':'Offline generation finished without an output for this prompt'})
                    return
                await asyncio.sleep(5)
            async with semaphore:
                try:
                    baseline=json.loads(baseline_path.read_text())
                    generation_model=args.model or baseline.get('generation_model',source['model'])
                    token_limit=args.recovery_max_tokens or args.max_tokens or baseline.get('max_tokens_per_turn',source['max_tokens_per_turn'])
                    initial=dict(baseline)
                    if 'pangram-'+str((initial.get('pangram') or {}).get('version'))!=args.retry_model:
                        initial['pangram']=await predict(detector,initial['generated_text'],args.retry_model,folder/'attempt_0'/'scores')
                    save(folder/'attempt_0.json',initial)

                    async def attempt(number):
                        path=folder/f'attempt_{number}';path.mkdir(exist_ok=True)
                        record=path/'record.json'
                        if record.exists(): return json.loads(record.read_text())
                        trace_path=path/'generation.json'
                        trace=json.loads(trace_path.read_text()) if trace_path.exists() else {}
                        if not trace.get('text'):
                            if trace_path.exists():
                                failed_number=len(list(path.glob('generation.failed*.json')))+1
                                trace_path.rename(path/f'generation.failed{failed_number}.json')
                            try:
                                trace=await run(row['prompt'],model=generation_model,url=args.mcp_url,system=SYSTEM,
                                    max_tokens=token_limit,checkpoint=lambda x:save(trace_path,x))
                            except Exception as exc:
                                save(path/'primary_error.json',{'type':type(exc).__name__,'message':str(exc)})
                                if not args.fallback_model or args.fallback_model==generation_model: raise
                                if trace_path.exists(): trace_path.rename(path/'generation.primary_failed.json')
                                trace=await run(row['prompt'],model=args.fallback_model,url=args.mcp_url,system=SYSTEM,
                                    max_tokens=token_limit,checkpoint=lambda x:save(trace_path,x))
                        text=trace['text']
                        result=await predict(detector,text,args.retry_model,path/'scores')
                        value={'index':index,'prompt':row,'generated_text':text,'pangram':result,
                               'word_count':len(text.split()),'generation_model':trace['model'],
                               'max_tokens_per_turn':trace.get('max_tokens_per_turn',token_limit),
                               'generation_stop_reason':trace['messages'][-1]['stop_reason'],
                               'tool_calls':trace['tool_calls'],'usage':dict(sum((Counter({k:v for k,v in m['usage'].items() if isinstance(v,int)}) for m in trace['messages']),Counter()))}
                        save(record,value);return value

                    history=await retry_flagged(initial,attempt,args.max_retries)
                    chosen=dict(history[-1])
                    chosen.update(selected_attempt=len(history)-1,attempt_count=len(history),
                                  retry_model=args.retry_model,attempt_labels=[a['pangram']['prediction_short'] for a in history])
                    target=folder/'pangram_models';target.mkdir(exist_ok=True)
                    for model in score_models:
                        if model==args.retry_model: score=chosen['pangram']
                        elif len(history)==1 and (args.offline/f'item_{index:03d}'/'pangram_models'/f'{model}.json').exists():
                            score=json.loads((args.offline/f'item_{index:03d}'/'pangram_models'/f'{model}.json').read_text())['result']
                        else: score=await predict(detector,chosen['generated_text'],model,folder/f'attempt_{len(history)-1}'/'scores')
                        save(target/f'{model}.json',{'model':model,'input_sha256':hashlib.sha256(chosen['generated_text'].encode()).hexdigest(),'result':score})
                    save(folder/'result.json',chosen)
                    (folder/'error.json').unlink(missing_ok=True)
                    print(f"{index+1}/{len(sample)}: {chosen['attempt_labels']} (selected attempt {chosen['selected_attempt']})",flush=True)
                except Exception as exc:
                    save(folder/'error.json',{'type':type(exc).__name__,'message':str(exc)})
                    print(f'{index+1}/{len(sample)} failed: {type(exc).__name__}',flush=True)
        selected={int(value) for value in args.indices.split(',')} if args.indices else set(range(len(sample)))
        await asyncio.gather(*(one(i,row) for i,row in enumerate(sample) if i in selected))
    report=summarize(args.output,manifest,args.scope_ratio)
    completed=[json.loads(p.read_text()) for p in args.output.glob('item_*/result.json')]
    report['online']={'retry_model':args.retry_model,'max_retries':args.max_retries,
                      'total_retries':sum(r['selected_attempt'] for r in completed),
                      'attempt_histogram':dict(Counter(r['attempt_count'] for r in completed)),
                      'stopping_rule':'AI retries; Mixed/Human stop; same prompt; no detector feedback added to prompt'}
    save(args.output/'summary.json',report)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--offline',type=Path,default=Path('data/evals/pangram_mcp_n60_seed42'))
    p.add_argument('--output',type=Path,default=Path('data/evals/pangram_mcp_online_n60_seed42'))
    p.add_argument('--n',type=int,default=60)
    p.add_argument('--max-retries',type=int,choices=range(4),default=3)
    p.add_argument('--extend-existing',action='store_true',help='Wait for the active initial batch, then extend only still-flagged tasks')
    p.add_argument('--retry-model',default='pangram-4.1')
    p.add_argument('--score-models',nargs='+')
    p.add_argument('--model',help='Override the generator model; otherwise inherit each initial output')
    p.add_argument('--fallback-model',default='claude-opus-4-8')
    p.add_argument('--max-tokens',type=int,help='Override the initial output token limit')
    p.add_argument('--recovery-max-tokens',type=int,help='Token limit only for newly generated recovery attempts; retain saved outputs')
    p.add_argument('--indices',help='Comma-separated zero-based indices to resume')
    p.add_argument('--mcp-url',default='http://127.0.0.1:3000/mcp')
    p.add_argument('--scope-ratio',type=float,default=1)
    p.add_argument('--concurrency',type=int,default=3)
    p.add_argument('--wait-for-offline',action='store_true',help='Consume offline outputs as they become available')
    args=p.parse_args()
    args.output.mkdir(parents=True,exist_ok=True)
    # A second resume command waits, then reuses the completed records instead
    # of issuing concurrent duplicate generations or detector submissions.
    with (args.output/'.run.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX)
        asyncio.run(main(args))
