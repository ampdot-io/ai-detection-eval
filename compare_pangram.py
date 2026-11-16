"""Score identical saved outputs with multiple account-authorized Pangram models."""
import argparse
import asyncio
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import time

from dotenv import load_dotenv
import httpx2 as httpx
from eval_pangram import PANGRAM, save, wilson
from evaluation_scope import excluded_indices


def summary(root, models, ratio=1.0):
    manifest=json.loads((root/'manifest.json').read_text())
    records=[]
    excluded=excluded_indices(manifest)
    for path in sorted(root.glob('item_*/result.json')):
        item=json.loads(path.read_text());original=manifest['sample'][item['index']]['original_prompt']
        if item['index'] in excluded:continue
        predictions={}
        digest=hashlib.sha256(item['generated_text'].encode()).hexdigest()
        for model in models:
            score=path.parent/'pangram_models'/f'{model}.json'
            if score.exists():
                value=json.loads(score.read_text());assert value['input_sha256']==digest
                predictions[model]=value['result']
        records.append({'index':item['index'],'ratio':len(item['generated_text'].strip())/len(original.strip()),'predictions':predictions})
    valid=lambda p:p.get('prediction_short') in ['AI','Mixed','Human']
    paired=[r for r in records if r['ratio']>=ratio and all(m in r['predictions'] and valid(r['predictions'][m]) for m in models)]
    metrics={}
    for model in models:
        labels=[r['predictions'][model]['prediction_short'] for r in paired]
        misses=sum(label in ['Mixed','Human'] for label in labels)
        metrics[model]={'available_scores':sum(model in r['predictions'] for r in records),
                       'paired_n':len(paired),'AI':labels.count('AI'),'Mixed':labels.count('Mixed'),'Human':labels.count('Human'),
                       'false_negatives':misses,'false_negative_rate':misses/len(paired) if paired else None,
                       'wilson95':wilson(misses,len(paired)),
                       'reported_versions':sorted({r['predictions'][model].get('version','') for r in records if model in r['predictions']})}
    result={'scope_ratio_threshold':ratio,'generated_outputs':len(records),'paired_n':len(paired),
            'paired_indices':[r['index'] for r in paired],'models':metrics,
            'definition':'False negative = Mixed or Human. All comparisons use the same in-scope documents.'}
    save(root/'comparison_summary.json',result)
    return result


async def compare(args):
    load_dotenv()
    if args.summary_only:
        models=json.loads((args.root/'pangram_models.json').read_text())['selected']
        print(json.dumps(summary(args.root,models,args.scope_ratio),indent=2));return
    async with httpx.AsyncClient(headers={'x-api-key':os.environ['PANGRAM_API_KEY']},timeout=60) as client:
        response=await client.get(PANGRAM+'/models');response.raise_for_status()
        available=response.json()['models']
        selected=args.models or [m for m in available if m!='default']
        if not set(selected)<=set(available):
            raise ValueError(f'Models must be in {available}')
        save(args.root/'pangram_models.json',{'available':available,'selected':selected,
              'created_at':datetime.now(timezone.utc).isoformat(),
              'method':'Identical generated text; paired comparison on common scored cohort. Existing default results are reused when version matches the explicit selector.'})
        semaphore=asyncio.Semaphore(args.concurrency)
        post_lock=asyncio.Lock()

        async def request(method,url,**kwargs):
            for attempt in range(8):
                r=await client.request(method,url,**kwargs)
                if r.status_code not in [429,500,502,503,504]:
                    r.raise_for_status();return r
                await asyncio.sleep(min(30,2**attempt))
            r.raise_for_status()

        async def score(path,model):
            item=json.loads(path.read_text());text=item['generated_text']
            folder=path.parent/'pangram_models';folder.mkdir(exist_ok=True)
            output=folder/f'{model}.json'
            if output.exists(): return
            digest=hashlib.sha256(text.encode()).hexdigest()
            baseline=item.get('pangram') or {}
            if model=='pangram-'+str(baseline.get('version','')):
                save(output,{'model':model,'input_sha256':digest,'result':baseline,'reused_default':True})
                return
            async with semaphore:
                try:
                    task_file=folder/f'{model}.task.json'
                    if task_file.exists():
                        task=json.loads(task_file.read_text())
                        assert task['input_sha256']==digest
                    else:
                        async with post_lock:
                            r=await request('POST',PANGRAM+'/task',json={'text':text,'model':model,'public_dashboard_link':False})
                            task={**r.json(),'input_sha256':digest,'model':model}
                            save(task_file,task)
                            await asyncio.sleep(.3)
                    deadline=time.monotonic()+900
                    while True:
                        r=await request('GET',PANGRAM+'/task/'+task['task_id'])
                        result=r.json()
                        if result.get('stage')=='STAGE_SUCCESS':
                            save(output,{'model':model,'input_sha256':digest,'result':result,'reused_default':False})
                            (folder/f'{model}.error.json').unlink(missing_ok=True)
                            print(f"{path.parent.name} {model}: {result.get('prediction_short')}",flush=True)
                            return
                        if result.get('stage')=='STAGE_FAILED':
                            raise ValueError(result.get('headline','Pangram task failed'))
                        if time.monotonic()>deadline: raise TimeoutError('Pangram task pending after 15 minutes')
                        await asyncio.sleep(3)
                except Exception as exc:
                    save(folder/f'{model}.error.json',{'type':type(exc).__name__,'message':str(exc)})
                    print(f'{path.parent.name} {model}: {type(exc).__name__}: {exc}',flush=True)
        while True:
            paths=sorted(args.root.glob('item_*/result.json'))
            await asyncio.gather(*(score(path,model) for path in paths for model in selected))
            summary(args.root,selected,args.scope_ratio)
            if not args.watch or (args.root/'generation_complete.json').exists():
                break
            await asyncio.sleep(5)
        print('Pangram model comparison scoring is current.',flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root',type=Path,default=Path('data/evals/pangram_mcp_n60_seed42'))
    p.add_argument('--models',nargs='+')
    p.add_argument('--concurrency',type=int,default=3)
    p.add_argument('--watch',action='store_true')
    p.add_argument('--summary-only',action='store_true')
    p.add_argument('--scope-ratio',type=float,default=1.0)
    asyncio.run(compare(p.parse_args()))
