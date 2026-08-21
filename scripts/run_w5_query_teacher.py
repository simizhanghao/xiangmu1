#!/usr/bin/env python3
"""Generate compact MISSING + QUERY targets without gold/reference input."""
import argparse, json, os, re, time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
import requests

STOP={"the","a","an","of","in","on","for","to","and","or","is","was","are","were","who","what","which","when","where","how","did","does","do","with","from","by"}
WORD=re.compile(r"[A-Za-z0-9][A-Za-z0-9_-]*")
def words(x): return {w.lower() for w in WORD.findall(str(x)) if w.lower() not in STOP}
def qnorm(x): return " ".join(WORD.findall(str(x).lower()))
def read(p): return [json.loads(x) for x in Path(p).open() if x.strip()]

def parse_args():
 p=argparse.ArgumentParser(); p.add_argument('--queue',default='results/74_w5_controller_data/query_queue.jsonl'); p.add_argument('--output-dir',type=Path,required=True); p.add_argument('--max-samples',type=int,default=0); p.add_argument('--smoke-per-origin',type=int,default=0); p.add_argument('--concurrency',type=int,default=4); p.add_argument('--retries',type=int,default=3); p.add_argument('--timeout',type=float,default=180); p.add_argument('--base-url',default=os.environ.get('TEACHER_BASE_URL','https://api.deepseek.com')); p.add_argument('--model',default=os.environ.get('TEACHER_MODEL','deepseek-v4-flash')); p.add_argument('--api-key',default=os.environ.get('TEACHER_API_KEY','')); return p.parse_args()

SYSTEM='''You are a research controller teacher. The current Web evidence is known to be insufficient. Return one JSON object only: {"decision":"CONTINUE","missing":"one specific unresolved fact","query":"one concise next Web query"}. The query must target missing, must not repeat Previous Queries, and should use a concrete entity learned from Current Observation whenever possible. Use only the supplied state. Never answer the question and never use reference answers.'''
def call(c,row):
 body={'model':c.model,'messages':[{'role':'system','content':SYSTEM},{'role':'user','content':row['controller_input']+'\n\nPrevious Queries: '+json.dumps(row['previous_queries'],ensure_ascii=False)}],'temperature':0,'seed':42,'max_tokens':300,'response_format':{'type':'json_object'},'thinking':{'type':'disabled'}}
 err=None
 for a in range(c.retries):
  try:
   z=requests.post(c.base_url.rstrip('/')+'/chat/completions',headers={'Content-Type':'application/json','Authorization':'Bearer '+c.api_key},json=body,timeout=c.timeout)
   if z.status_code>=400 and a==0: body.pop('thinking',None);body.pop('response_format',None);continue
   z.raise_for_status(); text=str(z.json()['choices'][0]['message']['content']).strip(); text=re.sub(r'^```(?:json)?\s*|\s*```$','',text,flags=re.I|re.S); v=json.loads(text)
   missing=' '.join(str(v.get('missing') or '').split()); query=' '.join(str(v.get('query') or '').split())
   if v.get('decision')!='CONTINUE' or not missing or not query: raise ValueError('invalid schema')
   duplicate=any(qnorm(query)==qnorm(x) for x in row['previous_queries'])
   question=row['controller_input'].split('\n\nCurrent Observation:',1)[0]
   observation=row['controller_input'].split('\n\nCurrent Observation:',1)[-1]
   obs_entity=bool((words(query)-words(question)) & words(observation))
   missing_refine=qnorm(query)!=qnorm(question) and bool(words(query)&words(missing))
   return {'ok':True,'decision':'CONTINUE','missing':missing,'query':query,'duplicate':duplicate,'conditioning':'observation_entity' if obs_entity else ('missing_state_refinement' if missing_refine else 'none')}
  except Exception as e: err=e; time.sleep(2**a)
 return {'ok':False,'error_type':type(err).__name__,'error':str(err)[:300]}

def main():
 c=parse_args()
 if not c.api_key: raise SystemExit('MISSING_TEACHER_API_KEY')
 rows=read(c.queue)
 if c.smoke_per_origin:
  natural=[x for x in rows if x['state_origin']=='natural_bocha_on_policy'][:c.smoke_per_origin]
  masked=[x for x in rows if x['state_origin']=='counterfactual_evidence_mask'][:c.smoke_per_origin]
  rows=natural+masked
 rows=rows[:c.max_samples] if c.max_samples else rows
 c.output_dir.mkdir(parents=True,exist_ok=True); out=c.output_dir/'queries.jsonl'; cache={x['state_id']:x for x in read(out)} if out.exists() else {}
 todo=[x for x in rows if x['state_id'] not in cache]
 with ThreadPoolExecutor(max_workers=c.concurrency) as pool:
  fs={pool.submit(call,c,x):x for x in todo}
  for f in as_completed(fs):
   x=fs[f]; y={'state_id':x['state_id'],'sample_id':x['sample_id'],'controller_split':x['controller_split'],'state_origin':x['state_origin'],'teacher':f.result()}; cache[x['state_id']]=y
   with out.open('a') as h:h.write(json.dumps(y,ensure_ascii=False)+'\n')
 sel=[cache[x['state_id']] for x in rows]; ok=[x for x in sel if x['teacher'].get('ok')]; valid=[x for x in ok if not x['teacher']['duplicate'] and x['teacher']['conditioning']!='none']
 summary={'gate':'W5_QUERY_TEACHER_PASS' if len(ok)==len(sel) and len(valid)/max(1,len(sel))>=.8 else 'W5_QUERY_TEACHER_FAIL','requested':len(sel),'successful':len(ok),'valid':len(valid),'valid_rate':len(valid)/max(1,len(sel)),'duplicate_rate':sum(x['teacher'].get('duplicate',False) for x in ok)/max(1,len(ok)),'state_conditioned_rate':sum(x['teacher'].get('conditioning')!='none' for x in ok)/max(1,len(ok)),'gold_visible':False,'output':str(out)}
 (c.output_dir/'summary.json').write_text(json.dumps(summary,indent=2)+'\n');print(json.dumps(summary,indent=2));
 if summary['gate'].endswith('FAIL'):raise SystemExit(1)
if __name__=='__main__':main()
