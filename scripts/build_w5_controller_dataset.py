#!/usr/bin/env python3
"""Build the frozen balanced W5 Controller SFT and natural behavior-dev sets."""
import hashlib, json
from collections import Counter
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
STATES=ROOT/'results/74_w5_controller_data/adjudicated_states.jsonl'
QUERIES=ROOT/'results/75_w5_query_teacher/full/queries.jsonl'
OUT=ROOT/'results/76_w5_controller_dataset'
SYSTEM='''You are a research controller. Given the question, current Web observation, ResearchMemory, previous queries, and remaining budget, output exactly one action. If evidence is sufficient: DECISION: STOP. Otherwise output DECISION: CONTINUE, one specific MISSING fact, and one concise non-duplicate QUERY grounded in the current state. Do not answer the research question.'''
def read(p):return [json.loads(x) for x in p.open() if x.strip()]
def order(x):return hashlib.sha256(('42:'+x['state_id']).encode()).hexdigest()
def valid(t):return t.get('ok') and not t.get('duplicate') and t.get('conditioning')!='none'
def target(s,q):
 if s['decision']=='STOP':return 'DECISION: STOP'
 t=q['teacher'];return f"DECISION: CONTINUE\nMISSING: {t['missing']}\nQUERY: {t['query']}"
def sharegpt(s,q,repeat=0):
 return {'system':SYSTEM,'conversations':[{'from':'human','value':s['controller_input']},{'from':'gpt','value':target(s,q)}],'metadata':{'state_id':s['state_id'],'sample_id':s['sample_id'],'controller_split':s['controller_split'],'state_origin':s['state_origin'],'decision':s['decision'],'repeat_index':repeat}}
def main():
 states=read(STATES); qmap={x['state_id']:x for x in read(QUERIES)}
 train_stop=sorted([x for x in states if x['controller_split']=='train' and x['decision']=='STOP'],key=order)
 natural=sorted([x for x in states if x['controller_split']=='train' and x['decision']=='CONTINUE' and x['state_origin']=='natural_bocha_on_policy' and x['state_id'] in qmap and valid(qmap[x['state_id']]['teacher'])],key=order)[:1200]
 masked=sorted([x for x in states if x['controller_split']=='train' and x['decision']=='CONTINUE' and x['state_origin']=='counterfactual_evidence_mask' and x['state_id'] in qmap and valid(qmap[x['state_id']]['teacher'])],key=order)[:342]
 assert len(train_stop)==771 and len(natural)==1200 and len(masked)==342
 train=[]
 for s in train_stop:
  train.extend([sharegpt(s,None,0),sharegpt(s,None,1)])
 train.extend(sharegpt(s,qmap[s['state_id']]) for s in natural+masked)
 train.sort(key=lambda x:(x['metadata']['sample_id'],x['metadata']['decision'],x['metadata']['repeat_index']))
 dev_states=sorted([x for x in states if x['controller_split']=='dev' and x['state_origin']=='natural_bocha_on_policy'],key=order)
 behavior=[];dev=[]
 for s in dev_states:
  q=qmap.get(s['state_id']); qok=s['decision']=='STOP' or (q is not None and valid(q['teacher']))
  behavior.append({'state_id':s['state_id'],'sample_id':s['sample_id'],'decision':s['decision'],'controller_input':s['controller_input'],'previous_queries':s['previous_queries'],'query_target':q['teacher']['query'] if q and valid(q['teacher']) else None,'query_target_valid':qok})
  if qok:dev.append(sharegpt(s,q))
 OUT.mkdir(parents=True,exist_ok=True)
 for name,rows in [('train.jsonl',train),('dev_ce.jsonl',dev),('behavior_dev500.jsonl',behavior)]:
  with (OUT/name).open('w') as f:
   for x in rows:f.write(json.dumps(x,ensure_ascii=False)+'\n')
 info={
  'w5_controller_train':{'file_name':'train.jsonl','formatting':'sharegpt','columns':{'messages':'conversations','system':'system'},'tags':{'role_tag':'from','content_tag':'value','user_tag':'human','assistant_tag':'gpt'}},
  'w5_controller_dev':{'file_name':'dev_ce.jsonl','formatting':'sharegpt','columns':{'messages':'conversations','system':'system'},'tags':{'role_tag':'from','content_tag':'value','user_tag':'human','assistant_tag':'gpt'}}}
 (OUT/'dataset_info.json').write_text(json.dumps(info,indent=2)+'\n')
 tc=Counter(x['metadata']['decision'] for x in train); ids_train={x['metadata']['sample_id'] for x in train};ids_dev={x['sample_id'] for x in behavior}
 manifest={'gate':'W5_CONTROLLER_DATA_GATE_PASS' if tc['STOP']==tc['CONTINUE']==1542 and len(behavior)==500 and not(ids_train&ids_dev) else 'W5_CONTROLLER_DATA_GATE_FAIL','train_rows':len(train),'train_unique_states':len({x['metadata']['state_id'] for x in train}),'train_decisions':dict(tc),'train_natural_continue':1200,'train_masked_continue':342,'stop_repeat':2,'dev_ce_rows':len(dev),'behavior_dev_rows':len(behavior),'behavior_dev_invalid_query_targets':sum(not x['query_target_valid'] for x in behavior),'question_overlap':len(ids_train&ids_dev),'structural_gold_fields':0,'global_batch_target':16,'epochs':2,'expected_optimizer_steps':386}
 (OUT/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n');print(json.dumps(manifest,indent=2))
 if manifest['gate'].endswith('FAIL'):raise SystemExit(1)
if __name__=='__main__':main()
