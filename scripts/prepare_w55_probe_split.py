#!/usr/bin/env python3
"""Freeze the natural-only W5.5 train/calibration/dev split."""
import hashlib,json
from collections import Counter
from pathlib import Path

SRC=Path('results/74_w5_controller_data/adjudicated_states.jsonl')
OUT=Path('results/79_w55_linear_probe')
def order(x):return hashlib.sha256(('w55:42:'+x['state_id']).encode()).hexdigest()
rows=[json.loads(x) for x in SRC.open() if x.strip()]
natural=[x for x in rows if x['state_origin']=='natural_bocha_on_policy']
train0=[x for x in natural if x['controller_split']=='train'];dev=[x for x in natural if x['controller_split']=='dev']
by={k:sorted([x for x in train0 if x['decision']==k],key=order) for k in ('STOP','CONTINUE')}
calib_stop=round(400*len(by['STOP'])/len(train0));calib=by['STOP'][:calib_stop]+by['CONTINUE'][:400-calib_stop]
calib_ids={x['state_id'] for x in calib};train=[x for x in train0 if x['state_id'] not in calib_ids]
for split,xs in [('train',train),('calib',calib),('dev',dev)]:
 for x in xs:x['probe_split']=split
OUT.mkdir(parents=True,exist_ok=True)
all_rows=sorted(train+calib+dev,key=lambda x:(x['probe_split'],order(x)))
with (OUT/'split.jsonl').open('w') as f:
 for x in all_rows:
  keep={k:x[k] for k in ('state_id','sample_id','decision','controller_input','previous_queries','probe_split')}
  f.write(json.dumps(keep,ensure_ascii=False)+'\n')
counts={s:dict(Counter(x['decision'] for x in all_rows if x['probe_split']==s)) for s in ('train','calib','dev')}
ids={s:{x['sample_id'] for x in all_rows if x['probe_split']==s} for s in ('train','calib','dev')}
summary={'gate':'W55_SPLIT_PASS' if len(train)==4098 and len(calib)==400 and len(dev)==500 and not(ids['train']&ids['calib']|ids['train']&ids['dev']|ids['calib']&ids['dev']) else 'W55_SPLIT_FAIL','natural_states':len(natural),'counts':counts,'overlap':{'train_calib':len(ids['train']&ids['calib']),'train_dev':len(ids['train']&ids['dev']),'calib_dev':len(ids['calib']&ids['dev'])},'seed':42,'output':str(OUT/'split.jsonl')}
(OUT/'split_summary.json').write_text(json.dumps(summary,indent=2)+'\n');print(json.dumps(summary,indent=2))
if summary['gate'].endswith('FAIL'):raise SystemExit(1)
