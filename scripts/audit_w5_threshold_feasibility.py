#!/usr/bin/env python3
import json
from collections import Counter
from pathlib import Path

rows=[json.loads(x) for x in Path('results/78_w5_controller_offline/predictions.jsonl').open()]
best=None;feasible=[]
for threshold in sorted({x['continue_score'] for x in rows}):
 stop=[x for x in rows if x['gold_decision']=='STOP'];cont=[x for x in rows if x['gold_decision']=='CONTINUE']
 sr=sum(x['continue_score']<threshold for x in stop)/len(stop)
 cr=sum(x['continue_score']>=threshold for x in cont)/len(cont)
 item=((sr+cr)/2,sr,cr,threshold)
 if best is None or item>best:best=item
 if sr>=.8 and cr>=.8:feasible.append(item)
summary={'gate':'W5_THRESHOLD_RESCUE_FAIL' if not feasible else 'W5_THRESHOLD_RESCUE_PASS','generated_confusion':{'|'.join(k):v for k,v in Counter((x['gold_decision'],x['prediction']) for x in rows).items()},'best_balanced_accuracy':best[0],'best_stop_recall':best[1],'best_continue_recall':best[2],'best_threshold':best[3],'thresholds_meeting_80_80':len(feasible)}
out=Path('results/78_w5_controller_offline/threshold_audit.json');out.write_text(json.dumps(summary,indent=2)+'\n');print(json.dumps(summary,indent=2))
