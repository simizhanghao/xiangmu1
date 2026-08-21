#!/usr/bin/env python3
"""Frozen natural-dev500 offline Gate for the W5 modular Controller."""
import argparse,json,re,sys
from pathlib import Path
import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM,AutoTokenizer

SYSTEM='''You are a research controller. Given the question, current Web observation, ResearchMemory, previous queries, and remaining budget, output exactly one action. If evidence is sufficient: DECISION: STOP. Otherwise output DECISION: CONTINUE, one specific MISSING fact, and one concise non-duplicate QUERY grounded in the current state. Do not answer the research question.'''
STOPWORDS={"the","a","an","of","in","on","for","to","and","or","is","was","are","were","who","what","which","when","where","how","did","does","do","with","from","by"}
WORD=re.compile(r"[A-Za-z0-9][A-Za-z0-9_-]*")
def words(x):return {w.lower() for w in WORD.findall(str(x)) if w.lower() not in STOPWORDS}
def qnorm(x):return " ".join(WORD.findall(str(x).lower()))
def auc_rank(y,scores):
 order=sorted(range(len(scores)),key=lambda i:scores[i]);ranks=[0.0]*len(scores);i=0
 while i<len(order):
  j=i+1
  while j<len(order) and scores[order[j]]==scores[order[i]]:j+=1
  rank=(i+1+j)/2
  for k in range(i,j):ranks[order[k]]=rank
  i=j
 n1=sum(y);n0=len(y)-n1
 return (sum(r for r,v in zip(ranks,y) if v)-n1*(n1+1)/2)/(n1*n0)
def main():
 p=argparse.ArgumentParser();p.add_argument('--base',default='model/Qwen3-1.7B');p.add_argument('--adapter',default='outputs/77_w5_controller_lora');p.add_argument('--data',default='results/76_w5_controller_dataset/behavior_dev500.jsonl');p.add_argument('--output-dir',type=Path,default=Path('results/78_w5_controller_offline'));p.add_argument('--batch-size',type=int,default=8);c=p.parse_args()
 rows=[json.loads(x) for x in Path(c.data).open()]
 tok=AutoTokenizer.from_pretrained(c.base,trust_remote_code=True);tok.padding_side='left';tok.pad_token=tok.eos_token
 assert tok.encode(' STOP',add_special_tokens=False)==[45537] and tok.encode(' CONTINUE',add_special_tokens=False)[0]==16120
 model=AutoModelForCausalLM.from_pretrained(c.base,dtype=torch.bfloat16,device_map='cuda',trust_remote_code=True);model=PeftModel.from_pretrained(model,c.adapter);model.eval()
 prompts=[tok.apply_chat_template([{'role':'system','content':SYSTEM},{'role':'user','content':x['controller_input']}],tokenize=False,add_generation_prompt=True,enable_thinking=False) for x in rows]
 preds=[]
 for start in range(0,len(rows),c.batch_size):
  batch=rows[start:start+c.batch_size]; texts=prompts[start:start+c.batch_size]
  enc=tok(texts,padding=True,truncation=True,max_length=4096,return_tensors='pt').to(model.device)
  with torch.inference_mode(): out=model.generate(**enc,max_new_tokens=96,do_sample=False,pad_token_id=tok.pad_token_id,eos_token_id=tok.eos_token_id)
  generated=tok.batch_decode(out[:,enc.input_ids.shape[1]:],skip_special_tokens=True)
  score_enc=tok([x+'DECISION:' for x in texts],padding=True,truncation=True,max_length=4096,return_tensors='pt').to(model.device)
  with torch.inference_mode(): logits=model(**score_enc).logits[:,-1,:].float()
  scores=(logits[:,16120]-logits[:,45537]).cpu().tolist()
  for row,text,score in zip(batch,generated,scores):
   m=re.search(r'DECISION\s*:\s*(STOP|CONTINUE)',text,re.I);decision=m.group(1).upper() if m else 'INVALID'
   mm=re.search(r'MISSING\s*:\s*(.+?)(?:\n|$)',text,re.I);qm=re.search(r'QUERY\s*:\s*(.+?)(?:\n|$)',text,re.I)
   missing=mm.group(1).strip() if mm else '';query=qm.group(1).strip() if qm else ''
   duplicate=decision=='CONTINUE' and (not query or any(qnorm(query)==qnorm(x) for x in row['previous_queries']))
   question=row['controller_input'].split('\n\nCurrent Observation:',1)[0];observation=row['controller_input'].split('\n\nCurrent Observation:',1)[-1]
   conditioned=decision=='CONTINUE' and bool(query) and (bool((words(query)-words(question))&words(observation)) or (qnorm(query)!=qnorm(question) and bool(words(query)&words(missing))))
   preds.append({'state_id':row['state_id'],'sample_id':row['sample_id'],'gold_decision':row['decision'],'prediction':decision,'continue_score':score,'missing':missing,'query':query,'duplicate':duplicate,'state_conditioned':conditioned,'raw_output':text})
   print(f"[{len(preds)}/500] gold={row['decision']} pred={decision}",flush=True)
 y=[int(x['gold_decision']=='CONTINUE') for x in preds];scores=[x['continue_score'] for x in preds]
 stop=[x for x in preds if x['gold_decision']=='STOP'];cont=[x for x in preds if x['gold_decision']=='CONTINUE'];pc=[x for x in preds if x['prediction']=='CONTINUE']
 stop_recall=sum(x['prediction']=='STOP' for x in stop)/len(stop);cont_recall=sum(x['prediction']=='CONTINUE' for x in cont)/len(cont);bal=(stop_recall+cont_recall)/2
 auc=auc_rank(y,scores)
 summary={'gate':'W5_CONTROLLER_OFFLINE_GATE_PASS' if auc>=.90 and stop_recall>=.80 and cont_recall>=.80 and bal>=.80 and sum(x['duplicate'] for x in pc)/max(1,len(pc))<=.10 and sum(x['state_conditioned'] for x in pc)/max(1,len(pc))>=.70 else 'W5_CONTROLLER_OFFLINE_GATE_FAIL','n':len(preds),'gold_stop':len(stop),'gold_continue':len(cont),'auroc':auc,'stop_recall':stop_recall,'continue_recall':cont_recall,'balanced_accuracy':bal,'parse_valid_rate':sum(x['prediction']!='INVALID' for x in preds)/len(preds),'predicted_continue':len(pc),'duplicate_query_rate':sum(x['duplicate'] for x in pc)/max(1,len(pc)),'state_conditioned_query_rate':sum(x['state_conditioned'] for x in pc)/max(1,len(pc))}
 c.output_dir.mkdir(parents=True,exist_ok=True)
 with (c.output_dir/'predictions.jsonl').open('w') as f:
  for x in preds:f.write(json.dumps(x,ensure_ascii=False)+'\n')
 (c.output_dir/'summary.json').write_text(json.dumps(summary,indent=2)+'\n');print(json.dumps(summary,indent=2))
if __name__=='__main__':main()
