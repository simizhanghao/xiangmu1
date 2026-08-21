#!/usr/bin/env python3
"""Extract frozen W5 Controller Decision-prefix hidden vectors."""
import argparse,hashlib,json
from pathlib import Path
import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM,AutoTokenizer

SYSTEM='''You are a research controller. Given the question, current Web observation, ResearchMemory, previous queries, and remaining budget, output exactly one action. If evidence is sufficient: DECISION: STOP. Otherwise output DECISION: CONTINUE, one specific MISSING fact, and one concise non-duplicate QUERY grounded in the current state. Do not answer the research question.'''
def main():
 p=argparse.ArgumentParser();p.add_argument('--base',default='model/Qwen3-1.7B');p.add_argument('--adapter',default='outputs/77_w5_controller_lora');p.add_argument('--split',default='results/79_w55_linear_probe/split.jsonl');p.add_argument('--output',default='results/79_w55_linear_probe/hidden.pt');p.add_argument('--batch-size',type=int,default=4);c=p.parse_args()
 rows=[json.loads(x) for x in Path(c.split).open()]
 tok=AutoTokenizer.from_pretrained(c.base,trust_remote_code=True);tok.padding_side='right';tok.pad_token=tok.eos_token
 model=AutoModelForCausalLM.from_pretrained(c.base,dtype=torch.bfloat16,device_map='cuda',trust_remote_code=True);model=PeftModel.from_pretrained(model,c.adapter);model.eval()
 vectors=[];max_len=0;truncated=0;prefix_ids=tok.encode('DECISION:',add_special_tokens=False)
 for start in range(0,len(rows),c.batch_size):
  batch=rows[start:start+c.batch_size]
  prompts=[tok.apply_chat_template([{'role':'system','content':SYSTEM},{'role':'user','content':x['controller_input']}],tokenize=False,add_generation_prompt=True,enable_thinking=False) for x in batch]
  prompt_ids=tok(prompts,add_special_tokens=False)['input_ids'];lengths=[len(x)+len(prefix_ids) for x in prompt_ids];max_len=max(max_len,max(lengths));truncated+=sum(x>4096 for x in lengths)
  # LlamaFactory supervised preprocessing keeps the source prefix (source_ids[:N]).
  # Preserve the decision marker explicitly so every extracted vector is at the
  # same classification position rather than at an arbitrary truncated token.
  seqs=[ids[:4096-len(prefix_ids)]+prefix_ids for ids in prompt_ids]
  enc=tok.pad({'input_ids':seqs},padding=True,return_tensors='pt').to(model.device);last_idx=enc.attention_mask.sum(dim=1)-1
  with torch.inference_mode():out=model(**enc,output_hidden_states=True,use_cache=False,return_dict=True)
  h=out.hidden_states[-1][torch.arange(len(batch),device=model.device),last_idx]
  vectors.append(h.float().cpu());print(f'[{min(start+c.batch_size,len(rows))}/{len(rows)}]',flush=True)
 payload={'hidden':torch.cat(vectors),'labels':torch.tensor([int(x['decision']=='CONTINUE') for x in rows]),'state_ids':[x['state_id'] for x in rows],'sample_ids':[x['sample_id'] for x in rows],'splits':[x['probe_split'] for x in rows],'system_sha256':hashlib.sha256(SYSTEM.encode()).hexdigest(),'decision_prefix':'DECISION:','padding_side':'right','max_raw_prompt_tokens':max_len,'truncated_prompts':truncated,'truncation_policy':'keep_source_prefix_then_append_decision_prefix','backbone_frozen':True,'adapter_frozen':True}
 Path(c.output).parent.mkdir(parents=True,exist_ok=True);torch.save(payload,c.output)
 summary={'gate':'W55_HIDDEN_EXTRACTION_PASS','vectors':len(rows),'hidden_dim':payload['hidden'].shape[1],'max_raw_prompt_tokens':max_len,'truncated_prompts':truncated,'truncation_policy':payload['truncation_policy'],'padding_side':'right','decision_prefix':'DECISION:','output':c.output}
 Path(c.output).with_suffix('.summary.json').write_text(json.dumps(summary,indent=2)+'\n');print(json.dumps(summary,indent=2))
if __name__=='__main__':main()
