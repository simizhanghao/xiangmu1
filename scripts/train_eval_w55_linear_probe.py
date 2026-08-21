#!/usr/bin/env python3
"""Train one weighted linear probe, calibrate once, evaluate frozen dev once."""
import json
from pathlib import Path
import torch

ROOT=Path('results/79_w55_linear_probe');data=torch.load(ROOT/'hidden.pt',map_location='cpu',weights_only=False)
x=data['hidden'];y=data['labels'];splits=data['splits'];idx={s:torch.tensor([i for i,v in enumerate(splits) if v==s]) for s in ('train','calib','dev')}
torch.manual_seed(42);device='cuda' if torch.cuda.is_available() else 'cpu';x=x.to(device);y=y.to(device)
head=torch.nn.Linear(x.shape[1],2).to(device);yt=y[idx['train']];counts=torch.bincount(yt,minlength=2).float();weights=(len(yt)/(2*counts)).to(device)
opt=torch.optim.LBFGS(head.parameters(),lr=1.0,max_iter=100,line_search_fn='strong_wolfe')
def closure():
 opt.zero_grad();loss=torch.nn.functional.cross_entropy(head(x[idx['train']]),yt,weight=weights);loss.backward();return loss
initial_loss=float(opt.step(closure).detach().cpu());head.eval()
with torch.inference_mode():
 final_train_loss=float(torch.nn.functional.cross_entropy(head(x[idx['train']]),yt,weight=weights).cpu())
with torch.inference_mode():scores=torch.softmax(head(x),dim=-1)[:,1].cpu()
def recalls(ids,t):
 yy=y[ids].cpu();ss=scores[ids];pred=ss>=t
 return float((~pred[yy==0]).float().mean()),float(pred[yy==1].float().mean())
cal=idx['calib'];candidates=sorted(set(float(v) for v in scores[cal]));best=None
for t in candidates:
 sr,cr=recalls(cal,t);item=(min(sr,cr),(sr+cr)/2,-abs(t-.5),t,sr,cr)
 if best is None or item>best:best=item
threshold=best[3]
def auc_rank(ids):
 yy=y[ids].cpu().tolist();ss=scores[ids].tolist();order=sorted(range(len(ss)),key=lambda i:ss[i]);ranks=[0.]*len(ss);i=0
 while i<len(order):
  j=i+1
  while j<len(order) and ss[order[j]]==ss[order[i]]:j+=1
  r=(i+1+j)/2
  for k in range(i,j):ranks[order[k]]=r
  i=j
 n1=sum(yy);n0=len(yy)-n1;return (sum(r for r,v in zip(ranks,yy) if v)-n1*(n1+1)/2)/(n1*n0)
dev=idx['dev'];sr,cr=recalls(dev,threshold);auc=auc_rank(dev);ba=(sr+cr)/2
summary={'gate':'W55_LINEAR_PROBE_PASS' if auc>=.90 and sr>=.80 and cr>=.80 and ba>=.80 else 'W55_LINEAR_PROBE_FAIL','initial_train_loss':initial_loss,'final_weighted_train_loss':final_train_loss,'train_n':len(idx['train']),'calib_n':len(cal),'dev_n':len(dev),'threshold':threshold,'calib_stop_recall':best[4],'calib_continue_recall':best[5],'auroc':auc,'stop_recall':sr,'continue_recall':cr,'balanced_accuracy':ba,'parse_valid_rate':1.0,'linear_only':True,'backbone_updated':False,'api_calls':0}
torch.save({'state_dict':head.cpu().state_dict(),'threshold':threshold,'hidden_dim':x.shape[1],'summary':summary},ROOT/'linear_head.pt');(ROOT/'summary.json').write_text(json.dumps(summary,indent=2)+'\n');print(json.dumps(summary,indent=2))
