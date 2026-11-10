"""Numerical contract audit required before the additional training run."""
import sys,json,torch
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from side_information import SideInformation
from simscore_make import SimScoreMAKE
from feature_store import FIELDS
torch.set_num_threads(2)
sizes={f:20 for f in FIELDS}
torch.manual_seed(42);a=SideInformation(sizes,mode='simscore')
torch.manual_seed(42);b=SimScoreMAKE(sizes)
f=torch.randint(2,20,(3,10));i=torch.randint(2,20,(3,8));c=i.clone();v=torch.ones_like(i,dtype=torch.bool);v[0]=False
q=torch.randn(3,128);h=torch.randn(3,8,128);qa=torch.tensor([True,False,True]);ha=v.clone();args=(f,i,c,v,q,h,qa,ha)
with torch.no_grad():
 a.side_projection.weight.normal_();b.side_projection.weight.copy_(a.side_projection.weight)
 assert torch.equal(a(*args),b(*args))
before={k:p.clone() for k,p in b.content_branch.state_dict().items()}
opt=torch.optim.AdamW([p for n,p in b.named_parameters() if p.requires_grad and not n.startswith('embeddings.')],lr=.001)
for _ in range(2):
 opt.zero_grad();b(*args).square().mean().backward()
 assert b.side_projection.weight.grad.abs().sum()>0
 assert b.content_projection.weight.grad.abs().sum()>0
 opt.step()
assert all(p.grad is None for p in b.content_branch.parameters())
assert all(torch.equal(before[k],p) for k,p in b.content_branch.state_dict().items())
d=SimScoreMAKE(sizes);d.load_state_dict(b.state_dict());assert torch.equal(b(*args),d(*args))
assert torch.isfinite(b(*args)).all()
result={'baseline_equivalence':True,'simscore_and_make_projection_gradients':True,'make_frozen':True,'reload_exact':True,'empty_history_and_missing_content_finite':True}
Path(sys.argv[1]).write_text(json.dumps(result,indent=2));print(json.dumps(result))
