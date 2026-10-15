"""Official-initialized 2x2. The step loop is the 2026-10-13 run.

Shared decode, metrics, and the held-out report live in vae8px.
Arms B and AB import PatchDiscWithContext from madebyollin/seraena
(https://github.com/madebyollin/seraena), MIT, Copyright (c) 2023 Ollin Boer Bohan.
The package is not vendored. Set SERAENA_PATH to a checkout of that repository.
"""
import argparse, concurrent.futures, gc, hashlib, json, os, random, signal, sys, time, traceback
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import lpips
import torch
import torch.nn.functional as F
from diffusers import AutoencoderKLQwenImage21
from torch.utils.tensorboard import SummaryWriter
from vae8px.corrupt import blurred, periodic
from vae8px.decode import decode_image
from vae8px.heldout import evaluate_available
from vae8px.io import atomic_json, read_rectangle, sample_online
from vae8px.metrics import horizontal_error, lattice_error, perceptual_patches
from vae8px.patchgan import NLayerDiscriminator
from vae8px.paths import PATHS
ROOT = PATHS.fourarm_run
ROOT.mkdir(parents=True, exist_ok=True)
SEED=2026101307
STOP=False

def stop(sig,frame):
 global STOP
 STOP=True
 print('STOP_REQUESTED',flush=True)

def digest(named):
 h=hashlib.sha256()
 for n,p in named:
  h.update(n.encode());h.update(p.detach().cpu().contiguous().numpy().tobytes())
 return h.hexdigest()

def sample(records,index):
 p=ROOT/'samples'/f'{index:07d}.json';p.parent.mkdir(exist_ok=True)
 if p.exists():
  meta=json.loads(p.read_text());x,t=read_rectangle(meta,meta['shape'],meta['crop_seed']);return dict(**meta,tensor=x,read_seconds=t,attempts=1)
 s=sample_online(records,index,SEED)
 atomic_json(p,{k:s[k] for k in ['source','shape','crop_seed','ratio']})
 return s

def patches(y,target,z,vae,index):
 h,w=y.shape[-2:];assert h//z.shape[-2]==16 and w//z.shape[-1]==16
 mu=torch.tensor(vae.config.latents_mean,device=z.device).view(1,64,1,1)
 sd=torch.tensor(vae.config.latents_std,device=z.device).view(1,64,1,1)
 ctx=(z[:,:,0]-mu)/sd
 rng=random.Random(SEED+index*101);fake=[];real=[];contexts=[]
 for size,n in [(256,4),(512,2)]:
  for _ in range(n):
   iy=rng.randrange((h-size)//16+1)*16;ix=rng.randrange((w-size)//16+1)*16
   fake.append(F.interpolate(y[:,:3,iy:iy+size,ix:ix+size],size=256,mode='area'))
   real.append(F.interpolate(target[:,:3,iy:iy+size,ix:ix+size],size=256,mode='area'))
   contexts.append(F.interpolate(ctx[:,:,iy//16:(iy+size)//16,ix//16:(ix+size)//16],size=16,mode='area'))
 fake.append(F.interpolate(y[:,:3],size=256,mode='area'));real.append(F.interpolate(target[:,:3],size=256,mode='area'));contexts.append(F.interpolate(ctx,size=16,mode='area'))
 return torch.cat(fake),torch.cat(real),torch.cat(contexts).detach()

def patch_disc_with_context():
 path = os.environ.get("SERAENA_PATH")
 if not path:
  raise RuntimeError("set SERAENA_PATH to a checkout of https://github.com/madebyollin/seraena")
 if path not in sys.path:
  sys.path.insert(0, path)
 from seraena import PatchDiscWithContext
 return PatchDiscWithContext

class Critic:
 def __init__(self,conditional):
  self.conditional=conditional;torch.manual_seed(SEED+99)
  if conditional:self.d=patch_disc_with_context()(3,64).cuda()
  else:
   self.d=NLayerDiscriminator(input_nc=3,ndf=64,n_layers=3,use_actnorm=False).cuda().eval()
   for i,m in enumerate(self.d.main):
    if isinstance(m,torch.nn.BatchNorm2d):
     self.d.main[i-1]=torch.nn.utils.fuse_conv_bn_eval(self.d.main[i-1],m);self.d.main[i]=torch.nn.Identity()
  self.opt=torch.optim.Adam(self.d.parameters(),lr=2e-5,betas=(.5,.9))
  self.buffer=[];self.buffer_limit=1024
  self.initial_digest=digest(self.d.named_parameters())
 def update(self,fake,real,ctx):
  self.d.train().requires_grad_(True);self.opt.zero_grad(set_to_none=True)
  if self.conditional:
   f=(fake.detach()+1)/2;r=(real.detach()+1)/2;n=len(f);k=min(n//2,len(self.buffer))
   ff=f.clone();cc=ctx.clone()
   if k:
    picked=random.choices(self.buffer,k=k)
    ff[-k:]=torch.stack([x[0] for x in picked]).to(f.device,dtype=f.dtype)
    cc[-k:]=torch.stack([x[1] for x in picked]).to(ctx.device,dtype=ctx.dtype)
   mask=torch.rand(n,1,1,1,device=f.device)<.5
   scores=self.d(torch.where(mask,ff,r),torch.where(mask,cc,ctx));targets=mask.float()*2-1
   loss=(scores-targets).square().mean()
   for fi,ci in zip(f,ctx):
    pair=(fi.detach().to(device='cpu',dtype=torch.float16),ci.detach().to(device='cpu',dtype=torch.float16))
    if len(self.buffer)<self.buffer_limit:self.buffer.append(pair)
    else:self.buffer[random.randrange(self.buffer_limit)]=pair
   info={'d_loss':float(loss.detach()),'replay_pairs':len(self.buffer)}
  else:
   dr,df,dc,db=self.d(torch.cat([real.detach(),fake.detach(),periodic(real.detach()),blurred(real.detach())])).chunk(4)
   adv=.5*(F.relu(1-dr).mean()+F.relu(1+df).mean());rank=F.relu(.2-dr+dc).mean();blur=F.relu(.2-dr+db).mean();loss=adv+rank+blur
   info=dict(d_loss=float(loss.detach()),d_real=float(dr.detach().mean()),d_fake=float(df.detach().mean()),d_periodic_preference=float((dr.detach().mean((1,2,3))>dc.detach().mean((1,2,3))).float().mean()))
  assert torch.isfinite(loss)
  loss.backward();gn=torch.nn.utils.clip_grad_norm_(self.d.parameters(),10,error_if_nonfinite=True);self.opt.step();self.opt.zero_grad(set_to_none=True)
  self.d.eval().requires_grad_(False);info['d_grad_norm']=float(gn)
  return info
 def generator_loss(self,fake,real,ctx):
  if not self.conditional:return -self.d(fake).mean(),{}
  f=(fake+1)/2;r=(real.detach()+1)/2;probe=f.detach().requires_grad_(True)
  with torch.no_grad():ref=self.d(r,ctx)
  score_loss=(self.d(probe,ctx)-ref).square().mean((1,2,3)).sum()
  correction=-torch.autograd.grad(score_loss,probe)[0].detach()
  std=correction.std(correction=0);correction=correction/(std+1e-5)
  target=f.detach()+correction
  return F.mse_loss(f,target),dict(correction_std=float(std),correction_rms=float(correction.square().mean().sqrt()),paired_score_error=float(score_loss.detach()))
 def state(self):return dict(model=self.d.state_dict(),optimizer=self.opt.state_dict(),buffer=self.buffer,initial_digest=self.initial_digest)
 def load(self,s):
  self.d.load_state_dict(s['model']);self.opt.load_state_dict(s['optimizer']);self.buffer=s['buffer'];self.initial_digest=s['initial_digest']


def save(run,vae,opt,critic,step,warmup,config):
 path=run/f'checkpoint_{step:06d}.pt';tmp=path.with_suffix('.pt.tmp')
 state=dict(step=step,warmup=warmup,decoder_state={n:p.detach().cpu() for n,p in vae.named_parameters() if p.requires_grad},optimizer=opt.state_dict(),critic=critic.state(),config=config,torch_rng=torch.get_rng_state(),cuda_rng=torch.cuda.get_rng_state(),python_rng=random.getstate())
 torch.save(state,tmp);os.replace(tmp,path)
 atomic_json(run/'latest_checkpoint.json',dict(step=step,path=str(path)))
 for p in run.glob('checkpoint_*.pt'):
  st=int(p.stem.split('_')[-1])
  if st!=step and st%1000!=0 and st!=1:p.unlink()
 print('CHECKPOINT',step,flush=True)


def evaluate(run,vae,net,config,writer,step,critic):
 gc.collect();torch.cuda.empty_cache();atomic_json(run/'status.json',dict(state='evaluating',step=step))
 evaluate_available(vae,net,config,writer,step,run)
 # Independent matched heldout patches: report both high-score-real conventions correctly.
 ds=[]
 for case in config['validation'][:2]:
  with torch.no_grad():
   x,_=read_rectangle(case,case['shape'],case['crop_seed']);x=x.cuda();z=vae.encode(x).latent_dist.mode();y=decode_image(vae,z)[:,:,0].float();f,r,c=patches(y,x[:,:,0],z,vae,990001)
   if critic.conditional:
    a=(r+1)/2;raw=critic.d(a,c);score=-raw.mean((1,2,3));pred=-critic.d((f+1)/2,c).mean((1,2,3));wrong=-critic.d(a,c.roll(1,0)).mean((1,2,3))
   else:score=critic.d(r).mean((1,2,3));pred=critic.d(f).mean((1,2,3));wrong=score
   row={'real_over_reconstruction':float((score>pred).float().mean()),'context_margin':float((score-wrong).mean())}
   for period in [2,4,8,16]:
    bad=periodic(r,period,4)
    scorebad=(-critic.d((bad+1)/2,c) if critic.conditional else critic.d(bad)).mean((1,2,3))
    row[f'p{period}_preference']=float((score>scorebad).float().mean())
   ds.append(row);del x,z,y,f,r,c
 atomic_json(run/f'eval_{step:06d}'/'critic_metrics.json',ds)
 for k in ds[0]:writer.add_scalar('critic_heldout/'+k,sum(d[k] for d in ds)/len(ds),step)
 writer.flush();gc.collect();torch.cuda.empty_cache()


def main(args):
 run=ROOT/('_smoke' if args.smoke else 'arms')/args.arm;run.mkdir(parents=True,exist_ok=True)
 signal.signal(signal.SIGTERM,stop);signal.signal(signal.SIGINT,stop)
 torch.set_num_threads(4);torch.cuda.set_per_process_memory_fraction(.85)
 torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
 torch.manual_seed(SEED);random.seed(SEED)
 protocol=json.loads((Path(__file__).resolve().parent/'protocol.json').read_text());names=protocol['trainable_names']
 config=dict(arm=args.arm,source=str(PATHS.official_vae),initialization='official VAE only; fresh G/D optimizers; no learned checkpoint at step0',trainable_names=names,seed=SEED,steps=args.steps,lr=5e-5,d_lr=2e-5,warmup_d=100 if not args.smoke else 1,validation=protocol['validation'],generated=protocol['generated'],conditional=args.arm in ('B','AB'),pixel_weight=.1 if args.arm in ('A','AB') else 1.,lpips_weight=.005 if args.arm in ('A','AB') else .05,coarse_weight=.25,phase_reference='unweakened original base; coefficient cap20',gan_reference='unweakened original base + same phase; output norm target10percent; no scalar cap (score scale differs between critics)',patches='4 native256, 2 resized512, 1 global256; all source origins factor16 aligned',replay_cpu_limit=1024,checkpoint_interval=100,evaluation_interval=250,quality_early_stop=False,upstream_commit='fb9a363889d6ba4d3699b121483189a3904812bb')
 vae=AutoencoderKLQwenImage21.from_pretrained(str(PATHS.official_vae),torch_dtype=torch.float32).cuda().eval().requires_grad_(False)
 named=dict(vae.named_parameters());params=[named[n] for n in names]
 for p in params:p.requires_grad_(True)
 config['active_parameters']=sum(p.numel() for p in params);config['official_trainable_sha256']=digest((n,named[n]) for n in names)
 frozen={n:p._version for n,p in vae.named_parameters() if not p.requires_grad}
 opt=torch.optim.AdamW(params,lr=5e-5,betas=(.9,.99),weight_decay=0,foreach=False);critic=Critic(config['conditional'])
 config['critic_initial_sha256']=critic.initial_digest
 start=0;warmup=0;checkpoints=sorted(run.glob('checkpoint_*.pt'))
 if checkpoints:
  state=torch.load(checkpoints[-1],map_location='cpu',weights_only=True,mmap=True);vae.load_state_dict(state['decoder_state'],strict=False);opt.load_state_dict(state['optimizer']);critic.load(state['critic']);start=state['step'];warmup=state['warmup'];torch.set_rng_state(state['torch_rng']);torch.cuda.set_rng_state(state['cuda_rng']);random.setstate(state['python_rng']);del state
 atomic_json(run/'config.json',config)
 if start>=args.steps:atomic_json(run/'status.json',dict(state='complete',step=start));return
 net=lpips.LPIPS(net='alex').cuda().eval().requires_grad_(False);writer=SummaryWriter(str(run/'events'),flush_secs=5,purge_step=start+1 if checkpoints else None)
 writer.add_text('config',json.dumps(config,indent=2))
 if not args.smoke:
  baseline=ROOT/'baseline'
  if not (baseline/'eval_000000/metrics.json').exists():
   baseline.mkdir(exist_ok=True);evaluate_available(vae,net,config,writer,0,baseline)
  if not (run/'eval_000000').exists():(run/'eval_000000').symlink_to(baseline/'eval_000000',target_is_directory=True)
 records=[json.loads(l) for l in (PATHS.photo_index/'train.jsonl').open()]
 for wi in range(warmup,config['warmup_d']):
  s=sample(records,1000000+wi);x=s.pop('tensor').cuda()
  with torch.no_grad():
   z=vae.encode(x).latent_dist.mode()
   with torch.autocast('cuda',dtype=torch.bfloat16):y=decode_image(vae,z)[:,:,0].float()
   f,r,c=patches(y,x[:,:,0],z,vae,1000000+wi)
  info=critic.update(f,r,c);warmup=wi+1
  atomic_json(run/'status.json',dict(state='critic_warmup',warmup=warmup,target=config['warmup_d'],**info))
  if warmup%10==0:print('WARMUP',warmup,info,flush=True)
  del x,z,y,f,r,c;gc.collect();torch.cuda.empty_cache()
  if STOP:save(run,vae,opt,critic,start,warmup,config);return
 if not checkpoints:save(run,vae,opt,critic,0,warmup,config)
 pool=concurrent.futures.ThreadPoolExecutor(max_workers=2);futures={i:pool.submit(sample,records,i) for i in range(start,min(start+2,args.steps))};prev=None;step=start
 with (run/'metrics.jsonl').open('a',buffering=1) as log:
  for step in range(start+1,args.steps+1):
   t=time.perf_counter();gc.collect();s=futures.pop(step-1).result();wait=time.perf_counter()-t
   if step+1<args.steps:futures[step+1]=pool.submit(sample,records,step+1)
   if tuple(s['shape'])!=prev:torch.cuda.empty_cache()
   prev=tuple(s['shape']);torch.cuda.reset_peak_memory_stats();x=s.pop('tensor').cuda();opt.zero_grad(set_to_none=True)
   ramp=min(1.,step/50)
   for g in opt.param_groups:g['lr']=5e-5*ramp
   with torch.no_grad():z=vae.encode(x).latent_dist.mode()
   with torch.autocast('cuda',dtype=torch.bfloat16):y=decode_image(vae,z)[:,:,0].float()
   target=x[:,:,0];e=y[:,:3]-target[:,:3];mse=e.square().mean();per=perceptual_patches(net,y[:,:3],target[:,:3]);coarse=F.avg_pool2d(e,8).square().mean();h8=horizontal_error(e);alpha=(y[:,3]-target[:,3]).square().mean()
   reference=mse+.05*per+.25*coarse+20*h8+.1*alpha;phase=lattice_error(e)
   bg=torch.autograd.grad(reference,y,retain_graph=True)[0];pg=torch.autograd.grad(phase,y,retain_graph=True)[0];phase_coef=(.5*bg.norm()/pg.norm().clamp_min(1e-12)).clamp(max=20).detach();del bg,pg
   full_reference=reference+phase_coef*phase
   reconstruction=config['pixel_weight']*mse+config['lpips_weight']*per+.25*coarse+20*h8+.1*alpha+phase_coef*phase
   f,r,c=patches(y,target,z,vae,step-1);info=critic.update(f,r,c);gl,ginfo=critic.generator_loss(f,r,c);info.update(ginfo)
   bg=torch.autograd.grad(full_reference,y,retain_graph=True)[0];ag=torch.autograd.grad(gl,y,retain_graph=True)[0];bn=bg.norm();an=ag.norm();gan_coef=(.1*ramp*bn/an.clamp_min(1e-12)).detach()
   info.update(gan_weight=float(gan_coef),gan_output_ratio=float(gan_coef*an/bn.clamp_min(1e-12)),gan_reference_cosine=float((bg*ag).sum()/(bn*an).clamp_min(1e-20)));del bg,ag
   loss=reconstruction+gan_coef*gl;assert torch.isfinite(loss);loss.backward();gn=torch.nn.utils.clip_grad_norm_(params,1,error_if_nonfinite=True);opt.step();torch.cuda.synchronize()
   row=dict(step=step,**s,**info,loss=float(loss.detach()),mse=float(mse.detach()),lpips=float(per.detach()),coarse=float(coarse.detach()),phase=float(phase.detach()),phase_weight=float(phase_coef),horizontal8=float(h8.detach()),gan_loss=float(gl.detach()),grad_norm=float(gn),lr=opt.param_groups[0]['lr'],wall_seconds=time.perf_counter()-t,data_wait_seconds=wait,peak_allocated_gib=torch.cuda.max_memory_allocated()/2**30,peak_reserved_gib=torch.cuda.max_memory_reserved()/2**30)
   log.write(json.dumps(row,ensure_ascii=False)+'\n');atomic_json(run/'status.json',dict(state='training',target=args.steps,**row));print('STEP',json.dumps(row),flush=True)
   for k,v in row.items():
    if isinstance(v,(int,float)):writer.add_scalar('train/'+k,v,step)
   del x,z,y,target,e,mse,per,coarse,h8,alpha,reference,phase,phase_coef,full_reference,reconstruction,f,r,c,gl,gan_coef,loss;opt.zero_grad(set_to_none=True)
   if step%100==0 or step%250==0 or step==args.steps or STOP:save(run,vae,opt,critic,step,warmup,config)
   if not args.smoke and (step%250==0 or step==args.steps) and not STOP:evaluate(run,vae,net,config,writer,step,critic);prev=None
   if STOP:break
 assert all(p._version==frozen[n] for n,p in vae.named_parameters() if not p.requires_grad)
 atomic_json(run/'status.json',dict(state='stopped' if STOP else 'complete',step=step,frozen_parameters_unchanged=True));writer.close();pool.shutdown(wait=True,cancel_futures=True)

if __name__=='__main__':
 parser=argparse.ArgumentParser();parser.add_argument('--arm',choices=['C','A','B','AB'],required=True);parser.add_argument('--steps',type=int,default=5000);parser.add_argument('--smoke',action='store_true');args=parser.parse_args()
 try:main(args)
 except Exception:
  run=ROOT/('_smoke' if args.smoke else 'arms')/args.arm;run.mkdir(parents=True,exist_ok=True);atomic_json(run/'failure.json',dict(traceback=traceback.format_exc()));atomic_json(run/'status.json',dict(state='failed'));raise
