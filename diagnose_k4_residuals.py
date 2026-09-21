#!/usr/bin/env python3
from __future__ import annotations
import argparse, csv, json, math, sys
from pathlib import Path
import numpy as np
from PIL import Image
from scipy import ndimage as ndi
from scipy.stats import spearmanr

ROOT = Path(__file__).resolve().parents[2] if 'experiments' in str(Path(__file__).resolve()) else Path('/mnt/c/depth_issue')
sys.path.insert(0, str(ROOT))
from experiments.sharp_hybrid_research.alignment import AlignmentConfig, _spread, restricted_smoothing, robust_affine, stratify_anchors

DEFAULT_INPUT = ROOT / 'results/sharp_hybrid/k4_context/far_1480_512'
DEFAULT_OUTPUT = ROOT / 'results/sharp_hybrid/k4_residual_diagnostics/far_1480_512'

def stats(v):
    v=np.asarray(v,float); v=v[np.isfinite(v)]
    if not len(v): return {'n':0,'median':None,'p90':None,'mean':None,'max':None,'gt_0.10':None,'gt_0.25':None}
    return {'n':int(len(v)),'median':float(np.median(v)),'p90':float(np.percentile(v,90)),
            'mean':float(np.mean(v)),'max':float(np.max(v)),
            'gt_0.10':float(np.mean(v>0.10)),'gt_0.25':float(np.mean(v>0.25))}

def group(rows,key):
    d={}
    for r in rows: d.setdefault(r[key],[]).append(r['err'])
    out=[]
    for k,v in d.items(): out.append({key:int(k) if isinstance(k,(int,np.integer)) else k, **stats(v)})
    return sorted(out,key=lambda r:-(r['p90'] if r['p90'] is not None else -1))

def corr(rows,key):
    a=np.array([r['err'] for r in rows]); b=np.array([r[key] for r in rows])
    m=np.isfinite(a)&np.isfinite(b)
    if m.sum()<3: return None
    z=spearmanr(a[m],b[m])
    return {'rho':float(z.statistic),'p':float(z.pvalue),'n':int(m.sum())}

def write_csv(path,rows,fields):
    with path.open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=fields); w.writeheader(); w.writerows(rows)

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--input',type=Path,default=DEFAULT_INPUT)
    ap.add_argument('--output',type=Path,default=DEFAULT_OUTPUT)
    a=ap.parse_args(); src=a.input; out=a.output
    if out.exists(): raise FileExistsError(out)
    rep=json.loads((src/'alignment.json').read_text())
    cfg=AlignmentConfig(**rep['config'])
    raw=np.load(src/'raw_tinyvim_relative.npy').astype(float)
    tgt=np.load(src/'sharp_visible_inverse_m.npy').astype(float)
    rf=np.load(src/'tinyvim_common_bandwidth.npy').astype(float)
    tf=np.load(src/'sharp_common_bandwidth.npy').astype(float)
    reg=np.load(src/'fitting_regions.npy')
    saved=np.load(src/'anchor_mask.npy').astype(bool)
    cand=np.array(Image.open(src/'anchor_candidates.png').convert('L'))>0
    valid=cand & np.isfinite(raw)&np.isfinite(tgt)&(tgt>0)&np.isfinite(rf)&np.isfinite(tf)&(tf>0)&(reg>0)
    sample=stratify_anchors(tf,valid,reg,cfg)
    idx,w,blocks,bins,sregs=(sample[k] for k in ('indices','weights','blocks','bins','regions'))
    recreated=np.zeros_like(cand); recreated.flat[idx]=True
    if not np.array_equal(recreated,saved): raise RuntimeError(f'selected mask mismatch: {(recreated^saved).sum()} px')
    xfull,yfull=rf.flat[idx],tf.flat[idx]; yspan=_spread(yfull)
    ub=np.unique(blocks); np.random.default_rng(cfg.seed).shuffle(ub)
    yy,xx=np.mgrid[:raw.shape[0],:raw.shape[1]]; ncols=math.ceil(raw.shape[1]/cfg.block_size)
    blockmap=(yy//cfg.block_size)*ncols + xx//cfg.block_size
    # distance to connected fitting-interior boundary
    rdist=np.zeros(raw.shape,float)
    for rid in np.unique(reg):
        if rid<=0: continue
        m=reg==rid; d=ndi.distance_transform_edt(m); rdist[m]=d[m]
    rows=[]; cur_all=[]; fixed_all=[]; folds=[]; fixed_folds=[]
    for fi,vblocks in enumerate(np.array_split(ub,cfg.folds)):
        val=np.isin(blocks,vblocks); tr=~val; vdomain=np.isin(blockmap,vblocks)
        xf=np.full_like(xfull,np.nan); yf=np.full_like(yfull,np.nan); mass=np.full_like(xfull,np.nan)
        for subset,domain in ((tr.copy(),~vdomain),(val.copy(),vdomain)):
            rsm,rm=restricted_smoothing(raw,valid & domain,reg,cfg)
            tsm,tm=restricted_smoothing(tgt,valid & domain,reg,cfg)
            sm=np.minimum(rm.flat[idx],tm.flat[idx])
            sup=(rm.flat[idx]>=cfg.min_kernel_mass)&(tm.flat[idx]>=cfg.min_kernel_mass)
            use=subset & sup
            xf[use]=rsm.flat[idx[use]]; yf[use]=tsm.flat[idx[use]]; mass[use]=sm[use]
        tr2=tr & np.isfinite(xf)&np.isfinite(yf); va2=val & np.isfinite(xf)&np.isfinite(yf)
        fa,fb=robust_affine(xf[tr2],yf[tr2],w[tr2],cfg)
        e=np.abs(fa*xf[va2]+fb-yf[va2])/yspan; cur_all.extend(e.tolist())
        folds.append({'fold':fi,'a':fa,'b':fb,'train':int(tr2.sum()),'val':int(va2.sum()),**stats(e)})
        va_pos=np.flatnonzero(va2)
        for j,si in enumerate(va_pos):
            flat=int(idx[si]); y0,x0=divmod(flat,raw.shape[1])
            rows.append({'fold':fi,'x':x0,'y':y0,'block':int(blocks[si]),'region':int(sregs[si]),'depth_bin':int(bins[si]),
                         'err':float(e[j]),'kernel_mass':float(mass[si]),'region_boundary_dist':float(rdist[y0,x0]),
                         'target':float(yf[si]),'raw_fit':float(xf[si])})
        # fixed full-context smoothing control: same held-out blocks, no validation point in affine fit
        ffa,ffb=robust_affine(xfull[tr],yfull[tr],w[tr],cfg)
        ef=np.abs(ffa*xfull[val]+ffb-yfull[val])/yspan; fixed_all.extend(ef.tolist())
        fixed_folds.append({'fold':fi,'a':ffa,'b':ffb,'train':int(tr.sum()),'val':int(val.sum()),**stats(ef)})
    cur=stats(cur_all); fixed=stats(fixed_all)
    if not np.isclose(cur['median'],rep['heldout_normalized_median'],rtol=1e-9,atol=1e-12): raise RuntimeError((cur['median'],rep['heldout_normalized_median']))
    if not np.isclose(cur['p90'],rep['heldout_normalized_p90'],rtol=1e-9,atol=1e-12): raise RuntimeError((cur['p90'],rep['heldout_normalized_p90']))
    out.mkdir(parents=True)
    by_block=group(rows,'block'); by_region=group(rows,'region'); by_depth=group(rows,'depth_bin')
    # kernel-mass quantiles, useful because sigma=24 vs 32px holdout blocks can shrink support
    masses=np.array([r['kernel_mass'] for r in rows]); q=np.quantile(masses,[0,.1,.25,.5,.75,.9,1])
    for r in rows: r['mass_quantile_bin']=int(np.searchsorted(q[1:-1],r['kernel_mass'],side='right'))
    by_mass=group(rows,'mass_quantile_bin')
    summary={'source':str(src),'reported':{'median':rep['heldout_normalized_median'],'p90':rep['heldout_normalized_p90'],'a':rep['a'],'b':rep['b']},
             'reproduced_disjoint_kernel_crossfit':cur,'reproduced_folds':folds,
             'fixed_full_context_smoothing_control_DIAGNOSTIC_ONLY':fixed,'fixed_folds':fixed_folds,
             'candidate_count':int(cand.sum()),'selected_count':int(saved.sum()),'yspan':yspan,
             'kernel_mass_quantile_edges':q.tolist(),'by_kernel_mass_quantile':by_mass,
             'by_region':by_region,'by_depth_bin':by_depth,'worst_20_blocks':by_block[:20],
             'spearman':{'err_vs_kernel_mass':corr(rows,'kernel_mass'),'err_vs_region_boundary_dist':corr(rows,'region_boundary_dist'),
                         'err_vs_x':corr(rows,'x'),'err_vs_y':corr(rows,'y'),'err_vs_target':corr(rows,'target'),'err_vs_raw_fit':corr(rows,'raw_fit')},
             'interpretation_guardrails':['fixed-smoothing control is diagnostic only','fitting_regions are connected interiors, not semantic surfaces','no inference/fusion/K5 performed']}
    (out/'summary.json').write_text(json.dumps(summary,indent=2)+'\n')
    write_csv(out/'worst_blocks.csv',by_block,['block','n','median','p90','mean','max','gt_0.10','gt_0.25'])
    write_csv(out/'regions.csv',by_region,['region','n','median','p90','mean','max','gt_0.10','gt_0.25'])
    write_csv(out/'depth_bins.csv',by_depth,['depth_bin','n','median','p90','mean','max','gt_0.10','gt_0.25'])
    write_csv(out/'kernel_mass_quantiles.csv',by_mass,['mass_quantile_bin','n','median','p90','mean','max','gt_0.10','gt_0.25'])
    print(json.dumps(summary,indent=2))

if __name__=='__main__': main()
