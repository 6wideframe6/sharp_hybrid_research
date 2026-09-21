#!/usr/bin/env python3
"""Evaluate C.3 outputs with the exact discrete objective used by integration.py."""
from __future__ import annotations
import argparse, csv, json
from pathlib import Path
import numpy as np
import sys
REPO_ROOT=Path(__file__).resolve().parents[1]; sys.path.insert(0,str(REPO_ROOT))
from k5.integration import pixel_vector_to_edge_targets, forward_edge_gradient

def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--c3-output',type=Path,required=True); ap.add_argument('--screenings',type=float,nargs='+',required=True); ap.add_argument('--output',type=Path,required=True); a=ap.parse_args()
    root=a.c3_output.resolve(); gx=np.load(root/'desired_gx.npy').astype(float); gy=np.load(root/'desired_gy.npy').astype(float)
    tx,ty=pixel_vector_to_edge_targets(gx,gy); denom=float(np.sum(tx*tx)+np.sum(ty*ty))
    if denom<=np.finfo(float).tiny: raise RuntimeError('empty desired edge field')
    rows=[]
    for lam in a.screenings:
        label=f'{lam:g}'.replace('.','p'); delta=np.load(root/f'delta_q_screening_{label}.npy').astype(float)
        ux,uy=forward_edge_gradient(delta); rx=ux-tx; ry=uy-ty
        rel=float(np.sqrt((np.sum(rx*rx)+np.sum(ry*ry))/denom))
        rows.append({'screening':lam,'relative_discrete_edge_l2_residual':rel})
    a.output.parent.mkdir(parents=True,exist_ok=True)
    with a.output.open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0].keys())); w.writeheader(); w.writerows(rows)
    print(json.dumps(rows,indent=2))
if __name__=='__main__': main()
