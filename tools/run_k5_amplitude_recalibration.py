#!/usr/bin/env python3
from __future__ import annotations
import argparse, json, sys
from pathlib import Path
import numpy as np
from PIL import Image
from scipy import ndimage as ndi
REPO_ROOT=Path(__file__).resolve().parents[1]; sys.path.insert(0,str(REPO_ROOT))
from k5.overlap import crop_to_native_box
from k5.sharp_fields import shared_full_context_gradients
from k5.types import NativeBox

def load_report(folder): return json.loads((folder/"alignment.json").read_text())
def load_context_array(folder,name,box):
    a=np.load(folder/name)
    if a.shape!=box.shape: raise ValueError(f"{folder/name}: {a.shape} != {box.shape}")
    return a
def map_wire_proxy(folder,target_box):
    meta=json.loads((folder/"metadata.json").read_text()); source_box=NativeBox.from_sequence(meta["box"])
    mask=np.asarray(Image.open(folder/"rgb_proxy_mask.png").convert("L"))>0
    out=np.zeros(target_box.shape,dtype=bool)
    try: common=source_box.intersect(target_box)
    except ValueError: return out
    sy,sx=source_box.local_slices(common); ty,tx=target_box.local_slices(common); out[ty,tx]=mask[sy,sx]
    return out
def stats(a):
    a=np.asarray(a,dtype=np.float64); a=a[np.isfinite(a)]
    if not len(a): return {"n":0}
    return {"n":int(len(a)),"mean":float(np.mean(a)),"median":float(np.median(a)),
            "p10":float(np.percentile(a,10)),"p25":float(np.percentile(a,25)),
            "p75":float(np.percentile(a,75)),"p90":float(np.percentile(a,90)),
            "p99":float(np.percentile(a,99)),"max":float(np.max(a))}
def cosine(dx,dy,gx,gy):
    gm=np.hypot(gx,gy); out=np.full(dx.shape,np.nan,dtype=np.float64); valid=gm>np.finfo(float).tiny
    out[valid]=(dx[valid]*gx[valid]+dy[valid]*gy[valid])/gm[valid]; return np.clip(out,-1.0,1.0)

def main():
    ap=argparse.ArgumentParser()
    for n in ("context-a","context-b","k5-output","wire-crop","output"): ap.add_argument(f"--{n}",type=Path,required=True)
    ap.add_argument("--fine-sigma",type=float,default=1.0); ap.add_argument("--coarse-sigma",type=float,default=2.0)
    ap.add_argument("--min-k5-confidence",type=float,default=0.025); ap.add_argument("--min-direction-cosine",type=float,default=0.7)
    ap.add_argument("--sharp-fine-percentile",type=float,default=70.0); ap.add_argument("--wire-exclusion-px",type=int,default=8)
    ap.add_argument("--border-px",type=int,default=0); args=ap.parse_args()
    output=args.output.resolve()
    if output.exists(): raise FileExistsError(f"Refusing to overwrite {output}")
    ca,cb=args.context_a.resolve(),args.context_b.resolve()
    ra,rb=load_report(ca),load_report(cb)
    ba=NativeBox.from_sequence(ra["provenance"]["box_native_half_open"]); bb=NativeBox.from_sequence(rb["provenance"]["box_native_half_open"])
    common=ba.intersect(bb)
    k5_root=args.k5_output.resolve(); ks=json.loads((k5_root/"summary.json").read_text())
    if NativeBox.from_sequence(ks["overlap_box_native"])!=common: raise ValueError("K5/context overlap mismatch")
    variant=k5_root/f"fine_{args.fine_sigma:g}_coarse_{args.coarse_sigma:g}"
    dx=np.load(variant/"direction_x.npy").astype(np.float64); dy=np.load(variant/"direction_y.npy").astype(np.float64)
    detail=np.load(variant/"detail_consensus.npy").astype(np.float64); confidence=np.load(variant/"final_confidence.npy").astype(np.float64)
    k5_valid=np.load(variant/"valid.npy").astype(bool)
    qa_full=load_context_array(ca,"sharp_visible_inverse_m.npy",ba); qb_full=load_context_array(cb,"sharp_visible_inverse_m.npy",bb)
    qa=crop_to_native_box(qa_full,ba,common); qb=crop_to_native_box(qb_full,bb,common); q=0.5*(qa+qb)
    f=shared_full_context_gradients(qa_full,ba,qb_full,bb,common,fine_sigma_px=args.fine_sigma,coarse_sigma_px=args.coarse_sigma)
    fx,fy=np.asarray(f["fine_x"]),np.asarray(f["fine_y"]); cx,cy=np.asarray(f["coarse_x"]),np.asarray(f["coarse_y"])
    sharp_safe=np.asarray(f["safe_mask"],dtype=bool); strict_halo=int(f["strict_halo_px"])
    fine_mag=np.hypot(fx,fy); coarse_mag=np.hypot(cx,cy)
    target=np.sqrt(np.maximum(fine_mag*fine_mag-coarse_mag*coarse_mag,0.0))
    dcos=cosine(dx,dy,fx,fy)
    finite=np.isfinite(q)&(q>0)&np.isfinite(detail)&np.isfinite(confidence)&np.isfinite(dcos)&np.isfinite(fine_mag)&np.isfinite(target)
    wire=map_wire_proxy(args.wire_crop.resolve(),common)
    excluded=ndi.binary_dilation(wire,iterations=args.wire_exclusion_px) if args.wire_exclusion_px>0 else wire.copy()
    border=np.ones(common.shape,dtype=bool)
    if args.border_px>0:
        b=args.border_px; border[:b]=False; border[-b:]=False; border[:,:b]=False; border[:,-b:]=False
    base=finite&sharp_safe&k5_valid&(~excluded)&border&(confidence>=args.min_k5_confidence)&(detail>0)&(dcos>=args.min_direction_cosine)
    if int(base.sum())<100: raise RuntimeError(f"Only {int(base.sum())} base pixels")
    threshold=float(np.percentile(fine_mag[base],args.sharp_fine_percentile))
    selected=base&(fine_mag>=threshold); positive=selected&(target>0)
    if int(positive.sum())<100: raise RuntimeError(f"Only {int(positive.sum())} positive calibration pixels")
    summary={
      "purpose":"K5-B.5b boundary-safe SHARP amplitude recalibration",
      "metric_target":"positive SHARP energy-excess magnitude","variant":variant.name,
      "overlap_box_native":[common.x0,common.y0,common.x1,common.y1],
      "config":{"fine_sigma":args.fine_sigma,"coarse_sigma":args.coarse_sigma,
                "min_k5_confidence":args.min_k5_confidence,"min_direction_cosine":args.min_direction_cosine,
                "sharp_fine_percentile":args.sharp_fine_percentile,"sharp_fine_threshold":threshold,
                "wire_exclusion_px":args.wire_exclusion_px,"border_px":args.border_px,"sharp_strict_halo_px":strict_halo},
      "support":{"overlap_pixels":int(common.width*common.height),"sharp_halo_safe_pixels":int(sharp_safe.sum()),
                 "sharp_halo_safe_fraction":float(sharp_safe.mean()),"base_pixels":int(base.sum()),
                 "selected_pixels":int(selected.sum()),"positive_calibration_pixels":int(positive.sum()),
                 "wire_pixels":int(wire.sum()),"target_positive":stats(target[positive]),
                 "fine_gradient_on_selected":stats(fine_mag[selected]),"direction_cosine_on_selected":stats(dcos[selected])},
      "guardrails":["SHARP filtering/gradients are computed on full contexts before overlap crop.",
                    "Both SHARP contexts must have complete Gaussian+gradient support.","K5-A.1b validity is required.",
                    "Wire pixels and exclusion radius are calibration exclusions only.","No correction is integrated."]}
    output.mkdir(parents=True,exist_ok=False)
    np.save(output/"sharp_halo_safe.npy",sharp_safe); np.save(output/"selected_support_mask.npy",selected)
    np.save(output/"positive_calibration_mask.npy",positive); np.save(output/"energy_excess_target.npy",target)
    np.save(output/"sharp_fine_magnitude.npy",fine_mag)
    (output/"summary.json").write_text(json.dumps(summary,indent=2)+"\n")
    print("K5-B.5b boundary-safe amplitude recalibration complete"); print(json.dumps(summary,indent=2))
if __name__=="__main__": main()
