from __future__ import annotations
import math
import numpy as np
from scipy import ndimage as ndi
from .overlap import crop_to_native_box, native_halo_support_mask
from .types import NativeBox

def gaussian_gradient_halo_px(sigma_px: float) -> int:
    if not np.isfinite(sigma_px) or sigma_px < 0:
        raise ValueError("sigma_px must be finite and non-negative")
    return int(math.ceil(4.0 * float(sigma_px))) + 1

def _full_gradient(array: np.ndarray, sigma_px: float):
    value=np.asarray(array,dtype=np.float64)
    if value.ndim!=2 or not np.isfinite(value).all():
        raise ValueError("SHARP field must be a dense finite 2-D array")
    filtered=ndi.gaussian_filter(value,sigma=float(sigma_px),mode="nearest") if sigma_px>0 else value.copy()
    gy,gx=np.gradient(filtered)
    return gx,gy

def shared_full_context_gradients(q_a_full, box_a: NativeBox, q_b_full, box_b: NativeBox,
                                  target_box: NativeBox, *, fine_sigma_px: float,
                                  coarse_sigma_px: float|None=None):
    if tuple(q_a_full.shape)!=box_a.shape or tuple(q_b_full.shape)!=box_b.shape:
        raise ValueError("SHARP array/box shape mismatch")
    if not box_a.contains(target_box) or not box_b.contains(target_box):
        raise ValueError("target_box must be contained in both source boxes")
    if coarse_sigma_px is not None and coarse_sigma_px <= fine_sigma_px:
        raise ValueError("coarse_sigma_px must be greater than fine_sigma_px")
    fax,fay=_full_gradient(q_a_full,fine_sigma_px); fbx,fby=_full_gradient(q_b_full,fine_sigma_px)
    out={
        "fine_x":0.5*(crop_to_native_box(fax,box_a,target_box)+crop_to_native_box(fbx,box_b,target_box)),
        "fine_y":0.5*(crop_to_native_box(fay,box_a,target_box)+crop_to_native_box(fby,box_b,target_box)),
    }
    max_sigma=float(fine_sigma_px)
    if coarse_sigma_px is not None:
        cax,cay=_full_gradient(q_a_full,coarse_sigma_px); cbx,cby=_full_gradient(q_b_full,coarse_sigma_px)
        out["coarse_x"]=0.5*(crop_to_native_box(cax,box_a,target_box)+crop_to_native_box(cbx,box_b,target_box))
        out["coarse_y"]=0.5*(crop_to_native_box(cay,box_a,target_box)+crop_to_native_box(cby,box_b,target_box))
        max_sigma=max(max_sigma,float(coarse_sigma_px))
    halo=gaussian_gradient_halo_px(max_sigma)
    out["safe_mask"]=native_halo_support_mask(box_a,target_box,halo_px=halo)&native_halo_support_mask(box_b,target_box,halo_px=halo)
    out["strict_halo_px"]=halo
    return out
