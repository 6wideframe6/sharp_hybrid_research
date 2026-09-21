"""Discrete-consistent screened integration for K5 detail gradients.

The desired K5 vector field is pixel-centred. It is converted to staggered
edge targets by averaging adjacent pixel components, then integrated by solving
exactly the normal equation of

    ||D(delta) - t||^2 + screening * ||delta_interior||^2

with a zero outer boundary. D is a forward edge-difference operator and the
right-hand side uses its exact algebraic adjoint D^T.
"""
from __future__ import annotations
from dataclasses import dataclass
import numpy as np
from scipy.sparse.linalg import LinearOperator, cg

@dataclass(frozen=True)
class IntegrationResult:
    delta_q: np.ndarray
    converged: bool
    cg_info: int

def saturated_gate(confidence: np.ndarray, *, low: float, high: float) -> np.ndarray:
    if not np.isfinite(low) or not np.isfinite(high):
        raise ValueError("gate thresholds must be finite")
    if low < 0 or high <= low:
        raise ValueError("require 0 <= low < high")
    c=np.asarray(confidence,dtype=np.float64)
    gate=np.clip((c-low)/(high-low),0.0,1.0)
    gate[~np.isfinite(c)]=0.0
    return gate

def pixel_vector_to_edge_targets(gx: np.ndarray, gy: np.ndarray):
    gx=np.asarray(gx,dtype=np.float64); gy=np.asarray(gy,dtype=np.float64)
    if gx.shape!=gy.shape or gx.ndim!=2:
        raise ValueError("gx and gy must be same-shape 2-D arrays")
    if not np.isfinite(gx).all() or not np.isfinite(gy).all():
        raise ValueError("gradient field must be finite")
    return 0.5*(gx[:,:-1]+gx[:,1:]), 0.5*(gy[:-1,:]+gy[1:,:])

def forward_edge_gradient(scalar: np.ndarray):
    u=np.asarray(scalar,dtype=np.float64)
    if u.ndim!=2: raise ValueError("scalar must be 2-D")
    return u[:,1:]-u[:,:-1], u[1:,:]-u[:-1,:]

def edge_adjoint(tx: np.ndarray, ty: np.ndarray, *, shape: tuple[int,int]) -> np.ndarray:
    h,w=shape; tx=np.asarray(tx,dtype=np.float64); ty=np.asarray(ty,dtype=np.float64)
    if tx.shape!=(h,w-1) or ty.shape!=(h-1,w):
        raise ValueError("edge field shape mismatch")
    out=np.zeros((h,w),dtype=np.float64)
    out[:,:-1]-=tx; out[:,1:]+=tx
    out[:-1,:]-=ty; out[1:,:]+=ty
    return out

def solve_screened_gradient_field(gx: np.ndarray, gy: np.ndarray, *, screening: float, rtol: float=1e-7, maxiter: int=5000) -> IntegrationResult:
    gx=np.asarray(gx,dtype=np.float64); gy=np.asarray(gy,dtype=np.float64)
    if gx.shape!=gy.shape or gx.ndim!=2: raise ValueError("gx and gy must be same-shape 2-D arrays")
    if min(gx.shape)<3: raise ValueError("gradient field must be at least 3x3")
    if not np.isfinite(gx).all() or not np.isfinite(gy).all(): raise ValueError("gradient field must be finite")
    if not np.isfinite(screening) or screening<0: raise ValueError("screening must be finite and non-negative")
    if rtol<=0 or maxiter<1: raise ValueError("invalid CG controls")
    h,w=gx.shape; ih,iw=h-2,w-2
    tx,ty=pixel_vector_to_edge_targets(gx,gy)
    rhs_full=edge_adjoint(tx,ty,shape=(h,w))
    rhs=rhs_full[1:-1,1:-1].reshape(-1)
    def matvec(v):
        u=np.asarray(v,dtype=np.float64).reshape(ih,iw)
        out=(4.0+screening)*u.copy()
        out[:,1:]-=u[:,:-1]; out[:,:-1]-=u[:,1:]
        out[1:,:]-=u[:-1,:]; out[:-1,:]-=u[1:,:]
        return out.reshape(-1)
    op=LinearOperator((ih*iw,ih*iw),matvec=matvec,dtype=np.float64)
    sol,info=cg(op,rhs,rtol=rtol,atol=0.0,maxiter=maxiter)
    delta=np.zeros((h,w),dtype=np.float64)
    delta[1:-1,1:-1]=sol.reshape(ih,iw)
    return IntegrationResult(delta,info==0,int(info))
