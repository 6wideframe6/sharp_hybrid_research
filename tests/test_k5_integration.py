from pathlib import Path
import sys, unittest
import numpy as np
REPO_ROOT=Path(__file__).resolve().parents[1]; sys.path.insert(0,str(REPO_ROOT))
from k5.integration import saturated_gate, solve_screened_gradient_field, forward_edge_gradient, edge_adjoint, pixel_vector_to_edge_targets

class IntegrationTests(unittest.TestCase):
    def test_saturated_gate(self):
        c=np.array([0.0,0.025,0.0625,0.1,1.0])
        np.testing.assert_allclose(saturated_gate(c,low=.025,high=.1),[0,0,.5,1,1])
    def test_zero(self):
        z=np.zeros((20,21)); r=solve_screened_gradient_field(z,z,screening=.05)
        self.assertTrue(r.converged); np.testing.assert_allclose(r.delta_q,0)
    def test_adjoint_identity(self):
        rng=np.random.default_rng(7); h,w=19,23
        u=rng.normal(size=(h,w)); tx=rng.normal(size=(h,w-1)); ty=rng.normal(size=(h-1,w))
        ux,uy=forward_edge_gradient(u)
        lhs=float(np.sum(ux*tx)+np.sum(uy*ty))
        rhs=float(np.sum(u*edge_adjoint(tx,ty,shape=(h,w))))
        self.assertAlmostEqual(lhs,rhs,places=11)
    def test_exact_normal_equation(self):
        rng=np.random.default_rng(11); gx=rng.normal(size=(28,31)); gy=rng.normal(size=(28,31)); lam=.05
        r=solve_screened_gradient_field(gx,gy,screening=lam,rtol=1e-10,maxiter=10000)
        self.assertTrue(r.converged)
        tx,ty=pixel_vector_to_edge_targets(gx,gy); ux,uy=forward_edge_gradient(r.delta_q)
        n=edge_adjoint(ux-tx,uy-ty,shape=gx.shape)[1:-1,1:-1]+lam*r.delta_q[1:-1,1:-1]
        self.assertLess(float(np.max(np.abs(n))),1e-7)
    def test_exact_recovery_for_exact_edge_targets(self):
        h,w=25,29; y=np.linspace(0,1,h); x=np.linspace(0,1,w); yy,xx=np.meshgrid(y,x,indexing='ij')
        truth=np.sin(np.pi*yy)*np.sin(np.pi*xx)
        tx,ty=forward_edge_gradient(truth)
        gx=np.zeros_like(truth); gy=np.zeros_like(truth)
        gx[:,0]=tx[:,0]
        for j in range(w-1): gx[:,j+1]=2*tx[:,j]-gx[:,j]
        gy[0,:]=ty[0,:]
        for i in range(h-1): gy[i+1,:]=2*ty[i,:]-gy[i,:]
        cx,cy=pixel_vector_to_edge_targets(gx,gy)
        np.testing.assert_allclose(cx,tx,atol=1e-13,rtol=0); np.testing.assert_allclose(cy,ty,atol=1e-13,rtol=0)
        r=solve_screened_gradient_field(gx,gy,screening=0,rtol=1e-11,maxiter=10000)
        self.assertTrue(r.converged); np.testing.assert_allclose(r.delta_q,truth,atol=2e-9,rtol=0)
if __name__=='__main__': unittest.main(verbosity=2)
