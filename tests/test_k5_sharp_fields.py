from pathlib import Path
import sys, unittest, numpy as np
REPO_ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(REPO_ROOT))
from k5.sharp_fields import gaussian_gradient_halo_px, shared_full_context_gradients
from k5.types import NativeBox

class SharpFieldTests(unittest.TestCase):
    def test_expected_halo(self):
        self.assertEqual(gaussian_gradient_halo_px(1.0),5)
        self.assertEqual(gaussian_gradient_halo_px(2.0),9)

    def test_full_context_linear_gradient_survives_overlap_crop(self):
        box_a=NativeBox(1000,2000,1100,2080)
        box_b=NativeBox(1010,1990,1110,2070)
        ya,xa=np.mgrid[:80,:100]
        yb,xb=np.mgrid[:80,:100]
        q_a=0.003*xa+0.007*ya+2.0
        q_b=0.003*(xb+box_b.x0-box_a.x0)+0.007*(yb+box_b.y0-box_a.y0)+2.0
        target=box_a.intersect(box_b)
        out=shared_full_context_gradients(q_a,box_a,q_b,box_b,target,fine_sigma_px=1.0,coarse_sigma_px=2.0)
        safe=out["safe_mask"]
        self.assertTrue(safe.any())
        np.testing.assert_allclose(np.asarray(out["fine_x"])[safe],0.003,atol=1e-10,rtol=0)
        np.testing.assert_allclose(np.asarray(out["fine_y"])[safe],0.007,atol=1e-10,rtol=0)

if __name__=="__main__":
    unittest.main(verbosity=2)
