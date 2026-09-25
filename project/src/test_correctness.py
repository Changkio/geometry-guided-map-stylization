"""Tests of mathematical endpoints and failure-prone edge cases."""
import sys,unittest
import torch
import numpy as np
from common import ROOT
sys.path.insert(0,str(ROOT/'project/third_party/AttentionDistillation'))
from losses import q_loss
from geometry_losses import geometry_q_loss,projected_prior,relative_q_loss,deduplicate_pairs

class GeometryTests(unittest.TestCase):
    def test_beta_zero_preserves_loss_and_latent_gradient(self):
        torch.manual_seed(7)
        z=torch.randn(1,4,requires_grad=True)
        transforms=[torch.randn(4,2*16*3),torch.randn(4,2*4*3)]
        qs=[(z@t).reshape(1,2,n,3) for t,n in zip(transforms,[16,4])]
        qcs=[torch.randn_like(q,requires_grad=True) for q in qs]
        a=q_loss(qs,qcs)
        b=geometry_q_loss(qs,qcs,[(4,4),(2,2)],torch.rand(8,8),0,q_loss)
        ga=torch.autograd.grad(a,z,retain_graph=True)[0]
        gb=torch.autograd.grad(b,z,retain_graph=True)[0]
        torch.testing.assert_close(a,b,rtol=1e-5,atol=1e-6)
        torch.testing.assert_close(ga,gb,rtol=1e-5,atol=1e-6)
        self.assertIsNone(torch.autograd.grad(b,qcs[0],allow_unused=True)[0])

    def test_weighting_is_normalized_and_layer_sum(self):
        q=torch.tensor([[[[1.],[2.],[3.],[4.]]]],requires_grad=True)
        qc=torch.zeros_like(q);prior=torch.tensor([[0.,1.],[0.,1.]])
        value=geometry_q_loss([q,q],[qc,qc],[(2,2),(2,2)],prior,2.,q_loss)
        expected=2*(1+3*2+3+3*4)/8
        self.assertAlmostEqual(float(value),expected,places=6)
        grad=torch.autograd.grad(value,q)[0].flatten()
        torch.testing.assert_close(grad,torch.tensor([.25,.75,.25,.75]))

    def test_max_projection_retains_thin_registered_feature(self):
        prior=torch.zeros(8,12);prior[5,9]=1
        projected=projected_prior(prior,(2,3)).reshape(2,3)
        self.assertEqual(float(projected.sum()),1.)
        self.assertEqual(float(projected[1,2]),1.)

    def test_empty_and_collapsed_relation_pairs(self):
        q=torch.randn(1,2,4,3,requires_grad=True);qc=torch.zeros_like(q)
        value=relative_q_loss([q],[qc],[(2,2)],[[],[[[0,0],[1,1]]]],(8,8))
        self.assertEqual(float(value),0.)
        self.assertTrue(torch.isfinite(torch.autograd.grad(value,q)[0]).all())
        pairs=[[[0,0],[7,7]],[[7,7],[0,0]],[[0,0],[1,1]]]
        self.assertEqual(deduplicate_pairs(pairs,(2,2),(8,8),'cpu').tolist(),[[0,3]])

    def test_wrong_grid_is_rejected(self):
        q=torch.randn(1,2,6,3)
        with self.assertRaises(ValueError):geometry_q_loss([q],[q],[(2,2)],torch.ones(8,8),1,q_loss)

if __name__=='__main__':unittest.main(verbosity=2)
