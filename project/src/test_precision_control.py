"""Check that the numerical control overrides geometry explicitly and only as specified."""
import unittest
from unittest.mock import patch
import torch
from precision_control import UniformFP32Pipeline
from map_pipeline import MapADPipeline
from geometry_losses import geometry_q_loss

class PrecisionControlTests(unittest.TestCase):
    def test_zero_prior_forwarding_and_diagnostic(self):
        cfg={'prior_override':'zero_for_uniform_fp32_control','content_mode':'geometry_q','beta':1,'content_weight':.25}
        image=torch.zeros(1,3,8,8);prior=torch.ones(8,8)
        # Avoid loading a diffusion model; inspect the exact base-method arguments.
        instance=object.__new__(UniformFP32Pipeline)
        with patch.object(MapADPipeline,'optimize_map',return_value=(image,{})) as call:
            _,diag=instance.optimize_map(image,image,prior,cfg)
            passed=call.call_args.args[2]
            self.assertEqual(int(torch.count_nonzero(passed)),0)
            self.assertTrue(torch.equal(prior,torch.ones_like(prior)))
            self.assertEqual(diag['precision_control']['uniform_token_weights'],1)
        with self.assertRaises(ValueError):instance.optimize_map(image,image,prior,{**cfg,'content_weight':.5})

    def test_uniform_fp32_loss_and_gradient(self):
        torch.manual_seed(91)
        q=torch.randn(1,2,16,3,requires_grad=True);qc=torch.randn_like(q)
        value=geometry_q_loss([q],[qc],[(4,4)],torch.zeros(8,8),1,None)
        reference=(q-qc).abs().mean()
        a=torch.autograd.grad(value,q,retain_graph=True)[0]
        b=torch.autograd.grad(reference,q)[0]
        torch.testing.assert_close(value,reference,rtol=1e-6,atol=1e-7)
        torch.testing.assert_close(a,b,rtol=1e-6,atol=1e-7)

if __name__=='__main__':unittest.main()
