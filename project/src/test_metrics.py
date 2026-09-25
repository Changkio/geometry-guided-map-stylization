import unittest
import numpy as np
from scipy.ndimage import binary_erosion
from evaluate import boundary_metrics

class BoundaryMetricTests(unittest.TestCase):
    def setUp(self):
        self.mask=np.zeros((64,64),bool);self.mask[10:54,29:35]=True
        self.boundary=self.mask & ~binary_erosion(self.mask)
    def test_empty_prediction_returns_zero(self):
        out=np.full((64,64,3),255,dtype=np.uint8)
        score=boundary_metrics(out,self.boundary)
        for tol in (1,2,3):
            self.assertEqual(score[f'boundary_f1_{tol}'],0.)
            self.assertEqual(score[f'boundary_precision_{tol}'],0.)
            self.assertEqual(score[f'boundary_recall_{tol}'],0.)
    def test_correct_edge_beats_displaced_edge(self):
        img=np.full((64,64,3),255,dtype=np.uint8);img[self.mask]=0
        displaced=np.roll(img,8,axis=1)
        good=boundary_metrics(img,self.boundary)['boundary_f1_2']
        bad=boundary_metrics(displaced,self.boundary)['boundary_f1_2']
        self.assertGreater(good,.99);self.assertGreater(good,bad+.5)
    def test_empty_reference_is_explicit_error(self):
        with self.assertRaises(ValueError):boundary_metrics(np.zeros((64,64,3),np.uint8),np.zeros((64,64),bool))

if __name__=='__main__':unittest.main(verbosity=2)
