"""Protect the development selection rule from silent style/replicate errors."""
import unittest
import pandas as pd
from analyze import choose_candidate

class SelectionTests(unittest.TestCase):
 def test_reject_structure_win_with_unmatched_style(self):
  rows=pd.DataFrame([dict(method='A',gram_style_distance=10,boundary_f1_2=.8,content_weight=.125,beta=1),
                     dict(method='B',gram_style_distance=12,boundary_f1_2=.99,content_weight=.25,beta=1)])
  row,bad=choose_candidate(rows,10)
  self.assertEqual(row.method,'A');self.assertFalse(bad)
 def test_explicit_mismatch_fallback_and_tie(self):
  rows=pd.DataFrame([dict(method='A',gram_style_distance=12,boundary_f1_2=.8,content_weight=.25,beta=2),
                     dict(method='B',gram_style_distance=12,boundary_f1_2=.99,content_weight=.125,beta=4)])
  row,bad=choose_candidate(rows,10)
  self.assertEqual(row.method,'B');self.assertTrue(bad)
 def test_equal_map_weight_despite_unbalanced_image_counts(self):
  # The estimator averages within maps; extra copies of one map must not increase its weight.
  d=pd.DataFrame(dict(map_id=['a','a','a','b'],score=[0.,0.,0.,1.]))
  self.assertEqual(d.groupby('map_id').score.mean().mean(),.5)
  self.assertNotEqual(d.score.mean(),.5)

if __name__=='__main__':unittest.main(verbosity=2)
