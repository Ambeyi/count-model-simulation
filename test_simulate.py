import unittest
import numpy as np
from simulate import fit_groups,simulate
class Tests(unittest.TestCase):
    def test_known_fit(self):
        p,g=fit_groups([0,0,1,1],[2,4,4,8])
        self.assertAlmostEqual(p['slope'],np.log(2));self.assertEqual(g['slope'],3)
        self.assertEqual(p['fitted_mean'],4.5);self.assertEqual(g['fitted_mean'],4.5)
        self.assertAlmostEqual(p['model_variance'],.25)
        self.assertAlmostEqual(g['model_variance'],5)
        self.assertAlmostEqual(g['hc0_variance'],2.5)
    def test_ols_independent_matrix(self):
        x=np.array([0,0,0,1,1,1]);y=np.array([1,3,5,2,4,9]);_,g=fit_groups(x,y)
        X=np.column_stack([np.ones(6),x]);beta=np.linalg.lstsq(X,y,rcond=None)[0]
        residual=y-X@beta;bread=np.linalg.inv(X.T@X)
        cov=bread@X.T@np.diag(residual**2)@X@bread
        self.assertAlmostEqual(g['slope'],beta[1]);self.assertAlmostEqual(g['hc0_variance'],cov[1,1])
    def test_invalid_or_boundary(self):
        for x,y in [([0,0,0],[1,2,3]),([0,1,1],[0,2,3]),([0,1,1],[1,-1,2]),([0,1,1],[1,2.2,3])]:
            with self.assertRaises(ValueError): fit_groups(x,y)
    def test_reproducibility_and_equal_mean_bias(self):
        a,s=simulate(20,123);b,t=simulate(20,123)
        self.assertTrue(a.equals(b));self.assertEqual(s,t)
        for _,g in a.groupby(['n','rate']): self.assertAlmostEqual(g.mean_bias.iloc[0],g.mean_bias.iloc[1])
    def test_repetition_validation(self):
        with self.assertRaises(ValueError): simulate(1)
if __name__=='__main__': unittest.main()
