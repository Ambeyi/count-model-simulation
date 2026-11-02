"""Check closed-form variances against design-matrix calculations."""
import numpy as np
from simulate import fit_groups
rng=np.random.default_rng(77)
for _ in range(100):
    x=rng.binomial(1,.5,50);y=rng.poisson(4,50)
    p,g=fit_groups(x,y);X=np.column_stack([np.ones(len(x)),x])
    beta=np.linalg.lstsq(X,y,rcond=None)[0];mu=X@beta;r=y-mu
    bread=np.linalg.inv(X.T@X)
    hc=bread@X.T@np.diag(r*r)@X@bread
    model=bread*(r@r)/(len(y)-2)
    assert np.isclose(g['slope'],beta[1])
    assert np.isclose(g['hc0_variance'],hc[1,1])
    assert np.isclose(g['model_variance'],model[1,1])
    info=X.T@np.diag(mu)@X;pbread=np.linalg.inv(info)
    phc=pbread@X.T@np.diag(r*r)@X@pbread
    assert np.isclose(p['model_variance'],pbread[1,1])
    assert np.isclose(p['hc0_variance'],phc[1,1])
    fitted=np.exp(np.log(mu[x==0][0])+p['slope']*x)
    assert np.allclose(fitted,mu)
print('100 independent matrix checks passed for both models and both variance estimators.')
