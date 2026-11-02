"""Reproducible Poisson and Gaussian binary-predictor simulation."""
import argparse
import json
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

def fit_groups(x,y):
    x=np.asarray(x);y=np.asarray(y,dtype=float)
    if x.ndim!=1 or y.shape!=x.shape or len(y)<3:
        raise ValueError('Equal one-dimensional arrays with at least three observations required')
    if not np.isin(x,[0,1]).all() or not np.isfinite(y).all() or (y<0).any() or (y!=np.floor(y)).any():
        raise ValueError('Binary predictors and finite nonnegative integer outcomes required')
    groups=[y[x==v] for v in [0,1]]
    if any(len(g)==0 or g.sum()==0 for g in groups):
        raise ValueError('Both groups must be present and have positive outcome totals')
    n=np.array([len(g) for g in groups]);m=np.array([g.mean() for g in groups])
    ss=np.array([np.sum((g-g.mean())**2) for g in groups])
    # Closed-form fits for a model with intercept and one binary predictor.
    return [dict(model='Poisson log link',slope=float(np.log(m[1]/m[0])),
                 model_variance=float(np.sum(1/(n*m))),hc0_variance=float(np.sum(ss/(n*n*m*m))),fitted_mean=float(y.mean())),
            dict(model='Gaussian identity link',slope=float(m[1]-m[0]),
                 model_variance=float(ss.sum()/(len(y)-2)*np.sum(1/n)),hc0_variance=float(np.sum(ss/(n*n))),fitted_mean=float(y.mean()))]

def simulate(repetitions=2000,seed=123):
    if isinstance(repetitions,bool) or not isinstance(repetitions,int) or repetitions<2:
        raise ValueError('At least two integer repetitions required')
    rng=np.random.default_rng(seed);records=[];skipped=[]
    for n in [20,200]:
        for rate in [4,10]:
            failures=0
            for replicate in range(repetitions):
                x=rng.binomial(1,.5,n);y=rng.poisson(rate,n)
                try: fits=fit_groups(x,y)
                except ValueError: failures+=1;continue
                for fit in fits:
                    records.append(dict(n=n,rate=rate,replicate=replicate,mean_error=fit.pop('fitted_mean')-rate,**fit))
            skipped.append(dict(n=n,rate=rate,attempted=repetitions,skipped=failures))
    raw=pd.DataFrame(records);rows=[]
    for (n,rate,model),g in raw.groupby(['n','rate','model']):
        row=dict(n=int(n),rate=int(rate),model=model,successful=len(g),mean_bias=g.mean_error.mean(),
                 bias_mcse=g.mean_error.std(ddof=1)/np.sqrt(len(g)),slope_mean=g.slope.mean(),
                 empirical_slope_variance=g.slope.var(ddof=1))
        for kind in ['model','hc0']:
            var=g[kind+'_variance'];covered=np.abs(g.slope)<=1.959963984540054*np.sqrt(var)
            p=covered.mean();row[kind+'_mean_variance']=var.mean();row[kind+'_coverage']=p
            row[kind+'_coverage_mcse']=np.sqrt(p*(1-p)/len(g))
        rows.append(row)
    return pd.DataFrame(rows),skipped

def run(output,repetitions=2000,seed=123):
    table,skipped=simulate(repetitions,seed);output=Path(output);output.mkdir(parents=True,exist_ok=True)
    table.to_csv(output/'simulation_summary.csv',index=False)
    (output/'settings.json').write_text(json.dumps(dict(seed=seed,repetitions=repetitions,scenarios=skipped),indent=2)+'\n')
    fig,axes=plt.subplots(1,2,figsize=(11,4.5),sharey=True)
    for ax,(model,g) in zip(axes,table.groupby('model')):
        labels=[f'n={r.n}, mean={r.rate}' for r in g.itertuples()];index=np.arange(len(g))
        ax.plot(index,g.model_coverage,'o-',label='Model-based')
        ax.plot(index,g.hc0_coverage,'s-',label='HC0 sandwich')
        ax.axhline(.95,color='gray',linestyle='--',label='Nominal 95%')
        ax.set_xticks(index,labels,rotation=25,ha='right');ax.set_title(model);ax.set_ylim(.8,1);ax.legend()
    axes[0].set_ylabel('Empirical interval coverage');fig.tight_layout();fig.savefig(output/'coverage.svg');plt.close(fig)
    return table

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',default='results');p.add_argument('--repetitions',type=int,default=2000);p.add_argument('--seed',type=int,default=123);a=p.parse_args()
    print(run(a.output,a.repetitions,a.seed).to_string(index=False))
