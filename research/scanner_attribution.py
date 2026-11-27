"""Read-only analysis of a completed scoped Gauntlet observations checkpoint."""
import argparse
import json
import sqlite3
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.special import softmax
from research.overlap_inference import overlap_robust_test
from research.calibration import reliability_bins

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("checkpoint", type=Path)
    checkpoint = parser.parse_args().checkpoint.resolve()
    from core.live_volume import on_live_volume
    if on_live_volume(checkpoint):
        raise ValueError("Use an isolated observations checkpoint")
    with sqlite3.connect(checkpoint.as_uri() + '?mode=ro', uri=True) as db:
        cutoff, end, grid, _, spec = json.loads(db.execute('SELECT spec FROM geometry_config').fetchone()[0])
        if not (spec['observations'] and spec['exploratory']
                and cutoff == '2025-01-01' and end == '2025-11-01'):
            raise ValueError("Expected the exploratory 2024–2025 observation protocol")
        if len(spec['symbols']) != db.execute('SELECT count(*) FROM geometry_done').fetchone()[0]:
            raise ValueError("Finish all symbol batches before scoring the study")
        data = pd.read_sql_query('''
            SELECT f.*, g.outcome AS exit, g.net_r, g.net_pct, g.holding,
                   longer.net_pct AS longer_net_pct, longer.outcome AS longer_exit
            FROM geometry_observations f LEFT JOIN geometry_outcomes g
            ON f.symbol=g.symbol AND f.day=g.day AND g.variant='scanner'
            LEFT JOIN geometry_outcomes longer
            ON f.symbol=longer.symbol AND f.day=longer.day AND longer.variant='long'
            ORDER BY f.day,f.symbol
        ''',db)
    dates = {d:i for i,d in enumerate(spec['sessions'])}
    data = data.drop(columns=['features','outcome']).join(pd.DataFrame(map(json.loads,data.features))).join(pd.DataFrame(map(json.loads,data.outcome)))
    data['ordinal'] = data.day.map(dates)
    data['log_turnover'] = np.log(data.turnover_20)
    data['flagged'] = data.signals.map(bool)
    data['buy'] = data.verdict.eq('BUY')
    data['positive'] = data.net_r.gt(0).astype(int)
    data['label'] = data.exit.map({'WIN':0,'TARGET_GAP_SCRATCH':0,'LOSS':1,'SCRATCH':1,'FLAT':2})
    data['setup_payoff'] = data.net_pct.where(data.exit.ne('NO_FILL'), 0.)
    data['entry_label'] = data.label.notna().astype(float).where(data.setup_payoff.notna())
    keys = ['score','momentum_20','volatility_20','log_turnover']
    horizon = max(x[3] for x in grid.values())

    def infer(series):
        by_day = series.groupby(level=0).agg(['sum','count'])
        return overlap_robust_test({dates[d]:tuple(row) for d,row in by_day.iterrows()},horizon)

    eligible = data[(data.close >= 20) & (data.turnover_20 >= 10_000_000)]
    attribution = {}
    for group in ('buy','flagged'):
        attribution[group] = {}
        for field in ('close_return','open_to_close_return','next_open_gap'):
            means = eligible.groupby(['day',group])[field].mean().unstack()
            pair = (means.reindex(columns=[False, True])[True]-means.reindex(columns=[False, True])[False]).dropna()
            attribution[group][field] = {'paired_dates':len(pair), 'mean_spread_pct':100*float(pair.mean()) if len(pair) else None, 'inference':infer(pair)}
    ranked = eligible[eligible.flagged & eligible.score.notna()].copy()
    ranked['top_five'] = ranked.groupby('day').score.rank(method='first', ascending=False) <= 5
    rank_means = ranked.groupby(['day','top_five']).open_to_close_return.mean().unstack()
    rank_means = rank_means.reindex(columns=[False, True])
    rank_spread = (rank_means[True] - rank_means[False]).dropna()
    attribution['rank_top_five'] = {'paired_dates':len(rank_spread),
                                    'mean_spread_pct':100*float(rank_spread.mean()) if len(rank_spread) else None, 'inference':infer(rank_spread)}
    # Sessionwise ranks avoid conflating market drift with stock ordering.
    ranked = ranked.dropna(subset=['open_to_close_return']).copy()
    by_day = ranked.groupby('day')
    rank_correlations = by_day.apply(
        lambda frame: frame.score.rank().corr(frame.open_to_close_return.rank()),
        include_groups=False,
    ).dropna()
    ranked['score_quintile'] = np.minimum((by_day.score.rank(pct=True) * 5).apply(np.ceil), 5).astype(int)
    attribution['score_rank_correlation'] = infer(rank_correlations)
    attribution['score_quintiles'] = {
        int(q): {'n': len(group), 'mean_pct': float(group.open_to_close_return.mean()) * 100,
                 'p10_pct': float(group.open_to_close_return.quantile(.1)) * 100,
                 'p90_pct': float(group.open_to_close_return.quantile(.9)) * 100}
        for q, group in ranked.groupby('score_quintile')
    }
    execution_groups = eligible[eligible.flagged & eligible.exit.isin(['LOSS', 'WIN', 'FLAT', 'SCRATCH', 'TARGET_GAP_SCRATCH', 'NO_FILL'])]
    group_means = execution_groups.groupby(['day', 'buy']).setup_payoff.mean().unstack().reindex(columns=[False, True])
    attribution['buy_vs_rejected_setups_net_pct'] = infer((group_means[True] - group_means[False]).dropna())
    paired_geometry = execution_groups.dropna(subset=['net_pct', 'longer_net_pct'])
    attribution['long_minus_scanner_on_paired_fills_pct'] = infer(pd.Series(
        (paired_geometry.longer_net_pct - paired_geometry.net_pct).to_numpy(), index=paired_geometry.day))
    buy = data[data.buy & data.net_r.notna()]
    execution = dict(fills=len(buy), mean_net_r=float(buy.net_r.mean()), mean_net_pct=float(buy.net_pct.mean()),
                     mean_gross_pct=float(buy.net_pct.mean()+spec['cnc_cost_pct']),
                     mean_holding=float(buy.holding.mean()), exit_counts=buy.exit.value_counts().to_dict())

    def design(train,test):
        if train.empty:
            raise ValueError('UNMEASURED: no training observations')
        a,b = (frame[keys].to_numpy(float) for frame in (train,test))
        center = a.mean(axis=0)
        scale = np.maximum(a.std(axis=0),1e-12)
        return [np.column_stack((np.ones(len(x)),np.clip((x-center)/scale,-5,5))) for x in (a,b)]

    def categorical(a,b,labels,classes):
        y=np.eye(classes)[labels.astype(int)]
        def objective(beta):
            w=beta.reshape(a.shape[1],classes)
            p=softmax(a@w,axis=1)
            loss=-np.mean(np.sum(y*np.log(p.clip(1e-15)),axis=1))+.05*np.sum(w[1:]**2)/2
            grad=a.T@(p-y)/len(a)
            grad[1:]+=.05*w[1:]
            return loss,grad.ravel()
        fit=minimize(objective,np.zeros(a.shape[1]*classes),jac=True,method='L-BFGS-B')
        if not fit.success: raise RuntimeError(fit.message)
        return softmax(b@fit.x.reshape(a.shape[1],classes),axis=1)

    def forecasts(train,test,field,classes):
        fit=train[train[field].notna()]
        a,b=design(fit,test)
        return categorical(a,b,fit[field].to_numpy(),classes)

    def payoff(train,test):
        fit=train[train.setup_payoff.notna()]
        a,b=design(fit,test)
        penalty=np.diag([0]+[.05]*(a.shape[1]-1))
        return b@np.linalg.solve(a.T@a/len(a)+penalty,a.T@fit.setup_payoff.to_numpy()/len(a))

    def scores(prob,y):
        actual=np.eye(prob.shape[1])[y]
        return {'brier':float(np.mean(np.sum((prob-actual)**2,axis=1))),
                'log_loss':float(-np.log(prob[np.arange(len(y)),y].clip(1e-15)).mean()),
                'reliability':[reliability_bins(prob[:,i],actual[:,i],5) for i in range(prob.shape[1])]}

    setups=data[(data.close>=20)&(data.turnover_20>=10_000_000)&data.score.notna()&data.exit.notna()]
    frozen=setups[setups.ordinal+horizon<min(i for d, i in dates.items() if d >= cutoff)]
    chunks=[]
    for first,last in zip(['2025-01-01','2025-04-01','2025-07-01','2025-10-01'],
                          ['2025-04-01','2025-07-01','2025-10-01',end]):
        if first>=end: continue
        bound=min(i for d,i in dates.items() if d>=first)
        train=setups[setups.ordinal+horizon<bound]
        test=setups[(setups.day>=first)&(setups.day<last)].copy()
        if test.empty: continue
        for mode,learn in [('frozen',frozen),('adaptive',train)]:
            fills=learn[learn.label.notna()]
            for name,field,n in [('exit','label',3),('positive','positive',2),('entry','entry_label',2)]:
                sample=fills if name=='positive' else learn
                p=forecasts(sample,test,field,n)
                observed = sample[field].dropna().astype(int)
                prior=(np.bincount(observed,minlength=n)+.5)/(len(observed)+n*.5)
                for i in range(n):
                    test[f'{mode}_{name}{i}']=p[:,i]
                    test[f'{mode}_{name}_prior{i}']=prior[i]
            test[f'{mode}_payoff']=payoff(learn,test)
        chunks.append(test)
    evals=pd.concat(chunks)
    filled=evals[evals.label.notna()]
    models={}
    for mode in ('frozen','adaptive'):
        models[mode]={}
        for name,field,n in [('exit','label',3),('positive','positive',2),('entry','entry_label',2)]:
            evaluated = evals[evals.entry_label.notna()] if name == 'entry' else filled
            for suffix in ('','_prior'):
                prob=evaluated[[f'{mode}_{name}{suffix}{i}' for i in range(n)]].to_numpy()
                models[mode][name+suffix]=scores(prob,evaluated[field].astype(int).to_numpy())
        models[mode]['payoff_rmse_pct']=float(np.mean((evals[f'{mode}_payoff']-evals.setup_payoff)**2)**.5)
        baseline_brier = np.sum((filled[[f'{mode}_exit_prior{i}' for i in range(3)]].to_numpy()-np.eye(3)[filled.label.astype(int)])**2,axis=1)
        model_brier = np.sum((filled[[f'{mode}_exit{i}' for i in range(3)]].to_numpy()-np.eye(3)[filled.label.astype(int)])**2,axis=1)
        models[mode]['paired_exit_brier_improvement']=infer(pd.Series(baseline_brier-model_brier,index=filled.day))
        # Policy dry run on a 20-session grid with five equal cash slots. Realized-only drawdown.
        nav = peak = basket_nav = 1.0
        max_dd = 0.0
        periods = censored = 0
        for day,frame in evals[evals.ordinal % horizon == 0].groupby('ordinal'):
            payoff_column = f'{mode}_payoff'
            pick = (frame[frame[payoff_column] > 0]
                    .sort_values([payoff_column, 'symbol'], ascending=[False, True]).head(5))
            censored+=int(pick.setup_payoff.isna().sum())
            nav *= 1 + pick.setup_payoff.fillna(-100).sum() / 500
            basket_nav *= 1 + frame.setup_payoff.fillna(-100).mean() * len(pick) / 500
            peak=max(peak,nav); max_dd=max(max_dd,(peak-nav)/peak); periods+=1
        models[mode].update(policy_periods=periods, policy_unresolved_slots=censored, policy_return_pct=100*(nav-1), policy_drawdown_pct=100*max_dd,
                            policy_matched_basket_return_pct=100*(basket_nav-1))
    print(json.dumps(dict(snapshot_code_sha256=spec['code'], cost_pct=spec['cnc_cost_pct'], observations=len(data),day_count=data.day.nunique(),symbols=data.symbol.nunique(),
                          fully_resolved_controls=int(data.close_return.notna().sum()),
                          attribution=attribution,execution=execution,model_eval_fills=len(filled),
                          model_eval_setups=len(evals),model_eval_dates=evals.day.nunique(),models=models),
                     indent=2,allow_nan=False))


if __name__ == '__main__':
    main()
