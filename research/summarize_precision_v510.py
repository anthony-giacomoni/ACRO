"""Combine independent V5.10 high-precision batches without reusing seeds.

Run the 5-rep batches with --high-precision --start-rep 0 / 5 / 10,
then use this script to aggregate all independently numbered replications.
"""
from pathlib import Path
import csv,json,math
import numpy as np
from scipy.stats import t

ROOT=Path(__file__).resolve().parents[1]
BATCHES=[ROOT/'results_precision_v510_high',
         ROOT/'results_precision_v510_high_chunk2',
         ROOT/'results_precision_v510_high_chunk3']


def summarize(files=BATCHES,destination=ROOT/'results_precision_v510_combined'):
    allrows=[]
    for source in files:
        with (Path(source)/'replications.csv').open(newline='',encoding='utf-8') as f:
            allrows.extend(dict(row) for row in csv.DictReader(f))
    reps=[int(z['rep']) for z in allrows]
    if len(set(reps))!=len(reps) or sorted(reps)!=list(range(1,len(reps)+1)):
        raise ValueError('Replications are duplicated or missing: check seed batches')
    allrows.sort(key=lambda z:int(z['rep']))
    stats={}
    for field in ('lower','upper','gap'):
        x=np.array([float(z[field]) for z in allrows]);n=len(x)
        m=float(x.mean());sd=float(x.std(ddof=1))
        hw=float(t.ppf(.975,n-1)*sd/math.sqrt(n))
        stats[field]={'mean':m,'sd_across_reps':sd,'ci95':[m-hw,m+hw],
                      'halfwidth':hw,'n':n}
    result={'version':'5.10','coupon':.18,'steps_per_year':96,
            'samples_independent_curve_pairs':len(reps),
            'inner_paths_each':4096,'gbm_reference_paths_per_scramble':8192,
            'gbm_reference_scrambles':6,'outer_paths_per_rep':8000,
            'results':stats,'finite_grid_statistical_halfwidth_under_0p02':stats['upper']['halfwidth']<.02,
            'global_error_tolerance_certified':False,
            'caveats':['Confidence interval reflects replication variability for the particular discrete model with nodal linear interpolation.',
                       'Upper/lower claims are in expectation; finite-sample Student-t intervals are diagnostics.',
                       'Bias of interpolation relative to conditional value, time step bias and financial model misspecification are not bounded.']}
    dest=Path(destination);dest.mkdir(parents=True,exist_ok=True)
    (dest/'AUDIT_GLOBAL_15_REPLICATIONS.json').write_text(json.dumps(result,indent=2,ensure_ascii=False),encoding='utf-8')
    with (dest/'replications_combined.csv').open('w',newline='',encoding='utf-8') as f:
        wr=csv.DictWriter(f,fieldnames=allrows[0].keys());wr.writeheader();wr.writerows(allrows)
    lines=['# ACRO V5.10 — Synthèse de 15 réplications indépendantes','',
           'Coupon contractuel 18 %, 96 pas/an, deux courbes indépendantes par réplication.',
           '', '| Estimateur | Moyenne €/100 | IC95 empirique | Demi-largeur |',
           '|---|---:|---|---:|']
    for name,key in [('Inférieur (en espérance)','lower'),('Supérieur (en espérance)','upper'),
                     ('Écart de décision Jensen','gap')]:
        z=stats[key]
        lines.append(f'| {name} | {z["mean"]:.5f} | [{z["ci95"][0]:.5f} ; {z["ci95"][1]:.5f}] | {z["halfwidth"]:.5f} |')
    lines +=['', 'Les intervalles concernent exclusivement la dispersion entre réplications sur le **modèle discret avec interpolation**.',
             '**La précision statistique ciblée est atteinte ici**, mais la précision totale ±0,02 €/100 ne l’est PAS (biais temporel, interpolation, modèle).',
             'Les bornes inférieure et supérieure sont valables en espérance, pas comme limites certaines pour ces réalisations.', '']
    (dest/'AUDIT_GLOBAL_15_REPLICATIONS_FR.md').write_text('\n'.join(lines),encoding='utf-8')
    print(json.dumps({'n':len(reps),'low':stats['lower']['mean'],'up':stats['upper']['mean'],
                      'upper_ci95':stats['upper']['ci95'],'gap':stats['gap']['mean']},indent=2))
    return result

if __name__=='__main__': summarize()
