"""One focused plot: distribution of coupled time-grid differences across banks."""
from pathlib import Path
import csv
import numpy as np
import matplotlib.pyplot as plt

ROOT=Path(__file__).resolve().parents[1]

def make_plot(output_dir=ROOT/'results_temporal_v5101_standard'):
    output_dir=Path(output_dir)
    with (output_dir/'banques_independantes.csv').open(encoding='utf-8',newline='') as f:
        rows=list(csv.DictReader(f))
    series=[('192 − 96 pas/an','delta_192_96_a'),('384 − 192 pas/an','delta_384_192_a')]
    fig,ax=plt.subplots(figsize=(9,5.1),layout='constrained')
    for i,(name,key) in enumerate(series):
        vals=np.asarray([float(x[key]) for x in rows]);
        jitter=np.linspace(-.095,.095,len(vals))
        ax.scatter(vals, np.full(len(vals),i)+jitter,alpha=.65,s=31,linewidths=.35,edgecolors='black')
        mean=vals.mean()
        sd=vals.std(ddof=1)
        from scipy.stats import t
        h=t.ppf(.975,len(vals)-1)*sd/np.sqrt(len(vals))
        ax.errorbar(mean,i,xerr=h,fmt='o',markersize=8,capsize=5,
                    color='black',linewidth=2.1,zorder=5)
        ax.text(.054,i+.10,f'{mean:+.4f}  ±{h:.4f}',fontsize=10)
    ax.axvline(0,linestyle='-',linewidth=1.15,color='black')
    for x in (-.02,.02):ax.axvline(x,linestyle='--',linewidth=1,alpha=.65,color='gray')
    ax.set_yticks(range(len(series)),[x[0] for x in series]);ax.invert_yaxis()
    ax.set_xlim(-.06,.10);ax.set_ylim(1.45,-.5)
    ax.set_xlabel('Différence couplée de prime ACRO (€/100)')
    ax.set_title('Stabilité temporelle : 24 banques indépendantes',loc='left',fontsize=13)
    ax.grid(axis='x',alpha=.2)
    ax.text(0,-.3,'Points = banques indépendantes    |    ● noir + barre = moyenne et IC 95 %',fontsize=9)
    fig.savefig(output_dir/'stabilite_temporelle_v5101.png',dpi=180)
    plt.close(fig)

if __name__=='__main__': make_plot()
