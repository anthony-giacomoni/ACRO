"""Recreate V5.8 diagnostic figures without rerunning Monte Carlo.

python plot_audit_v58.py [results_directory]
"""
from pathlib import Path
import sys, json, csv
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def plot(out):
    out=Path(out)
    data=json.loads((out/'AUDIT_VARIANCE_V5_8.json').read_text(encoding='utf-8'))
    anchor=data['independent_anchor_experiment']
    sources=[
        ('Trajectoires externes\n(courbe fixe)',data['external_only']['sd_across_reps']),
        ('Courbes indépendantes\n(méthode V5.7)',data['internal_only']['sd_across_reps']),
        ('Courbes ancrées\n(méthode V5.8)',anchor['anchored_cv']['sd_across_reps'])]
    fig,ax=plt.subplots(figsize=(9,5.5))
    bars=ax.bar(range(3),[z[1] for z in sources],width=.57)
    ax.set_xticks(range(3),[z[0] for z in sources])
    ax.set_ylabel('Écart-type des primes entre réplications (€/100)')
    ax.set_title('ACRO — D’où vient l’instabilité numérique ?')
    ax.grid(axis='y',alpha=.18);ax.set_axisbelow(True)
    ax.set_ylim(0,max(z[1] for z in sources)*1.24)
    for b,(_,v) in zip(bars,sources):
        ax.text(b.get_x()+b.get_width()/2,b.get_height()+.003,f'{v:.3f} €',
                ha='center',va='bottom',fontsize=11)
    fig.text(.09,.018,'Coupon 18 %, 96 pas/an ; six réplications. Référence GBM commune exclue.\nCes colonnes ne sont pas un intervalle de confiance global.',fontsize=9)
    fig.tight_layout(rect=(0,.065,1,1));fig.savefig(out/'01_decomposition_dispersion_v58.png',dpi=175)
    plt.close(fig)
    with (out/'anchor_comparison.csv').open(newline='',encoding='utf-8') as f:
        rows=list(csv.DictReader(f))
    reps=np.array([int(r['rep']) for r in rows]);old=np.array([float(r['old_premium']) for r in rows]);new=np.array([float(r['anchored_premium']) for r in rows])
    fig,ax=plt.subplots(figsize=(9,5.2))
    ax.plot(reps,old,'o-',label='V5.7 : contrôle GBM direct à chaque nœud')
    ax.plot(reps,new,'s-',label='V5.8 : ancrage indépendant + différences CRN')
    ax.set_xticks(reps)
    ax.set_xlabel('Réplication indépendante de la courbe')
    ax.set_ylabel('Prime ACRO calculée sur les mêmes scénarios externes (€/100)')
    ax.set_title('ACRO — Stabilité des deux méthodes conditionnelles')
    ax.grid(alpha=.22);ax.legend()
    fig.text(.09,.02,'Expérience de variance : scénarios externes fixés, même référence GBM commune.\nAucune des deux séries ne fournit à elle seule un prix certifié.',fontsize=9)
    fig.tight_layout(rect=(0,.068,1,1));fig.savefig(out/'02_comparaison_estimateurs_v58.png',dpi=175)
    plt.close(fig)

if __name__=='__main__':
    plot(Path(sys.argv[1]) if len(sys.argv)>1 else Path(__file__).resolve().parents[1]/'results_audit_v58')
