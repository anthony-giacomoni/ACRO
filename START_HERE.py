"""ACRO — point d'entrée recommandé. Ouvrir ce fichier dans Spyder puis F5.

Par défaut, lance une étude rapide illustrant la structure, avec sélection automatique du moteur compatible.
Pour les expériences chiffrées de recherche, consulter le README.md et notes/MODEL_NOTES.md.
Le mode rapide ne fournit PAS une cotation fiable ou comparable à l'audit
à coupon contractuel fixe de 18 %.
"""
from pathlib import Path
from run_demo import run

ROOT=Path(__file__).resolve().parent


def main():
    results=ROOT/'RESULTATS_DEMO'
    print('\n'+'='*64)
    print('ACRO | LABORATOIRE DE STRUCTURATION — DÉMONSTRATION')
    print('='*64)
    print('Mode rapide : modèle Q synthétique, coupon du Phoenix recalculé au pair.')
    print('Ce résultat ne doit PAS être confondu avec le benchmark 18 % / 12 pas.')
    values=run(config=ROOT/'config.json', output=results,quick=True,
               sensitivity=False, pricing_method='auto')
    print('\nRÉSULTATS DE LA SIMULATION (illustratifs)')
    print(f"Coupon annuel du Phoenix : {values['annual_coupon']:.2%}")
    print(f"Prime ACRO estimée : {values['acro_price']:.3f} € / contrat (nominal {values['initial_nominal']:g} €)")
    print(f"Rappel admissible sous Q : {values['eligible_call_probability_Q']:.1%}")
    print(f'Graphiques, données CSV et rapport : {results}')
    print('Guide : README.md')
    print('\nLes PNG sont consultables directement dans RESULTATS_DEMO.')
    # In Spyder/IPython, render two figure files in the console immediately.
    try:
        from IPython import get_ipython
        if get_ipython() is not None:
            from IPython.display import display, Image
            for name in ['01_carte_payoff_worstof_vol.png','02_payoff_au_rappel.png']:
                path=results/name
                if path.exists():
                    print('Visualisation :',name)
                    display(Image(filename=str(path),width=920))
    except (ImportError,OSError):
        pass
    return values

if __name__=='__main__':
    main()
