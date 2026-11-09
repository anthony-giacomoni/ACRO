"""Spyder: ouvrir ce fichier et appuyer F5 pour visualiser 3 graphiques ACRO.

Le programme START_HERE.py doit avoir été exécuté au moins une fois.
Changez la liste GRAPHIQUES pour afficher d'autres fichiers PNG du dossier.
"""
from pathlib import Path

ROOT=Path(__file__).resolve().parent
RESULTATS=ROOT/'RESULTATS_DEMO'
GRAPHIQUES=[
 '01_carte_payoff_worstof_vol.png',
 '02_payoff_au_rappel.png',
 '03_rappel_exercice_et_prix.png',
]

def main():
    for nom in GRAPHIQUES:
        fichier=RESULTATS/nom
        if not fichier.exists():
            raise FileNotFoundError('Lancez START_HERE.py d’abord : '+str(fichier))
    try:
        from IPython import get_ipython
        if get_ipython() is not None:
            from IPython.display import display,Image
            for nom in GRAPHIQUES:
                print('\n'+nom)
                display(Image(filename=str(RESULTATS/nom),width=1000))
            return
    except ImportError:
        pass
    import matplotlib.pyplot as plt
    import matplotlib.image as mpimg
    for nom in GRAPHIQUES:
        fig,ax=plt.subplots(figsize=(11,6))
        ax.imshow(mpimg.imread(RESULTATS/nom))
        ax.axis('off')
        fig.suptitle(nom)
    plt.show()

if __name__=='__main__': main()
