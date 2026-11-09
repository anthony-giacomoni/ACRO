"""Lancer l'interface ACRO en cliquant sur ▶ Exécuter dans Spyder.

Ce lanceur ouvre une fenêtre Python indépendante : Spyder reste utilisable.
Aucun fichier config.json original n'est modifié.
"""
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent

if __name__ == '__main__':
    print('Ouverture de la fenêtre ACRO…')
    print('Interpréteur :', sys.executable)
    print('Dossier :', ROOT)
    proc = subprocess.Popen([sys.executable, str(ROOT / 'acro_gui.py')], cwd=str(ROOT))
    print('Processus GUI démarré (PID :', proc.pid, '). Vérifie que la fenêtre ACRO apparaît ; sinon lance acro_gui.py depuis un Terminal.')
