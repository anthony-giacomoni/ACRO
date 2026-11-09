"""ACRO GUI -- local, standalone Python/Tkinter desktop interface.

Run `LAUNCH_ACRO_GUI.py` in Spyder, or `python acro_gui.py` in the ACRO env.
This is a control/view layer only. All financial calculations come from run_demo.py.
The original validated source and benchmark files are not modified.
"""
from __future__ import annotations

import json
import math
import os
import queue
import shutil
import subprocess
import sys
import tempfile
import threading
import time
from datetime import datetime
from pathlib import Path
import tkinter as tk
from tkinter import ttk, filedialog, messagebox

ROOT = Path(__file__).resolve().parent
BASE_CONFIG = ROOT / 'config.json'
RUNS = ROOT / 'GUI_RUNS'
FIGURES = [
    ('01_carte_payoff_worstof_vol.png', 'Déclenchement : panier × volatilité'),
    ('02_payoff_au_rappel.png', 'Valeur du droit au rappel'),
    ('03_rappel_exercice_et_prix.png', 'Probabilités et contributions annuelles'),
    ('04_distribution_coupon_renouvellement.png', 'Coupons du renouvellement'),
    ('05_couverture_versus_forfait.png', 'Déficit non financé : ACRO et forfait'),
    ('06_sensibilite_dependance.png', 'Sensibilité à la corrélation spot-vol (optionnel)'),
    ('07_prix_promesse_coupon.png', 'Prime selon coupon contractuel'),
    ('08_trajectoires_monte_carlo.png', 'Trajectoires Monte Carlo : référence panier'),
    ('09_neuf_scenarios_phoenix.png', '9 scénarios : actions et payoffs'),
    ('10_distribution_payoffs_acro.png', 'Distribution des payoffs ACRO'),
    ('11_risque_capital_et_coupon.png', 'Risque et origine du coupon'),
]

def validate_custom_coupon_selection(enabled, text):
    """Reject a custom coupon checkbox with no actual rate entered."""
    if enabled and not str(text).strip():
        raise ValueError('Coupon ACRO cible activé : saisis un pourcentage numérique, ou décoche la case.')


def available_figures(folder):
    """The correlation sensitivity chart is optional and exists only after a sensitivity run."""
    root = Path(folder)
    return [item for item in FIGURES if (root / item[0]).is_file()]

ILLUSTRATIVE_PATH_CHOICES = (48, 100, 500, 1000)
CORRELATION_PAIRS = ((0, 1), (0, 2), (1, 2))  # API historique pour les tests 3 actions
MAX_EQUITIES = 6
BASKET_CHOICES = ('worst_of', 'best_of', 'equal_weight')


def resize_market_config(source, n):
    """Resize equity vectors/matrix while preserving overlapping user inputs.

    Added pairs receive a neutral positive baseline, not a calibrated quote.
    The project validator performs the final positive-definiteness check.
    """
    import copy
    if not isinstance(n, int) or not 1 <= n <= MAX_EQUITIES:
        raise ValueError(f'Le nombre d’actions doit être compris entre 1 et {MAX_EQUITIES}.')
    data=copy.deepcopy(source)
    m=data['market']
    previous=len(m['initial_spots'])
    defaults={'initial_spots':100.,'volatilities':0.25,'dividends':0.025}
    for name,default in defaults.items():
        m[name]=(list(m[name])[:n]+[default]*max(0,n-previous))
    old=m['correlations']
    m['correlations']=[[1.0 if i==j else
                       (float(old[i][j]) if i<previous and j<previous else 0.30)
                       for j in range(n)] for i in range(n)]
    return data


def correlation_interval(other_a, other_b):
    """Open admissible interval for the third correlation of a 3x3 PD matrix.

    For a=Corr(X,Y) and b=Corr(X,Z), positive definiteness requires
    a*b - sqrt((1-a²)(1-b²)) < Corr(Y,Z) < a*b + sqrt(...).
    """
    a, b = float(other_a), float(other_b)
    if not all(math.isfinite(z) and abs(z) < 1 for z in (a, b)):
        raise ValueError('Les deux autres corrélations doivent être strictement entre −1 et +1.')
    half_width = math.sqrt(max(0.0, (1-a*a)*(1-b*b)))
    return max(-1.0, a*b-half_width), min(1.0, a*b+half_width)


def correlation_limits(r12, r13, r23):
    """Admissible open interval for each correlation, given the other two."""
    return {(0, 1): correlation_interval(r13, r23),
            (0, 2): correlation_interval(r12, r23),
            (1, 2): correlation_interval(r12, r13)}


def check_correlation_triple(r12, r13, r23):
    """Show a helpful message before the engine's definitive Cholesky check."""
    values = {(0, 1): float(r12), (0, 2): float(r13), (1, 2): float(r23)}
    if any(not math.isfinite(v) or abs(v) >= 1 for v in values.values()):
        raise ValueError('Chaque corrélation doit être strictement comprise entre −1 et +1.')
    limits = correlation_limits(r12, r13, r23)
    for (i, j), value in values.items():
        low, high = limits[i, j]
        if not low < value < high:
            raise ValueError(f'Corrélation {i+1}–{j+1} : {value:.3f} impossible avec les deux autres. '
                             f'Plage autorisée : {low:.3f} < ρ{i+1}{j+1} < {high:.3f}.')
    return limits


NAVY = '#0f172a'
BG = '#f4f7fb'
WHITE = '#ffffff'
BLUE = '#1769aa'
MUTED = '#627389'

def run_cancelable_process(cmd, cwd, cancelled, on_started=None, popen_factory=None):
    """Run the pricing child safely when Cancel is clicked before Popen returns.

    Return (status, stdout); status is 'success', 'cancelled' or 'error'.
    This helper is kept independent of Tk so the cancellation race is testable.
    """
    if cancelled.is_set():
        return 'cancelled', ''
    kwargs = {'cwd': cwd, 'stdout': subprocess.PIPE, 'stderr': subprocess.STDOUT, 'text': True}
    if os.name == 'nt':
        kwargs['creationflags'] = subprocess.CREATE_NO_WINDOW
    p = (popen_factory or subprocess.Popen)(cmd, **kwargs)
    if on_started is not None:
        on_started(p)
    if cancelled.is_set() and p.poll() is None:
        p.terminate()
    output, _ = p.communicate()
    if cancelled.is_set():
        return 'cancelled', output or ''
    if p.returncode != 0:
        return 'error', f'Code retour {p.returncode}\n{(output or "")[-2400:]}'
    return 'success', output or ''


def comparable_run_snapshots(current, previous):
    """Only display a monetary difference for like-sized contracts and coupon policies."""
    if not isinstance(previous, dict):
        return False
    return (float(current.get('initial_nominal', 100)) ==
            float(previous.get('initial_nominal', 100)) and
            current.get('initial_coupon_policy', 'fair') ==
            previous.get('initial_coupon_policy', 'fair') and
            current.get('assets',3) == previous.get('assets',3) and
            current.get('initial_basket_rule','worst_of') == previous.get('initial_basket_rule','worst_of') and
            current.get('renewal_basket_rule','worst_of') == previous.get('renewal_basket_rule','worst_of'))


class ScrollingFrame(ttk.Frame):
    def __init__(self, parent):
        super().__init__(parent)
        self.canvas = tk.Canvas(self, borderwidth=0, highlightthickness=0, bg=WHITE)
        sb = ttk.Scrollbar(self, orient='vertical', command=self.canvas.yview)
        self.canvas.configure(yscrollcommand=sb.set)
        self.canvas.pack(side='left', fill='both', expand=True)
        sb.pack(side='right', fill='y')
        self.content = ttk.Frame(self.canvas, padding=13)
        self.winid = self.canvas.create_window((0, 0), window=self.content, anchor='nw')
        self.content.bind('<Configure>', lambda _: self.canvas.configure(scrollregion=self.canvas.bbox('all')))
        self.canvas.bind('<Configure>', lambda e: self.canvas.itemconfigure(self.winid, width=e.width))
        self.canvas.bind('<Enter>', lambda _: self.canvas.bind_all('<MouseWheel>', self._wheel))
        self.canvas.bind('<Leave>', lambda _: self.canvas.unbind_all('<MouseWheel>'))

    def _wheel(self, event):
        self.canvas.yview_scroll(-1 if event.delta > 0 else 1, 'units')

class ACROApp(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title('ACRO • Laboratoire de structuration')
        self.geometry('1390x880')
        self.minsize(1000, 710)
        self.configure(bg=BG)
        self.style = ttk.Style(self)
        if 'clam' in self.style.theme_names(): self.style.theme_use('clam')
        self.style.configure('TFrame', background=WHITE)
        self.style.configure('TLabel', background=WHITE, font=('Helvetica', 10), foreground=NAVY)
        self.style.configure('Header.TLabel', font=('Helvetica', 20, 'bold'), foreground=NAVY)
        self.style.configure('Section.TLabel', font=('Helvetica', 11, 'bold'), foreground=BLUE)
        self.style.configure('Hint.TLabel', font=('Helvetica', 9), foreground=MUTED)
        self.style.configure('TNotebook', background=BG, borderwidth=0)
        self.style.configure('TNotebook.Tab', padding=(12, 8), font=('Helvetica', 10))
        self.style.configure('Run.TButton', padding=(10, 11), font=('Helvetica', 12, 'bold'))
        self.fields = {}
        self.corr_hints = {}
        self._dataset = None
        self._previous = None
        self._process = None
        self._queue = queue.Queue()
        self._running = False
        self._cancel_requested = threading.Event()
        self._images = None
        self._current_chart = 0
        self._original_config = json.loads(BASE_CONFIG.read_text(encoding='utf-8'))
        self._base = json.loads(json.dumps(self._original_config))
        self._layout()
        self._fill(self._base)
        self._last_run_snapshot = self._snapshot()
        self._pending_snapshot = None
        for var, _, _ in self.fields.values():
            var.trace_add('write', self._on_parameter_edit)
        self.quick.trace_add('write', self._on_parameter_edit)
        self.sensitivity.trace_add('write', self._on_parameter_edit)
        self.illustrative_count.trace_add('write', self._on_parameter_edit)
        self._update_correlation_hints()
        # The target-coupon checkbox affects the exported contract, not just its text field.
        self.custom_target.trace_add('write', self._on_parameter_edit)
        self.after(110, self._poll)
        self.after(250, self._load_preloaded)
        self.protocol('WM_DELETE_WINDOW', self._on_close)

    def _layout(self):
        banner = tk.Frame(self, bg=NAVY, height=78)
        banner.pack(side='top', fill='x')
        tk.Label(banner, text='ACRO', fg='white', bg=NAVY, font=('Helvetica', 23, 'bold')).pack(side='left', padx=(23, 12), pady=14)
        tk.Label(banner, text='Autocall-Contingent Reinvestment Option  |  Simulateur de recherche Python',
                 fg='#d1e3f8', bg=NAVY, font=('Helvetica', 11)).pack(side='left', pady=17)
        main = ttk.Panedwindow(self, orient='horizontal')
        main.pack(fill='both', expand=True, padx=14, pady=12)
        left = ttk.Frame(main, padding=(0, 0, 5, 0))
        right = ttk.Frame(main, padding=(5, 0, 0, 0))
        main.add(left, weight=3)
        main.add(right, weight=7)
        # left: tabs
        ttk.Label(left, text='Paramètres du produit', style='Header.TLabel').pack(anchor='w', pady=(3, 9))
        self.tabs = ttk.Notebook(left)
        self.tabs.pack(fill='both', expand=True)
        pages = {}
        for k, title in [('market', 'Marché'), ('initial', 'Phoenix'), ('renewal', 'Renouvelé'), ('acro', 'ACRO / calcul')]:
            page = ScrollingFrame(self.tabs)
            self.tabs.add(page, text=title)
            pages[k] = page.content
        self._market_fields(pages['market'])
        self._phoenix_fields(pages['initial'], 'initial_phoenix')
        self._phoenix_fields(pages['renewal'], 'renewal_phoenix')
        self._acro_fields(pages['acro'])
        controls = ttk.Frame(left, padding=(5, 8, 5, 5))
        controls.pack(fill='x')
        ttk.Button(controls, text='Copier Phoenix → Renouvelé', command=self._copy_terms).pack(fill='x', pady=(0, 5))
        options = ttk.Frame(controls)
        options.pack(fill='x', pady=(1, 7))
        self.quick = tk.BooleanVar(value=True)
        self.sensitivity = tk.BooleanVar(value=False)
        ttk.Checkbutton(options, text='Rapide (aperçu, plus bruité)', variable=self.quick).pack(side='left')
        ttk.Checkbutton(options, text='Sensibilités (plus lent)', variable=self.sensitivity).pack(side='right')
        view_options = ttk.Frame(controls)
        view_options.pack(fill='x', pady=(0, 5))
        ttk.Label(view_options, text='Trajectoires illustratives :').pack(side='left')
        self.illustrative_count = tk.StringVar(value='48')
        ttk.Combobox(view_options, textvariable=self.illustrative_count,
                     state='readonly', width=7, values=[str(n) for n in ILLUSTRATIVE_PATH_CHOICES]).pack(side='left', padx=(8, 6))
        ttk.Label(view_options, text='Illustration seule ; prix inchangé', style='Hint.TLabel').pack(side='left')
        self.run_button = ttk.Button(controls, text='▶  SIMULER AVEC CES PARAMÈTRES', style='Run.TButton', command=self._start)
        self.run_button.pack(fill='x', pady=5)
        line = ttk.Frame(controls)
        line.pack(fill='x', pady=2)
        self.cancel_button = ttk.Button(line, text='Arrêter', command=self._cancel, state='disabled')
        self.cancel_button.pack(side='left', padx=(0, 7))
        ttk.Button(line, text='Réinitialiser', command=self._reset).pack(side='left', padx=(0, 7))
        ttk.Button(line, text='Exporter paramètres…', command=self._export).pack(side='left')
        self.status = tk.StringVar(value='Prêt. Modifie les paramètres puis clique sur Simuler.')
        ttk.Label(controls, textvariable=self.status, wraplength=370, style='Hint.TLabel').pack(anchor='w', pady=(6, 1))
        self.progress = ttk.Progressbar(controls, mode='indeterminate')
        self.progress.pack(fill='x')
        # right: results
        ttk.Label(right, text='Résultats & graphiques', style='Header.TLabel').pack(anchor='w', pady=(3, 9))
        kpi_frame = ttk.Frame(right, padding=4)
        kpi_frame.pack(fill='x')
        self.kpis = {}
        self.kpi_titles = {}
        for i, (key, heading) in enumerate([('annual_coupon', 'Coupon initial (option séparée)'), ('acro_price', 'Prime séparée / 100 €'),
                                             ('eligible_call_probability_Q', 'P(rappel) sous Q'), ('bundled_fair_coupon', 'Coupon autofinancé (sans prime)')]):
            tile = tk.Frame(kpi_frame, bg='#eaf2fc', highlightbackground='#d9e5f3', highlightthickness=1)
            tile.grid(row=0, column=i, sticky='nsew', padx=3, pady=(1, 6))
            kpi_frame.columnconfigure(i, weight=1)
            title = tk.Label(tile, text=heading, bg='#eaf2fc', fg='#506782', font=('Helvetica', 10))
            title.pack(pady=(10, 3))
            self.kpi_titles[key] = title
            value = tk.Label(tile, text='—', bg='#eaf2fc', fg=BLUE, font=('Helvetica', 17, 'bold'))
            value.pack(pady=(0, 10))
            self.kpis[key] = value
        self.finance_info = tk.StringVar(value='Deux modes alternatifs : prime séparée OU réduction du coupon, jamais les deux cumulés.')
        ttk.Label(right, textvariable=self.finance_info, style='Hint.TLabel',
                  wraplength=850, justify='left').pack(anchor='w', padx=7, pady=(0, 4))
        self.decomp_info = tk.StringVar(value='Décomposition du Phoenix : calculée après une simulation (capital + coupons conditionnels).')
        ttk.Label(right, textvariable=self.decomp_info, style='Hint.TLabel',
                  wraplength=850, justify='left').pack(anchor='w', padx=7, pady=(0, 4))
        self.risk_info = tk.StringVar(value='Risque sur le nominal : diagnostic disponible après une simulation.')
        ttk.Label(right, textvariable=self.risk_info, wraplength=850, justify='left',
                  foreground='#9b302e', font=('Helvetica', 10, 'bold')).pack(anchor='w', padx=7, pady=(0, 4))
        self.put_info = tk.StringVar(value='Décomposition obligataire / pertes contingentes : calculée sur le même échantillon Q.')
        ttk.Label(right, textvariable=self.put_info, style='Hint.TLabel',
                  wraplength=850, justify='left').pack(anchor='w', padx=7, pady=(0, 4))
        self.snapshot_info = tk.StringVar(value='Les résultats affichés correspondent aux paramètres du dernier calcul.')
        ttk.Label(right, textvariable=self.snapshot_info, style='Hint.TLabel',
                  wraplength=850, justify='left').pack(anchor='w', padx=7, pady=(0, 2))
        navigation = ttk.Frame(right, padding=(4, 7))
        navigation.pack(fill='x')
        ttk.Button(navigation, text='◀', width=4, command=lambda: self._chart(-1)).pack(side='left')
        self.chart_choice = tk.StringVar()
        # Figure 06 exists only when a sensitivity run produced it.
        self._figures = [item for item in FIGURES if not item[0].startswith('06_')]
        self.chart_select = ttk.Combobox(navigation, state='readonly', textvariable=self.chart_choice,
                                         values=[v for _, v in self._figures])
        self.chart_select.pack(side='left', fill='x', expand=True, padx=8)
        self.chart_select.bind('<<ComboboxSelected>>', lambda _: self._choose_chart())
        self._current_chart = next(i for i, item in enumerate(self._figures) if item[0].startswith('08_'))
        self.chart_select.current(self._current_chart)
        ttk.Button(navigation, text='▶', width=4, command=lambda: self._chart(1)).pack(side='left')
        ttk.Button(navigation, text='Agrandir', command=self._open_chart).pack(side='left', padx=(9, 0))
        ttk.Button(navigation, text='Ouvrir résultats', command=self._open_results).pack(side='left', padx=(6, 0))
        self.image_box = tk.Frame(right, bg=WHITE, bd=1, relief='solid')
        self.image_box.pack(fill='both', expand=True, padx=4, pady=4)
        self.plot_label = tk.Label(self.image_box, bg=WHITE, fg=MUTED, text='Les graphiques apparaîtront ici après une simulation.', font=('Helvetica', 12))
        self.plot_label.pack(fill='both', expand=True, padx=6, pady=6)
        self.image_box.bind('<Configure>', lambda _: self.after(180, self._display_chart))
        self.details = tk.StringVar(value='Résultats en modèle synthétique Q, sans calibration ni cotation de marché.')
        ttk.Label(right, textvariable=self.details, wraplength=850, style='Hint.TLabel').pack(anchor='w', padx=6, pady=(4, 1))
        logs = ttk.LabelFrame(right, text='Journal d’exécution', padding=6)
        logs.pack(fill='x', pady=(8, 0))
        self.log = tk.Text(logs, height=4, font=('Menlo', 9), bg='#f7f9fc', fg=NAVY, wrap='word', state='disabled')
        self.log.pack(fill='x')
        footer = tk.Label(self, bg=BG, fg=MUTED, font=('Helvetica', 9),
                          text='ACRO = droit de renouveler un coupon conditionnel, pas garantie de coupons. Simulation éducative non calibrée.')
        footer.pack(side='bottom', fill='x', pady=(0, 7))

    def _section(self, parent, title):
        ttk.Label(parent, text=title, style='Section.TLabel').pack(anchor='w', pady=(10, 4))

    def _field(self, parent, path, label, units='', values=None):
        row = ttk.Frame(parent)
        row.pack(fill='x', pady=3)
        ttk.Label(row, text=label, width=23, anchor='w').pack(side='left')
        original = self._lookup(self._base, path)
        if isinstance(original, bool):
            var = tk.BooleanVar(value=original)
            ttk.Checkbutton(row, variable=var).pack(side='left')
        else:
            var = tk.StringVar()
            if values is None:
                ttk.Entry(row, textvariable=var, width=14).pack(side='left', padx=2)
            else:
                ttk.Combobox(row, textvariable=var, values=values, state='readonly', width=15).pack(side='left', padx=2)
        if units:
            ttk.Label(row, text=units, style='Hint.TLabel').pack(side='left', padx=6)
        self.fields[path] = (var, 'percent' if units == '%' else 'raw', values is not None)
        # A few financial parameters also have sliders: users can explore without typing.
        if units == '%' and (path[1] in ('volatilities', 'coupon_barrier',
                                         'capital_barrier', 'autocall_barrier', 'risk_free_rate')):
            low, high = ((1, 65) if path[1] == 'volatilities' else
                         (0, 10) if path[1] == 'risk_free_rate' else (0, 150))
            scale = ttk.Scale(row, from_=low, to=high, orient='horizontal', length=104)
            scale.pack(side='right', padx=(3, 2))
            state = {'busy': False}
            def slider_change(value, v=var, state=state):
                if state['busy']: return
                state['busy'] = True
                try: v.set(f'{float(value):.1f}')
                finally: state['busy'] = False
            def text_change(*_, v=var, sc=scale, state=state):
                if state['busy']: return
                try: value = float(v.get().replace(',', '.'))
                except ValueError: return
                if not math.isfinite(value): return
                state['busy'] = True
                try: sc.set(min(high, max(low, value)))
                finally: state['busy'] = False
            scale.configure(command=slider_change)
            var.trace_add('write', text_change)
        return var

    def _market_fields(self, parent):
        self._section(parent, 'Composition du panier')
        line=ttk.Frame(parent)
        line.pack(fill='x', pady=5)
        ttk.Label(line, text='Nombre d’actions', width=23).pack(side='left')
        self.asset_number=tk.StringVar(value=str(len(self._base['market']['initial_spots'])))
        selector=ttk.Combobox(line, textvariable=self.asset_number, state='readonly', width=14,
                              values=[str(i) for i in range(1,MAX_EQUITIES+1)])
        selector.pack(side='left', padx=2)
        selector.bind('<<ComboboxSelected>>', self._change_asset_count)
        ttk.Label(parent,text='1 à 6 actions ; les pourcentages sont calculés depuis les spots de départ.',
                  style='Hint.TLabel',wraplength=350).pack(anchor='w')
        preset=ttk.Frame(parent)
        preset.pack(fill='x',pady=(7,2))
        ttk.Label(preset,text='Type de panier',width=23).pack(side='left')
        self.basket_template=tk.StringVar(value=self._base['initial_phoenix'].get('basket_rule','worst_of'))
        ttk.Combobox(preset,textvariable=self.basket_template,state='readonly',width=14,
                     values=list(BASKET_CHOICES)).pack(side='left',padx=2)
        ttk.Button(parent,text='Appliquer ce type aux 2 Phoenix',command=self._apply_basket_type).pack(anchor='w',pady=(0,5))
        ttk.Label(parent,text='worst_of : minimum ; best_of : maximum ; equal_weight : moyenne des performances normalisées.',
                  style='Hint.TLabel',wraplength=350).pack(anchor='w')
        self.assets_holder=ttk.Frame(parent)
        self.assets_holder.pack(fill='x')
        self._build_asset_fields()
        self._section(parent, 'Modèle Q')
        self._field(parent, ('market', 'risk_free_rate'), 'Taux sans risque', '%')
        self._field(parent, ('market', 'dynamics'), 'Dynamique', values=['stochastic_vol', 'gbm'])
        self._field(parent, ('market', 'factor_volatility'), 'Vol. du facteur', '%')
        self._field(parent, ('market', 'factor_reversion'), 'Retour à la moyenne')
        self._field(parent, ('market', 'equity_factor_correlation'), 'Corrélation actions/facteur')
        self._field(parent, ('market', 'initial_logvol_factor'), 'Facteur de vol. initial')
        self._field(parent, ('market', 'factor_mean'), 'Moyenne facteur')

    def _build_asset_fields(self):
        # Destroy only market vector/matrix controls, never the fixed model controls.
        for path in list(self.fields):
            if len(path) > 2 and path[0]=='market':
                del self.fields[path]
        for child in self.assets_holder.winfo_children():
            child.destroy()
        self.corr_hints={}
        n=len(self._base['market']['initial_spots'])
        self.corr_pairs=tuple((i,j) for i in range(n) for j in range(i+1,n))
        self._section(self.assets_holder, 'Paramètres individuels')
        for j in range(n):
            self._field(self.assets_holder,('market','initial_spots',j),f'Spot initial action {j+1}')
            self._field(self.assets_holder,('market','volatilities',j),f'Volatilité action {j+1}','%')
            self._field(self.assets_holder,('market','dividends',j),f'Dividende action {j+1}','%')
        if n>1:
            self._section(self.assets_holder, 'Corrélations entre actions')
        for i,j in self.corr_pairs:
            path=('market','correlations',i,j)
            v=self._field(self.assets_holder,path,f'Corrélation {i+1}–{j+1}')
            v.trace_add('write',self._update_correlation_hints)
            hint=tk.StringVar()
            if n==3:
                ttk.Label(self.assets_holder,textvariable=hint,style='Hint.TLabel').pack(anchor='w',padx=14)
            self.corr_hints[i,j]=hint
        if n>1:
            ttk.Label(self.assets_holder,
                text='La matrice doit être définie positive ; validation avant simulation.',
                style='Hint.TLabel',wraplength=340).pack(anchor='w',pady=(4,2))
        for path,(var,_,_) in self.fields.items():
            if len(path)>2 and path[0]=='market':
                var.trace_add('write',self._on_parameter_edit)

    def _apply_basket_type(self):
        rule=self.basket_template.get()
        if rule not in BASKET_CHOICES:
            messagebox.showerror('Type de panier','Règle inconnue.')
            return
        for key in ('initial_phoenix','renewal_phoenix'):
            self.fields[(key,'basket_rule')][0].set(rule)
        self.status.set(f'Règle {rule} appliquée au Phoenix initial et au renouvelé.')

    def _change_asset_count(self, *_):
        try:
            # Previous fields still describe the previous dimension; collect them
            # before changing the underlying market array dimensions.
            old=self._config_from_ui()
            n=int(self.asset_number.get())
            self._base=resize_market_config(old,n)
            self._build_asset_fields()
            for path,(var,unit,_) in self.fields.items():
                if len(path)>2 and path[0]=='market':
                    value=self._lookup(self._base,path)
                    var.set(f'{value*100:.4g}' if unit=='percent' else str(value))
            self._update_correlation_hints()
            self._on_parameter_edit()
            self.status.set(f'Panier : {n} action(s). Relancer la simulation pour actualiser le prix.')
        except Exception as exc:
            self.asset_number.set(str(len(self._base['market']['initial_spots'])))
            messagebox.showerror('Composition du panier',str(exc))

    def _update_correlation_hints(self, *_):
        if len(self._base['market']['initial_spots']) != 3:
            return
        try:
            values={pair:float(self.fields[('market','correlations',*pair)][0].get().replace(',','.'))
                    for pair in CORRELATION_PAIRS}
            limits=correlation_limits(values[0,1],values[0,2],values[1,2])
            for pair in CORRELATION_PAIRS:
                low,high=limits[pair]
                ok=low<values[pair]<high
                self.corr_hints[pair].set(
                    f'{"✓" if ok else "⚠"} Plage : {low:+.3f} < ρ{pair[0]+1}{pair[1]+1} < {high:+.3f}')
        except (ValueError,OverflowError):
            for hint in self.corr_hints.values():
                hint.set('Corrélations invalides ; vérifier les valeurs.')

    def _phoenix_fields(self, parent, key):
        label = 'initial' if key == 'initial_phoenix' else 'renouvelé'
        self._section(parent, f'Phoenix {label}')
        self._field(parent, (key, 'maturity_years'), 'Maturité', 'ans')
        self._field(parent, (key, 'nominal'), 'Nominal', '€')
        self._field(parent, (key, 'basket_rule'), 'Référence panier', values=list(BASKET_CHOICES))
        self._field(parent, (key, 'coupon_barrier'), 'Barrière de coupon', '%')
        self._field(parent, (key, 'capital_barrier'), 'Barrière de capital', '%')
        self._field(parent, (key, 'autocall_barrier'), 'Seuil autocall', '%')
        self._field(parent, (key, 'observations_per_year'), 'Coupons par an')
        self._field(parent, (key, 'first_autocall_year'), 'Premier autocall', 'ans')
        self._field(parent, (key, 'autocall_every_years'), 'Fréquence autocall', 'ans')
        self._field(parent, (key, 'memory'), 'Coupons à mémoire')
        self._field(parent, (key, 'autocall_enabled'), 'Rappel possible')
        self._field(parent, (key, 'capital_barrier_monitoring'), 'Barrière de capital', values=['european', 'continuous'])
        self._field(parent, (key, 'coupon_rate'), 'Coupon fixé si option fixed', '%')

    def _acro_fields(self, parent):
        self._section(parent, 'Droit de réinvestissement ACRO')
        self._field(parent, ('acro', 'strike'), 'Prix de souscription', '€')
        self._field(parent, ('acro', 'renewal_units'), 'Quantité de notes')
        self._field(parent, ('acro', 'last_eligible_call_year'), 'Dernier rappel éligible', 'ans')
        self._field(parent, ('initial_coupon_policy',), 'Coupon initial', values=['fair', 'fixed'])
        self._section(parent, 'Coupon cible')
        self.custom_target = tk.BooleanVar(value=False)
        ttk.Checkbutton(parent, text='Fixer un coupon préservé différent du coupon initial', variable=self.custom_target).pack(anchor='w', pady=4)
        self.target = self._field(parent, ('acro', 'target_coupon_rate'), 'Coupon ACRO cible', '%')
        self._section(parent, 'Simulation')
        self._field(parent, ('simulation', 'steps_per_year'), 'Pas par an')
        self._field(parent, ('simulation', 'quote_paths'), 'Trajectoires cotation')
        self._field(parent, ('simulation', 'outer_paths'), 'Trajectoires externes')
        self._field(parent, ('simulation', 'conditional_paths'), 'Trajectoires internes')
        self._field(parent, ('simulation', 'curve_nodes'), 'Points de la courbe')
        self._field(parent, ('simulation', 'seed'), 'Graine aléatoire')
        ttk.Label(parent, text='Le mode rapide remplace les budgets de calcul (2500 / 1800 / 1200).',
                  style='Hint.TLabel', wraplength=330).pack(anchor='w', pady=10)

    @staticmethod
    def _lookup(d, path):
        for key in path:
            d = d[key]
        return d

    @staticmethod
    def _set(d, path, value):
        current = d
        for key in path[:-1]: current = current[key]
        current[path[-1]] = value

    def _fill(self, config):
        import copy
        self._base=copy.deepcopy(config)
        self.asset_number.set(str(len(config['market']['initial_spots'])))
        self.basket_template.set(config['initial_phoenix'].get('basket_rule','worst_of'))
        self._build_asset_fields()
        for path, (var, unit, _) in self.fields.items():
            value = self._lookup(config, path)
            if isinstance(var, tk.BooleanVar): var.set(bool(value))
            elif value is None: var.set('')
            elif unit == 'percent': var.set(f'{value*100:.4g}')
            else: var.set(str(value))
        self.custom_target.set(config['acro']['target_coupon_rate'] is not None)

    def _config_from_ui(self):
        validate_custom_coupon_selection(self.custom_target.get(), self.target.get())
        data = json.loads(json.dumps(self._base))
        for path, (var, unit, choice) in self.fields.items():
            raw = var.get()
            old = self._lookup(data, path)
            if path == ('acro', 'target_coupon_rate') and not self.custom_target.get():
                val = None
            elif isinstance(var, tk.BooleanVar):
                val = bool(raw)
            elif choice:
                val = raw
            elif raw.strip() == '' and old is None:
                val = None
            else:
                try:
                    number = float(raw.strip().replace(',', '.'))
                    if not math.isfinite(number): raise ValueError('non finite')
                    if (path[0] == 'simulation' or path[-1] == 'observations_per_year') and isinstance(old, int) and not isinstance(old, bool):
                        if not number.is_integer(): raise ValueError('entier attendu')
                        val = int(number)
                    else:
                        val = number / 100.0 if unit == 'percent' else number
                except ValueError as ex:
                    raise ValueError(f'Paramètre invalide {" / ".join(map(str,path))}: {raw!r}') from ex
            self._set(data, path, val)
        corr = data['market']['correlations']
        for i, j in self.corr_pairs: corr[j][i] = corr[i][j]
        if len(corr) == 3:
            check_correlation_triple(corr[0][1], corr[0][2], corr[1][2])
        # Authoritative project validation (same routines as pricing engine).
        from acro.config import load_project
        with tempfile.NamedTemporaryFile('w', suffix='.json', encoding='utf8', delete=False) as f:
            json.dump(data, f)
            temp = Path(f.name)
        try: load_project(temp)
        finally: temp.unlink(missing_ok=True)
        return data

    def _snapshot(self):
        """Snapshot of inputs; used only to flag stale displayed quotes."""
        items = tuple((str(path), str(var.get())) for path, (var, _, _) in sorted(self.fields.items()))
        return items, self.asset_number.get(), bool(self.quick.get()), bool(self.sensitivity.get()), bool(self.custom_target.get()), self.illustrative_count.get()

    def _on_parameter_edit(self, *_):
        if self._dataset is not None:
            if self._snapshot() != self._last_run_snapshot:
                self.snapshot_info.set('⚠ Paramètres modifiés depuis ce calcul : cliquer sur SIMULER pour actualiser les résultats.')
            else:
                self.snapshot_info.set('Résultats du dernier calcul : paramètres inchangés.')

    def _copy_terms(self):
        for path, (var, _, _) in self.fields.items():
            if path[0] == 'initial_phoenix':
                target = ('renewal_phoenix',) + path[1:]
                if target in self.fields: self.fields[target][0].set(var.get())
        self.status.set('Term sheet initiale copiée vers le Phoenix renouvelé.')

    def _reset(self):
        self._fill(self._original_config)
        self.quick.set(True)
        self.sensitivity.set(False)
        self.illustrative_count.set('48')
        self._update_correlation_hints()
        self.status.set('Valeurs initiales restaurées ; aucun fichier source modifié.')

    def _export(self):
        try: data = self._config_from_ui()
        except Exception as ex: messagebox.showerror('Paramètres', str(ex)); return
        dest = filedialog.asksaveasfilename(title='Exporter une configuration', defaultextension='.json',
                                            filetypes=[('JSON', '*.json')])
        if dest:
            Path(dest).write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding='utf8')
            self.status.set('Configuration exportée : ' + str(dest))

    def _log(self, line):
        self.log.configure(state='normal')
        self.log.insert('end', line.rstrip() + '\n')
        self.log.see('end')
        self.log.configure(state='disabled')

    def _start(self):
        if self._running: return
        if not self.quick.get() and not messagebox.askyesno('Calcul approfondi',
                'Le mode approfondi utilise les budgets complets de Monte Carlo et peut demander davantage de calcul. Continuer ?'):
            return
        try: config = self._config_from_ui()
        except Exception as ex:
            messagebox.showerror('Configuration invalide', f'{ex}\n\nCorrige les paramètres avant de simuler.')
            self.status.set('Validation refusée. Aucune simulation lancée.')
            return
        stamp = datetime.now().strftime('%Y%m%d_%H%M%S_%f')
        run_dir = RUNS / f'essai_{stamp}'
        run_dir.mkdir(parents=True, exist_ok=False)
        cfg = run_dir / 'input_config.json'
        cfg.write_text(json.dumps(config, indent=2, ensure_ascii=False), encoding='utf8')
        out = run_dir / 'resultats'
        cmd = [sys.executable, str(ROOT / 'run_demo.py'), '--config', str(cfg), '--output', str(out), '--pricing-method', 'auto',
               '--illustrative-paths', self.illustrative_count.get()]
        if self.quick.get(): cmd.append('--quick')
        if not self.sensitivity.get(): cmd.append('--no-sensitivity')
        self._pending_snapshot = self._snapshot()
        self._cancel_requested = threading.Event()
        self._running = True
        self._process = None
        self.run_button.configure(state='disabled')
        self.cancel_button.configure(state='normal')
        self.progress.start(9)
        self.status.set('Simulation en cours… Le moteur ACRO calcule avec les paramètres saisis.')
        self._log(f'▶ Simulation {run_dir.name} (rapide={self.quick.get()})')
        threading.Thread(target=self._worker, args=(cmd, out, run_dir, self._cancel_requested), daemon=False).start()

    def _worker(self, cmd, out, run_dir, cancel_event):
        try:
            status, output = run_cancelable_process(cmd, ROOT, cancel_event,
                                                     on_started=lambda p: setattr(self, '_process', p))
            if status == 'cancelled':
                # The child is fully finished before removing any incomplete run.
                shutil.rmtree(run_dir, ignore_errors=True)
                self._queue.put(('cancelled', None))
            elif status == 'error':
                self._queue.put(('error', output))
            else:
                self._queue.put(('success', (out, output)))
        except Exception as ex:
            if cancel_event.is_set():
                shutil.rmtree(run_dir, ignore_errors=True)
                self._queue.put(('cancelled', None))
            else:
                self._queue.put(('error', repr(ex)))

    def _poll(self):
        try:
            while True:
                kind, payload = self._queue.get_nowait()
                if kind == 'success' and self._cancel_requested.is_set():
                    out, _ = payload
                    shutil.rmtree(out.parent, ignore_errors=True)
                    kind = 'cancelled'
                if kind == 'success':
                    out, output = payload
                    self._load(out)
                    self._log(output[-750:] or 'Simulation terminée.')
                    self.status.set('Simulation terminée. Tes résultats sont dans ' + str(out))
                elif kind == 'cancelled':
                    self.status.set('Simulation annulée. Les résultats précédents sont conservés.')
                    self._log('Simulation annulée, aucun résultat partiel publié.')
                elif kind == 'error':
                    self.status.set('Échec de la simulation : les résultats précédents sont conservés.')
                    self._log('ERREUR : ' + str(payload))
                    messagebox.showerror('Simulation ACRO', str(payload)[-1800:])
                self._running = False
                self._process = None
                self._pending_snapshot = None
                self.progress.stop()
                self.run_button.configure(state='normal')
                self.cancel_button.configure(state='disabled')
        except queue.Empty:
            pass
        self.after(110, self._poll)

    def _cancel(self):
        if not self._running:
            return
        # Record cancellation BEFORE the worker has returned from Popen.
        self._cancel_requested.set()
        self.cancel_button.configure(state='disabled')
        self.status.set('Arrêt demandé…')
        proc = self._process
        if proc is not None and proc.poll() is None:
            try:
                proc.terminate()
            except ProcessLookupError:
                pass

    def _load_preloaded(self):
        if self._dataset is None:
            folder = ROOT / 'RESULTATS_DEMO'
            if (folder / 'metrics.json').exists():
                try: self._load(folder, previous=False)
                except Exception: pass

    def _load(self, folder, previous=True):
        folder = Path(folder)
        m = json.loads((folder / 'metrics.json').read_text(encoding='utf8'))
        if previous and self._dataset is not None:
            self._previous = self._dataset[1]
        self._dataset = (folder, m)
        self._last_run_snapshot = self._pending_snapshot or self._snapshot()
        self._on_parameter_edit()
        nominal = m.get('initial_nominal')
        if nominal is None:
            cfg_file = folder / 'effective_config.json'
            nominal = json.loads(cfg_file.read_text(encoding='utf-8'))['initial_phoenix']['nominal'] if cfg_file.exists() else 100
        nominal = float(nominal)
        denom = f'contrat nominal {nominal:g} €'
        self.kpi_titles['acro_price'].configure(text=f'Prime séparée / {denom}')
        policy = m.get('initial_coupon_policy', 'fair')
        self.kpi_titles['bundled_fair_coupon'].configure(
            text='Coupon autofinancé (sans prime)' if policy=='fair' else 'Package au pair : non comparable')
        bundle = m.get('bundled_fair_coupon')
        premium = m.get('acro_price')
        if policy == 'fixed':
            self.finance_info.set(
                f'Coupon contractuel FIXÉ à {m["annual_coupon"]:.3%} ; prime séparée ACRO : {premium:.2f} € / {denom}. '
                'Un nouveau package au pair aurait une autre économie : aucun coupon de réduction comparable n’est affiché.')
        elif bundle is None:
            self.finance_info.set('Mode A : option payée séparément, coupon initial inchangé. Mode B : autofinancement non disponible dans cette configuration.')
        else:
            self.finance_info.set(
                f'Mode A — coupon {m["annual_coupon"]:.3%} + prime séparée {premium:.2f} € / {denom}.   '
                f'OU mode B — coupon {bundle:.3%}, sans prime séparée '
                f'(ACRO valorisée {m.get("bundled_option_value",float("nan")):.2f} € / {denom} dans ce package).')
        initial_B=m.get('initial_principal_pv')
        initial_coupon_pv=m.get('initial_coupon_leg_pv')
        initial_A=m.get('initial_coupon_annuity_pv_per_unit_rate')
        if initial_B is not None and initial_coupon_pv is not None and initial_A is not None:
            self.decomp_info.set(
                f'Pourquoi ce coupon ? Prix Phoenix {m["initial_phoenix_price"]:.2f} € = '
                f'{initial_B:.2f} € de capital espéré actualisé + '
                f'{initial_coupon_pv:.2f} € de coupons conditionnels actualisés '
                f'(A = {initial_A:.2f} € par unité de taux annuel). Modèle Q, non garanti.')
        else:
            self.decomp_info.set('Décomposition indisponible pour cette ancienne simulation.')
        risk=m.get('phoenix_risk')
        if risk:
            n=risk['sample_paths']
            p_loss=risk['capital_loss_probability_Q']
            p_no_call=risk['no_autocall_probability_Q']
            loss_mean=risk.get('mean_nominal_loss_if_loss')
            self.risk_info.set(
                f'⚠ Risque en capital sous Q : P(perte sur nominal) ≈ {p_loss:.1%} ; '
                f'P(aucun autocall) ≈ {p_no_call:.1%} ; '
                f'perte moyenne si perte ≈ {loss_mean:.1f} €' if loss_mean is not None else
                f'⚠ Risque sous Q : P(perte sur nominal) ≈ {p_loss:.1%} ; aucun scénario perdant dans cet échantillon.')
            self.put_info.set(
                f'Décomposition pédagogique (mêmes {n:,} chemins bruts) : '
                f'{risk["pv_redemption_before_loss"]:.1f} € de capital sans pertes à la date de sortie '
                f'− {risk["pv_capital_loss"]:.1f} € de pertes contingentes sur panier '
                f'+ {risk["pv_conditional_coupons_uncontrolled"]:.1f} € de coupons '
                f'= {risk["pv_note_uncontrolled"]:.1f} € (échantillon brut). '
                'Ce n’est PAS un put vanilla coté ; les chiffres peuvent différer de la cotation contrôlée ci-dessus.')
        else:
            self.risk_info.set('Risque sur le nominal : lancer une nouvelle simulation pour l’estimer.')
            self.put_info.set('Décomposition des pertes conditionnelles indisponible pour cet ancien calcul.')
        for key, label in self.kpis.items():
            val = m.get(key)
            if val is None: text = 'n.d.'
            elif key == 'acro_price': text = f'{val:.3f} €'
            elif key == 'bundled_fair_coupon' and policy == 'fixed': text = '—'
            else: text = f'{100*val:.3f} %'
            label.configure(text=text)
        compare = ''
        if comparable_run_snapshots(m, self._previous):
            compare = (f" | vs précédent (même nominal et politique de coupon) : "
                       f"prime {m['acro_price']-self._previous['acro_price']:+.3f} € / {denom} "
                       "(inclut le bruit Monte Carlo)")
        positive=m.get('positive_payoff_probability_Q')
        exercise=(f' • P(payoff > 0)={positive:.1%}' if positive is not None else '')
        outer = int(m.get('n_outer_paths', 0))
        quote_n = int(m.get('n_quote_paths', 0))
        visible = int(m.get('illustrative_paths', 48))
        self.details.set(
            f"{m['market_dynamics']} • {m['assets']} actions • Pricing : {outer:,} chemins externes, "
            f"{quote_n:,} chemins de cotation ; {visible:,} trajectoires affichées séparément • "
            f"E.T. MC externe seulement {m['acro_outer_se_ONLY']:.4f} €{exercise}{compare}"
        )
        self._sync_available_figures(folder)
        self._display_chart()

    def _sync_available_figures(self, folder):
        """Expose only actually generated figures (notably optional sensitivity 06)."""
        current = self._figures[self._current_chart][0] if self._figures else '08_trajectoires_monte_carlo.png'
        self._figures = [
            (name, f'{int(self._dataset[1].get("illustrative_paths", 48))} trajectoires Monte Carlo : panier'
             if name.startswith('08_') else label)
            for name, label in available_figures(folder)
        ]
        self.chart_select.configure(values=[label for _, label in self._figures])
        if self._figures:
            self._current_chart = next((i for i, item in enumerate(self._figures) if item[0] == current), 0)
            self.chart_select.current(self._current_chart)
        else:
            self._current_chart = 0
            self.chart_choice.set('Aucune figure générée')

    def _choose_chart(self):
        self._current_chart = max(0, self.chart_select.current())
        self._display_chart()

    def _chart(self, direction):
        if not self._figures: return
        self._current_chart = (self._current_chart + direction) % len(self._figures)
        self.chart_select.current(self._current_chart)
        self._display_chart()

    def _display_chart(self):
        if not self._dataset or not self._figures: return
        folder, _ = self._dataset
        name = self._figures[self._current_chart][0]
        src = folder / name
        if not src.exists():
            self.plot_label.configure(image='', text='Figure indisponible pour cette simulation.')
            self._images = None
            return
        try:
            from PIL import Image, ImageTk
            # On macOS use actual dimensions, avoiding images too large for a Retina window.
            width = max(280, self.image_box.winfo_width() - 24)
            height = max(230, self.image_box.winfo_height() - 24)
            with Image.open(src) as im:
                im = im.convert('RGB')
                im.thumbnail((width, height), Image.Resampling.LANCZOS)
                self._images = ImageTk.PhotoImage(im)
            self.plot_label.configure(image=self._images, text='')
        except Exception as ex:
            self.plot_label.configure(image='', text=f'Image indisponible : {ex}')
            self._images = None

    def _open_chart(self):
        if not self._dataset or not self._figures:return
        source=self._dataset[0]/self._figures[self._current_chart][0]
        if not source.exists():return
        if sys.platform == 'darwin':subprocess.Popen(['open',str(source)])
        elif os.name == 'nt':os.startfile(str(source))
        else:subprocess.Popen(['xdg-open',str(source)])

    def _open_results(self):
        if not self._dataset: return
        folder = self._dataset[0]
        if sys.platform == 'darwin': subprocess.Popen(['open', str(folder)])
        elif os.name == 'nt': os.startfile(str(folder))
        else: subprocess.Popen(['xdg-open', str(folder)])

    def _on_close(self):
        if self._running:
            if not messagebox.askyesno('Simulation en cours', 'Arrêter la simulation et fermer la fenêtre ?'):
                return
            self._cancel()
        self.destroy()


def main():
    if not BASE_CONFIG.exists():
        raise SystemExit('Lance ce fichier dans le dossier du projet ACRO complet.')
    app = ACROApp()
    app.mainloop()

if __name__ == '__main__':
    main()
