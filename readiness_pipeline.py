# =============================================================================
#  EMPIRICAL IMPLEMENTATION-READINESS ANALYSIS  --  single-cell pipeline
#  "From problem discovery to empirical readiness assessment"
#
#  Operationalises the four readiness dimensions of the problem-discovery study
#  (Mitra et al.) on the Dhaka occupational-exposure survey:
#     E  Epistemic adequacy       -> label provenance, identifiability, data resolution
#     J  Distributive justice     -> exposure burden, outcome visibility, error equity
#     G  Ethical governance       -> explanation stability, expert-data concordance,
#                                    counterfactual actionability, re-identification risk
#     L  Institutional legitimacy -> discrimination, calibration, net benefit, transfer
#
#  Inputs (run from the repository root):
#     data/survey_harmonised.csv   -> harmonised survey (7,599 respondents)
#     data/area_pm25_lookup.csv    -> CAMS reanalysis PM2.5 by zone (2024 mean)
#
#  All outputs are written to  results/ :
#     numbers.tex          -> every value reported in the paper (key = value)
#     tab_*.tex            -> booktabs tabulars used in the paper
#     fig_*.pdf / .png     -> figures used in the paper
#     csv/*.csv            -> machine-readable copies of every table
#     run_info.json        -> config, package versions, runtime
#
#  Runtime: ~11-16 min on a laptop with FAST=False (5x5 repeated CV).
#           Set FAST=True for a ~2 min smoke test (numbers NOT paper-grade).
# =============================================================================

# ----------------------------------------------------------------------------- 0. setup
import os, sys, json, time, warnings, subprocess, importlib, platform
AUTO_INSTALL = True
for _mod, _pkg in {'xgboost': 'xgboost', 'shap': 'shap', 'statsmodels': 'statsmodels',
                   'sklearn': 'scikit-learn', 'seaborn': 'seaborn', 'scipy': 'scipy'}.items():
    try:
        importlib.import_module(_mod)
    except ImportError:
        if AUTO_INSTALL:
            subprocess.run([sys.executable, '-m', 'pip', 'install', '-q', _pkg], check=False)

warnings.filterwarnings('ignore')
try:
    sys.stdout.reconfigure(encoding='utf-8')   # Windows consoles default to cp1252
except Exception:
    pass
import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import statsmodels.api as sm
import statsmodels.formula.api as smf
import shap
import sklearn
import xgboost
from xgboost import XGBClassifier
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler, LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.dummy import DummyClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.utils.class_weight import compute_sample_weight
from sklearn.inspection import permutation_importance
from sklearn.metrics import (roc_auc_score, average_precision_score, balanced_accuracy_score,
                             f1_score, brier_score_loss, recall_score, precision_score,
                             accuracy_score, normalized_mutual_info_score)

T0 = time.time()
CFG = dict(
    DATA_PATH=os.path.join('data', 'survey_harmonised.csv'),
    PM25_LOOKUP=os.path.join('data', 'area_pm25_lookup.csv'),
    OUT_DIR='results',
    SEED=42,
    N_SPLITS=5,
    N_REPEATS=5,          # 5x5 repeated stratified CV
    N_TREES=300,
    FAST=False,           # True -> quick smoke test
    DEDUP=True,           # drop exact duplicate records within each scenario
    Y_TO_UNSPECIFIED=True,  # raw code "Y" (students) was mapped to Bronchiectasis in cleaning; undo it
    PRIMARY='B',          # primary label scenario for downstream analyses
    THRESH=0.5,           # decision threshold for binary metrics
    N_EEI_DRAWS=1000,     # Dirichlet draws for EEI weight sensitivity
    N_PERM=5,             # permutation-importance repeats
)
if os.environ.get('READINESS_FAST') == '1':
    CFG['FAST'] = True
if CFG['FAST']:
    CFG.update(N_REPEATS=1, N_TREES=100, N_EEI_DRAWS=200, N_PERM=2)

SEED = CFG['SEED']
np.random.seed(SEED)
OUT = CFG['OUT_DIR']
os.makedirs(os.path.join(OUT, 'csv'), exist_ok=True)

OKABE = ['#0072B2', '#D55E00', '#009E73', '#CC79A7', '#E69F00', '#56B4E9', '#F0E442', '#000000']
sns.set_theme(style='whitegrid', context='paper', font_scale=1.1)
plt.rcParams.update({'figure.dpi': 110, 'savefig.dpi': 300, 'axes.titleweight': 'bold',
                     'pdf.fonttype': 42, 'axes.spines.top': False, 'axes.spines.right': False})


def log(msg):
    print(f'[{time.time() - T0:7.1f}s] {msg}', flush=True)


# ----------------------------------------------------------------------------- output helpers
NUM = {}  # key -> formatted string, exported as \res{key}


def put(key, val, fmt='{:.3f}'):
    assert key.isalnum(), key
    if isinstance(val, str):
        NUM[key] = tex_escape(val)
    elif val is None or (isinstance(val, float) and np.isnan(val)):
        NUM[key] = '--'
    else:
        NUM[key] = fmt.format(val)


_TEX = {'&': r'\&', '%': r'\%', '$': r'\$', '#': r'\#', '_': r'\_', '{': r'\{', '}': r'\}',
        '~': r'\textasciitilde{}', '^': r'\textasciicircum{}', '\\': r'\textbackslash{}',
        '±': r'$\pm$', '–': '--', '≥': r'$\geq$', '≤': r'$\leq$', 'ρ': r'$\rho$', '×': r'$\times$',
        '→': r'$\rightarrow$', 'Δ': r'$\Delta$', '✓': r'\checkmark', 'χ': r'$\chi$', '²': r'$^2$'}


def tex_escape(s):
    return ''.join(_TEX.get(ch, ch) for ch in str(s))


def write_table(df, name, col_fmt=None, midrules=(), index=False):
    """Write a booktabs tabular to OUT/name.tex and a CSV copy."""
    d = df.reset_index() if index else df.copy()
    d.to_csv(os.path.join(OUT, 'csv', name + '.csv'), index=False)
    col_fmt = col_fmt or ('l' + 'r' * (d.shape[1] - 1))
    lines = [r'\begin{tabular}{' + col_fmt + '}', r'\toprule',
             ' & '.join(tex_escape(c) for c in d.columns) + r' \\', r'\midrule']
    for i, (_, row) in enumerate(d.iterrows()):
        if i in midrules:
            lines.append(r'\midrule')
        lines.append(' & '.join(tex_escape(v) for v in row.values) + r' \\')
    lines += [r'\bottomrule', r'\end{tabular}']
    with open(os.path.join(OUT, name + '.tex'), 'w', encoding='utf-8') as f:
        f.write('\n'.join(lines) + '\n')


def savefig(fig, name):
    fig.savefig(os.path.join(OUT, name + '.pdf'), bbox_inches='tight')
    fig.savefig(os.path.join(OUT, name + '.png'), bbox_inches='tight')
    plt.close(fig)


def fmt_ms(m, s, d=3):
    return f'{m:.{d}f} ± {s:.{d}f}'


# ----------------------------------------------------------------------------- 1. data
log('1. Loading data')
raw = pd.read_csv(CFG['DATA_PATH'])
if 'row_id' in raw.columns:
    raw = raw.drop(columns='row_id')

OCC_SHORT = {
    'Bus Driver/Private Car Driver (AC)': 'Driver (AC)',
    'Bus Driver (Non-AC)': 'Driver (non-AC)',
    'Rickshaw Puller/Biker': 'Rickshaw/biker',
    'Traffic Cop/Fire Fighter/Guard': 'Police/FF/guard',
    'Business Owner': 'Business owner',
    'Student': 'Student',
}
OCC_ORDER = ['Driver (AC)', 'Driver (non-AC)', 'Rickshaw/biker', 'Police/FF/guard', 'Business owner', 'Student']
raw['occ'] = raw['occupation'].map(OCC_SHORT)

raw['fine'] = raw['lung_disease']
if CFG['Y_TO_UNSPECIFIED']:
    # In the student sheet the raw code "Y" (n=65) was recoded to "Bronchiectasis" during
    # cleaning; no other student record had that label. Treat it as "Unspecified".
    raw.loc[(raw['occupation'] == 'Student') & (raw['fine'] == 'Bronchiectasis'), 'fine'] = 'Unspecified'

CATEGORY = {
    'No Disease': 'No disease', 'Asthma': 'Obstructive/allergic', 'Bronchitis': 'Obstructive/allergic',
    'Chronic Bronchitis': 'Obstructive/allergic', 'Bronchiectasis': 'Obstructive/allergic',
    'Allergy': 'Obstructive/allergic', 'Tuberculosis': 'Infectious', 'Pneumonia': 'Infectious',
    'Fungal Lung Infections': 'Infectious', 'Lung Cancer': 'Malignant', 'Mesothelioma': 'Malignant',
    'Cystic Fibrosis': 'Other', 'Sarcoidosis': 'Other', 'Unspecified': 'Other',
}

OPEN_AIR = {'Rickshaw', 'Bike', 'Motorbike', 'Bicycle', 'Walking', 'CNG', 'CNG Auto-rickshaw', 'Tempo', 'Leguna'}
ENCLOSED = {'Car', 'Private Car', 'Uber', 'Metro', 'Train'}
raw['vehicle_grp'] = np.where(raw['travel_vehicle'].isin(OPEN_AIR), 'Open-air',
                              np.where(raw['travel_vehicle'].isin(ENCLOSED), 'Enclosed', 'Bus'))
LOW_HOUSING = {'Tin Shed', 'Semi-pucca', 'Open House', 'Shared Dorm'}
raw['housing'] = np.where(raw['house_type'].isin(LOW_HOUSING), 'Low-quality', 'Standard')
raw['age_band'] = pd.cut(raw['age'], [0, 29, 44, 200], labels=['<30', '30-44', '45+']).astype(str)

# --- Environmental Exposure Index (identical weights to the discovery-phase pipeline)
W_OCC = {'Driver (AC)': 0.50, 'Driver (non-AC)': 0.90, 'Rickshaw/biker': 1.00,
         'Police/FF/guard': 0.95, 'Business owner': 0.40, 'Student': 0.30}
W_VEH = {'Bus': 0.9, 'CNG': 0.85, 'CNG Auto-rickshaw': 0.85, 'Tempo': 0.8, 'Leguna': 0.8,
         'Rickshaw': 0.5, 'Bike': 0.7, 'Motorbike': 0.75, 'Private Car': 0.4, 'Car': 0.4,
         'Uber': 0.35, 'Metro': 0.1, 'Train': 0.15, 'Walking': 0.6, 'Bicycle': 0.55}
# Layer-1 modifiers from build_exposure_features.py (second, independent expert scheme)
L1_VEH = {'Metro': 0.10, 'Train': 0.15, 'Uber': 0.25, 'Car': 0.25, 'Private Car': 0.25, 'Bus': 0.45,
          'CNG': 0.55, 'CNG Auto-rickshaw': 0.60, 'Tempo': 0.60, 'Leguna': 0.65, 'Motorbike': 0.80,
          'Bike': 0.80, 'Rickshaw': 0.85, 'Bicycle': 0.85, 'Walking': 0.90}
L1_TIER = {'Student': 1, 'Business owner': 2, 'Driver (AC)': 2, 'Police/FF/guard': 3,
           'Driver (non-AC)': 4, 'Rickshaw/biker': 4}
EEI_W = np.array([0.30, 0.25, 0.20, 0.15, 0.10])
EEI_NAMES = ['Travel time', 'Occupation', 'Vehicle', 'Workplace share', 'No AC']
EEI_COMP = np.column_stack([
    raw['travel_hours'] / 24.0,
    raw['occ'].map(W_OCC).fillna(0.5),
    raw['travel_vehicle'].map(W_VEH).fillna(0.5),
    raw['stay_office_hours'] / (raw['stay_home_hours'] + raw['stay_office_hours'] + 1e-5),
    raw['ac_service'].map({'Yes': 0.3, 'No': 1.0}).fillna(0.65),   # Unknown -> midpoint
])
raw['eei'] = 100 * EEI_COMP @ EEI_W
_w_noocc = EEI_W.copy(); _w_noocc[1] = 0; _w_noocc /= _w_noocc.sum()
raw['eei_noocc'] = 100 * EEI_COMP @ _w_noocc

FEAT_RAW = ['occupation', 'living_area', 'work_area', 'stay_home_hours', 'stay_office_hours',
            'travel_hours', 'travel_timing', 'travel_vehicle', 'ac_service', 'work_days',
            'house_type', 'age', 'gender', 'smoker_status']


def make_scenario(s):
    """A = as coded by the discovery pipeline (blank disease field excluded);
    B = confirmed reading (data collectors: a blank disease field means 'no disease'). B is primary."""
    d = raw.copy()
    n_unknown = int((d['fine'] == 'Unknown').sum())
    if s == 'A':
        d = d[d['fine'] != 'Unknown']
    else:
        d.loc[d['fine'] == 'Unknown', 'fine'] = 'No Disease'
    d['blank_nd'] = d['lung_disease'].eq('Unknown')
    n_before = len(d)
    if CFG['DEDUP']:
        d = d.drop_duplicates(subset=FEAT_RAW + ['fine'])
    d = d.reset_index(drop=True)
    d['y_bin'] = (d['fine'] != 'No Disease').astype(int)
    d['y_cat'] = d['fine'].map(CATEGORY)
    assert d['y_cat'].notna().all(), d.loc[d['y_cat'].isna(), 'fine'].unique()
    info = dict(n_unknown=n_unknown, n_before_dedup=n_before, n_dups=n_before - len(d), n=len(d))
    return d, info


D, DINFO = {}, {}
for s in ['A', 'B']:
    D[s], DINFO[s] = make_scenario(s)
    log(f'   scenario {s}: n={DINFO[s]["n"]} (dups removed {DINFO[s]["n_dups"]}), '
        f'prevalence={D[s]["y_bin"].mean():.3f}')
    put(f'N{s}', DINFO[s]['n'], '{:,}')
    put(f'Dups{s}', DINFO[s]['n_dups'], '{:,}')
    put(f'Prev{s}', 100 * D[s]['y_bin'].mean(), '{:.1f}')
put('NRaw', len(raw), '{:,}')
put('NUnknown', int((raw['fine'] == 'Unknown').sum()), '{:,}')
put('NSplits', CFG['N_SPLITS'], '{:d}'); put('NRepeats', CFG['N_REPEATS'], '{:d}')
put('NFolds', CFG['N_SPLITS'] * CFG['N_REPEATS'], '{:d}'); put('NTrees', CFG['N_TREES'], '{:d}')

# --- Table: cohort by occupation
rows = []
for g in OCC_ORDER + ['All']:
    d = raw if g == 'All' else raw[raw['occ'] == g]
    rows.append({'Occupation': g, 'n': f'{len(d):,}',
                 'Age': f'{d.age.mean():.1f} ({d.age.std():.1f})',
                 'Male %': f'{100 * (d.gender == "Male").mean():.1f}',
                 'Smoker %': f'{100 * (d.smoker_status == "Smoker").mean():.1f}',
                 'Travel h': f'{d.travel_hours.mean():.2f}',
                 'Work h': f'{d.stay_office_hours.mean():.1f}',
                 'Low-quality housing %': f'{100 * (d.housing == "Low-quality").mean():.1f}',
                 'EEI': f'{d.eei.mean():.1f} ({d.eei.std():.1f})',
                 'Disease %': f'{100 * (~d.fine.isin(["No Disease", "Unknown"])).mean():.1f}',
                 'No disease left blank %': f'{100 * (d.fine == "Unknown").mean():.1f}'})
write_table(pd.DataFrame(rows), 'tab_cohort', 'l' + 'r' * 10, midrules=(len(OCC_ORDER),))

# ----------------------------------------------------------------------------- 2. modelling utilities
NUM_EXPO = ['travel_hours', 'stay_home_hours', 'stay_office_hours', 'work_days']
CAT_EXPO = ['travel_vehicle', 'travel_timing', 'ac_service', 'house_type']
AREA = ['living_area', 'work_area']
FSETS = {
    'OCC': dict(num=[], cat=['occupation'], area=[]),
    'DEMO': dict(num=['age'], cat=['gender', 'smoker_status'], area=[]),
    'EXPO': dict(num=NUM_EXPO, cat=CAT_EXPO, area=AREA),
    'EXPODEMO': dict(num=NUM_EXPO + ['age'], cat=CAT_EXPO + ['gender', 'smoker_status'], area=AREA),
    'FULL': dict(num=NUM_EXPO + ['age', 'eei'], cat=CAT_EXPO + ['gender', 'smoker_status', 'occupation'], area=AREA),
}
FSET_LABEL = {'OCC': 'Occupation only', 'DEMO': 'Demographics only', 'EXPO': 'Exposure only',
              'EXPODEMO': 'Exposure + demographics', 'FULL': 'Full (+ occupation, EEI)'}
MODELS = ['LR', 'RF', 'XGB']
MODEL_LABEL = {'Prior': 'Prior (majority)', 'LR': 'Logistic regression', 'RF': 'Random forest', 'XGB': 'XGBoost'}


class FreqEncoder(BaseEstimator, TransformerMixin):
    """Frequency encoding fitted on the training fold only (unseen levels -> 0)."""
    def fit(self, X, y=None):
        X = pd.DataFrame(X)
        self.cols_ = list(X.columns)
        self.maps_ = {c: X[c].value_counts(normalize=True).to_dict() for c in self.cols_}
        return self

    def transform(self, X):
        X = pd.DataFrame(X, columns=self.cols_)
        return np.column_stack([X[c].map(self.maps_[c]).astype(float).fillna(0.0).values for c in self.cols_])

    def get_feature_names_out(self, input_features=None):
        return np.array([f'{c}_freq' for c in self.cols_])


def make_pre(fs):
    tr = []
    if fs['num']:
        tr.append(('num', StandardScaler(), fs['num']))
    if fs['cat']:
        tr.append(('cat', OneHotEncoder(handle_unknown='ignore', sparse_output=False), fs['cat']))
    if fs['area']:
        tr.append(('area', FreqEncoder(), fs['area']))
    return ColumnTransformer(tr, verbose_feature_names_out=False)


def make_model(name, k):
    if name == 'Prior':
        return DummyClassifier(strategy='prior')
    if name == 'LR':
        return LogisticRegression(max_iter=5000, class_weight='balanced')
    if name == 'RF':
        return RandomForestClassifier(n_estimators=CFG['N_TREES'], min_samples_leaf=3,
                                      class_weight='balanced_subsample', n_jobs=-1, random_state=SEED)
    if name == 'XGB':
        return XGBClassifier(n_estimators=CFG['N_TREES'], max_depth=4, learning_rate=0.05, subsample=0.8,
                             colsample_bytree=0.8, tree_method='hist', n_jobs=-1, random_state=SEED,
                             eval_metric='logloss' if k == 2 else 'mlogloss', verbosity=0)
    raise ValueError(name)


def make_pipe(fsname, mname, k):
    return Pipeline([('pre', make_pre(FSETS[fsname])), ('clf', make_model(mname, k))])


def fit_pipe(pipe, X, y, mname):
    if mname == 'XGB':
        pipe.fit(X, y, clf__sample_weight=compute_sample_weight('balanced', y))
    else:
        pipe.fit(X, y)
    return pipe


def metrics(y, P, k):
    if k == 2:
        p = P[:, 1]; yh = (p >= CFG['THRESH']).astype(int)
        return dict(AUC=roc_auc_score(y, p), AP=average_precision_score(y, p),
                    BalAcc=balanced_accuracy_score(y, yh), F1m=f1_score(y, yh, average='macro'),
                    Brier=brier_score_loss(y, p), Sens=recall_score(y, yh, zero_division=0),
                    Spec=recall_score(1 - y, 1 - yh, zero_division=0))
    yh = P.argmax(1)
    return dict(AUC=roc_auc_score(y, P, multi_class='ovr', average='macro', labels=np.arange(k)),
                BalAcc=balanced_accuracy_score(y, yh), F1m=f1_score(y, yh, average='macro'),
                F1w=f1_score(y, yh, average='weighted'), Acc=accuracy_score(y, yh))


def run_cv(df, ycol, fsname, mname, repeats=None, keep_models=False):
    repeats = repeats or CFG['N_REPEATS']
    le = LabelEncoder(); y = le.fit_transform(df[ycol]); k = len(le.classes_)
    rows, oof, models = [], np.full((len(df), k), np.nan), []
    for r in range(repeats):
        skf = StratifiedKFold(CFG['N_SPLITS'], shuffle=True, random_state=SEED + r)
        for f, (tr, te) in enumerate(skf.split(df, y)):
            pipe = fit_pipe(make_pipe(fsname, mname, k), df.iloc[tr], y[tr], mname)
            P = pipe.predict_proba(df.iloc[te])
            m = metrics(y[te], P, k); m.update(repeat=r, fold=f, n_train=len(tr), n_test=len(te))
            rows.append(m)
            if r == 0:
                oof[te] = P
                if keep_models:
                    models.append((pipe, tr, te))
    return pd.DataFrame(rows), oof, le, models


def nb_ttest(a, b, n_train, n_test):
    """Nadeau-Bengio corrected resampled t-test for paired CV scores."""
    d = np.asarray(a) - np.asarray(b); k = len(d)
    if k < 2 or np.var(d, ddof=1) == 0:
        return d.mean(), np.nan, np.nan
    t = d.mean() / np.sqrt((1 / k + n_test / n_train) * np.var(d, ddof=1))
    return d.mean(), t, 2 * stats.t.sf(abs(t), k - 1)


def cramers_v(x, y):
    """Bias-corrected Cramer's V (Bergsma 2013)."""
    ct = pd.crosstab(x, y).values
    chi2 = stats.chi2_contingency(ct, correction=False)[0]
    n = ct.sum(); r, k = ct.shape
    phi2 = max(0, chi2 / n - (k - 1) * (r - 1) / (n - 1))
    rc, kc = r - (r - 1) ** 2 / (n - 1), k - (k - 1) ** 2 / (n - 1)
    return np.sqrt(phi2 / max(1e-12, min(kc - 1, rc - 1)))


def var_of(name, variables):
    """Map a transformed column name back to its raw variable."""
    best = None
    for v in variables:
        if name == v or name.startswith(v + '_'):
            if best is None or len(v) > len(best):
                best = v
    return best


# ----------------------------------------------------------------------------- 3. EPISTEMIC ADEQUACY
log('3. Epistemic audit')
# 3a. label x source
labB = D['B']
ct = pd.crosstab(labB['fine'], labB['occ']).reindex(columns=OCC_ORDER, fill_value=0)
ct['Total'] = ct.sum(1)
ct = ct.sort_values('Total', ascending=False)
exclusive = [lab for lab in ct.index if (ct.loc[lab, OCC_ORDER] > 0).sum() == 1]
nd_sources = int(sum(((raw.occ == g) & (raw.fine == 'No Disease')).any() for g in OCC_ORDER))  # explicit code
tab = ct.reset_index().rename(columns={'fine': 'Label'})
_blank = [int(((labB.occ == g) & labB['blank_nd']).sum()) for g in OCC_ORDER]
tab.loc[len(tab)] = ['of which: no disease left blank'] + _blank + [sum(_blank)]
write_table(tab, 'tab_provenance', 'l' + 'r' * 7, midrules=(len(tab) - 1,))
put('NLabels', ct.shape[0], '{:d}')
put('NExclusive', len(exclusive), '{:d}')
put('NoDiseaseSources', nd_sources, '{:d}')
put('NBlankND', int(labB['blank_nd'].sum()), '{:,}')

for s in ['A', 'B']:
    put(f'CramerV{s}', cramers_v(D[s]['occ'], D[s]['fine']))
    put(f'NMI{s}', normalized_mutual_info_score(D[s]['occ'], D[s]['fine']))
    put(f'CramerVCat{s}', cramers_v(D[s]['occ'], D[s]['y_cat']))

unk = raw.groupby('occ')['fine'].apply(lambda x: 100 * (x == 'Unknown').mean()).reindex(OCC_ORDER)
dup = raw.groupby('occ').apply(lambda g: 100 * g.duplicated(subset=FEAT_RAW + ['fine']).mean()).reindex(OCC_ORDER)
put('UnknownMax', unk.max(), '{:.1f}'); put('UnknownMin', unk.min(), '{:.1f}')
put('DupBizPct', dup['Business owner'], '{:.1f}')
biz = D['A'][D['A']['occ'] == 'Business owner']
put('BizSmokerPctLabelled', 100 * (biz['smoker_status'] == 'Smoker').mean(), '{:.1f}')
put('BizSmokerPctAll', 100 * (raw.loc[raw.occ == 'Business owner', 'smoker_status'] == 'Smoker').mean(), '{:.1f}')

# 3b. source recoverability: can exposure variables alone identify the source sheet?
log('   source recoverability')
rec_rows = []
for fs in ['EXPO', 'DEMO']:
    le_o = LabelEncoder(); yo = le_o.fit_transform(raw['occ'])
    accs, f1s = [], []
    for f, (tr, te) in enumerate(StratifiedKFold(CFG['N_SPLITS'], shuffle=True, random_state=SEED).split(raw, yo)):
        p = fit_pipe(make_pipe(fs, 'XGB', 6), raw.iloc[tr], yo[tr], 'XGB')
        yh = p.predict(raw.iloc[te])
        accs.append(accuracy_score(yo[te], yh)); f1s.append(f1_score(yo[te], yh, average='macro'))
    rec_rows.append({'Features': FSET_LABEL[fs], 'Accuracy': fmt_ms(np.mean(accs), np.std(accs)),
                     'Macro-F1': fmt_ms(np.mean(f1s), np.std(f1s))})
    put(f'SourceRecAcc{fs}', np.mean(accs)); put(f'SourceRecFone{fs}', np.mean(f1s))
put('SourceRecChance', raw['occ'].value_counts(normalize=True).max())

# 3b'. deterministic artefacts: within a source sheet, does a single variable perfectly
#      determine reported disease status (disease vs. no disease)?  (checked on the raw survey)
ART_VARS = ['smoker_status', 'ac_service', 'gender', 'house_type', 'travel_vehicle', 'travel_timing', 'work_days']
art_rows, ARTEFACT = [], {}
for g in OCC_ORDER:
    dg = raw[raw['occ'] == g]
    lab = (dg['fine'] != 'Unknown')
    if lab.nunique() < 2:
        continue
    for v in ART_VARS:
        ctv = pd.crosstab(dg[v], lab)
        if ctv.shape[0] > 1 and ctv.shape[1] == 2 and (ctv == 0).any(axis=1).all():
            ARTEFACT.setdefault(g, []).append(v)
            desc = '; '.join(f'{lvl}: {"all report disease" if ctv.loc[lvl, False] == 0 else "all report no disease"} (n={int(ctv.loc[lvl].sum())})'
                             for lvl in ctv.index)
            art_rows.append({'Source sheet': g, 'Variable': v.replace('_', ' '), 'Pattern': desc})
if not art_rows:
    art_rows.append({'Source sheet': '--', 'Variable': '--', 'Pattern': 'none detected'})
write_table(pd.DataFrame(art_rows), 'tab_artefacts', 'llp{9cm}')
put('NArtefacts', sum(len(v) for v in ARTEFACT.values()), '{:d}')
put('ArtefactGroups', ', '.join(ARTEFACT) if ARTEFACT else 'none')
_bo = raw[raw['occ'] == 'Business owner']
put('BizNonSmokerBlank', int(((_bo.smoker_status == 'Non-smoker') & (_bo.fine == 'Unknown')).sum()), '{:,}')
put('BizNonSmoker', int((_bo.smoker_status == 'Non-smoker').sum()), '{:,}')
put('BizSmokerLabelled', int(((_bo.smoker_status == 'Smoker') & (_bo.fine != 'Unknown')).sum()), '{:,}')
put('BizSmoker', int((_bo.smoker_status == 'Smoker').sum()), '{:,}')

# 3c. ambient-exposure resolution
pm_distinct, pm_geocoded, pm_zones, pm_cov = np.nan, 0, 0, np.nan
if os.path.exists(CFG['PM25_LOOKUP']):
    pm = pd.read_csv(CFG['PM25_LOOKUP'])
    pm_zones = len(pm); pm_geocoded = int(pm['pm25_avg'].notna().sum())
    pm_distinct = int(pm['pm25_avg'].dropna().round(3).nunique())
    ok = set(pm.loc[pm['pm25_avg'].notna(), 'area'])
    pm_cov = 100 * (raw['living_area'].isin(ok) & raw['work_area'].isin(ok)).mean()
    raw['pm_covered'] = raw['living_area'].isin(ok) & raw['work_area'].isin(ok)
else:
    raw['pm_covered'] = False
put('PMZones', pm_zones, '{:d}'); put('PMGeocoded', pm_geocoded, '{:d}')
put('PMDistinct', pm_distinct, '{:d}' if not np.isnan(pm_distinct) else '{}')
put('PMCoverage', pm_cov, '{:.1f}')

# ----------------------------------------------------------------------------- 4. benchmark (feature ablation)
log('4. Repeated-CV benchmark (feature-set ablation)')
BENCH, OOF = [], {}
for s in ['A', 'B']:
    for task, ycol in [('bin', 'y_bin'), ('cat', 'y_cat')]:
        res, oof, le, _ = run_cv(D[s], ycol, 'OCC', 'Prior')
        res = res.assign(scenario=s, task=task, fset='PRIOR', model='Prior'); BENCH.append(res)
        for fs in FSETS:
            for m in MODELS:
                res, oof, le, _ = run_cv(D[s], ycol, fs, m)
                BENCH.append(res.assign(scenario=s, task=task, fset=fs, model=m))
                OOF[(s, task, fs, m)] = (oof, le)
            log(f'   {s}/{task}/{fs} done')
BENCH = pd.concat(BENCH, ignore_index=True)
BENCH.to_csv(os.path.join(OUT, 'csv', 'cv_fold_scores.csv'), index=False)


def score(s, task, fs, m, metric='AUC'):
    x = BENCH.query('scenario==@s and task==@task and fset==@fs and model==@m')[metric]
    return x.mean(), x.std(), x.values


# replication of the discovery-phase 13-class task under proper CV (scenario A)
log('   13-class replication')
rep_rows = []
for m in MODELS:
    res, _, _, _ = run_cv(D['A'], 'fine', 'FULL', m)
    rep_rows.append({'Model': MODEL_LABEL[m], 'Accuracy': fmt_ms(res.Acc.mean(), res.Acc.std()),
                     'Macro-F1': fmt_ms(res.F1m.mean(), res.F1m.std()),
                     'Weighted-F1': fmt_ms(res.F1w.mean(), res.F1w.std()),
                     'Macro AUC': fmt_ms(res.AUC.mean(), res.AUC.std())})
    put(f'Fine{m}FOne', res.F1m.mean()); put(f'Fine{m}AUC', res.AUC.mean())
write_table(pd.DataFrame(rep_rows), 'tab_fine_replication', 'lrrrr')

# main benchmark table
bench_rows = []
for s in ['A', 'B']:
    for fs in ['PRIOR'] + list(FSETS):
        for m in (['Prior'] if fs == 'PRIOR' else MODELS):
            r = {'Scen.': s, 'Features': 'Prior (no features)' if fs == 'PRIOR' else FSET_LABEL[fs], 'Model': m}
            for task, met in [('bin', 'AUC'), ('bin', 'AP'), ('bin', 'Brier'), ('cat', 'F1m'), ('cat', 'AUC')]:
                mu, sd, _ = score(s, task, fs, m, met)
                r[{('bin', 'AUC'): 'Any-disease AUC', ('bin', 'AP'): 'Any-disease AP', ('bin', 'Brier'): 'Brier',
                   ('cat', 'F1m'): 'Category macro-F1', ('cat', 'AUC'): 'Category AUC'}[(task, met)]] = fmt_ms(mu, sd)
            bench_rows.append(r)
            if fs != 'PRIOR':
                put(f'{s}{fs}{m}AUC', score(s, 'bin', fs, m, 'AUC')[0])
                put(f'{s}{fs}{m}AUCsd', score(s, 'bin', fs, m, 'AUC')[1])
                put(f'{s}{fs}{m}AP', score(s, 'bin', fs, m, 'AP')[0])
                put(f'{s}{fs}{m}CatFOne', score(s, 'cat', fs, m, 'F1m')[0])
                put(f'{s}{fs}{m}CatAUC', score(s, 'cat', fs, m, 'AUC')[0])
                put(f'{s}{fs}{m}Sens', score(s, 'bin', fs, m, 'Sens')[0])
                put(f'{s}{fs}{m}Spec', score(s, 'bin', fs, m, 'Spec')[0])
        put(f'{s}PriorCatFOne', score(s, 'cat', 'PRIOR', 'Prior', 'F1m')[0])
bt = pd.DataFrame(bench_rows)
per_s = 1 + len(FSETS) * len(MODELS)
write_table(bt, 'tab_benchmark', 'lllrrrrr',
            midrules=tuple([1 + i * len(MODELS) for i in range(len(FSETS))] + [per_s]
                           + [per_s + 1 + i * len(MODELS) for i in range(len(FSETS))]))

# statistical comparisons (XGB, any-disease AUC; corrected t-test)
cmp_rows = []
ntr = int(BENCH['n_train'].iloc[0]); nte = int(BENCH['n_test'].iloc[0])
for s in ['A', 'B']:
    for a, b, lab in [('FULL', 'EXPODEMO', 'Occupation + EEI over exposure + demographics'),
                      ('EXPODEMO', 'DEMO', 'Exposure over demographics'),
                      ('FULL', 'OCC', 'All features over occupation only'),
                      ('EXPO', 'OCC', 'Exposure only vs occupation only')]:
        _, _, va = score(s, 'bin', a, 'XGB'); _, _, vb = score(s, 'bin', b, 'XGB')
        d, t, p = nb_ttest(va, vb, ntr, nte)
        cmp_rows.append({'Scen.': s, 'Comparison (XGBoost, AUC)': lab, 'ΔAUC': f'{d:+.3f}',
                         't': f'{t:.2f}' if not np.isnan(t) else '--',
                         'p': ('<0.001' if p < 0.001 else f'{p:.3f}') if not np.isnan(p) else '--'})
        put(f'Delta{s}{a}{b}', d, '{:+.3f}')
        put(f'P{s}{a}{b}', p, '{:.3g}' if (not np.isnan(p) and p >= 0.001) else '{:.1e}')
write_table(pd.DataFrame(cmp_rows), 'tab_comparisons', 'llrrr', midrules=(4,))

# provenance share: fraction of above-chance discrimination reachable from occupation alone
for s in ['A', 'B']:
    full = score(s, 'bin', 'FULL', 'XGB')[0]; occ = score(s, 'bin', 'OCC', 'XGB')[0]
    put(f'ProvShare{s}', 100 * (occ - 0.5) / max(1e-9, full - 0.5), '{:.0f}')
    fullc = score(s, 'cat', 'FULL', 'XGB', 'F1m')[0]; occc = score(s, 'cat', 'OCC', 'XGB', 'F1m')[0]
    pri = score(s, 'cat', 'PRIOR', 'Prior', 'F1m')[0]
    put(f'ProvShareCat{s}', 100 * (occc - pri) / max(1e-9, fullc - pri), '{:.0f}')

# figure: ablation
fig, axes = plt.subplots(2, 2, figsize=(11, 7.2), sharey='row')
fsx = list(FSETS)
for j, s in enumerate(['A', 'B']):
    for i, (task, met, ylab) in enumerate([('bin', 'AUC', 'Any-disease ROC-AUC'), ('cat', 'F1m', 'Category macro-F1')]):
        ax = axes[i, j]
        for mi, m in enumerate(MODELS):
            mu = [score(s, task, f, m, met)[0] for f in fsx]; sd = [score(s, task, f, m, met)[1] for f in fsx]
            ax.errorbar(np.arange(len(fsx)) + (mi - 1) * 0.15, mu, yerr=sd, fmt='o', color=OKABE[mi],
                        label=MODEL_LABEL[m], capsize=3, ms=5)
        ax.axhline(score(s, task, 'PRIOR', 'Prior', met)[0], color='grey', ls='--', lw=1, label='Prior')
        ax.set_xticks(range(len(fsx))); ax.set_xticklabels([FSET_LABEL[f].replace(' (', '\n(').replace(' + ', '\n+ ') for f in fsx], fontsize=8)
        ax.set_title(f'Scenario {s}: {"as coded (blanks excluded)" if s == "A" else "confirmed (blank = no disease)"}' if i == 0 else '')
        ax.set_ylabel(ylab if j == 0 else '')
axes[0, 0].legend(fontsize=8, loc='lower right')
fig.tight_layout(); savefig(fig, 'fig_ablation')

# ----------------------------------------------------------------------------- 5. transfer: LOGO + within-occupation
log('5. Transfer and within-occupation analyses')
logo_rows = []
for s in ['A', 'B']:
    d = D[s]
    for g in OCC_ORDER:
        tr, te = d['occ'] != g, d['occ'] == g
        yte = d.loc[te, 'y_bin'].values
        row = {'Scen.': s, 'Held-out group': g, 'n': int(te.sum()), 'Prev. %': f'{100 * yte.mean():.1f}'}
        for fs, m in [('EXPODEMO', 'XGB'), ('EXPODEMO', 'LR'), ('FULL', 'XGB')]:
            key = f'{FSET_LABEL[fs].split(" (")[0]} / {m}'
            if len(np.unique(yte)) < 2 or len(np.unique(d.loc[tr, 'y_bin'])) < 2:
                row[key] = 'n/a'; continue
            p = fit_pipe(make_pipe(fs, m, 2), d[tr], d.loc[tr, 'y_bin'].values, m)
            auc = roc_auc_score(yte, p.predict_proba(d[te])[:, 1]); row[key] = f'{auc:.3f}'
        logo_rows.append(row)
logo = pd.DataFrame(logo_rows)
write_table(logo, 'tab_logo', 'llrrrrr', midrules=(len(OCC_ORDER),))
for s in ['A', 'B']:
    v = pd.to_numeric(logo.query('`Scen.`==@s')['Exposure + demographics / XGB'], errors='coerce')
    put(f'LOGOmean{s}', v.mean()); put(f'LOGOmin{s}', v.min()); put(f'LOGOmax{s}', v.max())
    put(f'LOGOfeasible{s}', int(v.notna().sum()), '{:d}')

within_rows = []
for s in ['A', 'B']:
    d = D[s]
    for g in OCC_ORDER:
        dg = d[d['occ'] == g].reset_index(drop=True)
        n1, n0 = int(dg['y_bin'].sum()), int((1 - dg['y_bin']).sum())
        gkey = g.replace(' ', '').replace('(', '').replace(')', '').replace('/', '').replace('-', '')
        row = {'Scen.': s, 'Group': g, 'Cases': n1, 'Non-cases': n0}
        reps = max(1, min(3, CFG['N_REPEATS']))
        for m in ['LR', 'XGB']:
            if min(n1, n0) < 2 * CFG['N_SPLITS']:
                row[f'AUC {m}'] = 'n/a'; continue
            res, _, _, _ = run_cv(dg, 'y_bin', 'EXPODEMO', m, repeats=reps)
            row[f'AUC {m}'] = fmt_ms(res.AUC.mean(), res.AUC.std())
            put(f'Within{s}{gkey}{m}', res.AUC.mean())
        # artefact-removed re-analysis (drop the variable that deterministically encodes labelling)
        row['AUC XGB, artefact removed'] = '--'
        if s == 'B' and g in ARTEFACT and min(n1, n0) >= 2 * CFG['N_SPLITS']:
            fs_clean = {k: [c for c in v if c not in ARTEFACT[g]] for k, v in FSETS['EXPODEMO'].items()}
            FSETS['_CLEAN'] = fs_clean
            res, _, _, _ = run_cv(dg, 'y_bin', '_CLEAN', 'XGB', repeats=reps)
            del FSETS['_CLEAN']
            row['AUC XGB, artefact removed'] = fmt_ms(res.AUC.mean(), res.AUC.std())
            put(f'Within{s}{gkey}Clean', res.AUC.mean())
        within_rows.append(row)
within = pd.DataFrame(within_rows)
write_table(within, 'tab_within', 'llrrrrr', midrules=(len(OCC_ORDER),))


def _auc_of(x):
    return float(x.split(' ')[0]) if isinstance(x, str) and x not in ('n/a', '--') else np.nan


for s in ['A', 'B']:
    ws = within.query('`Scen.`==@s')
    put(f'WithinFeasible{s}', int((ws['AUC XGB'] != 'n/a').sum()), '{:d}')
    vals = ws['AUC XGB'].map(_auc_of)
    put(f'WithinMean{s}', vals.mean()); put(f'WithinMax{s}', vals.max()); put(f'WithinMin{s}', vals.min())
    # artefact-corrected mean: replace artefact groups by their artefact-removed AUC
    clean = [(_auc_of(r['AUC XGB, artefact removed']) if r['Group'] in ARTEFACT and s == 'B' else _auc_of(r['AUC XGB']))
             for _, r in ws.iterrows()]
    put(f'WithinMeanClean{s}', np.nanmean(clean) if np.any(~np.isnan(clean)) else np.nan)
    wk = [c for c, (_, r) in zip(clean, ws.iterrows()) if r['Group'] != 'Student']
    put(f'WithinWorkersClean{s}', np.nanmean(wk) if np.any(~np.isnan(wk)) else np.nan)

# ----------------------------------------------------------------------------- 6. adjusted regression
log('6. Adjusted regression')


def fit_glm(d, formula, groups):
    mod = smf.glm(formula, data=d, family=sm.families.Binomial())
    try:
        return mod.fit(cov_type='cluster', cov_kwds={'groups': pd.factorize(groups)[0]}, maxiter=200)
    except Exception:
        return mod.fit(maxiter=200)


def prep_reg(d):
    d = d.copy()
    d['age10'] = d['age'] / 10
    d['male'] = (d['gender'] == 'Male').astype(int)
    d['smoker'] = (d['smoker_status'] == 'Smoker').astype(int)
    d['open_air'] = (d['vehicle_grp'] == 'Open-air').astype(int)
    d['bus'] = (d['vehicle_grp'] == 'Bus').astype(int)
    d['low_housing'] = (d['housing'] == 'Low-quality').astype(int)
    d['no_ac'] = (d['ac_service'] == 'No').astype(int)
    d['ac_unknown'] = (d['ac_service'] == 'Unknown').astype(int)
    d['occ_c'] = d['occ'].str.replace(r'[^A-Za-z]', '', regex=True)
    return d


REG_TERMS = {'age10': 'Age (per 10 y)', 'male': 'Male', 'smoker': 'Smoker', 'travel_hours': 'Travel (per h)',
             'stay_office_hours': 'Workplace (per h)', 'open_air': 'Open-air vehicle (vs enclosed)',
             'bus': 'Bus (vs enclosed)', 'low_housing': 'Low-quality housing', 'no_ac': 'No AC access'}
base = ' + '.join(REG_TERMS) + ' + ac_unknown'
reg_specs = [
    ('B, all respondents', 'B', None, f'y_bin ~ {base} + C(occ_c, Treatment("Student"))'),
    ('B, workers only', 'B', 'workers', f'y_bin ~ {base} + C(occ_c, Treatment("Businessowner"))'),
    ('B, workers excl. business owners', 'B', 'workers_nobo', f'y_bin ~ {base} + C(occ_c, Treatment("DriverAC"))'),
    ('A, students only', 'A', 'students', f'y_bin ~ {base}'),
]
reg_out, reg_notes = {}, {}
for lab, s, sub, form in reg_specs:
    d = prep_reg(D[s])
    if sub == 'workers':
        d = d[d['occ'] != 'Student']
    if sub == 'workers_nobo':
        d = d[~d['occ'].isin(['Student', 'Business owner'])]
    if sub == 'students':
        d = d[d['occ'] == 'Student']
    d = d.reset_index(drop=True)
    if d['y_bin'].nunique() < 2:
        reg_notes[lab] = 'outcome constant'; continue
    try:
        fit = fit_glm(d, form, d['living_area'])
        ci = fit.conf_int()
        reg_out[lab] = {t: (np.exp(fit.params[t]), np.exp(ci.loc[t, 0]), np.exp(ci.loc[t, 1]), fit.pvalues[t])
                        for t in REG_TERMS if t in fit.params.index}
        reg_notes[lab] = f'n={len(d)}, events={int(d.y_bin.sum())}'
    except Exception as e:
        reg_notes[lab] = f'not estimable ({type(e).__name__})'
# scenario A among workers: outcome is constant -> not identifiable
dA_w = D['A'][D['A']['occ'] != 'Student']
put('AWorkersPrev', 100 * dA_w['y_bin'].mean(), '{:.1f}')
reg_rows = []
for t, tl in REG_TERMS.items():
    row = {'Term': tl}
    for lab, *_ in reg_specs:
        if lab in reg_out and t in reg_out[lab]:
            o, lo, hi, p = reg_out[lab][t]
            star = '*' if p < 0.05 else ''
            row[lab] = f'{o:.2f} ({lo:.2f}–{hi:.2f}){star}' if hi < 1e3 else 'n.e.'
        else:
            row[lab] = 'n.e.'
    reg_rows.append(row)
reg_rows.append({'Term': 'Sample', **{lab: reg_notes.get(lab, '') for lab, *_ in reg_specs}})
write_table(pd.DataFrame(reg_rows), 'tab_regression', 'lrrrr', midrules=(len(REG_TERMS),))
for lab, key in [('B, all respondents', 'BAll'), ('B, workers only', 'BWork'),
                 ('B, workers excl. business owners', 'BWorkNoBO'), ('A, students only', 'AStud')]:
    for t, short in [('smoker', 'Smoker'), ('travel_hours', 'Travel'), ('open_air', 'OpenAir'),
                     ('low_housing', 'Housing'), ('no_ac', 'NoAC'), ('age10', 'Age'), ('bus', 'Bus')]:
        if lab in reg_out and t in reg_out[lab]:
            o, lo, hi, p = reg_out[lab][t]
            put(f'OR{key}{short}', o, '{:.2f}'); put(f'ORlo{key}{short}', lo, '{:.2f}')
            put(f'ORhi{key}{short}', hi, '{:.2f}'); put(f'ORp{key}{short}', p, '{:.3g}')

# ----------------------------------------------------------------------------- 7. ETHICAL GOVERNANCE
log('7. Explanation stability, expert-data concordance, counterfactuals, privacy')
PS = CFG['PRIMARY']
dP = D[PS]
IMP = {}  # (fset, method) -> DataFrame folds x variables
for fs in ['FULL', 'EXPODEMO']:
    variables = FSETS[fs]['num'] + FSETS[fs]['cat'] + FSETS[fs]['area']
    y = dP['y_bin'].values
    shap_f, perm_x, perm_r = [], [], []
    for f, (tr, te) in enumerate(StratifiedKFold(CFG['N_SPLITS'], shuffle=True, random_state=SEED).split(dP, y)):
        px = fit_pipe(make_pipe(fs, 'XGB', 2), dP.iloc[tr], y[tr], 'XGB')
        Xt = px['pre'].transform(dP.iloc[te]); names = px['pre'].get_feature_names_out()
        sv = shap.TreeExplainer(px['clf']).shap_values(Xt)
        sv = sv[1] if isinstance(sv, list) else sv
        s_abs = pd.Series(np.abs(sv).mean(0), index=names)
        shap_f.append(s_abs.groupby([var_of(n, variables) for n in names]).sum())
        pi = permutation_importance(px, dP.iloc[te][variables], y[te], scoring='roc_auc',
                                    n_repeats=CFG['N_PERM'], random_state=SEED, n_jobs=1)
        perm_x.append(pd.Series(pi.importances_mean, index=variables))
        pr = fit_pipe(make_pipe(fs, 'RF', 2), dP.iloc[tr], y[tr], 'RF')
        pi = permutation_importance(pr, dP.iloc[te][variables], y[te], scoring='roc_auc',
                                    n_repeats=CFG['N_PERM'], random_state=SEED, n_jobs=1)
        perm_r.append(pd.Series(pi.importances_mean, index=variables))
        if fs == 'FULL' and f == 0:
            SHAP_EXAMPLE = (sv, Xt, names)
    IMP[(fs, 'SHAP (XGB)')] = pd.DataFrame(shap_f)[variables]
    IMP[(fs, 'Permutation (XGB)')] = pd.DataFrame(perm_x)[variables]
    IMP[(fs, 'Permutation (RF)')] = pd.DataFrame(perm_r)[variables]
    log(f'   importance {fs} done')


def mean_pairwise_spearman(M):
    rs = [stats.spearmanr(M.iloc[i], M.iloc[j])[0] for i in range(len(M)) for j in range(i + 1, len(M))]
    return np.nanmean(rs)


VAR_LABEL = {'travel_hours': 'Travel hours', 'stay_home_hours': 'Home hours', 'stay_office_hours': 'Workplace hours',
             'work_days': 'Work days', 'age': 'Age', 'eei': 'EEI', 'travel_vehicle': 'Travel vehicle',
             'travel_timing': 'Travel timing', 'ac_service': 'AC access', 'house_type': 'House type',
             'gender': 'Sex', 'smoker_status': 'Smoking', 'occupation': 'Occupation',
             'living_area': 'Living area', 'work_area': 'Work area'}
stab_rows = []
for fs in ['FULL', 'EXPODEMO']:
    meths = ['SHAP (XGB)', 'Permutation (XGB)', 'Permutation (RF)']
    means = {m: IMP[(fs, m)].mean() for m in meths}
    row = {'Feature set': FSET_LABEL[fs]}
    for m in meths:
        row[f'Fold stability ρ: {m}'] = f'{mean_pairwise_spearman(IMP[(fs, m)]):.2f}'
    pairs = [(a, b) for i, a in enumerate(meths) for b in meths[i + 1:]]
    agree = np.mean([stats.spearmanr(means[a], means[b])[0] for a, b in pairs])
    row['Cross-method ρ'] = f'{agree:.2f}'
    top = means['SHAP (XGB)'].sort_values(ascending=False).index[:3]
    row['Top-3 (SHAP)'] = ', '.join(VAR_LABEL[v] for v in top)
    stab_rows.append(row)
    key = 'Full' if fs == 'FULL' else 'Expo'
    put(f'StabSHAP{key}', mean_pairwise_spearman(IMP[(fs, 'SHAP (XGB)')]), '{:.2f}')
    put(f'StabPermXGB{key}', mean_pairwise_spearman(IMP[(fs, 'Permutation (XGB)')]), '{:.2f}')
    put(f'StabPermRF{key}', mean_pairwise_spearman(IMP[(fs, 'Permutation (RF)')]), '{:.2f}')
    put(f'CrossMethod{key}', agree, '{:.2f}')
    put(f'TopOne{key}', VAR_LABEL[top[0]]); put(f'TopTwo{key}', VAR_LABEL[top[1]]); put(f'TopThree{key}', VAR_LABEL[top[2]])
    shap_share = means['SHAP (XGB)'] / means['SHAP (XGB)'].sum()
    put(f'ShapShareTop{key}', 100 * shap_share.max(), '{:.0f}')
    if fs == 'FULL':
        put('ShapShareOcc', 100 * shap_share.get('occupation', np.nan), '{:.0f}')
write_table(pd.DataFrame(stab_rows), 'tab_stability', 'lrrrrl')

fig, axes = plt.subplots(1, 2, figsize=(12, 5.2))
for ax, fs in zip(axes, ['FULL', 'EXPODEMO']):
    order = IMP[(fs, 'SHAP (XGB)')].mean().sort_values().index
    for mi, m in enumerate(['SHAP (XGB)', 'Permutation (XGB)', 'Permutation (RF)']):
        M = IMP[(fs, m)][order]; M = M.div(M.mean().abs().sum())
        for k, v in enumerate(order):
            ax.scatter(M[v], np.full(len(M), k) + (mi - 1) * 0.22, s=14, color=OKABE[mi], alpha=0.7,
                       label=m if k == 0 else None)
    ax.set_yticks(range(len(order))); ax.set_yticklabels([VAR_LABEL[v] for v in order])
    ax.set_xlabel('Normalised importance (per fold)'); ax.set_title(FSET_LABEL[fs])
axes[0].legend(fontsize=8, loc='lower right'); fig.tight_layout(); savefig(fig, 'fig_importance')

# 7b. expert vs data concordance ("Layer 3")
log('   expert-data concordance')
workers = dP[dP['occ'] != 'Student']
veh_n = workers['travel_vehicle'].value_counts()
vehs = [v for v in veh_n.index if veh_n[v] >= 30 and v in W_VEH]
veh_prev = workers.groupby('travel_vehicle')['y_bin'].mean().reindex(vehs)
# model-based: mean SHAP of the vehicle one-hot block per vehicle, exposure+demo XGB on all data
px_all = fit_pipe(make_pipe('EXPODEMO', 'XGB', 2), dP, dP['y_bin'].values, 'XGB')
Xt_all = px_all['pre'].transform(dP); nm_all = px_all['pre'].get_feature_names_out()
sv_all = shap.TreeExplainer(px_all['clf']).shap_values(Xt_all)
sv_all = sv_all[1] if isinstance(sv_all, list) else sv_all
veh_cols = [i for i, n in enumerate(nm_all) if n.startswith('travel_vehicle_')]
veh_shap = pd.Series(sv_all[:, veh_cols].sum(1), index=dP.index).groupby(dP['travel_vehicle']).mean().reindex(vehs)
occ_prev = dP.groupby('occ')['y_bin'].mean().reindex(OCC_ORDER)
occ_or = pd.Series({g: np.nan for g in OCC_ORDER})
if 'B, all respondents' in reg_out:
    # refit to extract occupation coefficients
    d_ = prep_reg(D['B'])
    fit_ = fit_glm(d_, reg_specs[0][3], d_['living_area'])
    for g in OCC_ORDER:
        key = f'C(occ_c, Treatment("Student"))[T.{g.replace(" ", "").replace("(", "").replace(")", "").replace("/", "").replace("-", "")}]'
        occ_or[g] = np.exp(fit_.params[key]) if key in fit_.params.index else (1.0 if g == 'Student' else np.nan)


def sp(a, b):
    a, b = pd.Series(a).astype(float), pd.Series(b).astype(float)
    m = a.notna() & b.notna()
    r, p = stats.spearmanr(a[m], b[m]) if m.sum() > 2 else (np.nan, np.nan)
    return r, p, int(m.sum())


conc = []
for lab, a, b in [
    ('Vehicle: EEI weights vs Layer-1 modifiers (expert vs expert)', pd.Series(W_VEH).reindex(vehs), pd.Series(L1_VEH).reindex(vehs)),
    ('Vehicle: EEI weights vs crude prevalence (workers)', pd.Series(W_VEH).reindex(vehs), veh_prev),
    ('Vehicle: Layer-1 modifiers vs crude prevalence (workers)', pd.Series(L1_VEH).reindex(vehs), veh_prev),
    ('Vehicle: EEI weights vs mean SHAP (exposure model)', pd.Series(W_VEH).reindex(vehs), veh_shap),
    ('Occupation: EEI weights vs Layer-1 tiers (expert vs expert)', pd.Series(W_OCC).reindex(OCC_ORDER), pd.Series(L1_TIER).reindex(OCC_ORDER)),
    ('Occupation: EEI weights vs crude prevalence', pd.Series(W_OCC).reindex(OCC_ORDER), occ_prev),
    ('Occupation: EEI weights vs adjusted OR', pd.Series(W_OCC).reindex(OCC_ORDER), occ_or),
]:
    r, p, n = sp(a, b)
    conc.append({'Comparison': lab, 'k': n, 'Spearman ρ': f'{r:.2f}' if not np.isnan(r) else '--',
                 'p': f'{p:.3f}' if not np.isnan(p) else '--'})
conc_df = pd.DataFrame(conc)
write_table(conc_df, 'tab_concordance', 'lrrr', midrules=(4,))
for i, key in enumerate(['VehExpExp', 'VehEEIPrev', 'VehLonePrev', 'VehEEIShap', 'OccExpExp', 'OccEEIPrev', 'OccEEIOR']):
    put(f'Rho{key}', float(conc_df['Spearman ρ'][i]) if conc_df['Spearman ρ'][i] != '--' else np.nan, '{:.2f}')
put('NVeh', len(vehs), '{:d}')

fig, axes = plt.subplots(1, 2, figsize=(11, 4.4))
ax = axes[0]
for i, v in enumerate(vehs):
    ax.scatter(W_VEH[v], veh_prev[v], s=40, color=OKABE[0]); ax.annotate(v, (W_VEH[v], veh_prev[v]), fontsize=8, xytext=(3, 3), textcoords='offset points')
    ax.scatter(L1_VEH[v], veh_prev[v], s=40, color=OKABE[1], marker='s')
ax.scatter([], [], color=OKABE[0], label='EEI weight'); ax.scatter([], [], color=OKABE[1], marker='s', label='Layer-1 modifier')
ax.set_xlabel('Expert exposure weight'); ax.set_ylabel(f'Observed disease prevalence (workers, scen. {PS})'); ax.legend(fontsize=8)
ax.set_title('Travel vehicle')
ax = axes[1]
for g in OCC_ORDER:
    ax.scatter(W_OCC[g], occ_prev[g], s=50, color=OKABE[2]); ax.annotate(g, (W_OCC[g], occ_prev[g]), fontsize=8, xytext=(3, 3), textcoords='offset points')
ax.set_xlabel('Expert occupational weight (EEI)'); ax.set_ylabel(f'Observed disease prevalence (scen. {PS})'); ax.set_title('Occupation')
fig.tight_layout(); savefig(fig, 'fig_concordance')

# 7c. counterfactual actionability
log('   counterfactuals')
cf_rows = []
for fs in ['FULL', 'EXPODEMO']:
    pf = fit_pipe(make_pipe(fs, 'XGB', 2), dP, dP['y_bin'].values, 'XGB')
    p0 = pf.predict_proba(dP)[:, 1]
    hi = dP[(p0 >= CFG['THRESH']) & (dP['occ'] != 'Student')].copy()
    base_p = pf.predict_proba(hi)[:, 1]
    interventions = {
        'AC access': lambda x: x.assign(ac_service='Yes'),
        'Smoking cessation': lambda x: x.assign(smoker_status='Non-smoker'),
        'Halve travel time': lambda x: x.assign(travel_hours=x['travel_hours'] / 2),
        'Enclosed transport': lambda x: x.assign(travel_vehicle=np.where(x['travel_vehicle'].isin(OPEN_AIR | {'Bus'}), 'Car', x['travel_vehicle'])),
        'All four combined': lambda x: x.assign(ac_service='Yes', smoker_status='Non-smoker', travel_hours=x['travel_hours'] / 2,
                                                travel_vehicle=np.where(x['travel_vehicle'].isin(OPEN_AIR | {'Bus'}), 'Car', x['travel_vehicle'])),
    }
    for iname, fn in interventions.items():
        xcf = fn(hi.copy())
        if fs == 'FULL':  # keep EEI coherent with the modified inputs
            comp = np.column_stack([xcf['travel_hours'] / 24, xcf['occ'].map(W_OCC), xcf['travel_vehicle'].map(W_VEH).fillna(0.5),
                                    xcf['stay_office_hours'] / (xcf['stay_home_hours'] + xcf['stay_office_hours'] + 1e-5),
                                    xcf['ac_service'].map({'Yes': 0.3, 'No': 1.0}).fillna(0.65)])
            xcf['eei'] = 100 * comp @ EEI_W
        pc = pf.predict_proba(xcf)[:, 1]
        dlt = pc - base_p
        nbo = (hi['occ'] != 'Business owner').values
        cf_rows.append({'Model': FSET_LABEL[fs], 'Intervention': iname, 'n': len(hi),
                        'Mean Δ risk (pp)': f'{100 * dlt.mean():+.1f}', 'Median Δ (pp)': f'{100 * np.median(dlt):+.1f}',
                        'Crossed below threshold %': f'{100 * (pc < CFG["THRESH"]).mean():.1f}',
                        'Mean Δ excl. business owners (pp)': f'{100 * dlt[nbo].mean():+.1f}'})
        key = ('Full' if fs == 'FULL' else 'Expo') + iname.replace(' ', '').replace('-', '')
        put(f'CF{key}', 100 * dlt.mean(), '{:+.1f}'); put(f'CFflip{key}', 100 * (pc < CFG['THRESH']).mean(), '{:.1f}')
        put(f'CFnobo{key}', 100 * dlt[nbo].mean(), '{:+.1f}')
write_table(pd.DataFrame(cf_rows), 'tab_counterfactual', 'llrrrrr', midrules=(len(interventions),))

# 7d. re-identification risk (k-anonymity) on the full survey
QI_SETS = {
    'Age, sex, occupation, living area, work area': (['age', 'gender', 'occupation', 'living_area', 'work_area'], raw),
    'Age, sex, occupation, living area': (['age', 'gender', 'occupation', 'living_area'], raw),
    '10-y age band, sex, occupation, living area': (['age10b', 'gender', 'occupation', 'living_area'], raw.assign(age10b=(raw['age'] // 10) * 10)),
    '10-y age band, sex, occupation': (['age10b', 'gender', 'occupation'], raw.assign(age10b=(raw['age'] // 10) * 10)),
}
priv_rows = []
for i, (lab, (qi, d)) in enumerate(QI_SETS.items()):
    k = d.groupby(qi)[qi[0]].transform('size')
    ldiv = d.groupby(qi)['fine'].transform('nunique')
    priv_rows.append({'Quasi-identifiers': lab, 'Unique (k=1) %': f'{100 * (k == 1).mean():.1f}',
                      'k<5 %': f'{100 * (k < 5).mean():.1f}', 'Median k': f'{k.median():.0f}',
                      'Health attribute disclosed %': f'{100 * (ldiv == 1).mean():.1f}'})
    put(f'PrivUnique{"ABCD"[i]}', 100 * (k == 1).mean(), '{:.1f}'); put(f'PrivKfive{"ABCD"[i]}', 100 * (k < 5).mean(), '{:.1f}')
    put(f'PrivDisc{"ABCD"[i]}', 100 * (ldiv == 1).mean(), '{:.1f}')
write_table(pd.DataFrame(priv_rows), 'tab_privacy', 'lrrrr')

# ----------------------------------------------------------------------------- 8. DISTRIBUTIVE JUSTICE
log('8. Distributive analyses')
q80 = raw['eei'].quantile(0.8)
burden = []
for col, levels in [('occ', OCC_ORDER), ('housing', ['Low-quality', 'Standard']), ('gender', ['Male', 'Female', 'Intersex'])]:
    for g in levels:
        d = raw[raw[col] == g]
        if len(d) == 0:
            continue
        burden.append({'Group': g, 'Pop. share %': f'{100 * len(d) / len(raw):.1f}',
                       'Top-EEI-quintile share %': f'{100 * (d.eei >= q80).sum() / (raw.eei >= q80).sum():.1f}',
                       'Mean EEI': f'{d.eei.mean():.1f}', 'Travel h': f'{d.travel_hours.mean():.2f}',
                       'No disease left blank %': f'{100 * (d.fine == "Unknown").mean():.1f}',
                       'CAMS PM2.5 covered %': f'{100 * d.pm_covered.mean():.1f}'})
bdf = pd.DataFrame(burden)
write_table(bdf, 'tab_burden', 'lrrrrrr', midrules=(len(OCC_ORDER), len(OCC_ORDER) + 2))
lq = raw['housing'] == 'Low-quality'
put('LowHousePop', 100 * lq.mean(), '{:.1f}')
put('LowHouseTopQ', 100 * (raw.loc[lq, 'eei'] >= q80).sum() / (raw['eei'] >= q80).sum(), '{:.1f}')
top_occ = raw.loc[raw['eei'] >= q80, 'occ'].value_counts(normalize=True)
put('TopQOccName', top_occ.index[0]); put('TopQOccShare', 100 * top_occ.iloc[0], '{:.1f}')
# concentration: rank respondents by EEI, share of top quintile held by outdoor workers
outdoor = raw['occ'].isin(['Driver (non-AC)', 'Rickshaw/biker', 'Police/FF/guard'])
put('OutdoorPop', 100 * outdoor.mean(), '{:.1f}')
put('OutdoorTopQ', 100 * (outdoor & (raw['eei'] >= q80)).sum() / (raw['eei'] >= q80).sum(), '{:.1f}')
cov_occ = raw.groupby('occ')['pm_covered'].mean().reindex(OCC_ORDER) * 100
put('CovMin', cov_occ.min(), '{:.1f}'); put('CovMax', cov_occ.max(), '{:.1f}')
put('CovMinGroup', cov_occ.idxmin()); put('CovMaxGroup', cov_occ.idxmax())

fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
sns.boxplot(data=raw, y='occ', x='eei', order=OCC_ORDER, ax=axes[0], color=OKABE[5], fliersize=1)
axes[0].axvline(q80, ls='--', color='k', lw=1); axes[0].set_xlabel('Environmental Exposure Index'); axes[0].set_ylabel('')
axes[0].set_title('Exposure burden (dashed: top quintile)')
b2 = bdf.iloc[:len(OCC_ORDER)].copy()
x = np.arange(len(b2)); w = 0.27
for i, (c, lab) in enumerate([('Pop. share %', 'Population share'), ('Top-EEI-quintile share %', 'Top-exposure share'), ('No disease left blank %', 'No disease left blank')]):
    axes[1].barh(x + (i - 1) * w, b2[c].astype(float), w, color=OKABE[i], label=lab)
axes[1].set_yticks(x); axes[1].set_yticklabels(b2['Group']); axes[1].invert_yaxis(); axes[1].set_xlabel('%')
axes[1].legend(fontsize=8); axes[1].set_title('Burden vs explicit health coding')
fig.tight_layout(); savefig(fig, 'fig_burden')

# 8b. error equity (out-of-fold, primary scenario, any-disease)
eq_rows = []
EQ = {}
for fs in ['FULL', 'EXPODEMO']:
    oof, _ = OOF[(PS, 'bin', fs, 'XGB')]
    p = oof[:, 1]; y = dP['y_bin'].values; yh = (p >= CFG['THRESH']).astype(int)
    tprs, fprs = [], []
    for col, levels in [('occ', OCC_ORDER), ('gender', ['Male', 'Female', 'Intersex']), ('housing', ['Low-quality', 'Standard']), ('age_band', ['<30', '30-44', '45+'])]:
        for g in levels:
            m = (dP[col] == g).values
            if m.sum() < 30:
                continue
            yy, pp, hh = y[m], p[m], yh[m]
            tpr = recall_score(yy, hh, zero_division=0) if yy.sum() > 0 else np.nan
            fpr = ((hh == 1) & (yy == 0)).sum() / max(1, (yy == 0).sum()) if (yy == 0).sum() > 0 else np.nan
            if col == 'occ':
                tprs.append(tpr); fprs.append(fpr)
            eq_rows.append({'Model': FSET_LABEL[fs], 'Attribute': {'occ': 'Occupation', 'gender': 'Sex', 'housing': 'Housing', 'age_band': 'Age'}[col],
                            'Group': g, 'n': int(m.sum()), 'Prev. %': f'{100 * yy.mean():.1f}',
                            'AUC': f'{roc_auc_score(yy, pp):.3f}' if len(np.unique(yy)) == 2 else 'n/a',
                            'TPR': f'{tpr:.2f}' if not np.isnan(tpr) else 'n/a', 'FPR': f'{fpr:.2f}' if not np.isnan(fpr) else 'n/a',
                            'Pred−obs (pp)': f'{100 * (pp.mean() - yy.mean()):+.1f}'})
    key = 'Full' if fs == 'FULL' else 'Expo'
    put(f'TPRgap{key}', np.nanmax(tprs) - np.nanmin(tprs), '{:.2f}'); put(f'FPRgap{key}', np.nanmax(fprs) - np.nanmin(fprs), '{:.2f}')
    EQ[fs] = (np.nanmax(tprs) - np.nanmin(tprs), np.nanmax(fprs) - np.nanmin(fprs))
eqdf = pd.DataFrame(eq_rows)
write_table(eqdf[eqdf['Model'] == FSET_LABEL['EXPODEMO']].drop(columns='Model'), 'tab_equity', 'llrrrrrr')
eqdf.to_csv(os.path.join(OUT, 'csv', 'tab_equity_all.csv'), index=False)

fig, axes = plt.subplots(1, 3, figsize=(13, 4.2), sharey=True)
e_occ = eqdf[eqdf['Attribute'] == 'Occupation']
for ax, met in zip(axes, ['TPR', 'FPR', 'Pred−obs (pp)']):
    for i, fs in enumerate(['FULL', 'EXPODEMO']):
        e = e_occ[e_occ['Model'] == FSET_LABEL[fs]].set_index('Group').reindex(OCC_ORDER)
        vals = pd.to_numeric(e[met], errors='coerce')
        ax.barh(np.arange(len(OCC_ORDER)) + (i - 0.5) * 0.38, vals, 0.38, color=OKABE[i], label=FSET_LABEL[fs])
    ax.set_title(met); ax.axvline(0, color='k', lw=0.8)
axes[0].set_yticks(range(len(OCC_ORDER))); axes[0].set_yticklabels(OCC_ORDER); axes[0].invert_yaxis()
axes[0].legend(fontsize=8); fig.tight_layout(); savefig(fig, 'fig_equity')

# ----------------------------------------------------------------------------- 9. INSTITUTIONAL LEGITIMACY
log('9. Calibration, decision curves, risk strata')


def calib_stats(y, p, bins=10):
    p = np.clip(p, 1e-6, 1 - 1e-6)
    edges = np.linspace(0, 1, bins + 1); idx = np.clip(np.digitize(p, edges) - 1, 0, bins - 1)
    ece = sum(abs(p[idx == b].mean() - y[idx == b].mean()) * (idx == b).mean() for b in range(bins) if (idx == b).any())
    lg = np.log(p / (1 - p))
    slope = sm.GLM(y, sm.add_constant(lg), family=sm.families.Binomial()).fit().params[1]
    return ece, slope, p.mean() - y.mean(), brier_score_loss(y, p)


def net_benefit(y, p, ts):
    n = len(y)
    return np.array([((p >= t) & (y == 1)).sum() / n - ((p >= t) & (y == 0)).sum() / n * t / (1 - t) for t in ts])


cal_rows = []
fig, axes = plt.subplots(1, 2, figsize=(11, 4.4))
ts = np.linspace(0.02, 0.8, 79)
for i, (s, fs) in enumerate([('B', 'FULL'), ('B', 'EXPODEMO'), ('B', 'OCC'), ('A', 'FULL'), ('A', 'EXPODEMO')]):
    oof, _ = OOF[(s, 'bin', fs, 'XGB')]; p = oof[:, 1]; y = D[s]['y_bin'].values
    ece, slope, citl, br = calib_stats(y, p)
    cal_rows.append({'Scen.': s, 'Features': FSET_LABEL[fs], 'AUC': f'{roc_auc_score(y, p):.3f}', 'Brier': f'{br:.3f}',
                     'ECE': f'{ece:.3f}', 'Cal. slope': f'{slope:.2f}', 'Cal.-in-large (pp)': f'{100 * citl:+.1f}'})
    put(f'ECE{s}{fs}', ece); put(f'Slope{s}{fs}', slope, '{:.2f}'); put(f'Brier{s}{fs}', br)
    if s == PS:
        fr = pd.DataFrame({'p': p, 'y': y}); fr['bin'] = pd.qcut(fr['p'].rank(method='first'), 10, labels=False)
        g = fr.groupby('bin').mean()
        axes[0].plot(g['p'], g['y'], 'o-', color=OKABE[i], label=FSET_LABEL[fs], ms=4)
        nb = net_benefit(y, p, ts)
        axes[1].plot(ts, nb, color=OKABE[i], label=FSET_LABEL[fs])
        mask = (ts >= 0.1) & (ts <= 0.5)
        prev = y.mean(); nb_all = prev - (1 - prev) * ts / (1 - ts)
        put(f'NBgain{fs}', np.mean(nb[mask] - np.maximum(nb_all[mask], 0)), '{:.3f}')
        put(f'NBpositive{fs}', 100 * np.mean(nb[mask] > np.maximum(nb_all[mask], 0) + 1e-9), '{:.0f}')
y = dP['y_bin'].values; prev = y.mean()
axes[1].plot(ts, prev - (1 - prev) * ts / (1 - ts), color='grey', ls='--', label='Treat all')
axes[1].axhline(0, color='k', lw=0.8, label='Treat none'); axes[1].set_ylim(-0.05, max(prev, 0.05) + 0.05)
axes[1].set_xlabel('Threshold probability'); axes[1].set_ylabel('Net benefit'); axes[1].set_title(f'Decision curves (scen. {PS})')
axes[0].plot([0, 1], [0, 1], 'k--', lw=0.8); axes[0].set_xlabel('Mean predicted risk'); axes[0].set_ylabel('Observed prevalence')
axes[0].set_title(f'Reliability (out-of-fold, scen. {PS})'); axes[0].legend(fontsize=8); axes[1].legend(fontsize=8)
fig.tight_layout(); savefig(fig, 'fig_calibration_dca')
write_table(pd.DataFrame(cal_rows), 'tab_calibration', 'llrrrrr', midrules=(3,))

# risk strata (out-of-fold)
strata_rows = []
for fs in ['FULL', 'EXPODEMO']:
    oof, _ = OOF[(PS, 'bin', fs, 'XGB')]; p = 100 * oof[:, 1]
    cat = pd.cut(p, [-1, 25, 50, 75, 101], labels=['Low (≤25)', 'Moderate (25–50]', 'High (50–75]', 'Very high (>75)'])
    for c in cat.categories:
        m = np.asarray(cat == c)
        strata_rows.append({'Model': FSET_LABEL[fs], 'Stratum': c, 'n': int(m.sum()),
                            'Share %': f'{100 * m.mean():.1f}',
                            'Observed prevalence %': f'{100 * y[m].mean():.1f}' if m.any() else '--'})
write_table(pd.DataFrame(strata_rows), 'tab_strata', 'llrrr', midrules=(4,))

# ----------------------------------------------------------------------------- 10. EEI weight sensitivity
log('10. EEI weight sensitivity')
# align components to primary-scenario rows (dedup may drop rows): recompute on dP
compP = np.column_stack([dP['travel_hours'] / 24, dP['occ'].map(W_OCC), dP['travel_vehicle'].map(W_VEH).fillna(0.5),
                         dP['stay_office_hours'] / (dP['stay_home_hours'] + dP['stay_office_hours'] + 1e-5),
                         dP['ac_service'].map({'Yes': 0.3, 'No': 1.0}).fillna(0.65)])
yP = dP['y_bin'].values
auc_expert = roc_auc_score(yP, compP @ EEI_W)
auc_noocc = roc_auc_score(yP, compP @ _w_noocc)
rng = np.random.default_rng(SEED)
W = rng.dirichlet(np.ones(5), CFG['N_EEI_DRAWS'])
aucs = np.array([roc_auc_score(yP, compP @ w) for w in W])
comp_auc = [roc_auc_score(yP, compP[:, j]) for j in range(5)]
put('EEIAUCExpert', auc_expert); put('EEIAUCNoOcc', auc_noocc)
put('NEEIDraws', CFG['N_EEI_DRAWS'], '{:,}')
put('EEIAUCPctile', 100 * (aucs < auc_expert).mean(), '{:.0f}')
put('EEIAUCmed', np.median(aucs)); put('EEIAUClo', np.percentile(aucs, 2.5)); put('EEIAUChi', np.percentile(aucs, 97.5))
best = W[aucs.argmax()]
put('EEIAUCBest', aucs.max())
for j, nm in enumerate(['Travel', 'Occ', 'Veh', 'Work', 'AC']):
    put(f'EEIComp{nm}AUC', comp_auc[j]); put(f'EEIBestW{nm}', best[j], '{:.2f}')
# occupation-ranking stability of mean EEI under perturbation
base_rank = pd.Series(compP @ EEI_W).groupby(dP['occ'].values).mean().rank()
taus = [stats.kendalltau(base_rank, pd.Series(compP @ w).groupby(dP['occ'].values).mean().rank())[0] for w in W[:300]]
put('EEIRankTau', np.nanmean(taus), '{:.2f}')
fig, ax = plt.subplots(figsize=(7, 3.8))
ax.hist(aucs, bins=40, color=OKABE[5], edgecolor='white')
ax.axvline(auc_expert, color=OKABE[1], lw=2, label=f'Expert weights ({auc_expert:.3f})')
ax.axvline(auc_noocc, color=OKABE[2], lw=2, ls='--', label=f'Without occupation term ({auc_noocc:.3f})')
ax.set_xlabel(f'Any-disease AUC of the EEI (scen. {PS})'); ax.set_ylabel('Random weight vectors'); ax.legend(fontsize=8)
ax.set_title('EEI weight sensitivity (Dirichlet(1) draws)'); fig.tight_layout(); savefig(fig, 'fig_eei_sensitivity')

# ----------------------------------------------------------------------------- 11. provenance heatmap
fig, ax = plt.subplots(figsize=(8, 6))
hm = ct[OCC_ORDER]
sns.heatmap(hm.div(hm.sum(1), axis=0), annot=hm.values, fmt='d', cmap='Blues', cbar_kws={'label': 'Row share'}, ax=ax, linewidths=0.5)
ax.set_xlabel('Source sheet (occupation)'); ax.set_ylabel('Reported label (confirmed coding)')
ax.set_title('Label provenance'); fig.tight_layout(); savefig(fig, 'fig_provenance')

# ----------------------------------------------------------------------------- 12. READINESS SCORECARD
log('12. Readiness scorecard')


def status(val, good, partial, higher_better=True):
    if val is None or (isinstance(val, float) and np.isnan(val)):
        return 'Not assessable'
    if higher_better:
        return 'Met' if val >= good else ('Partial' if val >= partial else 'Not met')
    return 'Met' if val <= good else ('Partial' if val <= partial else 'Not met')


cvB = float(NUM['CramerVB']); prov = float(NUM[f'ProvShare{PS}'])
auc_expo = score(PS, 'bin', 'EXPODEMO', 'XGB')[0]
logo_mean = float(NUM[f'LOGOmean{PS}']) if NUM[f'LOGOmean{PS}'] != '--' else np.nan
within_mean = float(NUM[f'WithinMeanClean{PS}']) if NUM[f'WithinMeanClean{PS}'] != '--' else np.nan
n_art = sum(len(v) for v in ARTEFACT.values())
unk_gap = unk.max() - unk.min()
SC = [
    ('Epistemic adequacy', 'E1', 'Label–source association (Cramér’s V, confirmed coding)', cvB, '≤ 0.30', status(cvB, 0.30, 0.50, False)),
    ('Epistemic adequacy', 'E2', 'Outcome labels exclusive to one source', len(exclusive), '0', status(len(exclusive), 0, 2, False)),
    ('Epistemic adequacy', 'E3', f'Discrimination attributable to occupation alone (%, scen. {PS})', prov, '≤ 50', status(prov, 50, 80, False)),
    ('Epistemic adequacy', 'E4', 'Distinct ambient PM2.5 values across zones', pm_distinct, '≥ 10', status(pm_distinct, 10, 5)),
    ('Epistemic adequacy', 'E5', 'Single-variable labelling artefacts within a source', n_art, '0', status(n_art, 0, 0, False)),
    ('Distributive justice', 'J1', 'Gap across groups in no-disease status left blank (pp)', unk_gap, '≤ 10', status(unk_gap, 10, 20, False)),
    ('Distributive justice', 'J2', 'Respondents with ambient PM2.5 (CAMS) coverage (%)', pm_cov, '≥ 90', status(pm_cov, 90, 50)),
    ('Distributive justice', 'J3', 'TPR gap across occupations (exposure model)', EQ['EXPODEMO'][0], '≤ 0.10', status(EQ['EXPODEMO'][0], 0.10, 0.20, False)),
    ('Distributive justice', 'J4', 'FPR gap across occupations (exposure model)', EQ['EXPODEMO'][1], '≤ 0.10', status(EQ['EXPODEMO'][1], 0.10, 0.20, False)),
    ('Ethical governance', 'G1', 'SHAP rank stability across folds (ρ, full model)', float(NUM['StabSHAPFull']), '≥ 0.80', status(float(NUM['StabSHAPFull']), 0.80, 0.60)),
    ('Ethical governance', 'G2', 'Cross-method importance agreement (ρ, full model)', float(NUM['CrossMethodFull']), '≥ 0.70', status(float(NUM['CrossMethodFull']), 0.70, 0.50)),
    ('Ethical governance', 'G3', 'Expert–data concordance, vehicle (ρ, EEI vs prevalence)', float(NUM['RhoVehEEIPrev']) if NUM['RhoVehEEIPrev'] != '--' else np.nan, '≥ 0.50', status(float(NUM['RhoVehEEIPrev']) if NUM['RhoVehEEIPrev'] != '--' else np.nan, 0.50, 0.20)),
    ('Ethical governance', 'G4', 'Records unique on age, sex, occupation, living area (%)', float(NUM['PrivUniqueB']), '≤ 5', status(float(NUM['PrivUniqueB']), 5, 20, False)),
    ('Institutional legitimacy', 'L1', f'Exposure-model AUC, pooled (scen. {PS})', auc_expo, '≥ 0.70', status(auc_expo, 0.70, 0.60)),
    ('Institutional legitimacy', 'L2', f'Exposure-model AUC within occupations (mean, artefact-corrected, scen. {PS})', within_mean, '≥ 0.70', status(within_mean, 0.70, 0.60)),
    ('Institutional legitimacy', 'L3', f'Transfer AUC to unseen occupation (mean, scen. {PS})', logo_mean, '≥ 0.65', status(logo_mean, 0.65, 0.55)),
    ('Institutional legitimacy', 'L4', f'Calibration error, ECE (exposure model, scen. {PS})', float(NUM[f'ECE{PS}EXPODEMO']), '≤ 0.05', status(float(NUM[f'ECE{PS}EXPODEMO']), 0.05, 0.10, False)),
    ('Institutional legitimacy', 'L5', 'Thresholds 0.1–0.5 with net benefit > treat-all/none (%)', float(NUM['NBpositiveEXPODEMO']), '≥ 80', status(float(NUM['NBpositiveEXPODEMO']), 80, 40)),
]
sc = pd.DataFrame(SC, columns=['Dimension', 'ID', 'Indicator', 'Value', 'Criterion', 'Status'])
def _fmt_val(v):
    if v is None or (isinstance(v, (float, np.floating)) and np.isnan(v)):
        return '--'
    if isinstance(v, (float, np.floating)) and not float(v).is_integer() and abs(v) < 10:
        return f'{v:.2f}'
    return f'{v:.0f}'


sc['Value'] = [_fmt_val(t[3]) for t in SC]
_b = sc['Dimension'].ne(sc['Dimension'].shift()).to_numpy().nonzero()[0][1:]
write_table(sc.drop(columns='Dimension'), 'tab_scorecard', 'lp{7.2cm}rll', midrules=tuple(int(i) for i in _b))
pts = {'Met': 1.0, 'Partial': 0.5, 'Not met': 0.0}
dim_score = sc.assign(pt=sc['Status'].map(pts)).groupby('Dimension', sort=False)['pt'].mean()
for dname, key in [('Epistemic adequacy', 'E'), ('Distributive justice', 'J'), ('Ethical governance', 'G'), ('Institutional legitimacy', 'L')]:
    put(f'Score{key}', 100 * dim_score.get(dname, np.nan), '{:.0f}')
    sub = sc[sc['Dimension'] == dname]['Status']
    put(f'Met{key}', int((sub == 'Met').sum()), '{:d}'); put(f'Tot{key}', len(sub), '{:d}')
put('ScoreOverall', 100 * sc['Status'].map(pts).mean(), '{:.0f}')
put('NMet', int((sc.Status == 'Met').sum()), '{:d}'); put('NPartial', int((sc.Status == 'Partial').sum()), '{:d}')
put('NNotMet', int((sc.Status == 'Not met').sum()), '{:d}'); put('NIndicators', len(sc), '{:d}')

fig, ax = plt.subplots(figsize=(9, 0.38 * len(sc) + 1))
colors = {'Met': '#009E73', 'Partial': '#E69F00', 'Not met': '#D55E00', 'Not assessable': '#999999'}
for i, r in sc.iterrows():
    ax.barh(i, 1, color=colors[r['Status']])
    ax.text(0.02, i, f"{r['ID']}  {r['Indicator']}", va='center', fontsize=8, color='white', fontweight='bold')
    ax.text(0.98, i, f"{r['Value']}  ({r['Status']})", va='center', ha='right', fontsize=8, color='white')
ax.set_yticks([]); ax.set_xticks([]); ax.invert_yaxis(); ax.set_xlim(0, 1)
for sp_ in ax.spines.values():
    sp_.set_visible(False)
ax.set_title('Empirical implementation-readiness scorecard'); fig.tight_layout(); savefig(fig, 'fig_scorecard')

# ----------------------------------------------------------------------------- 13. export
log('13. Writing numbers.tex and run_info.json')
put('RuntimeMin', (time.time() - T0) / 60, '{:.1f}')
put('Mode', 'FAST (smoke test)' if CFG['FAST'] else 'full')
with open(os.path.join(OUT, 'numbers.tex'), 'w', encoding='utf-8') as f:
    f.write('% Auto-generated by readiness pipeline -- do not edit by hand\n')
    for k in sorted(NUM):
        f.write(f'\\expandafter\\def\\csname res{k}\\endcsname{{{NUM[k]}}}\n')
info = dict(config=CFG, python=sys.version, platform=platform.platform(),
            versions={m.__name__: getattr(m, '__version__', '?') for m in [np, pd, sklearn, xgboost, shap, sm, matplotlib, sns]},
            scenarios=DINFO, runtime_min=(time.time() - T0) / 60, n_numbers=len(NUM))
with open(os.path.join(OUT, 'run_info.json'), 'w') as f:
    json.dump(info, f, indent=2, default=str)

print('\n' + '=' * 78)
print(f'DONE in {(time.time() - T0) / 60:.1f} min  ({"FAST smoke test" if CFG["FAST"] else "full run"})')
print(f'Outputs in {os.path.abspath(OUT)}: {len(NUM)} numbers, '
      f'{len([f for f in os.listdir(OUT) if f.endswith(".tex")])} .tex files, '
      f'{len([f for f in os.listdir(OUT) if f.endswith(".pdf")])} figures')
print('=' * 78)
print(sc[['ID', 'Indicator', 'Value', 'Status']].to_string(index=False))
print('\nDimension scores (% of criteria met; partial = 0.5):')
print((100 * dim_score).round(0).to_string())
print('\nAll outputs are in results/ (numbers.tex, tab_*.tex, fig_*.pdf/png, csv/, run_info.json).')
