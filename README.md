# SpendSense

SpendSense suggests a category for a student expense from a short description or a receipt photo. There are
8 categories: Food, Groceries, Transport, Shopping, Bills, Health, Education and Entertainment. Two pre-trained
models are used as they are. Tesseract OCR reads receipts, and the MiniLM sentence-embedding model turns a
description into 384 numbers, which a Logistic Regression model turns into a category. The model was trained on
one Indian household's transactions. The student confirms or corrects every suggestion in a small Streamlit app,
every prediction is logged, and the monitoring page shows when live accuracy has dropped enough to retrain with
the student's own corrections.

## Setup

Runs on CPU only. Python 3.11+ is recommended; Python 3.9 and 3.10 also work (tested on 3.9 and 3.11).

```bash
cd spendsense
python3 -m venv .venv                # on macOS, `python3.11 -m venv .venv` after `brew install python@3.11`
source .venv/bin/activate            # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

Tesseract OCR is a separate program, not a Python package:

| OS | Command |
|---|---|
| macOS | `brew install tesseract` |
| Ubuntu / Debian | `sudo apt install tesseract-ocr` |
| Windows | Run the installer from https://github.com/UB-Mannheim/tesseract/wiki, then add `C:\Program Files\Tesseract-OCR` to PATH |

Check it with `tesseract --version`.

The MiniLM model (`sentence-transformers/all-MiniLM-L6-v2`) downloads automatically the first time it is used.
If Hugging Face is blocked, put a copy in `models/all-MiniLM-L6-v2/` and the code will use that instead.

## Run it

1. **Notebooks, in order** (`jupyter notebook`, then open each file and choose *Run All*):
   1. `notebooks/01_pretrained_models.ipynb`: OCR accuracy on your receipts, and the zero-shot baseline.
   2. `notebooks/02_eda.ipynb`: charts, findings and the decisions they led to.
   3. `notebooks/03_train.ipynb`: features, 5-fold CV tuning; saves `models/model.joblib` and `metadata.json`.
   4. `notebooks/04_evaluate.ipynb`: held-out and transfer results, errors, threshold; writes `reports/results.csv`.
2. **App:** `streamlit run app.py`. It has three pages: Add expense, This month, and Model monitoring.
   A saved model only loads with the scikit-learn version that built it. If yours differs (or `models/` is
   empty), the app opens a setup screen with a **Build model now** button that trains it in about a minute.
   You can also do this from the terminal with `python -m src.train`.
3. **Tests:** `pytest`.
4. **Retrain by hand** (the same thing the app's button does): `python -m src.retrain`.

## Results

These are copied from `notebooks/04_evaluate.ipynb` (`reports/results.csv`), reference run on Python 3.11 with
scikit-learn 1.9. Other library versions can shift the third decimal (Python 3.9 with scikit-learn 1.6.1 gave
0.905 accuracy and 0.880 macro-F1 for the chosen model); the notebook you run is always the source of truth. The chosen model is MiniLM + Logistic
Regression (C = 10, class_weight = "balanced"), picked by 5-fold cross-validation on the training split
(CV macro-F1 0.786).

| Model | Household test accuracy | Household test macro-F1 | Personal finance macro-F1 | Own expenses macro-F1 (5 rows) |
|---|---|---|---|---|
| Majority baseline | 0.314 | 0.060 | 0.068 | 0.067 |
| Zero-shot MiniLM | 0.365 | 0.310 | 0.819 | 0.333 |
| TF-IDF + LogReg | 0.876 | 0.812 | 0.138 | 0.800 |
| **MiniLM + LogReg** | **0.902** | **0.878** | 0.360 | 0.600 |

- **Household test:** the latest 20% of household dates (315 rows, 2018-03-17 to 2018-09-20).
- **Personal finance:** 604 rows from a different person, in USD.
- **Review threshold:** 0.6. At this threshold, 91% of test predictions are accepted automatically and 94.8% of
  those are correct.
- **Model card:** see `reports/model_card.md` for the details.

## Data

| File | Source | Used for |
|---|---|---|
| `data/raw/daily_household_transactions.csv` | Daily Household Transactions (Kaggle), from the MIT-licensed mirror github.com/sanjanakrishnaraj0-lang/Daily-Transaction-Analysis | Training and time-based test split (INR) |
| `data/raw/personal_finance_transactions.csv` | Personal Finance Transaction Analysis (Kaggle, MIT), from github.com/FERIC19/Personal-Finance-Dashboard | Transfer test on another person's data (USD) |
| `data/own_expenses.csv` | Your own hand-labelled expenses: `date, description, amount_pkr, category` | Final test |
| `data/receipts/labels.csv` | One row per photo in `data/receipts/`: `file, total, date (YYYY-MM-DD), merchant` | OCR accuracy |
| `data/category_map.csv` | Written by hand after inspecting the source categories | Source category → SpendSense category |

**The 5 rows in `own_expenses.csv` are examples. Replace them with real expenses.** The receipts folder is empty
until you add photos.

## Project layout

```
src/config.py      paths, categories, seed (RANDOM_SEED = 42), confidence threshold
src/data.py        load, map categories, clean, time_split
src/features.py    clean_text, TF-IDF, MiniLM embeddings (cached), amount/date features
src/zero_shot.py   zero-shot classifier (no training)
src/train.py       build, tune, score and save models (used by notebook 03 and retrain.py)
src/ocr.py         receipt photo -> text, total, date
src/predict.py     load the saved model once, predict one expense
src/db.py          SQLite tables: expenses, prediction_log
src/monitoring.py  weekly metrics and the retrain rule
src/retrain.py     retrain with logged corrections; keep only if macro-F1 does not drop
app.py             Streamlit app (look: .streamlit/config.toml + CSS in app.py, logo in assets/)
```

## Limitations

- **One household's habits.** The model is trained on one Indian household (2015–2018). On another person's data
  its macro-F1 drops to 0.360, below zero-shot. One reason is that the household's own labels call "grocery" Food
  6 times out of 7.
- **Optimistic test score.** 60% of household test descriptions also appear in training, and anonymised notes like
  "Place 0 to Place 4" make Transport easy.
- **Pakistani words.** Words like "Jazz load", "Careem" and "Imtiaz" were never in training. Expect corrections
  until the model has been retrained on your own data.
- **Food vs Groceries** is the most common confusion.
- **Amounts are not converted between currencies.** Training is in INR and the app is in PKR.
- **OCR accuracy is not measured yet.** It needs labelled receipt photos. The total and date regexes cover common
  formats only.
- **Rare categories are scored on few rows.** Education has 3 test rows and Entertainment has 7, so their scores
  are noisy.
- **Not a financial tool.** See the model card for what it must not be used for.
