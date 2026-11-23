# Model card: SpendSense expense categoriser (v1)

**Purpose.** Suggest one of 8 categories (Food, Groceries, Transport, Shopping, Bills, Health, Education,
Entertainment) for a student's expense from its short description, amount and date. The user always sees the
suggestion and confirms or corrects it; low-confidence suggestions are flagged for review.

**Data.**

- *Training:* Daily Household Transactions (one Indian household, INR). After keeping expenses only, mapping to the
  8 categories and removing empty or duplicate rows: 1,573 rows. The earliest 1,258 rows (2015-01-01 to 2018-03-15)
  are for training; the latest 315 (2018-03-17 to 2018-09-20) are held out for testing.
- *Very imbalanced:* Food 367 and Groceries 347 training rows, Education only 20.
- *Transfer test:* Personal Finance Transaction Analysis (one US person, USD, 604 mapped debit rows, no Health or Education).
- *Own expenses:* currently 5 example rows only.

**Model.** The description is cleaned (lower-case, digits and punctuation removed), turned into a 384-number MiniLM
sentence embedding (`all-MiniLM-L6-v2`, used as-is), and joined with `log1p(amount)`, weekday and an is-weekend flag.
A Logistic Regression model (C = 10, class_weight = "balanced") makes the prediction. These settings were chosen by
5-fold cross-validation on the training split (CV macro-F1 0.786), ahead of TF-IDF + Logistic Regression (0.743).
Predictions below confidence **0.6** are flagged for review. That threshold is the lowest at which predictions
were at least 90% correct in cross-validation.

**Results** (from `notebooks/04_evaluate.ipynb`, saved in `reports/results.csv`):

| Model | Household test acc. | Household test macro-F1 | Personal finance macro-F1 | Own (5 rows) macro-F1 |
|---|---|---|---|---|
| Majority baseline | 0.314 | 0.060 | 0.068 | 0.067 |
| Zero-shot MiniLM | 0.365 | 0.310 | **0.819** | 0.333 |
| TF-IDF + LogReg | 0.876 | 0.812 | 0.138 | 0.800 |
| **MiniLM + LogReg (chosen)** | **0.902** | **0.878** | 0.360 | 0.600 |

At the 0.6 threshold, 91% of test predictions are accepted automatically, and 94.8% of those are correct.

**Known weaknesses.**

- *It learns one household's habits.* On another person's data it drops to 0.360 macro-F1, and zero-shot does far
  better. The household labels call "grocery" Food 6 times out of 7, so every "Grocery Store" was predicted Food.
- *Food and Groceries are confused* (10 of 20 sampled test errors). The boundary is fuzzy, and the source labels
  are themselves inconsistent.
- *It has never seen Pakistani words or brands* ("Jazz load", "Careem", "Imtiaz"). It was trained on Indian notes and
  anonymised places ("Place 0 to Place 4"), and 60% of test descriptions also appear in training, so the household
  test score is optimistic.
- *Amounts are taken in whatever currency arrives* (INR in training, PKR in the app). A check on the personal-finance
  data showed that the amount has little effect on predictions there, but it is still unscaled.
- *Rare categories are measured on very few rows* (3 Education and 7 Entertainment test rows), so their scores are noisy.
- *Some mistakes are confident* (0.94 for "Ginger + green chilli"), so the threshold cannot catch every error.
- *OCR accuracy has not been measured yet*, because no labelled receipt photos exist.

**Must not be used for.**

- Financial, tax, credit or legal decisions.
- Judging or scoring a person's spending.
- Categorising anyone's data without their consent.
- Automatic bookkeeping without a human confirming each category.

It is a learning project trained on one household's data, not a validated financial tool.
