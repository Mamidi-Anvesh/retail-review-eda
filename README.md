# Retail Product Review Analysis Using Exploratory Data Analysis (EDA)

A beginner-friendly data analytics project that analyses **23,486 women's clothing e-commerce reviews** to understand customer satisfaction, product performance, common complaints and praise, and how written sentiment compares with star ratings.

**Tools:** Python · Pandas · NumPy · Matplotlib · Seaborn · SciPy · VADER sentiment · Scikit-learn (optional extension)

---

## What the project answers

| Question | Where in the notebook |
|---|---|
| How satisfied are customers overall? | 4.1, 4.2 |
| Which departments, categories and products perform best and worst? | 4.3 – 4.5 |
| Does having more reviews relate to a higher rating? | 4.6 |
| Who writes the reviews, and how do age, length and helpful votes relate to ratings? | 4.8 |
| What words and themes separate happy from unhappy customers? | 5.1 – 5.4 |
| How well does text sentiment match star ratings? | 6.1 – 6.5 |
| Can review text predict "would recommend"? | 7 |
| What should a retailer do about it? | 8, 9 |

## Dataset

**Women's E-Commerce Clothing Reviews** (public Kaggle dataset; customer-identifying details are anonymised).

| Column | Meaning |
|---|---|
| `Clothing ID` | Product identifier (no product names are provided) |
| `Age` | Reviewer age |
| `Title`, `Review Text` | Written review |
| `Rating` | 1 – 5 stars |
| `Recommended IND` | 1 if the reviewer recommends the item |
| `Positive Feedback Count` | Helpful votes |
| `Division Name`, `Department Name`, `Class Name` | Product hierarchy |

**Not in this dataset:** review dates and verified-purchase status. The notebook detects this and skips those analyses (the code is included for datasets that have them).

Please follow the dataset's licence and terms on Kaggle when sharing the data.

## Project structure

```
retail-review-eda/
├── Retail_Review_EDA.ipynb      # full notebook (already executed, outputs included)
├── REPORT.md                    # project report
├── VIVA_AND_PRESENTATION.md     # 10 viva questions + presentation outline
├── requirements.txt
├── data/
│   ├── Womens_Clothing_E-Commerce_Reviews.csv   # raw data
│   └── cleaned_reviews.csv                      # produced by the notebook
├── figures/                     # 11 charts saved by the notebook
└── results/
    ├── summary.json                       # key numbers produced by the notebook
    └── manual_validation_sample.csv       # 99 reviews for YOU to label (see below)
```

## Project workflow

1. **Load and detect columns**: the notebook maps your column names to standard names, so it adapts to other review datasets.
2. **Inspect**: shape, types, missing values, duplicates, odd values.
3. **Clean**: fix typos, validate ratings and ages, remove duplicates, handle missing values, normalise text. Every step is logged and explained.
4. **EDA**: ratings, departments, categories, products (with a minimum-review rule), review count vs rating, age, length, helpful votes.
5. **Text analysis**: common words and phrases, distinctive terms, complaint and praise themes.
6. **Sentiment**: VADER vs star ratings, with agreement metrics, a "predict everything positive" baseline and a manual validation sample.
7. **Optional ML**: TF-IDF + logistic regression predicts "would recommend" and is compared with VADER on unseen reviews.
8. **Insights and recommendations**: generated from the computed results.

## How to run

### Option A: Google Colab (easiest)
1. Go to <https://colab.research.google.com> → **File → Upload notebook** → choose `Retail_Review_EDA.ipynb`.
2. Click the folder icon in the left sidebar and upload `Womens_Clothing_E-Commerce_Reviews.csv` (or put it in a `data` folder). If the file is not found, the notebook opens an upload dialog by itself.
3. **Runtime → Run all.** The first cell installs `vaderSentiment` automatically if needed. Expect roughly 1–3 minutes.

### Option B: VS Code (or Jupyter)
```bash
# 1. open a terminal in the project folder
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements.txt

# 2. open the notebook in VS Code (install the "Python" and "Jupyter" extensions),
#    choose the .venv kernel, then click "Run All"
code Retail_Review_EDA.ipynb
```
Or run it from the terminal without opening it:
```bash
jupyter nbconvert --to notebook --execute --inplace Retail_Review_EDA.ipynb
```

The notebook was executed successfully with pandas 2.3.3 and pandas 3.x, and gave identical results on both.

## Key results (all produced by the notebook)

- **23,464 reviews** remain after cleaning (22 duplicates removed). **1,206 products**, but the median product has only **2 reviews**.
- Mean rating **4.20 / 5**; **77.5%** of reviews are 4–5 stars, **10.3%** are 1–2 stars; **82.2%** of reviewers recommend the item.
- Lowest-rated department: **Trend** (3.82, only 119 reviews). Among large categories, **Dresses, Blouses and Knits** have the lowest averages (about 4.15–4.16).
- The complaint theme most over-represented in low-rated reviews is **"returned / disappointed"** (about 8.5× more common than in high-rated reviews), followed by **poor shape / unflattering cut** (4.4×) and **fabric / quality problems** (2.8×).
- **VADER** agrees with a star-based proxy on 79.0% of reviews, but simply labelling everything "positive" would score 77.1%. VADER labels about 70% of 1–2 star reviews as positive, so it should not be used alone.
- A trained text model predicts "would recommend" with ROC-AUC **0.946** vs **0.787** for VADER on 4,528 unseen reviews.

See `REPORT.md` for the full results and limitations.

## Important: the sentiment accuracy is NOT yet validated

The brief requires that we do not claim accuracy without validation. Star-based agreement is **not** accuracy. To measure real accuracy:

1. Open `results/manual_validation_sample.csv`.
2. For each of the 99 reviews, type `Positive`, `Neutral` or `Negative` in the `human_label` column (judge the **text**, not the stars).
3. Save, then re-run the cell in notebook section 6.5. It prints VADER's accuracy against your labels.

## Limitations (short version)

No dates or verified flags; no product names; most products have very few reviews; reviewers are self-selected; keyword themes are approximate; VADER is not tuned for clothing reviews; results show association, not causation. Details are in the report.

## Author and licence

Add your name, institution and chosen licence here.
