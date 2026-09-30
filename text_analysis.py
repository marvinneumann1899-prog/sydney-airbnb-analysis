"""Airbnb in Sydney — sentiment analysis and the guests' top 5 concerns (CRISP-DM phases 3-5).

Design decision: guests rarely write fully negative reviews. Complaints usually sit inside
otherwise positive reviews ("great location, but the street was noisy"). Scoring whole reviews
would hide them. So the pipeline scores every SENTENCE with VADER and runs the topic model only
on clearly negative sentences. The five topics of those sentences are the guests' five concerns.

Run:  .venv/bin/python text_analysis.py
"""
from __future__ import annotations

import re
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.decomposition import NMF, LatentDirichletAllocation
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS, CountVectorizer, TfidfVectorizer
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

HERE = Path(__file__).parent
RAW, RES, FIG = HERE / "data" / "raw", HERE / "results", HERE / "figures"
SEED = 42
WINDOW_START = pd.Timestamp("2025-06-16")     # last 12 months before the snapshot
NEG_THRESHOLD = -0.30                          # clearly negative sentence (VADER compound)
N_TOPICS = 15  # discovery: fit 15, label by reading, group into concerns, report the top 5
AUTOMATED = re.compile(r"this is an automated posting", re.I)
EN_HINTS = {"the", "and", "was", "we", "very", "place", "great", "to", "is", "in", "it", "stay"}

DOMAIN_STOP = {"stay", "stayed", "place", "sydney", "airbnb", "host", "hosts", "apartment", "house",
               "home", "room", "really", "just", "would", "did", "didn", "don", "got", "bit", "little",
               "time", "day", "days", "night", "nights", "one", "also", "however", "though", "even",
               "thing", "things", "get", "us", "could", "unit", "property", "definitely"}
# Generic evaluation words carry the sentiment, not the concern. Left in, they form their own
# "topics" ("unfortunately", "disappointing") that say nothing a planner could act on.
GENERIC_NEG = {"unfortunately", "disappointing", "disappointed", "disappointment", "experience", "issue",
               "issues", "problem", "problems", "difficult", "overall", "recommend", "expectations", "worst",
               "extremely", "quite", "wasn", "work", "booked", "downside", "bad", "poor", "negative",
               "terrible", "horrible", "awful", "sadly", "complaint", "complaints", "minor", "only", "need",
               "felt", "better", "hard", "lack", "left", "use", "wasn", "isn", "doesn", "aren", "weren",
               "couldn", "wouldn", "shame", "pity", "annoying", "hate", "worse", "unpleasant"}

# Topic -> concern mapping, set by reading the top terms and the 8 highest-weight sentences of each
# NMF topic (results/topics_nmf_terms.csv, topics_nmf_examples.csv; run of 2026-09-24, seed 42).
# "praise misread" = VADER scored praise as negative ("couldn't fault", "could not be better",
# "a block away"). Those topics are excluded from the concerns and reported as a method limit.
CONCERNS = {
    1: "Cleanliness", 6: "Cleanliness", 5: "Cleanliness",
    2: "Beds and sleep comfort", 9: "Beds and sleep comfort",
    13: "Broken or poorly maintained facilities", 14: "Broken or poorly maintained facilities",
    10: "Broken or poorly maintained facilities",
    12: "Kitchen equipment and supplies",
    7: "Noise",
    4: "Parking",
    3: "praise misread", 8: "praise misread", 15: "praise misread", 11: "praise misread",
}

plt.rcParams.update({"font.family": "Arial", "font.size": 9, "axes.spines.top": False,
                     "axes.spines.right": False, "figure.dpi": 200})


def clean(text: str) -> str:
    text = re.sub(r"<br\s*/?>", " ", str(text))
    return re.sub(r"\s+", " ", text).strip()


def is_english(text: str) -> bool:
    """Cheap filter: at least three common English function words among the first 40 tokens."""
    tokens = re.findall(r"[a-z']+", text.lower())[:40]
    return sum(t in EN_HINTS for t in tokens) >= 3


def load_reviews() -> tuple[pd.DataFrame, dict]:
    rev = pd.read_csv(RAW / "reviews.csv.gz", usecols=["listing_id", "date", "comments"], parse_dates=["date"])
    stats = {"reviews_total": len(rev)}
    rev = rev[rev["date"] >= WINDOW_START].copy()
    stats["reviews_last_12m"] = len(rev)
    rev["comments"] = rev["comments"].map(clean)
    auto = rev["comments"].str.contains(AUTOMATED)
    stats["automated_cancellation_posts"] = int(auto.sum())
    rev = rev[~auto & (rev["comments"].str.split().str.len() >= 3)]
    en = rev["comments"].map(is_english)
    stats["non_english_or_short_dropped"] = int((~en).sum())
    rev = rev[en].reset_index(drop=True)
    stats["reviews_analysed"] = len(rev)
    return rev, stats


def host_name_stopwords() -> set[str]:
    """Host first names would otherwise dominate topics ("Anna was lovely")."""
    names = pd.read_csv(RAW / "listings.csv.gz", usecols=["host_name"])["host_name"].dropna()
    return {w for n in names for w in re.findall(r"[a-z]+", n.lower()) if len(w) > 2}


def sentences(rev: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for rid, lid, text in zip(rev.index, rev["listing_id"], rev["comments"]):
        for s in re.split(r"(?<=[.!?])\s+", text):
            letters = re.findall(r"[^\W\d_]", s)
            ascii_share = sum(c.isascii() for c in letters) / max(len(letters), 1)
            toks = re.findall(r"[a-z']+", s.lower())
            en_share = sum(t in ENGLISH_STOP_WORDS for t in toks) / max(len(toks), 1)
            if len(toks) >= 4 and ascii_share >= 0.95 and en_share >= 0.25:
                rows.append((rid, lid, s))
    return pd.DataFrame(rows, columns=["review", "listing_id", "sentence"])


def fit_topics(neg: pd.Series, stop: set[str]) -> tuple[pd.DataFrame, np.ndarray, str]:
    """Fit NMF (TF-IDF) and LDA (counts) with k=5. Both are reported; NMF is used for the
    concern shares because its topics were more distinct on short sentences (checked by reading
    the top sentences per topic, see results/topics_*_examples.csv)."""
    stop_words = list(ENGLISH_STOP_WORDS | DOMAIN_STOP | GENERIC_NEG | stop)
    tfidf = TfidfVectorizer(stop_words=stop_words, min_df=20, max_df=0.3, ngram_range=(1, 2),
                            token_pattern=r"(?u)\b[a-z]{3,}\b")
    X = tfidf.fit_transform(neg.str.lower())
    nmf = NMF(n_components=N_TOPICS, random_state=SEED, init="nndsvda", max_iter=400)
    W = nmf.fit_transform(X)
    terms = np.array(tfidf.get_feature_names_out())
    nmf_top = pd.DataFrame({f"T{k+1}": terms[np.argsort(nmf.components_[k])[::-1][:12]]
                            for k in range(N_TOPICS)})

    cv = CountVectorizer(stop_words=stop_words, min_df=20, max_df=0.3, token_pattern=r"(?u)\b[a-z]{3,}\b")
    C = cv.fit_transform(neg.str.lower())
    lda = LatentDirichletAllocation(n_components=N_TOPICS, random_state=SEED, learning_method="online")
    lda.fit(C)
    cterms = np.array(cv.get_feature_names_out())
    lda_top = pd.DataFrame({f"T{k+1}": cterms[np.argsort(lda.components_[k])[::-1][:12]]
                            for k in range(N_TOPICS)})
    lda_top.to_csv(RES / "topics_lda_terms.csv", index=False)
    return nmf_top, W, "NMF"


def main() -> None:
    rev, stats = load_reviews()
    sia = SentimentIntensityAnalyzer()
    rev["compound"] = [sia.polarity_scores(t)["compound"] for t in rev["comments"]]
    rev["label"] = pd.cut(rev["compound"], [-1.01, -0.05, 0.05, 1.01], labels=["negative", "neutral", "positive"])

    sent = sentences(rev)
    sent["compound"] = [sia.polarity_scores(s)["compound"] for s in sent["sentence"]]
    neg = sent[sent["compound"] <= NEG_THRESHOLD].reset_index(drop=True)
    stats.update({
        "review_positive_%": round((rev["label"] == "positive").mean() * 100, 1),
        "review_neutral_%": round((rev["label"] == "neutral").mean() * 100, 1),
        "review_negative_%": round((rev["label"] == "negative").mean() * 100, 1),
        "sentences": len(sent),
        "negative_sentences": len(neg),
        "negative_sentence_%": round(len(neg) / len(sent) * 100, 2),
        "reviews_with_negative_sentence_%": round(neg["review"].nunique() / len(rev) * 100, 1),
    })

    top, W, method = fit_topics(neg["sentence"], host_name_stopwords())
    neg["topic"] = W.argmax(axis=1) + 1
    neg["weight"] = W.max(axis=1)
    # no topic if every token was a stop word, or if the best topic is only weakly present
    # (bottom 30 % of max weights) -- otherwise weak sentences inflate whichever topic wins argmax
    unassigned = (neg["weight"] == 0) | (neg["weight"] < neg.loc[neg["weight"] > 0, "weight"].quantile(0.30))
    stats["negative_sentences_without_topic"] = int(unassigned.sum())
    neg = neg[~unassigned].copy()
    top.to_csv(RES / "topics_nmf_terms.csv", index=False)
    # prevalence = share of total topic weight (soft assignment). Argmax counts inflate
    # whichever topic absorbs weakly-fitting sentences; weight mass does not.
    Wk = W[~unassigned.to_numpy()]
    shares = pd.Series(Wk.sum(axis=0) / Wk.sum() * 100, index=range(1, N_TOPICS + 1)).round(1)
    shares.to_csv(RES / "topics_nmf_shares.csv")

    # group topics into concerns, per area (Top 10 LGAs vs rest)
    listings = pd.read_csv(RAW / "listings.csv.gz", usecols=["id", "neighbourhood_cleansed"])
    top10 = listings["neighbourhood_cleansed"].value_counts().head(10).index
    area = listings.set_index("id")["neighbourhood_cleansed"].isin(top10).map({True: "Top 10", False: "Rest"})
    wdf = pd.DataFrame(Wk, columns=range(1, N_TOPICS + 1))
    wdf["area"] = neg["listing_id"].map(area).to_numpy()
    grouped = wdf.groupby("area").sum().T.groupby(pd.Series(CONCERNS)).sum()
    grouped["All"] = grouped.sum(axis=1)
    misread = grouped.loc["praise misread", "All"] / grouped["All"].sum() * 100
    stats["topic_weight_praise_misread_%"] = round(misread, 1)
    concerns = grouped.drop(index="praise misread")
    concerns = (concerns / concerns.sum() * 100).round(1).sort_values("All", ascending=False)
    concerns.to_csv(RES / "concerns_ranked.csv")
    print("concerns (% of informative topic weight)\n", concerns.to_string(), "\n")

    fig, ax = plt.subplots(figsize=(6.3, 2.3))
    c = concerns.iloc[::-1]
    y = np.arange(len(c))
    ax.barh(y + 0.2, c["Top 10"], 0.4, color="#1f4e79", label="Top 10 LGAs")
    ax.barh(y - 0.2, c["Rest"], 0.4, color="#a6a6a6", label="Rest")
    ax.set_yticks(y, c.index, fontsize=8)
    ax.set_xlabel("Share of complaint content (%)")
    ax.set_title(f"Guest concerns in negative review sentences, last 12 months (n = {len(neg):,})",
                 loc="left", fontsize=9)
    ax.legend(frameon=False, fontsize=7, loc="lower right")
    fig.tight_layout()
    fig.savefig(FIG / "fig4_concerns.png")
    plt.close(fig)
    # examples for manual labelling: 8 highest-weight sentences per topic
    ex = (neg.sort_values("weight", ascending=False).groupby("topic").head(8)
             .sort_values(["topic", "weight"], ascending=[True, False])[["topic", "sentence"]])
    ex.to_csv(RES / "topics_nmf_examples.csv", index=False)
    pd.Series(stats).to_csv(RES / "sentiment_stats.csv")

    print(pd.Series(stats).to_string(), "\n")
    print(f"{method} top terms\n", top.to_string(), "\n")
    print("LDA top terms\n", pd.read_csv(RES / "topics_lda_terms.csv").to_string(), "\n")
    print("topic shares %\n", shares.to_string(), "\n")
    for k, g in ex.groupby("topic"):
        print(f"--- T{k}")
        for s in g["sentence"].head(4):
            print("   ", s[:160])


if __name__ == "__main__":
    main()
