"""
Computes overall blau score for all the repos (k8s, tf, vscode). Excludes golang
and tells what percentile is the bot score compared to humans
"""
from audioop import reverse
import pandas as pd, numpy as np
from IPython import display
import warnings

warnings.filterwarnings("ignore")


def extract_org(url):
    prefix = "https://github.com/"
    org = url[len(prefix) :].split("/")[1]
    return org.strip().lower()


def compute_blau_df(df, binary_classification=True):
    # major is 1 minor is 0
    if df.shape[0] == 0:
        return {"score": None, "major": 0, "total": 0}
    # pd.set_option('display.max_rows', None)
    # print(display(df))
    major_count = np.count_nonzero(df["Ethnicity-Assignee"].values == "W_NL")
    major_prob = major_count / df.shape[0]
    # print(major_prob)
    pis = np.array([major_prob, 1 - major_prob])
    # print blau scores
    binary = {"score": 1 - np.sum(pis**2), "major": major_count, "total": df.shape[0]}
    if binary_classification:
        return binary
    pis = []
    for race in df["Ethnicity-Assignee"].unique():
        _prob = np.count_nonzero(df["Ethnicity-Assignee"].values == race) / df.shape[0]
        pis.append(_prob)
    pis = np.array(pis)
    fourway = {
        "score": 1 - np.sum(pis**2),
        "major": major_count,
        "total": df.shape[0],
    }
    return fourway


def fix_df_dates(df):
    # as per the fix here: https://datascientyst.com/outofboundsdatetime-out-of-bounds-nanosecond-timestamp-pandas-pd-to_datetime/
    # create a temp column
    df["date2"] = pd.to_datetime(df["Date"], errors="coerce")
    # get the problematic
    temp = df[df["date2"].isna()]
    print(temp.head())
    df.loc[temp.index, "Date"] = temp["Date"] + " 2021"
    df["Date Object"] = pd.to_datetime(df["Date"])
    return df


def get_percentile_rank(x, dataset):
    n = len(dataset)
    v = sum(1 if _x >= x else 0 for _x in dataset)
    return (n - v) / n


def pipeline(df):
    """
    computes overall BLAU score for individual assignors (different humans and bots) for the time
    period where we have bot-assigned issues as well.
    """
    bot_df = df[
        df["Ethnicity-Assignor"].str.lower() == "bot"
    ]  # do this to get the bot-assigned issue period
    # filtering out bot issues
    df = df[df["Ethnicity-Assignor"].str.lower() != "bot"]
    print(df.shape)
    print(f"{len(np.unique(df['Assignor']))} unique assignors")
    # add a column with dates as datetime objects
    try:
        df["Date Object"] = pd.to_datetime(df["Date"])
        bot_df["Date Object"] = pd.to_datetime(bot_df["Date"])
    except:
        df = fix_df_dates(df)
        bot_df = fix_df_dates(bot_df)
    df = df.sort_values(by="Date Object")
    df.head()
    bot_df = bot_df.sort_values(by="Date Object")

    start_date = bot_df.iloc[0]["Date Object"]
    max_date = bot_df.iloc[-1]["Date Object"]

    day_diff = max_date - start_date
    print(day_diff)
    print("*" * 30)

    # filtered_df = df[(df["Date Object"] >= start_date) & (df["Date Object"] < max_date)]
    print(f"Computing BLAU scores for ALL assignors in the following period:")
    print(f"START DATE: {start_date} | END DATE: {max_date}")
    human_blaus = []
    for assignor in df["Assignor"].unique():
        _df = df[df["Assignor"] == assignor]
        scores = compute_blau_df(_df, binary_classification=False)
        blau, total = scores["score"], scores["total"]
        human_blaus.append((blau, assignor, total))
    threshold = np.percentile([c[2] for c in human_blaus], q=90)
    print("THRESHOLD:", threshold)
    human_blaus = [o for o in human_blaus if o[2] >= threshold]
    human_blaus.sort()
    bot_blau = compute_blau_df(bot_df, binary_classification=False)
    human_blau = compute_blau_df(df, binary_classification=False)
    best_human, worst_human = human_blaus[-1], human_blaus[0]
    _human_scores = [c[0] for c in human_blaus]
    median_human = np.median(_human_scores)
    bot_percentile_rank = get_percentile_rank(bot_blau["score"], _human_scores)

    print(f"BOT BLAU: {bot_blau}")
    print(f"HUMAN BLAU: {human_blau}")
    print(f"BOT Percentile Rank: {bot_percentile_rank}")
    print(f"Best human: {best_human}")
    print(f"Worst Human: {worst_human}")
    print(f"Median Human: {median_human}")
    return human_blaus


def combine_bot(eth):
    if eth.lower() == "bot":
        return "BOT"
    return eth


def main():
    file_path = "./data/processed/issuewithdate.csv"
    df = pd.read_csv(file_path)

    df["org"] = df["IssueURL"].apply(extract_org)
    df["Ethnicity-Assignor"] = df["Ethnicity-Assignor"].apply(combine_bot)
    # remove Other race and self assignments
    df = df[
        (df["Ethnicity-Assignor"].isin(["W_NL", "B_NL", "A", "HL", "BOT"]))
        & (df["Ethnicity-Assignee"].isin(["W_NL", "B_NL", "A", "HL"]))
        & (df["Assignor"] != df["Assignee"])
        & (df["Ispr"].isin(["no"]))
    ]
    print("Excluding go and elasticsearch")
    df = df[df["org"].isin(["vscode", "tensorflow", "kubernetes"])]
    print(df.shape)
    print(pipeline(df))
    display.display(df.head())
    for org in df["org"].unique():
        print(f"{org.upper()}")
        _df = df[df["org"] == org]
        pipeline(_df)
        print("\n====")


if __name__ == "__main__":
    main()
