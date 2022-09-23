"""
Computes overall blau score for all the repos (k8s, tf, vscode). Excludes golang
"""
from audioop import bias
from typing import Counter
import pandas as pd, numpy as np
from IPython import display
import warnings

warnings.filterwarnings("ignore")


def extract_org(url):
    prefix = "https://github.com/"
    org = url[len(prefix) :].split("/")[1]
    return org.strip().lower()


def combine_bot(eth):
    if eth.lower() == "bot":
        return "BOT"
    return eth


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


def pipeline(df, race=None):
    """
    computes overall BLAU score for all assignors (humans and bots) for the time
    period where we have bot-assigned issues as well.
    """
    total_contributors = len(pd.concat([df["Assignor"], df["Assignee"]]).unique())
    print(df.shape)
    print(f"Total contributors: {total_contributors}")
    # do this to get the bot-assigned issue period
    bot_df = df[df["Ethnicity-Assignor"].str.lower() == "bot"]
    print(
        bot_df[bot_df["Assignor"].str.lower() == "vscode-toolsbot"]["IssueURL"].tolist()
    )
    unique_assignees = len(bot_df["Assignee"].unique())
    human_assignees = len(
        df[df["Ethnicity-Assignor"].str.lower() != "bot"]["Assignee"].unique()
    )
    assignee_eth_dist = Counter(bot_df["Ethnicity-Assignee"].tolist())
    human_assignee_eth_dist = Counter(
        df[df["Ethnicity-Assignor"].str.lower() != "bot"]["Ethnicity-Assignee"].tolist()
    )
    print(f"Unique Assignees by bots | humans: {unique_assignees} | {human_assignees}")
    print(f"Assignees Ethnicity distribution for bots: {assignee_eth_dist}")
    print(
        f"Human Assignees Ethnicity distribution for humans: {human_assignee_eth_dist}"
    )
    if race and race != "bot":
        # filter out bot issues
        df = df[df["Ethnicity-Assignor"].str.lower() != "bot"]
    print(df.shape)
    if race == "bot":
        print(df["Assignor"].unique())
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
    # compute BLAU scores for the df
    return compute_blau_df(df, binary_classification=False), df


def main():
    file_path = "./data/processed/issuewithdate.csv"
    df = pd.read_csv(file_path)

    df["org"] = df["IssueURL"].apply(extract_org)
    df["Ethnicity-Assignor"] = df["Ethnicity-Assignor"].apply(combine_bot)

    print("Excluding go and elasticsearch")
    df = df[df["org"].isin(["vscode", "tensorflow", "kubernetes"])]
    # remove Other race and self assignments and PRs
    df = df[
        (df["Ethnicity-Assignor"].isin(["W_NL", "B_NL", "A", "HL", "BOT"]))
        & (df["Ethnicity-Assignee"].isin(["W_NL", "B_NL", "A", "HL"]))
        & (df["Assignor"] != df["Assignee"])
        & (df["Ispr"].isin(["no"]))
    ]
    print(df.shape)
    display.display(df.head())

    print(pipeline(df)[0])
    print("\n====")

    print("Orgwise overall BLAU scores")
    print("=" * 50)
    new_df = []
    for org in df["org"].unique():
        print(f"{org.upper()}")
        _df = df[df["org"] == org]
        score, _filtered_df = pipeline(_df)
        new_df.append(_filtered_df)
        print(score)
        print("\n====")

    print("Overall BLAU Score")
    print(pipeline(pd.concat(new_df))[0])


if __name__ == "__main__":
    main()
