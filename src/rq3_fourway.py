import collections
from datetime import datetime, timedelta
import warnings

warnings.filterwarnings("ignore")
import pandas as pd, numpy as np, os, seaborn as sns, matplotlib.pyplot as plt, matplotlib._color_data as mcd
from scipy.ndimage import gaussian_filter1d
from IPython import display


def extract_org(url):
    prefix = "https://github.com/"
    org = url[len(prefix) :].split("/")[1]
    return org.strip().lower()


def compute_blau_df(
    df,
    binary_classification=True,
    classes=["W_NL", "B_NL", "A", "HL"],
):
    # major is 1 minor is 0
    if df.shape[0] == 0:
        return {"score": None, "major": 0, "total": 0}, {k: 0 for k in classes}
    # pd.set_option('display.max_rows', None)
    # print(display(df))
    major_count = np.count_nonzero(df["Ethnicity-Assignee"].values == "W_NL")
    major_prob = major_count / df.shape[0]
    # print(major_prob)
    pis = np.array([major_prob, 1 - major_prob])
    # print blau scores
    binary = {"score": 1 - np.sum(pis**2), "major": major_count, "total": df.shape[0]}
    if binary_classification:
        return binary, {"major": pis[0], "minor": pis[1]}
    pis = []
    pis_dict = {}
    for race in classes:
        _prob = np.count_nonzero(df["Ethnicity-Assignee"].values == race) / df.shape[0]
        pis.append(_prob)
        pis_dict[race] = _prob
    pis = np.array(pis)
    fourway = {
        "score": 1 - np.sum(pis**2),
        "major": major_count,
        "total": df.shape[0],
    }
    # make sure that there are four probability values
    assert len(pis_dict) == 4
    return fourway, pis_dict


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


def pipeline(df, bot=True, interval=30):
    assignor_type = "bot" if bot else "human"
    print(f"Filtered DF for {assignor_type.upper()} assignors")
    bot_df = df[
        df["Ethnicity-Assignor"].str.lower() == "bot"
    ]  # do this to get the bot-assigned issue period
    filtered_df = bot_df if bot else df[df["Ethnicity-Assignor"].str.lower() != "bot"]
    print(filtered_df.shape)
    print(f"{len(np.unique(filtered_df['Assignor']))} unique assignors")
    # add a column with dates as datetime objects
    try:
        filtered_df["Date Object"] = pd.to_datetime(filtered_df["Date"])
        bot_df["Date Object"] = pd.to_datetime(bot_df["Date"])
    except:
        filtered_df = fix_df_dates(filtered_df)
        bot_df = fix_df_dates(bot_df)
    filtered_df = filtered_df.sort_values(by="Date Object")
    filtered_df.head()
    bot_df = bot_df.sort_values(by="Date Object")

    print(f"Splitting DF in interval of {interval} days")
    split_dfs = []
    start_date = bot_df.iloc[0]["Date Object"]
    max_date = bot_df.iloc[-1]["Date Object"]

    day_diff = max_date - start_date
    print(day_diff)
    print("*" * 30)

    print(f"START DATE: {start_date} | END DATE: {max_date}")

    while start_date < max_date:
        end_date = start_date + timedelta(days=interval)
        split_dfs.append(
            filtered_df[
                (filtered_df["Date Object"] >= start_date)
                & (filtered_df["Date Object"] < end_date)
            ]
        )
        start_date = end_date
    print(f"{len(split_dfs)} split DFs")

    print(f"Computing BLAU scores for {assignor_type.upper()}")
    # compute BLAU scores for the split dfs
    scores = []
    prob_dict = collections.defaultdict(list)
    for _df in split_dfs:
        ret_dict, _prob_dict = compute_blau_df(_df, binary_classification=False)
        # print(f'{score:.4f}', end=', ')
        for k, v in _prob_dict.items():
            prob_dict[k].append(v)
        scores.append(ret_dict)
    return scores, prob_dict


def plot_probs(probs, assignor_type, org, x_multiplier):
    print("Plotting probabilities scores")
    sns.set(style="whitegrid")
    plt.rcParams["figure.dpi"] = 300
    plt.rcParams["savefig.dpi"] = 300
    plt.rcParams["lines.markersize"] = 2
    plt.rcParams["text.usetex"] = True
    COLORS = [k for k in mcd.TABLEAU_COLORS.keys()]
    PATTERNS = ["d:", "x-", "*-", "o-", ".-", "d--", "x-.", ",--", "*:"]
    plt.figure(figsize=(6, 4))

    first_key = list(probs.keys())[0]
    x = list(range(len(probs[first_key])))
    x = [i * x_multiplier for i in x]
    i = 0
    plt.ylim(-0.015, 1.015)
    for eth, scores in probs.items():
        y = smooth_line([float(x) if x else x for x in scores], sigma=0)
        print(org.upper(), eth, y)
        plt.plot(
            x,
            y,
            marker=PATTERNS[i][0],
            linestyle=PATTERNS[i][1:],
            # alpha=0.7,
            color=COLORS[i],
            lw=1.0,
            label=eth,
        )
        i += 1
    plt.legend(fontsize=14)
    plt.xlabel("Month", fontsize=17)
    plt.ylabel("Probability", fontsize=17)
    plt.title(r"Probability$_{%s} \ | $ %s" % (assignor_type, org), fontsize=17)
    plt.tight_layout()
    plt.savefig(f"./figures/rq3/{assignor_type}_probs_{org}_4way.png")
    plt.savefig(f"./figures/rq3/{assignor_type}_probs_{org}_4way.pdf")
    plt.close()


def smooth_line(y, sigma=2):
    """Smooth the line defined by `y` with a gaussian filter."""
    if sigma == 0:
        return y
    if None not in y:
        return gaussian_filter1d(y, sigma=sigma)
    # remove Nones here and add again after applying the filter
    none_pos, y_new = [], []
    for idx, _ in enumerate(y):
        if _ is None:
            none_pos.append(idx)
        else:
            y_new.append(_)
    y_new = gaussian_filter1d(y_new, sigma)

    # reinsert the None values
    for pos in none_pos:
        first_half = y_new[:pos]
        second_half = y_new[pos:]
        y_new = np.concatenate((first_half, [None], second_half), -1)
    return y_new


def plot(bot_dicts, human_dicts, org, x_multiplier):
    def extract_data(dicts):
        blau_scores, hist_data = [], []
        for _, d in enumerate(dicts):
            blau_scores.append(d["score"])
            hist_data.append((d["major"], d["total"]))
        return blau_scores, hist_data

    bot_blau_scores, bot_hist_data = extract_data(bot_dicts)
    human_blau_scores, human_hist_data = extract_data(human_dicts)
    race_scores = []

    # plot histograms
    assert len(bot_hist_data) == len(human_hist_data)
    plot_histograms(bot_hist_data, human_hist_data, org, x_multiplier)

    assert len(bot_blau_scores) == len(human_blau_scores)
    plot_scores(bot_blau_scores, human_blau_scores, org, x_multiplier)


def plot_histograms(bot_data, human_data, org, x_multiplier=1):
    print("Plotting histogram")
    sns.set(style="whitegrid")
    plt.rcParams["figure.dpi"] = 300
    plt.rcParams["savefig.dpi"] = 300
    plt.rcParams["lines.markersize"] = 2
    plt.rcParams["text.usetex"] = True
    COLORS = [k for k in mcd.TABLEAU_COLORS.keys()]
    PATTERNS = ["d:", "x-", "*-", "o-", ".-", "d--", "x-.", ",--", "*:"]
    plt.figure(figsize=(6, 4))
    plt.title(f"Human v/s Bot Histogram")
    x = list(range(len(bot_data)))
    x = np.array([i * x_multiplier for i in x])
    i = 0
    for assignor_type, counts in {"human": human_data, "bot": bot_data}.items():
        print(org.upper(), assignor_type, counts)
        offset = -0.6 if assignor_type == "human" else 0.2
        y = [c[0] for c in counts]
        y_total = [c[1] for c in counts]
        plt.bar(x + offset, y_total, color=COLORS[i], label=assignor_type + "-minor")
        plt.bar(x + offset, y, color=COLORS[i + 1], label=assignor_type + "-major")
        i += 2
    plt.legend()
    plt.xlabel("Month", fontsize=17)
    plt.ylabel("\#major and minor issues", fontsize=17)
    plt.title(f"Human v/s Bot issue dist. | {org}")
    plt.tight_layout()
    plt.savefig(f"./figures/rq3/issue_hist_{org}_4way.png")
    plt.close()


def plot_scores(bot_scores, human_scores, org, x_multiplier=1):
    print("Plotting BLAU scores")
    sns.set(style="whitegrid")
    plt.rcParams["figure.dpi"] = 300
    plt.rcParams["savefig.dpi"] = 300
    plt.rcParams["lines.markersize"] = 2
    plt.rcParams["text.usetex"] = True

    COLORS = [k for k in mcd.TABLEAU_COLORS.keys()]
    PATTERNS = ["d:", "x-", "*-", "o-", ".-", "d--", "x-.", ",--", "*:"]
    plt.figure(figsize=(6, 4))
    x = list(range(len(bot_scores)))
    x = [i * x_multiplier for i in x]
    i = 0
    scores_dict = {"human": human_scores, "bot": bot_scores}

    plt.ylim(-0.015, 1.015)
    for assignor_type, scores in scores_dict.items():
        y = smooth_line([float(x) if x else x for x in scores], sigma=0)
        print(org.upper(), assignor_type, y)
        plt.plot(
            x,
            y,
            marker=PATTERNS[i][0],
            linestyle=PATTERNS[i][1:],
            # alpha=0.7,
            color=COLORS[i],
            lw=1.0,
            label=assignor_type,
        )
        i += 1
    plt.axhline(y=0.75, color="black", label="y=0.75", lw=1.0)
    plt.legend(fontsize=15)
    plt.xlabel("Month", fontsize=17)
    plt.ylabel("BLAU Score", fontsize=17)
    plt.title(
        r"BLAU$_{human}$ v/s BLAU$_{bot} \ | $ " + f"{org}",
        fontsize=17,
    )
    plt.tight_layout()
    plt.savefig(f"./figures/rq3/blau_scores_{org}_4way.png")
    plt.savefig(f"./figures/rq3/blau_scores_{org}_4way.pdf")
    plt.close()


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
    display.display(df.head())

    for org in df["org"].unique():
        print(f"{org.upper()}")
        _df = df[df["org"] == org]
        interval = 60
        x_multiplier = interval // 30
        bot_dicts, bot_probs = pipeline(_df, bot=True, interval=interval)
        print("\n====")
        human_dicts, human_probs = pipeline(_df, bot=False, interval=interval)
        # plot BLAU scores and probabilities
        plot(bot_dicts, human_dicts, org, x_multiplier)
        plot_probs(human_probs, "human", org, x_multiplier)
        plot_probs(bot_probs, "bot", org, x_multiplier)


if __name__ == "__main__":
    main()
