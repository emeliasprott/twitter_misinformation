# Deplatforming and Misinformation: Efficacy of Twitter's January 2021 User Suspensions

## Introduction

This analysis builds on the study, *Post-January 6th deplatforming reduced the reach of misinformation on Twitter* (McCabe et al. 2024), which examined Twitter's large-scale account suspensions following the January 6th, 2021 insurrection. Noting the significant role misinformation and conspiracy theories played, Twitter and other social media platforms implemented sweeping account suspensions to prevent their spread.

The goal of this project is to replicate and extend the findings of McCabe et al. using an anonymized replication dataset provided by the authors. I aim to further susbtantiate their conclusions to gauge the actual efficacy of Twitter's post-January 6th user suspensions. By doing so, I hope to improve the understanding of how social media platforms can mitigate the spread of misinformation and its impact on public discourse.

### Background and Context

The original research team compiled a pool of over 500,000 active Twitter users that could be cross-verified with a voter registration database, then assembled a dataset containing all of these users' activity between 2019 and 2021.

Using multiple pre-curated lists of websites known to be sources of misinformation, the researchers focused specifically on tweets and retweets containing links to these websites.

Notes
- Aggregated Twitter data from late 2019 through 2021, focusing on URLs identified as misinformation
- Classified users into categories based on their activity levels and misinformation spread
- Used Difference-in-Differences (DiD) to measure the causal effect of deploatforming on misinformation spread


```python
import pandas as pd
import numpy as np
import seaborn as sns
import re
import json
import warnings
import statsmodels.api as sm
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import matplotlib_venn as venn
import matplotlib.dates as mdates
import base64
import io, requests
from IPython.display import Image, display
from PIL import Image as im
from datetime import datetime, timedelta, date


warnings.filterwarnings(action="ignore", category=pd.errors.SettingWithCopyWarning)
warnings.filterwarnings(action="ignore", category=FutureWarning)
warnings.filterwarnings(action="ignore", category=RuntimeWarning)
warnings.filterwarnings(action="ignore", category=DeprecationWarning)
```

## Data Preparation and Organization


```python
# load data
mccabe = mccabe = pd.read_csv(
    "mccabe-public-data.csv", on_bad_lines="skip"
).reset_index(names=["ID"])
mccabe["group"] = mccabe["group"]
mccabe["date"] = pd.to_datetime(mccabe["date"], format="%Y-%m-%d")
```

### De-aggregating the data

To protect users' privacy, the replication data is available in an anonymized, aggregated format. The researchers divided the users into overlapping categories based on their activity levels and misinformation spread, then supplied the observed (mostly) daily counts for each group.

Select Groups:
| Name | Group | Description |
|------|-------|-------------|
| FNS | misinformation sharers | users who share at least 1 URL with misinformation |
| DU | suspended users | users removed between January 6th and January 12th |
| HA | high activity | users who sent at least 3,200 tweets during a six-week collection interval between 2018 and April 2020 |
| MA | medium activity | the most active 500,000 users who didn't meet the high activity threshold |
| LA | low activity | all users who didn't meet the high or medium activity thresholds |
| A | Trump-only followers | non-suspended misinformation sharers who follow Trump but no other deplatformed users |
| B | deplatformed followers | non-suspended misinformation sharers who follow at least one deplatformed user (can include Trump) |
| D | 4+ deplatformed followers | non-suspended misinformation sharers who follow at least four deplatformed users (can include Trump) |
| F | not deplatformed followers | non-suspended misinformation sharers who do not follow any deplatformed users |

I used some probability rules to reorganize the data into a more usable (mutually exclusive) format.


```python
fig, ax = plt.subplots(1, 3, figsize=(18, 10))

v = venn.venn2(subsets=(4, 6, 3), set_labels=(' Group A ', ' Group B '), set_colors=('yellow', 'blue'), alpha=0.5, ax=ax[0])
v.get_label_by_id('10').set_text('')
v.get_label_by_id('01').set_text('')
v.get_label_by_id('11').set_text('')
v.get_patch_by_id('11').set_color('green')

w = venn.venn2(subsets=(4, 6, 3), set_labels=(' Group D ', ' Group B '), set_colors=('red', 'blue'), alpha=0.5, ax=ax[1])
w.get_label_by_id('10').set_text('')
w.get_label_by_id('01').set_text('')
w.get_label_by_id('11').set_text('')
w.get_patch_by_id('11').set_color('purple')

x = venn.venn2(subsets=(4, 4, 0), set_labels=(' Group A ', ' Group D '), set_colors=('yellow', 'red'), alpha=0.5, ax=ax[2])
x.get_label_by_id('10').set_text('')
x.get_label_by_id('01').set_text('')

fig.suptitle('Group Relationships', fontsize=16)
plt.tight_layout()
plt.show()
```



![png](cleaned-project-2_files/cleaned-project-2_12_0.png)




```python
# one entry per group per day
numbers = [col for col in mccabe.columns if col not in ["ID", "date", "stat", "group"]]

mccabe_full = mccabe.groupby(["date", "group", "stat"])[numbers].sum().reset_index()
```

I built several functions to handle the disaggregation of the data. In addition to comparing and subtracting the subsets, I also added empty rows to the dataframes to make sure that the dataframes had the same number of rows. I will be aggregating the data later on, so this will not imapct the inteegrity of the data. It's also important to note that the disaggregation only breaks up the data into mutually exclusive categories, not into individual, user-level observations.


```python
# add empty rows when subset has no activity (assume this is observed in the data)
def add_missing_level(set, set_name):
    """
    Adds missing activity level rows to a DataFrame when a subset has no activity.

    Args:
        set (pandas.DataFrame): The DataFrame to add missing rows to.
        set_name (str): The name of the grouping for the DataFrame.

    Returns:
        pandas.DataFrame: The updated DataFrame with missing rows added.
    """
    if len(set.loc[:, "level"].unique()) < 3:
        level = [
            l for l in ["ha", "ma", "la"] if l not in set.loc[:, "level"].unique()
        ][0]
        empty_row = pd.DataFrame(
            {col: [0 if col != "level" else level] for col in set.columns}
        )
        empty_row["level"] = level
        empty_row["stat"] = "total"
        empty_row["grouping"] = set_name
        if level == "la":
            set = pd.concat([set, empty_row], ignore_index=True)
        if level == "ha":
            set = pd.concat([empty_row, set], ignore_index=True)
    return set


def preprocessing(df_raw, date):
    """
    Preprocesses a raw DataFrame by filtering to a specific date, creating mutually exclusive groups, and processing subsets by activity level.

    Args:
        df_raw (pandas.DataFrame): The raw DataFrame to preprocess.
        date (str): The date to filter the DataFrame to.

    Returns:
        tuple:
            ha (pandas.DataFrame): The 'ha' group DataFrame.
            ma (pandas.DataFrame): The 'ma' group DataFrame.
            la (pandas.DataFrame): The 'la' group DataFrame.
            processed_groups (dict): A dictionary of processed group DataFrames.
            sub_total (pandas.DataFrame): The total subset DataFrame.
            groups (list): A list of group names.
    """
    # data for the given date
    df = df_raw.loc[df_raw["date"] == date].reset_index(drop=True)

    # start with mutually exclusive groups
    ha, ma, la = [
        df.loc[(df["group"] == group) & (df["stat"] == "total")]
        for group in ["ha", "ma", "la"]
    ]

    # subset by activity level
    su = df.loc[df["group"].str.contains(r"\_[hml]a")]
    groupings = (
        su.copy()
        .loc[:, "group"]
        # split the group column into two columns
        .str.split("_", expand=True)
        .rename(columns={0: "grouping", 1: "level"})
        .apply(lambda x: x.str.strip())
    )
    # turn into separate columns
    sub = pd.concat(
        [su.drop(columns=["grouping", "level", "group"], errors="ignore"), groupings],
        axis=1,
    )

    # filter to sum only
    sub_total = sub.copy().loc[sub["stat"] == "total"]

    def process_group(sub_total, group_name):
        """
        Checking for/adding 'ha', 'ma', 'la' levels.
        """
        if "grouping" not in sub_total.columns:
            sub_total["grouping"] = sub_total["group"]
        group = sub_total.loc[sub_total["grouping"] == group_name]
        group = add_missing_level(group, group_name)
        if group_name == "A":
            group["date"] = date
        return group

    # A, D, F, and nfns groups
    groups = ["A", "D", "F", "nfns"]
    processed_groups = {}
    for group in groups:
        result = process_group(sub_total, group)
        if isinstance(result, tuple):
            processed_groups[group] = list(result)
        else:
            processed_groups[group] = [result]

    return ha, ma, la, processed_groups, sub_total, groups


def process_suspended(suspended):
    """Processes the suspended data by creating a common DataFrame with the 'total', 'suspended', and 'level' columns.

    Args:
        suspended (pandas.DataFrame): The suspended DataFrame.

    Returns:
        pandas.DataFrame: The processed suspended DataFrame with the common columns added.
    """

    suspended_common = pd.DataFrame(
        {
            "stat": "total",
            "grouping": "suspended",
            "level": ["ha", "ma", "la"],
        }
    )
    suspended = suspended.reset_index().join(suspended_common, rsuffix="_common")
    return suspended


def pull_B(sub_total, processed_groups):
    """
    Pulls the 'B' group from the sub_total DataFrame and calculates the difference between 'B' and the union of 'A' and 'D' groups.

    Args:
        sub_total (pandas.DataFrame): The total subset DataFrame.
        processed_groups (dict): A dictionary of processed group DataFrames.

    Returns:
        tuple:
            numeric_columns (pandas.Index): The numeric columns in the sub_total DataFrame.
            B (pandas.DataFrame): The 'B' group DataFrame.
    """
    numeric_columns = sub_total.select_dtypes(include=["number"]).columns
    B_union = sub_total[sub_total["grouping"] == "B"]
    B_union = add_missing_level(B_union, "B")

    A_numeric = processed_groups["A"][0].set_index("level")[numeric_columns]
    D_numeric = processed_groups["D"][0].set_index("level")[numeric_columns]

    A_union_D = A_numeric.add(
        D_numeric.reindex(A_numeric.index, fill_value=0), fill_value=0
    )

    B = B_union.set_index("level")[numeric_columns].sub(A_union_D, fill_value=0)

    B_common = pd.DataFrame(
        {"stat": "total", "grouping": "B", "level": ["ha", "ma", "la"]}
    )

    B = B.reset_index().join(B_common, rsuffix="_common")

    return numeric_columns, B


def impute_NDU(B, ha, ma, la, processed_groups, groups):
    """
    Imputes the non-suspended users; values by calculating the difference between the sum of all non-suspended groups and the sum of the 'ha', 'ma', and 'la' groups.

    Args:
        B (pandas.DataFrame): The 'B' group DataFrame.
        ha (pandas.DataFrame): The 'ha' group DataFrame.
        ma (pandas.DataFrame): The 'ma' group DataFrame.
        la (pandas.DataFrame): The 'la' group DataFrame.
        processed_groups (dict): A dictionary of processed group DataFrames.
        groups (list): A list of group names.

    Returns:
        tuple:
            all_levels (pandas.DataFrame): A DataFrame containing all activity levels.
            non_suspended (pandas.DataFrame): A DataFrame containing the sum of all non-suspended groups.
    """
    all_levels = (
        pd.concat([ha, ma, la]).rename(columns={"group": "level"}).set_index("level")
    )
    non_suspended = (
        pd.concat([processed_groups[group][0] for group in groups] + [B])
        .groupby("level")
        .sum(numeric_only=True)
    )
    non_suspended["stat"] = "total"
    return all_levels, non_suspended

def recombine(date, processed_groups, B, suspended):
    """
    Recombines the processed data groups and suspended data into a single DataFrame.

    Args:
        date (datetime): The date for which the data is being processed.
        processed_groups (dict): A dictionary containing the processed data groups.
        B (pd.DataFrame): The B DataFrame.
        suspended (pd.DataFrame): The suspended DataFrame.

    Returns:
        pd.DataFrame: The final DataFrame containing the recombined data.
    """
    dfs_to_concat = []
    for group in processed_groups.values():
        for item in group:
            if isinstance(item, pd.DataFrame):
                dfs_to_concat.append(item)

    if not isinstance(B, pd.DataFrame):
        B = pd.DataFrame(B)

    if date < datetime(2021, 1, 12):
        suspended_p = process_suspended(suspended)
        dfs_to_concat.append(suspended_p)

    exclusive_groups = pd.concat(dfs_to_concat, ignore_index=True)

    return exclusive_groups


def aggregation_func(df_raw, date):
    ha, ma, la, processed_groups, sub_total, groups = preprocessing(df_raw, date)

    numeric_columns, B = pull_B(sub_total, processed_groups)

    all_levels, non_suspended = impute_NDU(B, ha, ma, la, processed_groups, groups)

    suspended = all_levels[numeric_columns].sub(
        non_suspended[numeric_columns], fill_value=0
    )
    suspended["date"] = date

    final = recombine(date, processed_groups, B, suspended)

    return final
```


```python
mut_exclusive_groups = []

for day in mccabe_full["date"].unique():
    mut_exclusive_groups.append(aggregation_func(mccabe_full, day))

total = pd.concat(mut_exclusive_groups).reset_index(drop=True)

# Create a new column 'subsets' by combining 'grouping' and 'level'
total.loc[:, "subsets"] = total["grouping"] + "_" + total["level"]
total = total.drop(columns=["grouping", "level", "stat", "level_common"]).reset_index(
    drop=True
)
# total.to_csv("total.csv") checkpoint
```


```python
total = pd.read_csv("total.csv").drop(columns=["Unnamed: 0"])
```


```python
total['date'] = pd.to_datetime(total['date'].apply(lambda x: datetime.strptime(x, "%Y-%m-%d").date()))
total['subset_group'] = total['subsets'].apply(lambda x: x.split('_')[0])
total['subset_activity'] = total['subsets'].apply(lambda x: x.split('_')[1])
```

Data collection was an inexhaustive process, so it's important to verify the sudden changes in each groups' behavior. Some changes can be explained by parallel changes in other groups; the number of low activity users sharply declines around July 2020, but at the same time the number of high activity users sharply increases.


```python
fig, ax = plt.subplots(3, 3, figsize=(15, 10), sharex=True)
for L in total['subset_activity'].unique():
    if L == 'ha':
        i = 0
        title = 'High'
    elif L == 'ma':
        i = 1
        title = 'Moderate'
    else:
        i = 2
        title = 'Low'
    d = total.loc[total['subset_activity'] == L]
    sns.lineplot(d, x='date', y='nusers', hue='subset_group', palette='Set2', ax=ax[0, i])
    sns.lineplot(d, x='date', y='n', hue='subset_group', palette='Set2', ax=ax[1, i])
    sns.lineplot(d, x='date', y='fake_merged', hue='subset_group', palette='Set2', ax=ax[2, i])

fig.text(-0.01, 0.75, 'Number of Users', rotation=90, fontsize=14)
fig.text(-0.01, 0.475, 'Total Tweets', rotation=90, fontsize=14)
fig.text(-0.01, 0.15, 'Fake Tweets', rotation=90, fontsize=14)

fig.text(0.15, 1.01, 'High Activity', fontsize=14)
fig.text(0.475, 1.01, 'Moderate Activity', fontsize=14)
fig.text(0.825, 1.01, 'Low Activity', fontsize=14)

handles, labels = ax[0, 0].get_legend_handles_labels()
for h in handles:
    h.set_linewidth(3)
fig.legend(handles, labels, loc='upper right', ncol=1, bbox_to_anchor=(1.1, 1))

for a in ax.flatten():
    a.xaxis.set_major_locator(mdates.AutoDateLocator())
    a.xaxis.set_major_formatter(mdates.ConciseDateFormatter(a.xaxis.get_major_locator()))
    plt.setp(a.xaxis.get_majorticklabels(), rotation=45, ha='right')
    a.get_legend().remove()
    a.yaxis.set_label_text('')

plt.tight_layout()
plt.show()
```



![png](cleaned-project-2_files/cleaned-project-2_20_0.png)




```python
features = ['fake_merged_initiation', 'fake_merged_rt', 'not_fake_conservative', 'not_fake_liberal', 'not_fake_shopping', 'not_fake_sports', 'n', 'nusers', 'subset_group', 'subset_activity', 'date']

df = total.copy().loc[:, features].pivot_table(index='date', columns=['subset_group', 'subset_activity'], values=['fake_merged_initiation', 'fake_merged_rt', 'not_fake_conservative', 'not_fake_liberal', 'not_fake_shopping', 'not_fake_sports', 'n', 'nusers'], fill_value=0).asfreq('D').fillna(0)
df.columns = ['_'.join(col).strip() for col in df.columns.values]

treatment_start = '2021-01-12'
treatment_end = '2021-01-19'

pretreatment_df = df.loc[:treatment_start]
posttreatment_df = df.loc[treatment_end:]
```


```python
from adtk.detector import MinClusterDetector
from adtk.pipe import Pipeline
from adtk.data import validate_series
from adtk.transformer import PcaProjection
from sklearn.cluster import KMeans
from sklearn.linear_model import LinearRegression

min_cluster_detector = MinClusterDetector(KMeans(n_clusters=4))
steps = [
    ("projection", PcaProjection(k=2)),
    ("detector", min_cluster_detector)
]
pipeline = Pipeline(steps)
pre_treated = validate_series(pretreatment_df)
pre_anomalies = pipeline.fit_detect(pre_treated).reset_index().rename(columns={0: "anomaly"})
```


```python
pre_outlier_data = total.loc[total['date'] < '2021-01-12'].join(pre_anomalies.set_index('date'), on='date')
plt.style.use('seaborn-v0_8-whitegrid')

fig, ax = plt.subplots(1, 3, figsize=(15, 10))

for L in ['ha', 'ma', 'la']:
    if L == 'ha':
        i = 0
        title = 'High'
    elif L == 'ma':
        i = 1
        title = 'Moderate'
    else:
        i = 2
        title = 'Low'
    name = f"{title} Activity Users"
    sns.lineplot(x='date', y='nusers', data=pre_outlier_data.loc[pre_outlier_data['subset_activity'] == L], ax=ax[i], hue='subset_group', palette='Set2')
    sns.scatterplot(x='date', y='nusers', data=pre_outlier_data.loc[(pre_outlier_data['subset_activity'] == L) & (pre_outlier_data['anomaly'] == True)], ax=ax[i], color='red', legend=False, size=800)
    ax[i].set_title(name)
fig.suptitle('Daily # Users', fontsize=16)

handles, labels = ax[0].get_legend_handles_labels()
fig.legend(handles, labels, loc='upper right', ncol=1, bbox_to_anchor=(1.0, 0.95))
for a in ax:
    a.xaxis.set_major_locator(mdates.MonthLocator())
    a.xaxis.set_major_formatter(mdates.ConciseDateFormatter(a.xaxis.get_major_locator()))
    plt.setp(a.xaxis.get_majorticklabels(), rotation=45, ha='right')
    a.get_legend().remove()
    a.set_ylim(0)
    a.yaxis.set_label_text('')
```



![png](cleaned-project-2_files/cleaned-project-2_23_0.png)



The main outliers seem to be the earliest observations, as well as the general downward trend in November 2020.

## Modeling


```python
df = (
    pd.read_csv("total.csv", parse_dates=["date"])
      .assign(
          post = lambda d: (d["date"] >= "2021-01-06").astype(int),
          t_idx = lambda d: d.groupby("subsets").cumcount(),
          logN = lambda d: np.log(d["n"].clip(lower=1))
      )
)

s_idx = pd.Series(df['subsets'].str.split('_').str[0].values).astype('category').cat.codes
df['s_idx'] = s_idx
a_idx = pd.Series(df['subsets'].str.split('_').str[1].values).astype('category').cat.codes
df['a_idx'] = a_idx
sa_idx = df.groupby(["s_idx","a_idx"]).ngroup()
df['sa_idx'] = sa_idx
```


```python
sns.relplot(data=df.reset_index(), x="date", y="fake_merged", hue="a_idx", col="s_idx", col_wrap=2, kind="line", height=5, aspect=1.2, facet_kws={"sharex": True, "sharey": True})
plt.axvline(pd.to_datetime("2021-01-06"), ls="--")
plt.show()
```



![png](cleaned-project-2_files/cleaned-project-2_27_0.png)



### Model Notation

| Symbol | Description |
| ----------------------------- | ------------------------------ |
| $g = 1,\dots,G$ | aggregated **user‑type groups** (cohorts) |
| $t = 1,\dots,T$ | calendar days |
| $Y_{gt}$ | misinformation‑tweet **count** for group $g$ on day $t$ |
| $N_{gt}$ | **exposure** (total tweets **or** active accounts) for the same cell |
| $G_g$ | first suspension day in group $g$ (∞ if never) |
| $D_{gt}= \mathbf 1[t\ge G_g]$ | post‑suspension indicator |
| $k=t-G_g$ | relative‑day index (negative = leads, positive = lags) |
| $H_g$ | **group‑level attributes** (e.g., follower‑size bin, ideology score) |
| $K_+$, $K_-$ | max lags / leads kept (e.g., $K_+=30,\;K_- =15$) |


```python
from patsy import dmatrix
import statsmodels.api as sm
import linearmodels.panel as plm
from econml.dml import LinearDML
from sklearn.ensemble import GradientBoostingRegressor, GradientBoostingClassifier
from econml.dr import DRLearner
```


```python
df["g"] = df["subsets"].astype("category").cat.codes
df["time"] = df["date"].rank(method="dense").astype(int)
df["rate_fake"] = 1e4 * df["fake_merged"] / df["n"]
df['rate_fake'] = df['rate_fake'].replace([np.inf, -np.inf], np.nan).fillna(0)
df["misinfo"] = df["fake_merged"]
df["clean"] = df["not_fake"]
df['post'] = (df['date'] >= '2021-01-06').astype(int)
df["pre_time"] = df["time"] * (df["post"]==0)

panel = df.copy()
```


```python
Y = df["rate_fake"].values
T = df['post'].values
X = pd.get_dummies(df[["g", "time"]])

# ② fit DML
dml = LinearDML(model_y=GradientBoostingRegressor(max_depth=3),
                model_t=GradientBoostingClassifier(max_depth=3),
                discrete_treatment=True)
dml.fit(Y, T, X=X)

# ③ effect for each row, then aggregate
df["tau_hat"] = dml.effect(X)
att_gt = df.groupby(["g", "time"])["tau_hat"].mean().reset_index()
```


```python
att_gt["rel_day"] = df.loc[att_gt.index, "t_idx"].values
att_gt['rel_day'] = att_gt['rel_day'] - 399
att_t = att_gt.loc[att_gt['rel_day'].between(-180, 180)].groupby("rel_day")["tau_hat"]

mu = att_t.mean()
se = att_t.std(ddof=1) / np.sqrt(att_t.count())
ci = 1.96 * se
fig, ax = plt.subplots(figsize=(8,5))
ax.plot(mu.index, mu.values, lw=2)
ax.fill_between(mu.index, mu-ci, mu+ci, alpha=.2)
ax.axvline(0, ls="--")
ax.set_xlabel("Days since suspension wave")
ax.set_ylabel("Average causal effect on misinformation rate\n(percentage-point change)")
ax.set_title("Event-study: Twitter January 2021 suspensions")
ax.grid(True)
plt.tight_layout()
plt.show()
```



![png](cleaned-project-2_files/cleaned-project-2_32_0.png)




```python
att_gt['g_'] = att_gt['g'].map({row['g']: row['subsets'] for _, row in df[['subsets', 'g']].drop_duplicates().iterrows()})
att_gt['time2'] = att_gt['time'] - 399
heat = att_gt.pivot_table(index="g_", columns="time2", values="tau_hat", aggfunc='mean').iloc[:, 330:-2]
```


```python
fig, ax = plt.subplots(figsize=(10,6))
sns.heatmap(heat, ax=ax, cmap="coolwarm", center=-250000)

ax.set_xlabel("Days since suspension")
ax.set_ylabel("Cohort")
ax.set_title("Cohort-by-time causal effects")
plt.tight_layout()
plt.show()
```



![png](cleaned-project-2_files/cleaned-project-2_34_0.png)




```python
meta = df[['g','subsets']].drop_duplicates().set_index('g')
meta['activity'] = meta['subsets'].str.extract(r'_(ha|ma|la)$')[0]
meta['followers']= meta['subsets'].str.extract(r'^(A|B|D|F|nfns|suspended)')[0]
sns.set_style('whitegrid')

def analyze_suspension_impact(b, ax):
    errorbars = []
    labels = ['Up to day 14', 'Up to day 28', 'Up to day 42']
    for right, color, label in zip(b, ['blue', 'orange', 'green'], labels):
        att_post = att_gt[att_gt['time'].between(399,right)]
        avg_g = att_post.groupby('g')['tau_hat'].mean()
        stats = meta.join(avg_g).reset_index(drop=True)
        group_stats = stats.groupby(['followers','activity'])['tau_hat'].agg(['mean','count','std']).reset_index()
        group_stats['se'] = group_stats['std'] / np.sqrt(group_stats['count'])
        group_stats['ci'] = 1.96 * group_stats['se']

        ypos = np.arange(len(group_stats))
        eb = ax.errorbar(group_stats['mean'], ypos, xerr=group_stats['ci'], fmt='o', capsize=4, color=color, label=label)
        errorbars.append(eb)
    ax.axvline(0, ls='--', color='gray', zorder=-1)
    ax.set_yticks(ypos)
    ax.set_yticklabels(group_stats['followers'] + " x " + group_stats['activity'])
    ax.set_xlabel("Average treatment effect")
    ax.set_title("Suspension impact on average treatment effect")
    ax.legend(handles=errorbars, loc='best')

fig, ax = plt.subplots(figsize=(16, 8))
analyze_suspension_impact([413, 428, 443], ax)
plt.show()

```



![png](cleaned-project-2_files/cleaned-project-2_35_0.png)



## Sources
McCabe, S.D., Ferrari, D., Green, J. et al. Post-January 6th deplatforming reduced the reach of misinformation on Twitter. Nature 630, 132–140 (2024). https://doi.org/10.1038/s41586-024-07524-8
