---
layout: posts
title:  "Analyzing historical NFL over-under data to identify trends"
date:   2021-09-18 18:00:00 -0400
categories: sports
tags: sports-betting nfl
excerpt: "Comparing at the over/under line from 2010 with weather, team ratings, weeks, etc"
---

# Introduction

While getting historical betting data can be prohibitively expensive, I recently found a great NFL betting dataset on Kaggle. It has every NFL game going back several decades, as well as their point spread and over-under line. It also has weather information for some of the years although I haven't yet vetted this. However unlikely, I decided it would be a good opportunity to see if there were any trends we could pick out of games that went above or below the over-under (20+ or 30+ points). If I could identify any predictive trends on these types of games, that would allow me to predict future games with a high degree of confidence.

# Ideation

As this was more of an exploration experiment, I didn't come into this with any explicit ideas. Instead, it was more about statistical analysis on the dataset.

# Implementation

Again, I will not go through all the boilerplate <a href="https://github.com/BenPollock/SportsBettingExploration/blob/main/NFL-over-under.ipynb">but the full jupyter notebook is available here.</a>

First, I pulled in the CSV dataset, which has games from 1979 through 2021 (2021 has empty results as it hadn't yet been played).

I dropped the 2021 season because I can't use the data yet, and I also dropped everything before 2010. I may reconsider this in the future, but from my past experimentation and from reading other's experiments it doesn't seem like data that old is very predictive.

```python
global_df = pd.read_csv("nfl_games_and_bets.csv")
pre_2021_df = global_df.drop(global_df[global_df.schedule_season == 2021].index)
recent_df = pre_2021_df.drop(pre_2021_df[pre_2021_df.schedule_season < 2010].index)
```

Then, I generated separate dataframes for my outliers (>20 points different from the O/U spread) and the super outliers (>30 points).

```python
pre_2021_df_with_score = recent_df
pre_2021_df_with_score['total_score'] = pre_2021_df_with_score.apply (lambda row: row.score_home + row.score_away, axis=1)

outliers_df = pre_2021_df_with_score[abs(pre_2021_df_with_score.total_score - pre_2021_df_with_score.over_under_line) > 20]
super_outliers_df = pre_2021_df_with_score[abs(pre_2021_df_with_score.total_score - pre_2021_df_with_score.over_under_line) > 30]
```

Using `.describe()` I can generate some quick statistical looks at each dataset.

**All Games**
```
schedule_season 	score_home 	score_away 	spread_fav 	ou_line 	weather_temp	wind_mph 	humidity 	total_score
count 	2939.000000 	2939.000000 	2939.000000 	2939.000000 	2939.000000 	2501.000000 	2499.000000 	672.000000 	2939.000000
mean 	2015.003403 	23.912215 	21.844505 	-5.290915 	45.321028 	62.834866 	4.591437 	58.436012 	45.756720
std 	3.164428 	10.289965 	9.921303 	3.394564 	4.424480 	15.414497 	4.630540 	18.927473 	13.995296
min 	2010.000000 	0.000000 	0.000000 	-26.500000 	33.000000 	-6.000000 	0.000000 	4.000000 	6.000000
25% 	2012.000000 	17.000000 	15.000000 	-7.000000 	42.500000 	53.000000 	0.000000 	45.000000 	36.000000
50% 	2015.000000 	24.000000 	21.000000 	-4.000000 	45.000000 	70.000000 	4.000000 	57.000000 	45.000000
75% 	2018.000000 	31.000000 	28.000000 	-3.000000 	48.000000 	72.000000 	7.000000 	72.250000 	54.500000
max 	2020.000000 	62.000000 	59.000000 	0.000000 	63.500000 	97.000000 	40.000000 	100.000000 	105.000000
```

**Outliers Only**
```
schedule_season 	score_home 	score_away 	spread_fav 	ou_line 	weather_temp 	wind_mph 	humidity 	total_score
count 	360.000000 	360.000000 	360.000000 	360.000000 	360.000000 	311.000000 	311.000000 	85.000000 	360.000000
mean 	2015.038889 	26.819444 	24.880556 	-5.213889 	45.762500 	61.733119 	4.498392 	61.223529 	51.700000
std 	3.083542 	15.694087 	14.656120 	3.393846 	4.576628 	17.017339 	4.568387 	19.342389 	26.690092
min 	2010.000000 	0.000000 	0.000000 	-20.000000 	33.000000 	-6.000000 	0.000000 	23.000000 	6.000000
25% 	2013.000000 	13.000000 	10.750000 	-7.000000 	42.500000 	52.000000 	0.000000 	47.000000 	23.000000
50% 	2015.000000 	30.000000 	27.000000 	-4.000000 	45.500000 	69.000000 	4.000000 	61.000000 	65.000000
75% 	2018.000000 	39.000000 	37.000000 	-3.000000 	48.500000 	72.000000 	7.000000 	78.000000 	73.000000
max 	2020.000000 	59.000000 	59.000000 	0.000000 	63.500000 	89.000000 	23.000000 	100.000000 	105.000000
```

**'Super' Outliers Only**
```
schedule_season 	score_home 	score_away 	spread_fav 	ou_line 	weather_temp 	wind_mph 	humidity 	total_score
count 	74.000000 	74.000000 	74.000000 	74.000000 	74.000000 	65.000000 	65.000000 	22.000000 	74.000000
mean 	2014.891892 	33.945946 	32.621622 	-4.912162 	45.817568 	61.415385 	3.846154 	52.454545 	66.567568
std 	3.414584 	17.166303 	15.236674 	2.987827 	4.811925 	16.944664 	4.047494 	16.247277 	29.933754
min 	2010.000000 	0.000000 	0.000000 	-15.500000 	36.000000 	18.000000 	0.000000 	32.000000 	6.000000
25% 	2012.000000 	28.750000 	24.500000 	-6.500000 	42.500000 	54.000000 	0.000000 	39.500000 	71.250000
50% 	2015.500000 	39.000000 	35.500000 	-4.000000 	45.500000 	68.000000 	3.000000 	48.500000 	78.500000
75% 	2018.000000 	45.000000 	43.000000 	-3.000000 	48.500000 	72.000000 	7.000000 	55.750000 	85.000000
max 	2020.000000 	56.000000 	59.000000 	-1.000000 	63.500000 	86.000000 	17.000000 	90.000000 	105.000000
```

Unfortunately, there are no discernible trends I can pick out. The value of the OU line, the point spread, and weather were all consistent across the outliers and the main dataset.

For fun, I cut the dataset down to just the last few years and added the Madden Offence/Defence ratings for each team to see if it made a difference. I assumed it would not since that should be baked into the line already, but it’s worth a try.

```python
df_2017_to_2020 = global_df.drop(global_df[global_df.schedule_season == 2021].index)
df_2017_to_2020 = df_2017_to_2020.drop(df_2017_to_2020[df_2017_to_2020.schedule_season < 2017].index)

madden_ratings = pd.read_csv("madden_team_ratings.csv")

df_with_madden = pd.merge(df_2017_to_2020, madden_ratings, how='left', left_on=['schedule_season', 'team_home'], right_on=['Year', 'Team']) \
    .drop(columns=['Team', 'Overall', 'Year']) \
    .rename(columns={'Offense': 'home_off', 'Defense' : 'home_def'})

df_with_madden = pd.merge(df_with_madden, madden_ratings, how='left', left_on=['schedule_season', 'team_away'], right_on=['Year', 'Team']) \
    .drop(columns=['Team', 'Overall', 'Year']) \
    .rename(columns={'Offense': 'away_off', 'Defense' : 'away_def'})

df_with_madden['total_score'] = df_with_madden.apply (lambda row: row.score_home + row.score_away, axis=1)

df_with_madden_outliers_over = df_with_madden[df_with_madden.total_score - df_with_madden.over_under_line > 20]
df_with_madden_outliers_under = df_with_madden[df_with_madden.over_under_line - df_with_madden.total_score > 20]
```
**All Games**
```
schedule_season 	score_home 	score_away 	spread_fav	ou_line 	weather_temp 	wind_mph 	humidity 	home_off 	home_def 	away_off 	away_def 	total_score
count 	1070.000000 	1070.000000 	1070.000000 	1070.000000 	1070.000000 	666.000000 	665.00000 	1.0 	1062.000000 	1062.000000 	1062.000000 	1062.000000 	1070.000000
mean 	2018.502804 	23.790654 	22.557944 	-5.458879 	46.038318 	63.995495 	4.47218 	78.0 	81.293785 	81.629944 	81.298493 	81.515066 	46.348598
std 	1.119389 	10.234076 	10.117916 	3.573522 	4.582415 	15.303987 	5.01800 	NaN 	5.757292 	5.246207 	5.722319 	5.224395 	14.361658
min 	2017.000000 	0.000000 	0.000000 	-21.500000 	35.000000 	10.000000 	0.00000 	78.0 	66.000000 	69.000000 	66.000000 	69.000000 	6.000000
25% 	2018.000000 	17.000000 	16.000000 	-7.000000 	43.000000 	55.000000 	0.00000 	78.0 	78.000000 	78.000000 	78.000000 	78.000000 	37.000000
50% 	2019.000000 	24.000000 	23.000000 	-4.500000 	46.000000 	72.000000 	3.00000 	78.0 	81.000000 	82.000000 	81.000000 	82.000000 	46.000000
75% 	2020.000000 	31.000000 	30.000000 	-3.000000 	49.000000 	72.000000 	8.00000 	78.0 	85.000000 	85.000000 	85.000000 	85.000000 	55.000000
max 	2020.000000 	57.000000 	59.000000 	0.000000 	63.500000 	97.000000 	24.00000 	78.0 	97.000000 	93.000000 	97.000000 	93.000000 	105.000000
```

**Under Outliers with Madden**
```
schedule_season 	score_home 	score_away 	spread_fav 	ou_line 	weather_temp 	wind_mph 	humidity 	home_off 	home_def 	away_off 	away_def 	total_score
count 	62.000000 	62.00000 	62.000000 	62.000000 	62.000000 	42.000000 	42.000000 	0.0 	62.000000 	62.000000 	61.000000 	61.000000 	62.000000
mean 	2018.193548 	10.50000 	11.209677 	-5.088710 	46.725806 	64.047619 	6.285714 	NaN 	81.467742 	81.580645 	82.721311 	82.557377 	21.709677
std 	1.083992 	7.05424 	6.606348 	3.432635 	4.460829 	18.126860 	5.848851 	NaN 	5.553666 	5.271477 	5.320185 	4.934655 	5.781222
min 	2017.000000 	0.00000 	0.000000 	-13.500000 	39.000000 	10.000000 	0.000000 	NaN 	67.000000 	71.000000 	67.000000 	72.000000 	6.000000
25% 	2017.000000 	6.00000 	7.000000 	-7.500000 	42.625000 	50.750000 	0.000000 	NaN 	79.000000 	77.000000 	80.000000 	79.000000 	19.250000
50% 	2018.000000 	10.00000 	10.000000 	-3.500000 	46.500000 	72.000000 	5.000000 	NaN 	81.500000 	81.000000 	83.000000 	83.000000 	21.500000
75% 	2019.000000 	15.75000 	16.000000 	-2.500000 	49.875000 	72.000000 	9.000000 	NaN 	85.000000 	85.000000 	86.000000 	86.000000 	26.000000
max 	2020.000000 	27.00000 	23.000000 	-1.000000 	56.500000 	89.000000 	18.000000 	NaN 	92.000000 	93.000000 	92.000000 	92.000000 	33.000000
```

I was unable to discern any trends with this additional data either.

Shortly after performing this analysis, I read the book *The Everything Guide to Sports Betting* which gave me a different perspective on what to look for. I’ll cover this in my next analysis.