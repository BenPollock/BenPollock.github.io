---
layout: posts
title:  "Identifying profitable statistical trends against the NFL spread"
date:   2021-10-02 10:00:00 -0400
categories: sports
tags: sports-betting nfl stats data
excerpt: "Discovering several profitable trends that consistently produce positive returns yearly"
---

# Ideation

In previous articles I collected and cleaned data, generated auxillary columns, and then ran the data through a gradient boosting regressor. The problem with machine learning is that it is effectively a 'black box' and really should be a last resort after standard statistical analysis is exhausted.

In this article, I'll again clean up data and generate auxillary columns. I'll then run through a series of statistical tests and identify if any of them are profitable. Spoiler: there are a few!

Boilerplate available on <a href="https://github.com/BenPollock/SportsBettingExploration/blob/main/hockey_data_cleanup.ipynb">GitHub</a>

# Implementation

I'll be using the CSV of NFL betting data from Kaggle that I previously used. I'll drop some unneeded columns and map consistency between team names. Additionally, some teams moved over the years so I need to account for their new names.

```python
global_df = pd.read_csv("nfl_games_and_bets.csv")
global_df = global_df.drop(global_df[global_df.schedule_season == 2021].index)
global_df = global_df.drop(columns=['stadium','weather_temperature', 'weather_wind_mph','weather_humidity','weather_detail'])
global_df = global_df.drop(global_df[global_df.schedule_season < 2000].index)

# Account for team moves
old_to_new_team_name = {"San Diego Chargers": "Los Angeles Chargers", "St. Louis Rams": "Los Angeles Rams", \
"Washington Redskins" : "Washington Football Team", "Oakland Raiders": "Las Vegas Raiders"}
global_df = global_df.replace({"team_away": old_to_new_team_name}).replace({"team_home": old_to_new_team_name})

# Maintain consistency between favourite and team name columns
short_form_to_team_name = {"GB": "Green Bay Packers", "HOU": "Houston Texans", "KC": "Kansas City Chiefs", "BUF": "Buffalo Bills", \
 "TEN": "Tennessee Titans", "NO": "New Orleans Saints", "SEA": "Seattle Seahawks", "MIN": "Minnesota Vikings", \
 "TB": "Tampa Bay Buccaneers", "LVR": "Las Vegas Raiders", "BAL": "Baltimore Ravens", "LAC": "Los Angeles Chargers", \
 "IND": "Indianapolis Colts", "DET": "Detroit Lions", "CLE": "Cleveland Browns", "JAX": "Jacksonville Jaguars", "MIA": "Miami Dolphins", \
 "ARI": "Arizona Cardinals", "PIT": "Pittsburgh Steelers", "CHI": "Chicago Bears","ATL": "Atlanta Falcons", "CAR": "Carolina Panthers", \
 "LAR": "Los Angeles Rams", "CIN": "Cincinnati Bengals", "DAL": "Dallas Cowboys", "SF": "San Francisco 49ers", "NYG": "New York Giants", \
 "WAS": "Washington Football Team", "DEN": "Denver Broncos", "PHI": "Philadelphia Eagles", "NYJ": "New York Jets", "NE": "New England Patriots"}
team_name_to_short_form = {value: key for key, value in short_form_to_team_name.items()}

global_df = global_df.replace({'team_away': team_name_to_short_form}).replace({"team_home": team_name_to_short_form})
```

With the data cleaned up (there isn't too much cleanup needed, the dataset is otherwise very clean) I can now generate some auxillary columns. Intuitively, some fields that may be interesting are if the teams are intradivisional, as well as their last few games played. First, we can create a map of teams to divisions and apply it to a new column. In this case, we only care for the binary value if the game is intradivisional or not, so we'll temporarily create auxillary columns and then drop them.

```python
team_to_division = {"ARI": "NW", "LAR": "NW", "SF": "NW", "SEA": "NW", "CAR": "NS", "TB": "NS", "NO": "NS", "ATL": "NS", \
 "GB": "NN", "CHI": "NN", "MIN": "NN", "DET": "NN", "WAS": "NE", "DAL": "NE", "PHI": "NE", "NYG": "NE", \
 "TEN": "AS", "HOU": "AS", "IND": "AS", "JAX": "AS", "BUF": "AE", "MIA": "AE", "NE": "AE", "NYJ": "AE", \
 "BAL": "AN", "PIT": "AN", "CLE": "AN", "CIN": "AN", "LVR": "AW", "DEN": "AW", "KC": "AW", "LAC": "AW"}

global_df2 = global_df
global_df2['home_division'] = global_df2.apply(lambda row: team_to_division[row.team_home], axis=1)
global_df2['away_division'] = global_df2.apply(lambda row: team_to_division[row.team_away], axis=1)
global_df2['intra_division'] = global_df2.apply(lambda row: row.home_division == row.away_division, axis=1)
global_df2 = global_df2.drop(columns=['home_division', 'away_division'])
```

Next, I created some point and spread differential columns to make he next calculations easier.

```python
global_df3 = global_df2
global_df3['home_point_diff'] = global_df2.apply(lambda row: row.score_home - row.score_away, axis=1)
global_df3['away_point_diff'] = global_df3.apply(lambda row: row.score_away - row.score_home, axis=1)
global_df3['home_spread'] = global_df3.apply(lambda row: row.spread_favorite * -1 if row.team_favorite_id == row.team_away else row.spread_favorite, axis=1)
```

Next, I looped through the dataset and to get the point differential for the home and away team for the last 1 and 3 games.

```python
team_to_games = {}

for index, row in global_df3.iterrows():

    if row.team_home not in team_to_games:
        team_to_games.update({row.team_home : deque([0,0,0])})

    if row.team_away not in team_to_games:
        team_to_games.update({row.team_away : deque([0,0,0])})

    last_games = team_to_games.get(row.team_home)
    home_last_3 = last_games[0] + last_games[1] + last_games[2]
    home_last_1 = last_games[0]
    last_games.pop()
    last_games.appendleft(row.home_point_diff)

    last_games = team_to_games.get(row.team_away)
    away_last_3 = last_games[0] + last_games[1] + last_games[2]
    away_last_1 = last_games[0]
    last_games.pop()
    last_games.appendleft(row.away_point_diff)

    global_df3.at[index, 'home_last_3'] = home_last_3
    global_df3.at[index, 'away_last_3'] = away_last_3
    global_df3.at[index, 'home_last_1'] = home_last_1
    global_df3.at[index, 'away_last_1'] = away_last_1
```

Our dataset now looks as follows

|       | schedule_date | schedule_season | schedule_week | schedule_playoff | team_home | score_home | score_away | team_away | team_favorite_id | spread_favorite | over_under_line | stadium_neutral | intra_division | home_point_diff | away_point_diff | home_spread | home_last_3 | away_last_3 | home_last_1 | away_last_1 |
|:-----:|:-------------:|:---------------:|:-------------:|:----------------:|:---------:|:----------:|:----------:|:---------:|:----------------:|:---------------:|:---------------:|:---------------:|:--------------:|:---------------:|:---------------:|:-----------:|:-----------:|:-----------:|:-----------:|:-----------:|
|  4853 | 9/3/2000      | 2000            | 1             | False            | ATL       | 36.0       | 28.0       | SF        | ATL              | -6.5            | 46.5            | False           | False          | 8.0             | -8.0            | -6.5        | 0.0         | 0.0         | 0.0         | 0.0         |
|  4854 | 9/3/2000      | 2000            | 1             | False            | BUF       | 16.0       | 13.0       | TEN       | BUF              | -1.0            | 40.0            | False           | False          | 3.0             | -3.0            | -1.0        | 0.0         | 0.0         | 0.0         | 0.0         |
|  4855 | 9/3/2000      | 2000            | 1             | False            | CLE       | 7.0        | 27.0       | JAX       | JAX              | -10.5           | 38.5            | False           | False          | -20.0           | 20.0            | 10.5        | 0.0         | 0.0         | 0.0         | 0.0         |
|  4856 | 9/3/2000      | 2000            | 1             | False            | DAL       | 14.0       | 41.0       | PHI       | DAL              | -6.0            | 39.5            | False           | True           | -27.0           | 27.0            | -6.0        | 0.0         | 0.0         | 0.0         | 0.0         |
|  4857 | 9/3/2000      | 2000            | 1             | False            | GB        | 16.0       | 20.0       | NYJ       | GB               | -2.5            | 44.0            | False           | False          | -4.0            | 4.0             | -2.5        | 0.0         | 0.0         | 0.0         | 0.0         |
|  ...  | ...           | ...             | ...           | ...              | ...       | ...        | ...        | ...       | ...              | ...             | ...             | ...             | ...            | ...             | ...             | ...         | ...         | ...         | ...         | ...         |
| 10441 | 1/17/2021     | 2020            | Division      | True             | KC        | 22.0       | 17.0       | CLE       | KC               | -8.0            | 56.0            | False           | False          | 5.0             | -5.0            | -8.0        | -11.0       | 6.0         | -17.0       | 11.0        |
| 10442 | 1/17/2021     | 2020            | Division      | True             | NO        | 20.0       | 30.0       | TB        | NO               | -2.5            | 53.0            | False           | True           | -10.0           | 10.0            | -2.5        | 57.0        | 65.0        | 12.0        | 8.0         |
| 10443 | 1/24/2021     | 2020            | Conference    | True             | GB        | 26.0       | 31.0       | TB        | GB               | -3.0            | 53.0            | False           | False          | -5.0            | 5.0             | -3.0        | 59.0        | 35.0        | 14.0        | 10.0        |
| 10444 | 1/24/2021     | 2020            | Conference    | True             | KC        | 38.0       | 24.0       | BUF       | KC               | -3.0            | 55.0            | False           | False          | 14.0            | -14.0           | -3.0        | -9.0        | 47.0        | 5.0         | 14.0        |
| 10445 | 2/7/2021      | 2020            | Superbowl     | True             | TB        | 31.0       | 9.0        | KC        | KC               | -3.0            | 54.5            | False           | False          | 22.0            | -22.0           | 3.0         | 23.0        | 2.0         | 5.0         | 14.0        |

Next, I'll create some helper functions for the back-test. We'll start with the statistical analysis function. Besides calculating the record and ending money, the function also determines the statistical significance of the results. Assuming that betting against the spread should provide a 50% win-rate, based on the number of records and the actual mean/deviation, there is enough to determine if the results are significant.

```python
def print_basic_stats(money, won, loss, push, sample_set):
    std_dev = statistics.stdev(sample_set)
    predicted_std_dev = std_dev / math.sqrt(len(sample_set))
    mean = statistics.mean(sample_set)
    # expected mean should be 0.5, if we have 50% win/loss
    z_score = (mean-0.5) / predicted_std_dev
    confidence = st.norm.cdf(abs(z_score))
    win_percent = won / (won+loss) * 100
    print ("Ending money is %d" %money)
    print ("Record: %d-%d-%d" %(won, loss, push))
    print ("Win Percent: %f" %win_percent)
    print ("Confidence %f" %confidence)
```

Next, I create the betting functions and the variable setter. For now, the functions are hardcoded with a betsize of $10 and a starting bankroll of $200. This is an aggressive betting pattern by leveraging 5% of bankroll per unit.

```python
def bet_home(money, won, loss, push, sample_set, year_to_record):
    if row.away_point_diff - row.home_spread > 0:
        money = money - 1000
        loss += 1
        sample_set.append(0)
        year_to_record.update({row.schedule_season: year_to_record.get(row.schedule_season, 0) -1 })
    elif row.away_point_diff - row.home_spread == 0:
        push +=1
    else:
        money = money + 909
        won += 1
        sample_set.append(1)
        year_to_record.update({row.schedule_season: year_to_record.get(row.schedule_season, 0) +1 })
    return money, won, loss, push, sample_set, year_to_record


def bet_away(money, won, loss, push, sample_set, year_to_record):
    if row.away_point_diff - row.home_spread > 0:
        money = money + 909
        won += 1
        sample_set.append(1)
        year_to_record.update({row.schedule_season: year_to_record.get(row.schedule_season, 0) +1 })
    elif row.away_point_diff - row.home_spread == 0:
        push +=1
    else:
        money = money - 1000
        loss += 1
        sample_set.append(0)
        year_to_record.update({row.schedule_season: year_to_record.get(row.schedule_season, 0) -1 })
    return money, won, loss, push, sample_set, year_to_record

def set_vars():
    money = 20000
    won = 0
    loss = 0
    push = 0
    sample_set = []
    year_to_record = {}
    return money, won, loss, push, sample_set, year_to_record
```

With all the information, we can now run some backtests. For this article, I've only posted a couple of the worst and best ones that I've found. However, I tested over a dozen strategies which can be found in the github link.

```python
money, won, loss, push, sample_set, year_to_record = set_vars()

for row in global_df_final.itertuples():

    if (row.away_last_1 <= -14 and row.away_last_1 >=-18 and row.intra_division and row.home_spread < 0): #bet on road underdog 
        money, won, loss, push, sample_set, year_to_record = bet_away(money, won, loss, push, sample_set, year_to_record)

print("\nPicking the road division underdog when they most recently lost between 14 and 18 points")
print_basic_stats(money, won, loss, push, sample_set)
print(year_to_record)

#-------------------------------------

money, won, loss, push, sample_set, year_to_record = set_vars()

for row in global_df_final.itertuples():

    if (row.away_last_1 <= -14 and row.away_last_1 >=-18 and row.home_spread < 0): #bet on road underdog 
        money, won, loss, push, sample_set, year_to_record = bet_away(money, won, loss, push, sample_set, year_to_record)

print("\nPicking the road underdog when they most recently lost between 14 and 18 points")
print_basic_stats(money, won, loss, push, sample_set)
print(year_to_record)
```

```
Picking the road division underdog when they most recently lost between 14 and 18 points
Ending money is 43539
Record: 71-41-4
Win Percent: 63.392857
Confidence 0.998300
{2000: 1, 2001: -1, 2002: 3, 2003: 1, 2004: -1, 2005: -2, 2006: 3, 2007: 0, 2008: 0, 2009: 3, 2010: 4, 2011: 4, 2012: 3, 2013: 0, 2014: 3, 2015: 0, 2016: 0, 2017: 2, 2018: -2, 2019: 4, 2020: 5}

Picking the road underdog when they most recently lost between 14 and 18 points
Ending money is 54166
Record: 174-124-11
Win Percent: 58.389262
Confidence 0.998322
{2000: 10, 2001: -5, 2002: 5, 2003: -2, 2004: 4, 2005: -4, 2006: 6, 2007: 1, 2008: 1, 2009: 2, 2010: 7, 2011: 1, 2012: 7, 2013: 2, 2014: -1, 2015: 7, 2016: -3, 2017: 4, 2018: 1, 2019: 6, 2020: 1}
```

Above is the best strategy I was able to find. Road underdogs that lost by between 14 and 18 points the previous game are consistently undervalued. It is important to note that teams that lost by less than 14 points the previous game no longer return positive returns. And teams that have lost by more than 18 are unlikely to cover the spread consistently. Although only betting on intradivisional games will produce a higher ROI, it matches against less games and so removing this qualifier produces higher winnings. This strategy would have return 2.7x on an initial investment from 2000.

```python
money, won, loss, push, sample_set, year_to_record = set_vars()

for row in global_df_final.itertuples():

    if (row.home_spread < 0):
        money, won, loss, push, sample_set, year_to_record = bet_home(money, won, loss, push, sample_set, year_to_record)
    
print("\nPicking the home favourite")
print_basic_stats(money, won, loss, push, sample_set)
print(year_to_record)

```

```
Picking the home favourite
Ending money is -281520
Record: 1720-1865-106
Win Percent: 47.977685
Confidence 0.992311
{2000: -11, 2001: -7, 2002: -14, 2003: 0, 2004: -8, 2005: 22, 2006: -18, 2007: 4, 2008: -14, 2009: -11, 2010: -7, 2011: -7, 2012: -11, 2013: 7, 2014: -8, 2015: -20, 2016: 11, 2017: 1, 2018: -19, 2019: -21, 2020: -14}
```

Above is the worst strategy I was able to find by far. Always picking the home favourite leads to a 48% win rate, and because it bets against every single game, it gets absolutely eaten by the <50% win percentage and vig. This would have cost you 14x your money since 2000. Definitely don't do this :)

While I was able to find profitable strategies, 2.7x on an investment from 2000 isn't great when considering it would have been less effort to put that money into an index fund. That being said, there are a lot more strategies that can be evaluated, so feel free to fork the repo and explore at your leisure.