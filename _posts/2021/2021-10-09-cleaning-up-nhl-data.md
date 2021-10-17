---
layout: posts
title:  "Getting ready for machine learning - cleaning up free NHL game and odds datasets"
date:   2021-10-09 10:00:00 -0400
categories: sports
tags: sports-betting nhl machine-learning data
excerpt: "Before beginning any feature engineering or ML, it's necessary to clean up the data first. In this article we work through a real-life example"
---

# Introduction

While it’s always glamorous to focus on the few lines of taking in a dataset, optimizing some hyperparameters, and fitting it to a machine learning model, that is only a small fraction of the whole process. In the real world you’re not dealing with pre-processed and nicely packaged example data you can find in scikit-learn. Instead, the data can come in a variety of formats that may need to be manipulated to fit your goals, and often data is missing or incorrect and needs to be filled in. In many cases the data doesn’t exist at all and requires creating via API, web scraping, or manual efforts.

I was looking into creating some models to predict NHL games, but only found one viable source of betting information that was also free. It has yearly historical odds in excel format: <a href="https://www.sportsbookreviewsonline.com/scoresoddsarchives/nhl/nhloddsarchives.htm">Sports Book Reviews Online</a>. The data is mostly complete but has some issues and comes in a bit of a weird format. In this article I’ll document my process to cleaning up, merging, and manipulating the data for use in future analysis.

As always, full code is available on <a href="https://github.com/BenPollock/SportsBettingExploration/blob/main/hockey_data_cleanup.ipynb">GitHub</a>

# Implementation

First, I downloaded the data from the newest to oldest years and explored it. It has quite a bit of data, including the Over/Under and opening lines. The PuckLine is only in the latter half of years.

| Date | Rot | VH |     Team     | 1st | 2nd | 3rd | Final | Open | Close | PuckLine |      | OpenOU |      | CloseOU |      |
|:----:|:---:|:--:|:------------:|:---:|:---:|:---:|:-----:|:----:|:-----:|:--------:|:----:|:------:|:----:|:-------:|:----:|
|  113 |  41 |  V |  Pittsburgh  |  1  |  1  |  1  |   3   | -110 |  -115 |    1.5   | -310 |    6   | -110 |    6    |  105 |
|  113 |  42 |  H | Philadelphia |  2  |  1  |  3  |   6   |  100 |  -105 |   -1.5   |  260 |    6   | -110 |    6    | -125 |
|  113 |  43 |  V |   Montreal   |  2  |  1  |  1  |   0   |  132 |  120  |    1.5   | -220 |    6   | -110 |    6    | -110 |
|  113 |  44 |  H |    Toronto   |  1  |  2  |  1  |   1   | -145 |  -140 |   -1.5   |  195 |    6   | -110 |    6    | -110 |
|  113 |  45 |  V |    Chicago   |  0  |  0  |  1  |   1   |  230 |  210  |    1.5   | -125 |   6.5  | -110 |   6.5   | -110 |
|  113 |  46 |  H |   TampaBay   |  3  |  1  |  1  |   5   | -260 |  -250 |   -1.5   |  105 |   6.5  | -110 |   6.5   | -110 |
|  113 |  47 |  V |   Vancouver  |  1  |  1  |  3  |   5   |  135 |  130  |    1.5   | -215 |   5.5  | -110 |   6.5   |  100 |
|  113 |  48 |  H |   Edmonton   |  0  |  1  |  2  |   3   | -150 |  -150 |   -1.5   |  190 |   5.5  | -110 |   6.5   | -120 |
|  113 |  49 |  V |   St.Louis   |  2  |  0  |  2  |   4   |  135 |  125  |    1.5   | -200 |   5.5  | -110 |    6    | -110 |

There are a couple issues with the data that I manually had to manipulate with excel, before I could start using the files in Python.

* PuckLine, OpenOU, and CloseOU headers occupy two columns each
* The data is in XLSX and it’s easier to manipulate CSV

Although I’m sure there is a way to do this programmatically, since it was only a few years of data I manually split each of these columns and exported the file to .csv.

From there, I can import them into pandas. I loop through the CSVs per season and combine them all into a single dataframe. There is some inconsistent naming across columns, namely OpenOU/Open OU and CloseOU / CloseOU, so I had to first apply a consistent naming. Additionally, Puck Line is only in later seasons, so I dropped it entirely so each year would have the same amount of data.

```python
combined_df = pd.read_csv('nhl_odds/2008.csv')
combined_df['Year'] = '2008'
combined_df.rename(columns={'Open OU': 'OpenOU', 'Close OU': 'CloseOU'}, inplace = True)


for i in range(2009, 2022):
    temp_df = pd.read_csv('nhl_odds/' + str(i) + '.csv')
    temp_df['Year'] = str(i)

    # Address inconsistencies in column naming
    temp_df.rename(columns={'Open OU': 'OpenOU', 'Close OU': 'CloseOU'}, inplace = True)

    combined_df = combined_df.append(temp_df)

# Drop columns that are only in later data
combined_df.drop(columns=['Puck Line', 'PuckLineOdds', 'PuckLine'], inplace = True)


combined_df
```

|   | Date | Rot | VH |    Team    | 1st | 2nd | 3rd | Final | Open | Close | OpenOU | OpenOUOdds | CloseOU | CloseOUOdds | Year |
|:-:|:----:|:---:|:--:|:----------:|:---:|:---:|:---:|:-----:|:----:|:-----:|:------:|:----------:|:-------:|:-----------:|:----:|
| 0 | 929  | 1   | V  | Anaheim    | 0   | 0   | 1   | 1     | -155 | -123  | 6.0    | 100        | 6.0     | -105        | 2008 |
| 1 | 929  | 2   | H  | LosAngeles | 1   | 1   | 2   | 4     | 135  | 103   | 6.0    | -120       | 6.0     | -115        | 2008 |
| 2 | 930  | 51  | V  | LosAngeles | 0   | 1   | 0   | 1     | 115  | 132   | 6.0    | -115       | 6.0     | 112         | 2008 |
| 3 | 930  | 52  | H  | Anaheim    | 2   | 2   | 0   | 4     | -135 | -152  | 6.0    | -105       | 6.0     | -132        | 2008 |


The main concern I have with the data is that it has 2 rows per game: one for the visitor and one for the home team. It would be much easier to do analysis by combining each game into a single row. However, before I looked into merging the rows I needed to evaluate the data and clean up any errors.

I noticed there were a few issues with the data:
* An instance of the Over/Under being set at the odds, not the actual value
* Missing 'final score' in one of the rows
* Some odds have no lines (comes up as NL)

When there was a specific instance, I updated the location itself. When there were multiple instances, I mass-replaced the offending data.

```python
combined_df['Open'].iloc[33752] = -110
combined_df['Open'].iloc[33753] = -110
combined_df['Open'] = pd.to_numeric(combined_df['Open'])

combined_df['Final'].iloc[33875] = 5
combined_df['Final'] = pd.to_numeric(combined_df['Final'])

combined_df['CloseOUOdds'].iloc[34658] = 100

combined_df.loc[combined_df.Open == 0, 'Open'] = np.nan
combined_df.fillna(-110, inplace=True)
combined_df.loc[combined_df.Close == 0, 'Close'] = np.nan
combined_df.fillna(-110, inplace=True)
combined_df.loc[combined_df.OpenOU == 'NL', 'OpenOU'] = np.nan
combined_df.fillna(5.5, inplace=True)
combined_df.loc[combined_df.CloseOU == 'NL', 'CloseOU'] = np.nan
combined_df.fillna(5.5, inplace=True)
combined_df.loc[combined_df.OpenOUOdds == 'NL', 'OpenOUOdds'] = np.nan
combined_df.fillna(-110, inplace=True)
combined_df.loc[combined_df.CloseOUOdds == 'NL', 'CloseOUOdds'] = np.nan
combined_df.fillna(-110, inplace=True)

combined_df['OpenOU'] = pd.to_numeric(combined_df['OpenOU'])
combined_df['CloseOU'] = pd.to_numeric(combined_df['CloseOU'])
combined_df['OpenOUOdds'] = pd.to_numeric(combined_df['OpenOUOdds'])
combined_df['CloseOUOdds'] = pd.to_numeric(combined_df['CloseOUOdds'])

combined_df.loc[combined_df.OpenOU < 0, 'OpenOU'] = np.nan
combined_df.fillna(5.5, inplace=True)
combined_df.loc[combined_df.CloseOU < 0, 'CloseOU'] = np.nan
combined_df.fillna(5.5, inplace=True)
```

Next, I looked at team names as those can have different names (eg Tampa, TampaBay, Tampa Bay). Because I was looking at games prior to Seattle joining the league, I expect to have 31 unique teams. But looking at the number of teams, I found a larger number. I looked at the list of unique teams and applied a unique mapping to get the number of teams down to 31.

```python
combined_df.Team.unique()


fixed_names = {'NY Islanders': 'NYIslanders', 'Tampa Bay': 'TampaBay', 'Tampa': 'Tampa Bay', 'Arizonas': 'Arizona', \
    'WinnipegJets': 'Winnipeg', 'Atlanta': 'Winnipeg', 'Phoenix': 'Arizona'}

combined_df.replace({'Team': fixed_names}, inplace=True)
```

Finally, I could combine the games into a single row per game. The data was formatted such that consistently the visiting team would be the first row, and the home team would be the row immediately after. This simplified the merger because I could just use `shift` to get the row under, rather than do a search for the row with the home team further in the data. After merging the data, I renamed the columns to match the new schema.

```python
combined_df['HomeTeam'] = combined_df.Team.shift(-1)
combined_df['Home1st'] = combined_df['1st'].shift(-1)
combined_df['Home2nd'] = combined_df['2nd'].shift(-1)
combined_df['Home3rd'] = combined_df['3rd'].shift(-1)
combined_df['HomeFinal'] = combined_df.Final.shift(-1)
combined_df['MlHomeOpen'] = combined_df.Open.shift(-1)
combined_df['MlHomeClose'] = combined_df.Close.shift(-1)

# For over-under, the first row (visitors) is over, then the second row (home) is under
# Because the over will always equal the under, I only need to pull the opening and closing odds
combined_df['OpenUOdds'] = combined_df.OpenOUOdds.shift(-1)
combined_df['CloseUOdds'] = combined_df.CloseOUOdds.shift(-1)

# Now we've pulled the needed data into a single row, so we can drop the extra (now in-correct) Home row
combined_df.drop(combined_df[combined_df['VH'] == 'H'].index, inplace = True)

# Clean up and rename columns
combined_df.drop(columns=['Rot','VH'], inplace=True)
combined_df.rename(columns={'Team': 'AwayTeam', '1st': 'Away1st', '2nd': 'Away2nd', '3rd': 'Away3rd', 'Final': 'AwayFinal', \
  'Open': 'MlAwayOpen', 'Close': 'MlAwayClose', 'OpenOUOdds': 'OpenOOdds', 'CloseOUOdds': 'CloseOOdds'}, inplace=True)

combined_df
```

|      | Date |  AwayTeam  | Away1st | Away2nd | Away3rd | AwayFinal | MlAwayOpen | MlAwayClose | OpenOU | OpenOOdds | ... | Year |  HomeTeam  | Home1st | Home2nd | Home3rd | HomeFinal | MlHomeOpen | MlHomeClose | OpenUOdds | CloseUOdds |
|:----:|:----:|:----------:|:-------:|:-------:|:-------:|:---------:|:----------:|:-----------:|:------:|:---------:|:---:|:----:|:----------:|:-------:|:-------:|:-------:|:---------:|:----------:|:-----------:|:---------:|:----------:|
|   0  | 929  | Anaheim    | 0       | 0       | 1       | 1         | -155.0     | -123.0      | 6.0    | 100       | ... | 2008 | LosAngeles | 1.0     | 1.0     | 2.0     | 4.0       | 135.0      | 103.0       | -120.0    | -115.0     |
|   2  | 930  | LosAngeles | 0       | 1       | 0       | 1         | 115.0      | 132.0       | 6.0    | -115      | ... | 2008 | Anaheim    | 2.0     | 2.0     | 0.0     | 4.0       | -135.0     | -152.0      | -105.0    | -132.0     |
|   4  | 1003 | Anaheim    | 0       | 2       | 0       | 2         | 165.0      | 190.0       | 5.5    | 105       | ... | 2008 | Detroit    | 1.0     | 0.0     | 1.0     | 3.0       | -185.0     | -230.0      | -125.0    | -140.0     |
|   6  | 1003 | Montreal   | 1       | 0       | 1       | 3         | 130.0      | 140.0       | 6.0    | -105      | ... | 2008 | Carolina   | 1.0     | 0.0     | 1.0     | 2.0       | -150.0     | -160.0      | -115.0    | -130.0     |
|   8  | 1003 | Ottawa     | 2       | 0       | 1       | 4         | -120.0     | -125.0      | 6.0    | -120      | ... | 2008 | Toronto    | 2.0     | 1.0     | 0.0     | 3.0       | 100.0      | 105.0       | 100.0     | 115.0      |
|  ... | ...  | ...        | ...     | ...     | ...     | ...       | ...        | ...         | ...    | ...       | ... | ...  | ...        | ...     | ...     | ...     | ...       | ...        | ...         | ...       | ...        |
| 1894 | 628  | Montreal   | 0       | 1       | 0       | 1         | 185.0      | 180.0       | 5.0    | -120      | ... | 2021 | TampaBay   | 1.0     | 1.0     | 3.0     | 5.0       | -225.0     | -200.0      | 100.0     | 105.0      |
| 1896 | 630  | Montreal   | 0       | 1       | 0       | 1         | 180.0      | 177.0       | 5.0    | -120      | ... | 2021 | TampaBay   | 0.0     | 2.0     | 1.0     | 3.0       | -220.0     | -197.0      | 100.0     | 100.0      |
| 1898 | 702  | TampaBay   | 2       | 2       | 2       | 6         | -145.0     | -123.0      | 5.0    | -120      | ... | 2021 | Montreal   | 1.0     | 1.0     | 1.0     | 3.0       | 125.0      | 113.0       | 100.0     | 100.0      |
| 1900 | 705  | TampaBay   | 0       | 1       | 1       | 2         | -145.0     | -150.0      | 5.0    | -135      | ... | 2021 | Montreal   | 1.0     | 0.0     | 1.0     | 3.0       | 125.0      | 135.0       | 115.0     | 120.0      |
| 1902 | 707  | Montreal   | 0       | 0       | 0       | 0         | 180.0      | 215.0       | 5.0    | -120      | ... | 2021 | TampaBay   | 0.0     | 1.0     | 0.0     | 1.0       | -220.0     | -245.0      | 100.0     | 120.0      |


Finally, with the data clean and in the format we want, we can export the data to CSV and use it in a future article.

```python
combined_df.to_csv('x.csv', index=True)
```



