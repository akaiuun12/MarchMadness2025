# %% 2025 March Madness Kaggle Competition
# [Reference Notebook](https://www.kaggle.com/code/paultimothymooney/simple-starter-notebook-for-march-mania-2025)

# %%
# March Mania 2025 - Starter Notebook
## Goal of the competition

# The goal of this competition is to predict that probability that the smaller ``TeamID`` will win a given matchup. You will predict the probability for every possible matchup between every possible team over the past 4 years. You'll be given a sample submission file where the ```ID``` value indicates the year of the matchup as well as the identities of both teams within the matchup. For example, for an ```ID``` of ```2025_1101_1104``` you would need to predict the outcome of the matchup between ```TeamID 1101``` vs ```TeamID 1104``` during the ```2025``` tournament. Submitting a ```PRED``` of ```0.75``` indicates that you think that the probability of ```TeamID 1101``` winning that particular matchup is equal to ```0.75```.

# ## Overview of our submission strategy 
# For this starter notebook, we will make a simple submission.

# We can predict the winner of a match by considering the respective rankings of the opposing teams, only. Since the largest possible difference is 15 (which is #16 minus #1), we use a rudimentary formula that's 0.5 plus 0.03 times the difference in seeds, leading to a range of predictions spanning from 5% up to 95%. The stronger-seeded team (with a lower seed number from 1 to 16) will be the favorite and will have a prediction above 50%. 

# Starter Code
# %%  Step 1: Import Python packages
import numpy as np
import pandas as pd
from sklearn.metrics import brier_score_loss, mean_squared_error

# %%  Step 2: Explore the data
w_seed = pd.read_csv('competition_data/WNCAATourneySeeds.csv')
m_seed = pd.read_csv('competition_data/MNCAATourneySeeds.csv')
seed_df = pd.concat([m_seed, w_seed], axis=0).fillna(0.05)
submission_df = pd.read_csv('competition_data/SampleSubmissionStage2.csv')

# %% Team rankings are present in the files WNCAATourneySeeds.csv and MNCAATourneySeeds.csv. 
# - The "Season" column indicates the year
# - The "Seed" column indicates the ranking for a given conference (W01 = ranking 1 in conference W)

# - The "TeamID" column contains a unique identifier for every team
seed_df.head()

# The sample_submission.csv file contains an "ID" column with the format year_teamID1_teamID2.
submission_df.head()

# %% EDA by RED
# Regular Season Results
regular_season_results = pd.read_csv('competition_data/MRegularSeasonCompactResults.csv')

ncaa_2024 = regular_season_results[regular_season_results['Season'] == 2024]
ncaa_2024['margin'] = ncaa_2024['WScore'] - ncaa_2024['LScore']

W_2024 = ncaa_2024[['Season', 'DayNum', 'WTeamID', 'WScore', 'WLoc', 'margin']] \
    .rename(columns={'WTeamID': 'TeamID', 'WScore': 'Score', 'WLoc': 'Loc'}, inplace=False)

L_2024 = ncaa_2024[['Season', 'DayNum', 'LTeamID', 'LScore', 'WLoc', 'margin']] \
    .rename(columns={'LTeamID': 'TeamID', 'LScore': 'Score', 'WLoc': 'Loc'}, inplace=False)

L_2024['margin'] = - L_2024['margin']

ncaa_2024_flatten = pd.concat([W_2024, L_2024], axis=0)
ncaa_2024_flatten

# %% 
team_name = pd.read_csv('competition_data/MTeams.csv')
team_name

# %% 
ncaa_2024_team_margin = ncaa_2024_flatten.groupby('TeamID')['margin'].mean().sort_values(ascending=False)

df_answer = pd.merge(ncaa_2024_team_margin, team_name, 
                     left_on='TeamID', right_on='TeamID', how='left')
df_answer

# %%  Step 3: Extract game info and team rankings
def extract_game_info(id_str):
    # Extract year and team_ids
    parts = id_str.split('_')
    year = int(parts[0])
    teamID1 = int(parts[1])
    teamID2 = int(parts[2])
    return year, teamID1, teamID2

def extract_seed_value(seed_str):
    # Extract seed value
    try:
        return int(seed_str[1:])
    # Set seed to 16 for unselected teams and errors
    except ValueError:
        return 16

# Reformat the data
submission_df[['Season', 'TeamID1', 'TeamID2']] = submission_df['ID'].apply(extract_game_info).tolist()
seed_df['SeedValue'] = seed_df['Seed'].apply(extract_seed_value)

# Merge seed information for TeamID1
submission_df = pd.merge(submission_df, seed_df[['Season', 'TeamID', 'SeedValue']],
                         left_on=['Season', 'TeamID1'], right_on=['Season', 'TeamID'],
                         how='left')
submission_df = submission_df.rename(columns={'SeedValue': 'SeedValue1'}).drop(columns=['TeamID'])

# Merge seed information for TeamID2
submission_df = pd.merge(submission_df, seed_df[['Season', 'TeamID', 'SeedValue']],
                         left_on=['Season', 'TeamID2'], right_on=['Season', 'TeamID'],
                         how='left')
submission_df = submission_df.rename(columns={'SeedValue': 'SeedValue2'}).drop(columns=['TeamID'])

# %%  Step 4: Make your predictions
# Calculate seed difference
submission_df['SeedDiff'] = submission_df['SeedValue1'] - submission_df['SeedValue2']

# Update 'Pred' column
submission_df['Pred'] = 0.5 + (0.03 * submission_df['SeedDiff'])

# Drop unnecessary columns
submission_df = submission_df[['ID', 'Pred']].fillna(0.5)

# Preview your submission
submission_df.head()
stats = submission_df.iloc[:, 1].describe()
print(stats)

# %%  Step 5: Understand the metric
# We don't know the outcomes of the games, so instead let's assume that the team that was listed first won every single matchup. This is what we'll call our "true value". Next, we'll calculate the average squared difference between the probabilities in our submission and that ground truth value. We'll call this the "Brier score". https://en.wikipedia.org/wiki/Brier_score

# Create a dataframe of ground truth values
solution_df = submission_df.copy()
solution_df['Pred'] = 1

# Now calculate the Brier score
y_true = solution_df['Pred']
y_pred = submission_df['Pred']
brier_score = brier_score_loss(y_true, y_pred)
print(f"Brier Score: {brier_score}")

# %%  Step 6: Make your submission
submission_df.to_csv('/kaggle/working/submission.csv', index=False)