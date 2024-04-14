import pandas as pd
import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

# read data from csv
df = pd.read_csv('2022-2023 NBA Player Stats - Regular.csv', encoding='ISO-8859-1', sep=";")
# keep only TOT (total) stats for players on multiple teams
df = df.drop_duplicates('Player')

# some players' names have question marks
rows_with_question_mark = df[df['Player'].str.contains('\?', regex=True)]
# correct their names manually
df.at[49, 'Player'] = 'Davis Bertans'
df.at[62, 'Player'] = 'Bogdan Bogdanovic'
df.at[63, 'Player'] = 'Bojan Bogdanovic'
df.at[105, 'Player'] = 'Vlatko Cancar'
df.at[160, 'Player'] = 'Luka Doncic'
df.at[167, 'Player'] = 'Goran Dragic'
df.at[318, 'Player'] = 'Nikola Jokic'
df.at[330, 'Player'] = 'Nikola Jovic'
df.at[391, 'Player'] = 'Boban Marjanovic'
df.at[466, 'Player'] = 'Jusuf Nurkic'
df.at[502, 'Player'] = 'Kristaps Porzingis'
df.at[550, 'Player'] = 'Luka Samanic'
df.at[551, 'Player'] = 'Dario Saric'
df.at[614, 'Player'] = 'Jonas Valanciunas'
df.at[622, 'Player'] = 'Nikola Vucevic'

# function to run logistic regression
def run_logistic_regression(df, stats):
    # lists the stats used
    print("Stats: " + str(stats))

    # scales the data
    df_scaled = scale(df)

    # creates features and target arrays
    X = df_scaled[stats].to_numpy()
    y = df_scaled[['allstar_selected']].to_numpy()

    # fits arrays to logistic regression
    reg = LogisticRegression(random_state=16)
    reg.fit(X, np.ravel(y))

    # prints coefficients
    print("Coefficients: " + str(reg.coef_))
    y_pred = reg.predict(X)

    # prints accuracy on training set
    score = accuracy_score(y, y_pred)
    print("Accuracy on training set: " + str(score))
    
    # newline for organization
    print("")

# function to standardize data
def scale(df):
    scaler = StandardScaler()
    
    # excludes non-continuous data
    players = df['Player']
    pos = df['Pos']
    team = df['Tm']
    rank = df['Rk']
    allstar_selected = df['allstar_selected']
    df_numeric = df.drop(columns=['Rk', 'Player', "Pos", 'Tm', 'allstar_selected'])

    # standardizes data
    scaled_numeric = scaler.fit_transform(df_numeric)
    scaled_df = pd.DataFrame(scaled_numeric, index=df.index, columns=df_numeric.columns)

    # reconcatenates non-continuous data
    new_columns_df = pd.DataFrame({'Rk': rank, 'Player': players, 'Pos': pos, 'Tm': team, 'allstar_selected': allstar_selected})
    scaled_df = pd.concat([new_columns_df, scaled_df], axis=1)
    return scaled_df

# function to add allstar selections to data
def add_allstars(df, allstars):
    df['allstar_selected'] = 0
    df.loc[df['Player'].isin(allstars), 'allstar_selected'] = 1

allstars = ["Kyrie Irving",
            "Donovan Mitchell",
            "Giannis Antetokounmpo",
            "Kevin Durant",
            "Jayson Tatum",
            "Jaylen Brown",
            "DeMar DeRozan",
            "Tyrese Haliburton",
            "Jrue Holiday",
            "Julius Randle",
            "Bam Adebayo",
            "Joel Embiid",
            "Pascal Siakam",
            "Stephen Curry",
            "Luka Doncic",
            "Nikola Jokic",
            "LeBron James",
            "Zion Williamson",
            "Shai Gilgeous-Alexander",
            "Damian Lillard",
            "Ja Morant",
            "Paul George",
            "Jaren Jackson Jr.",
            "Lauri Markkanen",
            "Domantas Sabonis",
            "Anthony Edwards",
            "De'Aaron Fox"]

add_allstars(df, allstars)

run_logistic_regression(df, ['TRB', 'AST', 'STL', 'BLK', 'PTS'])

run_logistic_regression(df, ['TRB', 'AST', 'PTS'])

run_logistic_regression(df, ['AST', 'PTS'])

run_logistic_regression(df, ['FT', 'FTA', 'FT%'])

print("Centers with big 5 stats: ")
run_logistic_regression(df.loc[df['Pos'] == 'C'], ['TRB', 'AST', 'STL', 'BLK', 'PTS'])

print("Shooting guards with big 5 stats: ")
run_logistic_regression(df.loc[df['Pos'] == 'SG'], ['TRB', 'AST', 'STL', 'BLK', 'PTS'])

print("Power forwards with big 5 stats: ")
run_logistic_regression(df.loc[df['Pos'] == 'PF'], ['TRB', 'AST', 'STL', 'BLK', 'PTS'])

print("Small forwards with big 5 stats: ")
run_logistic_regression(df.loc[df['Pos'] == 'SF'], ['TRB', 'AST', 'STL', 'BLK', 'PTS'])

print("Point guards with big 5 stats: ")
run_logistic_regression(df.loc[df['Pos'] == 'PG'], ['TRB', 'AST', 'STL', 'BLK', 'PTS'])