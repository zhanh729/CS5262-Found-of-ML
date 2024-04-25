import pandas as pd
import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.metrics import recall_score
from itertools import combinations

# read training data from csv
df_train = pd.read_csv('2022-2023 NBA Player Stats - Regular.csv', encoding='ISO-8859-1', sep=";")
# keep only TOT (total) stats for players on multiple teams
df_train = df_train.drop_duplicates('Player')
# some players' names have question marks
# print(df_train[df_train['Player'].str.contains('\?', regex=True)])
# correct their names manually
df_train.at[49, 'Player'] = 'Davis Bertans'
df_train.at[62, 'Player'] = 'Bogdan Bogdanovic'
df_train.at[63, 'Player'] = 'Bojan Bogdanovic'
df_train.at[105, 'Player'] = 'Vlatko Cancar'
df_train.at[160, 'Player'] = 'Luka Doncic'
df_train.at[167, 'Player'] = 'Goran Dragic'
df_train.at[318, 'Player'] = 'Nikola Jokic'
df_train.at[330, 'Player'] = 'Nikola Jovic'
df_train.at[391, 'Player'] = 'Boban Marjanovic'
df_train.at[466, 'Player'] = 'Jusuf Nurkic'
df_train.at[502, 'Player'] = 'Kristaps Porzingis'
df_train.at[550, 'Player'] = 'Luka Samanic'
df_train.at[551, 'Player'] = 'Dario Saric'
df_train.at[614, 'Player'] = 'Jonas Valanciunas'
df_train.at[622, 'Player'] = 'Nikola Vucevic'
# some players have multiple positions
# print(df_train[df_train['Pos'].str.contains('-', regex=True)])
# correct their positions manually
df_train.at[174, 'Pos'] = 'PF'
df_train.at[254, 'Pos'] = 'SF'
df_train.at[346, 'Pos'] = 'PF'
df_train.at[463, 'Pos'] = 'SG'
df_train.at[631, 'Pos'] = 'SF'

# read testing data from csv
df_test_A = pd.read_csv('2021-2022 NBA Player Stats - Regular.csv', encoding='ISO-8859-1', sep=";")
# keep only TOT (total) stats for players on multiple teams
df_test_A = df_test_A.drop_duplicates('Player')
# some players' names have question marks
# print(df_test_A[df_test_A['Player'].str.contains('\?', regex=True)])
# correct their names manually
df_test_A.at[54, 'Player'] = 'Davis Bertans'
df_test_A.at[65, 'Player'] = 'Bogdan Bogdanovic'
df_test_A.at[66, 'Player'] = 'Bojan Bogdanovic'
df_test_A.at[110, 'Player'] = 'Vlatko Cancar'
df_test_A.at[178, 'Player'] = 'Luka Doncic'
df_test_A.at[189, 'Player'] = 'Goran Dragic'
df_test_A.at[391, 'Player'] = 'Nikola Jokic'
df_test_A.at[472, 'Player'] = 'Boban Marjanovic'
df_test_A.at[557, 'Player'] = 'Jusuf Nurkic'
df_test_A.at[601, 'Player'] = 'Kristaps Porzingis'
df_test_A.at[651, 'Player'] = 'Tomas Satoransky'
df_test_A.at[743, 'Player'] = 'Jonas Valanciunas'
df_test_A.at[751, 'Player'] = 'Nikola Vucevic'
# some players have multiple positions
# print(df_test_A[df_test_A['Pos'].str.contains('-', regex=True)])
# correct their positions manually
df_test_A.at[81, 'Pos'] = 'SG'
df_test_A.at[89, 'Pos'] = 'SG'
df_test_A.at[146, 'Pos'] = 'PF'
df_test_A.at[276, 'Pos'] = 'PG'
df_test_A.at[282, 'Pos'] = 'PG'
df_test_A.at[294, 'Pos'] = 'SF'
df_test_A.at[315, 'Pos'] = 'SG'
df_test_A.at[325, 'Pos'] = 'SG'
df_test_A.at[329, 'Pos'] = 'SF'
df_test_A.at[343, 'Pos'] = 'SG'
df_test_A.at[350, 'Pos'] = 'C'
df_test_A.at[435, 'Pos'] = 'SG'
df_test_A.at[458, 'Pos'] = 'SG'
df_test_A.at[487, 'Pos'] = 'PG'
df_test_A.at[607, 'Pos'] = 'SG'
df_test_A.at[647, 'Pos'] = 'C'
df_test_A.at[657, 'Pos'] = 'SG'
df_test_A.at[698, 'Pos'] = 'SG'

# read testing data from csv
df_test_B = pd.read_csv('2023-2024 NBA Player Stats - Regular.csv', encoding='ISO-8859-1', sep=";")
# keep only TOT (total) stats for players on multiple teams
df_test_B = df_test_B.drop_duplicates('Player')
# some players' names have question marks
# print(df_test_B[df_test_B['Player'].str.contains('\?', regex=True)])
# correct their names manually
df_test_B.at[55, 'Player'] = 'Davis Bertans'
df_test_B.at[69, 'Player'] = 'Bogdan Bogdanovic'
df_test_B.at[70, 'Player'] = 'Bojan Bogdanovic'
df_test_B.at[162, 'Player'] = 'Luka Doncic'
df_test_B.at[321, 'Player'] = 'Nikola Jokic'
df_test_B.at[331, 'Player'] = 'Nikola JoVic'
df_test_B.at[384, 'Player'] = 'Boban Marjanovic'
df_test_B.at[414, 'Player'] = 'Vasilije Micic'
df_test_B.at[466, 'Player'] = 'Jusuf Nurkic'
df_test_B.at[507, 'Player'] = 'Kristaps Porzingis'
df_test_B.at[549, 'Player'] = 'Luka Samanic'
df_test_B.at[553, 'Player'] = 'Dario Saric'
df_test_B.at[628, 'Player'] = 'Jonas Valanciunas'
df_test_B.at[634, 'Player'] = 'Nikola Vucevic'
# some players have multiple positions
# print(df_test_B[df_test_B['Pos'].str.contains('-', regex=True)])
# correct their positions manually
df_test_B.at[0, 'Pos'] = 'PF'
df_test_B.at[70, 'Pos'] = 'SF'
df_test_B.at[91, 'Pos'] = 'SG'
df_test_B.at[136, 'Pos'] = 'SF'
df_test_B.at[203, 'Pos'] = 'C'
df_test_B.at[206, 'Pos'] = 'PF'
df_test_B.at[270, 'Pos'] = 'SF'
df_test_B.at[279, 'Pos'] = 'C'
df_test_B.at[396, 'Pos'] = 'PG'
df_test_B.at[518, 'Pos'] = 'PG'
df_test_B.at[542, 'Pos'] = 'PF'
df_test_B.at[587, 'Pos'] = 'SG'
df_test_B.at[613, 'Pos'] = 'C'

def regress_sets_of_n(df, stats, test_list, n):
    sets = list(combinations(stats, n))
    max_score = 0
    max_recall = 0
    max_set = []
    max_recall_set = []
    pred = []
    for set in sets:
        score, recall, pred_allstars_list = run_logistic_regression(df, list(set), test_list)
        if recall > max_recall:
            max_recall = recall
            max_recall_set = set
            max_score = score
            pred = pred_allstars_list
    print("The best set of stats is " + str(max_recall_set))
    print("Accuracy: " + str(max_score))
    print("Recall: " + str(max_recall))
    for i in range(len(pred)):
        print("Predicted All-Stars (Testing Set " + str(i + 1) + "):")
        print(', '.join(pred[i]))

# function to run logistic regression
def run_logistic_regression(df, stats, test_list):
    # scales the data
    df_reduced = reduce_players(df, 2)
    df_scaled = scale(df_reduced)

    # creates features and target arrays
    X = df_scaled[stats].to_numpy()
    y = df_scaled[['allstar_selected']].to_numpy()

    # fits arrays to logistic regression
    reg = LogisticRegression(random_state=16)
    reg.fit(X, np.ravel(y))

    y_pred = reg.predict(X)

    # prints accuracy on training set
    score = accuracy_score(y, y_pred)

    test_scores = []
    recalls = []
    pred_allstars_list = []
    for i in range(len(test_list)):
        df_test_scaled = scale(test_list[i])
        X_test = df_test_scaled[stats].to_numpy()
        y_test = df_test_scaled[['allstar_selected']].to_numpy()
        y_test_pred = reg.predict(X_test)
        
        test_score = accuracy_score(y_test, y_test_pred)
        test_scores.append(test_score)

        recall = recall_score(y_test, y_test_pred)
        recalls.append(recall)

        mask = y_test_pred.astype(bool)
        pred_allstars = test_list[i][mask]['Player']
        pred_allstars_list.append(pred_allstars)

    avg_test_score = sum(test_scores) / len(test_scores)
    avg_recall = sum(recalls) / len(recalls)
    
    return avg_test_score, avg_recall, pred_allstars_list

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

def reduce_players(df, num):
    # reduce non-allstar players to num
    df_non_allstar = df.loc[df['allstar_selected'] == 0]
    df_reduced = df_non_allstar.sample(n = len(df_non_allstar) // num)

    # combine allstars back into dataframe
    df_allstar = df.loc[df['allstar_selected'] == 1]
    df_combined = pd.concat([df_reduced, df_allstar], axis=0)

    # shuffle dataframe
    return df_combined.sample(frac = 1)

def get_pos(df, pos):
    return df.loc[df['Pos'].isin(pos)]

# function to add allstar selections to data
def add_allstars(df, allstars):
    df['allstar_selected'] = 0
    df.loc[df['Player'].isin(allstars), 'allstar_selected'] = 1

allstars_train = ["Kyrie Irving",
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

add_allstars(df_train, allstars_train)

allstars_test_A = ["Trae Young",
                   "DeMar DeRozan",
                   "Joel Embiid",
                   "Kevin Durant",
                   "Giannis Antetokounmpo",
                   "LaMelo Ball",
                   "Darius Garland",
                   "James Harden",
                   "Zach LaVine",
                   "Fred VanVleet",
                   "Jimmy Butler",
                   "Khris Middleton",
                   "Jayson Tatum",
                   "Jarrett Allen",
                   "Stephen Curry",
                   "Ja Morant",
                   "Nikola Jokic",
                   "LeBron James",
                   "Andrew Wiggins",
                   "Devin Booker",
                   "Luka Doncic",
                   "Donovan Mitchell",
                   "Dejounte Murray",
                   "Chris Paul",
                   "Draymond Green",
                   "Rudy Gobert",
                   "Karl-Anthony Towns"]

add_allstars(df_test_A, allstars_test_A)

allstars_test_B = ["Tyrese Haliburton",
                   "Damian Lillard",
                   "Giannis Antetokounmpo",
                   "Jayson Tatum",
                   "Joel Embiid",
                   "Jalen Brunson",
                   "Tyrese Maxey",
                   "Donovan Mitchell",
                   "Trae Young",
                   "Paolo Banchero",
                   "Scottie Barnes",
                   "Jaylen Brown",
                   "Julius Randle",
                   "Bam Adebayo",
                   "Luka Doncic",
                   "Shai Gilgeous-Alexander",
                   "Kevin Durant",
                   "LeBron James",
                   "Nikola Jokic",
                   "Devin Booker",
                   "Stephen Curry",
                   "Anthony Edwards",
                   "Paul George",
                   "Kawhi Leonard",
                   "Karl-Anthony Towns",
                   "Anthony Davis"]

add_allstars(df_test_B, allstars_test_B)

important_stats = ['eFG%', 'ORB', 'DRB', 'TRB', 'AST', 'STL', 'BLK', 'PTS']

n = 4

print("All positions:")
regress_sets_of_n(df_train, important_stats, [df_test_A, df_test_B], n)

front = ['C', 'PF', 'SF']
print("Front court:")
regress_sets_of_n(get_pos(df_train, front), important_stats, [get_pos(df_test_A, front), get_pos(df_test_B, front)], n)

back = ['PG', 'SG']
print("Back court:")
regress_sets_of_n(get_pos(df_train, back), important_stats, [get_pos(df_test_A, back), get_pos(df_test_B, back)], n)

print("Center:")
regress_sets_of_n(get_pos(df_train, ['C']), important_stats, [get_pos(df_test_A, ['C']), get_pos(df_test_B, ['C'])], n)

print("Power forward:")
regress_sets_of_n(get_pos(df_train, ['PF']), important_stats, [get_pos(df_test_A, ['PF']), get_pos(df_test_B, ['PF'])], n)

print("Small forward:")
regress_sets_of_n(get_pos(df_train, ['SF']), important_stats, [get_pos(df_test_A, ['SF']), get_pos(df_test_B, ['SF'])], n)

print("Point guard:")
regress_sets_of_n(get_pos(df_train, ['PG']), important_stats, [get_pos(df_test_A, ['PG']), get_pos(df_test_B, ['PG'])], n)

print("Shooting guard:")
regress_sets_of_n(get_pos(df_train, ['SG']), important_stats, [get_pos(df_test_A, ['SG']), get_pos(df_test_B, ['SG'])], n)