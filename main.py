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

def regress_sets_of_n(df, stats, test_list, n):
    sets = list(combinations(stats, n))
    max_score = 0
    max_recall = 0
    max_set = []
    max_recall_set = []
    for set in sets:
        score, recall = run_logistic_regression(df, list(set), test_list)
        if score > max_score:
            max_score = score
            max_set = set
        if recall > max_recall:
            max_recall = recall
            max_recall_set = set
    print("The most accurate set is " + str(max_set) + " with an accuracy of " + str(max_score))
    print("The set with the best recall is " + str(max_recall_set) + " with a recall of " + str(max_recall))

# function to run logistic regression
def run_logistic_regression(df, stats, test_list):
    # scales the data
    df_scaled = scale(df)

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
    for i in range(len(test_list)):
        df_test_scaled = scale(test_list[i])
        X_test = df_test_scaled[stats].to_numpy()
        y_test = df_test_scaled[['allstar_selected']].to_numpy()
        y_test_pred = reg.predict(X_test)
        
        test_score = accuracy_score(y_test, y_test_pred)
        test_scores.append(test_score)

        recall = recall_score(y_test, y_test_pred)
        recalls.append(recall)

    avg_test_score = sum(test_scores) / len(test_scores)
    avg_recall = sum(recalls) / len(recalls)
    
    return avg_test_score, avg_recall

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