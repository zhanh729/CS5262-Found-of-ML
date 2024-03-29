import pandas as pd
import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
df = pd.read_csv('2022-2023 NBA Player Stats - Regular.csv', encoding='ISO-8859-1', sep=";")

print(df.head())
print(df.shape)

# Keep only TOT (total) stats for players on multiple teams
df = df.drop_duplicates('Player')

# Some Players' name have question mark
rows_with_question_mark = df[df['Player'].str.contains('\?', regex=True)]
print(rows_with_question_mark)
# Correct their names manually
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

# check null values
null_counts = df.isnull().sum()
print(null_counts)

# Standardization
scaler = StandardScaler()
# Players, tm, pos are string
players = df['Player']
pos = df['Pos']
team = df['Tm']
rank = df['Rk']
df_numeric = df.drop(columns=['Rk', 'Player', "Pos", 'Tm'])
scaled_numeric = scaler.fit_transform(df_numeric)
scaled_df = pd.DataFrame(scaled_numeric, index=df.index, columns=df_numeric.columns)

# combine new columns
new_columns_df = pd.DataFrame({'Rk': rank, 'Player': players, 'Pos': pos, 'Tm': team})
scaled_df = pd.concat([new_columns_df, scaled_df], axis=1)
print(scaled_df)


# PCA( if needed)
# pca = PCA()
# pca.fit(scaled_df.drop(columns=['Rk', 'Player', "Pos", 'Tm']))
# explained_variance = pca.explained_variance_ratio_
# print(explained_variance)

# Add new column "allstar_selected"
scaled_df['allstar_selected'] = 0
# Update allstar list
selected_name_list_2023 = ["Kyrie Irving",
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
scaled_df.loc[scaled_df['Player'].isin(selected_name_list_2023), 'allstar_selected'] = 1

print(scaled_df.loc[scaled_df['allstar_selected'] == 1])

# saved as csv file
scaled_df.to_csv('2022-2023 NBA Player Stats cleaned.csv', index=False, encoding='utf-8-sig')

# Place big five stats into numpy array for features
X = scaled_df[['TRB', 'AST', 'STL', 'BLK', 'PTS']].to_numpy()
print(X.shape)

# Place allstar_selected into numpy array for label
y = scaled_df[['allstar_selected']].to_numpy()
print(y.shape)

# Make logistic regression model and print coefficients
logreg = LogisticRegression(random_state=16)
logreg.fit(X, np.ravel(y))
print("Coefficients: " + str(logreg.coef_))