import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
df = pd.read_csv('2022-2023 NBA Player Stats - Regular.csv', encoding='ISO-8859-1', sep=";")

print(df.head())
print(df.shape)

# Add new column "allstar_selected"
df['allstar_selected'] = 0
# Update allstar list
selected_name_list_2023 = ["Jaylen Brown",
                     'Stephen Curry',
                     'Luka Doncic',
                     'Anthony Edwards',
                     'Joel Embiid',
                     'De\'Aaron Fox',
                     'Paul George',
                     'Tyrese Haliburton',
                     'Kyrie Irving',
                     'Jaren Jackson, Jr.',
                     'LeBron James',
                     'Nikola Jokic',
                     'Julius Randle',
                     'Zion Williamson']
df.loc[df['Player'].isin(selected_name_list_2023), 'allstar_selected'] = 1

print(df.loc[df['allstar_selected']==1])
print(df.loc[df['Player']=='Kyrie Irving'])


# Some Players' name have question mark
rows_with_question_mark = df[df['Player'].str.contains('\?', regex=True)]
print(rows_with_question_mark)

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

# saved as csv file
scaled_df.to_csv('2022-2023 NBA Player Stats cleaned.csv', index=False, encoding='utf-8-sig')
