#####################
# pandas
#####################
# pd.read_csv(data_path, delimiter = ",")


# pd.get_dummies(T,prefix=['Actor'])
"""
One hot encoding of actor
"""

# for column in X.columns:
#     X[column].fillna(X[column].mode()[0], inplace=True)
""" 
we replace every empty value with the column's mode 
since we have categorical 
"""

# X['Oldpeak'] = pd.Series(X['Oldpeak']).str.replace(',', '.')
""" 
Some of the data is written in a European format, 
that doesn't work in python
"""
