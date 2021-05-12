###### import and load data:
# import pandas as pd 
# import os
# Example:
# fileName = 'Police_Department_Incident_Reports__Historical_2003_to_May_2018.csv'
# filePath = os.path.abspath(os.path.join(os.getcwd(), '..' ,'Datasets', fileName))
# Data = pd.read_csv(filePath)
 



###### create
# pd.DataFrame(series or lst,index=lst, columns=lst)
# pd.DataFrame(dic) """" example of dic = {'x': [1, 2, 3], 'y': [3, 4, 5]}""""
# pd.Series(lst, index=lst)


###### columns rows
# data.index
# data.columns


###### slicing and accessing
# data['colName']
# data.loc[rowIndex] 

# data[['colName1','colName2']]
# data.loc[[rowIndex1,rowIndex2]]

# data.loc[rowIndexStart:rowIndexEnd,:]
# data.loc[:,'colStart':'colStart']

# data.iloc same but not index

# data[data['colName']==value]
# data[(data['colName'] == val) & (data['colName'] == val)]
# data[data['colName'].isin(lst)]


####### apply
# data['colName'].apply(lambda x : e)
# data.apply(lambda row : row['colName1']+row['colName2'], axis = 1)


####### copy
# data.copy()
# pd.Series.copy() """ pd.Series is a data['colNme'] """


####### date
# pd.to_datetime(Data['Date']) # comparsion is okay
# pd.to_datetime(data['Date']).dt.month
# pd.to_datetime(data['Date']).dt.day
# pd.to_datetime(data['Time']).dt.hour
# pd.to_datetime(data['Time']).dt.minute

####### functions
# data.reset_index(drop=True)

# pd.Series.sort_index()
# pd.Series.sort_values(ascending=False)

# data.drop(columns='colName')
# data.drop(index=rowindex)

# data['colName'].unique()
# data['colName'].nunique()

# data.groupby(['colName','colName'])['ColName'].count()
# data.groupby(['colName','colName'])['ColName'].size().unstack()

# counts = df[df.Category == 'WEAPON LAWS']['DayOfWeek'].value_counts()
# weekdays = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
# counts = counts.reindex(weekdays)

# data.shape

# data['colName'].values()


####### examples: 
# fileName = 'MVC_SL_W_Final.csv'
# filePath = os.path.abspath(os.path.join(os.getcwd(), fileName))
# Data.to_csv(filePath)


####### ref 
# .groupby()
# .count()
# .size()
# .unstack()

# .reset_index()
# .sort_index()
# .sort_values()
# .drop()

# .value_counts()
# .shape
# .values
# .index
# .columns
# .copy()
# .apply()

# .loc()
# .iloc()

# .sample(frac=1) """ shuffle rows"""
# .drop(columns=[])
# .rename(columns={colName:toColName})


# Table = pd.pivot_table(df_crimes, index = "Hour", columns = "Category", values = 'PdId' ,aggfunc = 'count')
# The above is equivalent to using .groupby(), then using .unstack().

# .select_dtypes(include=object)

# .to_frame()




