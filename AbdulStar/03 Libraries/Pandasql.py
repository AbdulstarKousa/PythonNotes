# ================== install:
# pip install pandasql
# conda install pandasql

# ================== Libraries: 
# for pandasql 
import pandasql as ps

# related 
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
sns.set()


# ==================  Methods:
# Load: 
df_shootings = pd.read_csv('shootings.csv')
df_states = pd.read_csv('shootings_states.csv')


# Comment 
"""
/*comment*/
"""

# SELECT 
ps.sqldf("""
SELECT id, 
       date 
FROM df_shootings 
LIMIT 2
""")

ps.sqldf("""
SELECT *
FROM df_shootings 
WHERE state = 'VT'
""")

# WHERE
ps.sqldf("""
SELECT date
FROM df_shootings 
WHERE state = 'VT'
""")

ps.sqldf("""
SELECT *
FROM df_shootings 
WHERE state = 'KS' AND ( 
              arms_category = 'Unarmed' 
              OR 
              arms_category = 'Other unusual objects'
              )
""")


# ORDER BY
ps.sqldf("""
SELECT *
FROM df_shootings 
ORDER BY age DESC, 
         date DESC
LIMIT 10
""")


# GROUP BY
ps.sqldf("""
SELECT gender,
       COUNT(*)
FROM df_shootings 
GROUP BY gender
""")

ps.sqldf("""
SELECT date,
       COUNT(*)
FROM df_shootings 
GROUP BY date
ORDER BY COUNT(*) DESC
LIMIT 5
""")


# MIN AVG MAX
ps.sqldf("""
SELECT threat_level,
       MIN(age),
       AVG(age), 
       MAX(age)
FROM df_shootings 
GROUP BY threat_level
""")

# HAVING
ps.sqldf("""
SELECT race,
       flee,
       COUNT(*)
FROM df_shootings 
GROUP BY race, 
         flee
HAVING COUNT(*) > 75
ORDER BY flee, race
""")


# CAST SUM AS
ps.sqldf("""
SELECT race,
       CAST(SUM(signs_of_mental_illness) AS FLOAT) / COUNT(*) AS ratio,
       SUM(signs_of_mental_illness) AS signs_of_mental_illness_count,
       COUNT(*) AS total_count
FROM df_shootings 
GROUP BY race
ORDER BY ratio DESC
""")


# STRFTIME
query46 = ps.sqldf("""
SELECT STRFTIME('%Y', date) AS year,
       flee,
       CAST(SUM(signs_of_mental_illness) AS FLOAT) / COUNT(*) AS ratio
FROM df_shootings 
GROUP BY year, flee
ORDER BY year ASC
""")

query46


# pivot_table plot
ax = query46.pivot_table(index='year', columns='flee', values='ratio') \
            .plot(title='Ratio of mental illness over time by flee method', figsize=(16, 4))


# INNER JOIN
ps.sqldf("""
SELECT region,
       COUNT(*)
FROM df_shootings 
INNER JOIN df_states ON df_shootings.state = df_states.state
GROUP BY region
""")

# INNER JOIN WHERE LIKE OR
ps.sqldf("""
SELECT state_long,
       COUNT(*)
FROM df_shootings 
INNER JOIN df_states ON df_shootings.state = df_states.state
WHERE df_states.state_long LIKE 'New%'
OR df_states.state_long LIKE 'North%'
GROUP BY df_states.state
""")


# INNER JOIN USING 
ps.sqldf("""
SELECT city,
       df_states.state_long,
       COUNT(*)
FROM df_shootings 
INNER JOIN df_states USING (state)
WHERE df_states.region = 'south'
GROUP BY city
ORDER BY COUNT(*) DESC
LIMIT 10
""")


# CASE
ps.sqldf("""
SELECT COUNT(*),
       CASE 
              WHEN STRFTIME('%m', date) < '07' THEN 'first'
              ELSE 'last'
       END first_or_last
FROM df_shootings
GROUP BY first_or_last
""")



# WITH CROSS JOIN
ps.sqldf("""
WITH first_six AS (
       SELECT COUNT(*) AS count,
              STRFTIME('%m', date) AS month
       FROM df_shootings
       WHERE month < '07'
), 
last_six AS (
       SELECT COUNT(*) AS count,
              STRFTIME('%m', date) AS month
       FROM df_shootings
       WHERE month >= '07'
)

SELECT first_six.count AS first_six_count,
       last_six.count AS last_six_count

FROM first_six CROSS JOIN last_six
""")


# IN ROUND ||
ps.sqldf("""
WITH first_three AS (
       SELECT state,
              COUNT(*) AS count,
              AVG(age) AS avg_age
       FROM df_shootings
       WHERE STRFTIME('%Y', date) IN ('2015', '2016', '2017')
       GROUP BY state
), 
last_three AS (
       SELECT state,
              COUNT(*) AS count,
              AVG(age) AS avg_age
       FROM df_shootings
       WHERE STRFTIME('%Y', date) IN ('2018', '2019', '2020')
       GROUP BY state
)

SELECT state_long,
       ROUND(first_three.avg_age, 2) || ' (' || first_three.count || ')' AS '2015 to 2017',
       ROUND(last_three.avg_age, 2) || ' (' || last_three.count || ')' AS '2018 to 2020'
FROM first_three

INNER JOIN last_three USING (state)

INNER JOIN df_states USING (state)

WHERE first_three.count > 40 AND last_three.count > 40

ORDER BY state
""")

clean_transactions\
       .groupBy(['company', 'year', 'quarter'])\
       .agg(f.sum('price').alias('purchases'))