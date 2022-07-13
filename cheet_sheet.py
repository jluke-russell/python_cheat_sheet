# Filtering a dataset 
luke_history = names.query("name == 'Luke' & year >= 1965 & year <= 1999 ")[['name','year','MD']]
#print(luke_history)

# Making a Table for Markdown
print(luke
    .head(5)
    .filter(["name", "year", "MD"])
    .to_markdown(index=False))

# Multiple names (or states) and it's graph
bible_names = names.query('name in ["Mary", "Martha", "Paul", "Peter"]  & year >= 1920  & year <= 2000')
#print(bible_names)

bible_chart = (alt.Chart(bible_names).properties(title='Popularity of Select Christian Names')
  .mark_line()
    .encode(
        x= alt.X('year', axis= alt.Axis(format="d", title="Year")),
        y= alt.Y('Total', axis = alt.Axis(title='Total')),
        color = 'name'
  )
)

# adds vertical and horizontal line 
xrule = (
    alt.Chart()
    .mark_rule(color="cyan", strokeWidth=2)
    .encode(x=alt.datum(alt.DateTime(year=2006, month="November")))
)

yrule = (
    alt.Chart().mark_rule(strokeDash=[12, 6], size=2).encode(y=alt.datum(350))
)

# puts it all together 
bible_chart + xrule + yrule

# Challenge 2 

mister = pd.Series([np.nan, 15, 22, 45, 31, np.nan, 85, 38, 129, 8000, 21, 2])
mister.median() find median first without NA
mister.mean() find mean after replacing NA


# Challenge 3 




# Challenge 4 
tidy_flights = (flights
.replace(-999, np.nan)
.replace("1500+", 1500)
.replace("n/a", np.nan).ffill()
.replace("Febuary", "Feburary")
)
tidy_flights.interpolate().reset_index()


# Challenge 5 
import pandas as pd 
import altair as alt
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn import tree
from sklearn.ensemble import GradientBoostingClassifier
from sklearn import metrics

dwellings_ml = pd.read_csv("https://github.com/byuidatascience/data4dwellings/raw/master/data-raw/dwellings_ml/dwellings_ml.csv")

X_pred = dwellings_ml.drop(dwellings_ml.filter(regex = 'basement|finbsmnt|BASEMENT').columns, axis = 1)

y_pred = dwellings_ml.basement
y_pred[y_pred > 0] = 1 


x = dwellings.filter(['livearea', 'finbsmnt', 
    'basement', 'nocars', 'numbdrm', 'numbaths', 
    'stories', 'sprice','gartype_Att','arcstyle_ONE-STORY'])
y = dwellings['before1980']

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.2, random_state = 2022)

x_train.head()
y_train.head()

# RGradient Boosting Classifier

# create the model
classifier = GradientBoostingClassifier()()

# train the model
classifier.fit(x_train, y_train)

# make predictions
y_predictions = classifier.predict(x_test)