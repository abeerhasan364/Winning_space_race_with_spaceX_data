
# SpaceX Data Analysis and Machine Learning Combined Script

# --- Section: SpaceX Folium Map ---
import piplite
await piplite.install(['folium'])
await piplite.install(['pandas'])
import folium
import pandas as pd
# Import folium MarkerCluster plugin
from folium.plugins import MarkerCluster
# Import folium MousePosition plugin
from folium.plugins import MousePosition
# Import folium DivIcon plugin
from folium.features import DivIcon
# Download and read the `spacex_launch_geo.csv`
from js import fetch
import io

URL = 'https://cf-courses-data.s3.us.cloud-object-storage.appdomain.cloud/IBM-DS0321EN-SkillsNetwork/datasets/spacex_launch_geo.csv'
resp = await fetch(URL)
spacex_csv_file = io.BytesIO((await resp.arrayBuffer()).to_py())
spacex_df=pd.read_csv(spacex_csv_file)
# Select relevant sub-columns: `Launch Site`, `Lat(Latitude)`, `Long(Longitude)`, `class`
spacex_df = spacex_df[['Launch Site', 'Lat', 'Long', 'class']]
launch_sites_df = spacex_df.groupby(['Launch Site'], as_index=False).first()
launch_sites_df = launch_sites_df[['Launch Site', 'Lat', 'Long']]
launch_sites_df
# Start location is NASA Johnson Space Center
nasa_coordinate = [29.559684888503615, -95.0830971930759]
site_map = folium.Map(location=nasa_coordinate, zoom_start=10)
# Create a blue circle at NASA Johnson Space Center's coordinate with a popup label showing its name
circle = folium.Circle(nasa_coordinate, radius=1000, color='#d35400', fill=True).add_child(folium.Popup('NASA Johnson Space Center'))
# Create a blue circle at NASA Johnson Space Center's coordinate with a icon showing its name
marker = folium.map.Marker(
    nasa_coordinate,
    # Create an icon as a text label
    icon=DivIcon(
        icon_size=(20,20),
        icon_anchor=(0,0),
        html='<div style="font-size: 12; color:#d35400;"><b>%s</b></div>' % 'NASA JSC',
        )
    )
site_map.add_child(circle)
site_map.add_child(marker)
# Initial the map
site_map = folium.Map(location=nasa_coordinate, zoom_start=5)
# For each launch site, add a Circle object based on its coordinate (Lat, Long) values. In addition, add Launch site name as a popup label
ls = folium.map.FeatureGroup()
for lat, lng in zip (launch_sites_df.Lat, launch_sites_df.Long):
    ls.add_child(
        folium.vector_layers.CircleMarker(
            [lat,lng],
             radius=5, # define how big you want the circle markers to be
            color='yellow',
            fill=True,
            fill_color='blue',
            fill_opacity=0.6
        )
    )
site_map.add_child(ls) 
spacex_df["Launch Site"].unique()
marker_cluster = MarkerCluster()


# Initialize an empty list to store marker colors
marker_colors = []

# Loop through each row in the dataframe
for index, row in spacex_df.iterrows():
    # Check the value of 'class' and assign the appropriate marker color
    if row['class'] == 1:
        marker_colors.append('green')
    else:
        marker_colors.append('red')

# Add the 'marker_color' list as a new column to the dataframe
spacex_df['marker_color'] = marker_colors

# Display the first few rows to verify
spacex_df.head()

# Add marker_cluster to current site_map
site_map.add_child(marker_cluster)


# Loop through each row in the spacex_df DataFrame
for index, record in spacex_df.iterrows():
    # Create a folium Marker with the appropriate icon color based on the 'marker_color' column
    marker = folium.Marker(
        location=[record['Lat'], record['Long']],  # Coordinates of the launch site
        icon=folium.Icon(color='white', icon_color=record['marker_color']),  # Customize the marker icon color
        popup=f"Launch Site: {record['Launch Site']}\nSuccess: {record['class']}"  # Add a popup with details
    )
    
    # Add the marker to the marker cluster
    marker_cluster.add_child(marker)

# Display the map
site_map

# Add Mouse Position to get the coordinate (Lat, Long) for a mouse over on the map
formatter = "function(num) {return L.Util.formatNum(num, 5);};"
mouse_position = MousePosition(
    position='topright',
    separator=' Long: ',
    empty_string='NaN',
    lng_first=False,
    num_digits=20,
    prefix='Lat:',
    lat_formatter=formatter,
    lng_formatter=formatter,
)

site_map.add_child(mouse_position)
site_map
import folium
from folium.features import DivIcon
from math import sin, cos, sqrt, atan2, radians

# Function to calculate the distance between two coordinates (Haversine formula)
def calculate_distance(lat1, lon1, lat2, lon2):
    # Approximate radius of Earth in km
    R = 6373.0

    lat1 = radians(lat1)
    lon1 = radians(lon1)
    lat2 = radians(lat2)
    lon2 = radians(lon2)

    dlon = lon2 - lon1
    dlat = lat2 - lat1

    a = sin(dlat / 2)**2 + cos(lat1) * cos(lat2) * sin(dlon / 2)**2
    c = 2 * atan2(sqrt(a), sqrt(1 - a))

    distance = R * c
    return distance

# Coordinates of the closest coastline
coastline_lat = 28.56367
coastline_lon = -80.57163

# Create a map object
site_map = folium.Map(location=[28.5, -80.6], zoom_start=10)

# Create a MarkerCluster to add multiple markers
marker_cluster = folium.plugins.MarkerCluster()

# Loop through each row in spacex_df DataFrame to add markers for launch sites
for index, record in spacex_df.iterrows():
    # Create a marker for each launch site
    marker = folium.Marker(
        location=[record['Lat'], record['Long']],  # Coordinates of the launch site
        icon=folium.Icon(color='white', icon_color=record['marker_color']),  # Customize the icon
        popup=f"Launch Site: {record['Launch Site']}\nSuccess: {record['class']}"  # Popup information
    )
    marker_cluster.add_child(marker)  # Add the marker to the marker cluster

# Add the marker cluster to the map
site_map.add_child(marker_cluster)

# Calculate the distance between the launch site and the coastline
launch_lat = spacex_df.loc[0, 'Lat']  # Example launch site coordinates
launch_lon = spacex_df.loc[0, 'Long']
distance_coastline = calculate_distance(launch_lat, launch_lon, coastline_lat, coastline_lon)

# Create a marker at the coastline and display the distance to the launch site
distance_marker = folium.Marker(
    location=[coastline_lat, coastline_lon],
    icon=DivIcon(
        icon_size=(20, 20),
        icon_anchor=(0, 0),
        html='<div style="font-size: 12; color:#d35400;"><b>{:.2f} KM</b></div>'.format(distance_coastline)
    )
)
site_map.add_child(distance_marker)

# Create and add a PolyLine to represent the line between the launch site and the coastline
coordinates = [[launch_lat, launch_lon], [coastline_lat, coastline_lon]]
line = folium.PolyLine(locations=coordinates, weight=2, color='blue')
site_map.add_child(line)

# Display the map
site_map
# Example coordinates for the closest city, railway, and highway (replace with actual coordinates)
closest_city_lat = 28.3922  # Example: Orlando, FL
closest_city_lon = -81.2298

closest_railway_lat = 28.5721  # Example railway coordinates
closest_railway_lon = -80.5850

closest_highway_lat = 28.4550  # Example highway coordinates
closest_highway_lon = -80.7090

# Calculate distances to each of these locations from the launch site
distance_city = calculate_distance(launch_lat, launch_lon, closest_city_lat, closest_city_lon)
distance_railway = calculate_distance(launch_lat, launch_lon, closest_railway_lat, closest_railway_lon)
distance_highway = calculate_distance(launch_lat, launch_lon, closest_highway_lat, closest_highway_lon)

# Create a marker for the closest city and display the distance
city_marker = folium.Marker(
    location=[closest_city_lat, closest_city_lon],
    icon=DivIcon(
        icon_size=(20, 20),
        icon_anchor=(0, 0),
        html='<div style="font-size: 12; color:#2980b9;"><b>{:.2f} KM to City</b></div>'.format(distance_city)
    )
)
site_map.add_child(city_marker)

# Create a marker for the closest railway and display the distance
railway_marker = folium.Marker(
    location=[closest_railway_lat, closest_railway_lon],
    icon=DivIcon(
        icon_size=(20, 20),
        icon_anchor=(0, 0),
        html='<div style="font-size: 12; color:#e74c3c;"><b>{:.2f} KM to Railway</b></div>'.format(distance_railway)
    )
)
site_map.add_child(railway_marker)

# Create a marker for the closest highway and display the distance
highway_marker = folium.Marker(
    location=[closest_highway_lat, closest_highway_lon],
    icon=DivIcon(
        icon_size=(20, 20),
        icon_anchor=(0, 0),
        html='<div style="font-size: 12; color:#27ae60;"><b>{:.2f} KM to Highway</b></div>'.format(distance_highway)
    )
)
site_map.add_child(highway_marker)

# Draw lines between the launch site and the closest city, railway, and highway
city_line = folium.PolyLine(locations=[[launch_lat, launch_lon], [closest_city_lat, closest_city_lon]], color='blue', weight=2)
railway_line = folium.PolyLine(locations=[[launch_lat, launch_lon], [closest_railway_lat, closest_railway_lon]], color='red', weight=2)
highway_line = folium.PolyLine(locations=[[launch_lat, launch_lon], [closest_highway_lat, closest_highway_lon]], color='green', weight=2)

# Add the lines to the map
site_map.add_child(city_line)
site_map.add_child(railway_line)
site_map.add_child(highway_line)

# Display the updated map
site_map



# --- Section: SpaceX Exploratory Data Analysis with SQL ---
!pip install sqlalchemy==1.3.9

!pip install ipython-sql
%load_ext sql
import csv, sqlite3

con = sqlite3.connect("my_data1.db")
cur = con.cursor()
!pip install -q pandas
%sql sqlite:///my_data1.db
import pandas as pd
df = pd.read_csv("https://cf-courses-data.s3.us.cloud-object-storage.appdomain.cloud/IBM-DS0321EN-SkillsNetwork/labs/module_2/data/Spacex.csv")
df.to_sql("SPACEXTBL", con, if_exists='replace', index=False,method="multi")
#DROP THE TABLE IF EXISTS

%sql DROP TABLE IF EXISTS SPACEXTABLE;
%sql create table SPACEXTABLE as select * from SPACEXTBL where Date is not null
df["Launch_Site"].unique()
df1 = df[df["Launch_Site"].str.startswith("CCA")]
df1.head(5)
df2 = df[df["Customer"]=="NASA (CRS)"]
df2 = df2["PAYLOAD_MASS__KG_"].sum()
df2
df3 = df[df["Booster_Version"] == "F9 v1.1"]["PAYLOAD_MASS__KG_"].mean()
df3
df4 = df[df["Landing_Outcome"] == "Success (ground pad)"]["Date"].min()
df4

# Filter the DataFrame for specific conditions
boosters = df[
    (df["Landing_Outcome"] == "Success (drone ship)") & 
    (df["PAYLOAD_MASS__KG_"] > 4000) & 
    (df["PAYLOAD_MASS__KG_"] < 6000)
]

# Display the filtered DataFrame
boosters = boosters["Booster_Version"].tolist()
boosters
df_outcome = df["Mission_Outcome"].value_counts()
df_outcome
# Step 1: Find the maximum payload mass
max_payload_mass = df["PAYLOAD_MASS__KG_"].max()

# Step 2: Use a subquery to get booster versions with the maximum payload mass
booster_versions_max_payload = df[df["PAYLOAD_MASS__KG_"] == max_payload_mass]["Booster_Version"].unique()

# Display the result
print("Booster Versions that have carried the maximum payload mass:")
booster_versions_max_payload

import pandas as pd

# Convert Date to datetime format if it's not already
df['Date'] = pd.to_datetime(df['Date'])

# Filter for records from 2015 with failure landing outcomes on drone ships
failure_drone_ship = df[
    (df['Landing_Outcome'] == 'Failure (drone ship)') &
    (df['Date'].dt.year == 2015)
]

# Create a new column for month names
failure_drone_ship['Month'] = failure_drone_ship['Date'].dt.month_name()

# Select relevant columns
result = failure_drone_ship[['Month', 'Landing_Outcome', 'Booster_Version', 'Launch_Site']]

# Display the result
result

import pandas as pd

# Assuming your DataFrame is already loaded as 'df'

# Convert Date column to datetime if it's not already in datetime format
df['Date'] = pd.to_datetime(df['Date'])

# Step 1: Filter data between 2010-06-04 and 2017-03-20
filtered_df = df[
    (df['Date'] >= '2010-06-04') & 
    (df['Date'] <= '2017-03-20')
]

# Step 2: Count the occurrences of each landing outcome
landing_outcome_counts = filtered_df['Landing_Outcome'].value_counts()

# Step 3: Sort the counts in descending order
ranked_landing_outcomes = landing_outcome_counts.sort_values(ascending=False)

# Display the ranked landing outcomes
print(ranked_landing_outcomes)



# --- Section: SpaceX Machine Learning Prediction ---
# -*- coding: utf-8 -*-
"""spaceX_Machine Learning Prediction.ipynb

Automatically generated by Colab.

Original file is located at
    https://colab.research.google.com/drive/16ZSixHpKJBSO2Zy1cINWNlaVuy9UWdpH

# **Space X  Falcon 9 First Stage Landing Prediction**

Space X advertises Falcon 9 rocket launches on its website with a cost of 62 million dollars; other providers cost upward of 165 million dollars each, much of the savings is because Space X can reuse the first stage. Therefore if we can determine if the first stage will land, we can determine the cost of a launch. This information can be used if an alternate company wants to bid against space X for a rocket launch.

## Import Libraries and Define Auxiliary Functions
"""

!pip install numpy
!pip install pandas
!pip install seaborn

# Pandas is a software library written for the Python programming language for data manipulation and analysis.
import pandas as pd
# NumPy is a library for the Python programming language, adding support for large, multi-dimensional arrays and matrices, along with a large collection of high-level mathematical functions to operate on these arrays
import numpy as np
# Matplotlib is a plotting library for python and pyplot gives us a MatLab like plotting framework. We will use this in our plotter function to plot data.
import matplotlib.pyplot as plt
#Seaborn is a Python data visualization library based on matplotlib. It provides a high-level interface for drawing attractive and informative statistical graphics
import seaborn as sns
# Preprocessing allows us to standarsize our data
from sklearn import preprocessing
# Allows us to split our data into training and testing data
from sklearn.model_selection import train_test_split
# Allows us to test parameters of classification algorithms and find the best one
from sklearn.model_selection import GridSearchCV
# Logistic Regression classification algorithm
from sklearn.linear_model import LogisticRegression
# Support Vector Machine classification algorithm
from sklearn.svm import SVC
# Decision Tree classification algorithm
from sklearn.tree import DecisionTreeClassifier
# K Nearest Neighbors classification algorithm
from sklearn.neighbors import KNeighborsClassifier

"""This function is to plot the confusion matrix.

"""

def plot_confusion_matrix(y,y_predict):
    "this function plots the confusion matrix"
    from sklearn.metrics import confusion_matrix

    cm = confusion_matrix(y, y_predict)
    ax= plt.subplot()
    sns.heatmap(cm, annot=True, ax = ax); #annot=True to annotate cells
    ax.set_xlabel('Predicted labels')
    ax.set_ylabel('True labels')
    ax.set_title('Confusion Matrix');
    ax.xaxis.set_ticklabels(['did not land', 'land']); ax.yaxis.set_ticklabels(['did not land', 'landed'])
    plt.show()

"""## Load the dataframe

"""

import requests
import io

URL1 = "https://cf-courses-data.s3.us.cloud-object-storage.appdomain.cloud/IBM-DS0321EN-SkillsNetwork/datasets/dataset_part_2.csv"
resp1 = requests.get(URL1)
text1 = io.BytesIO(resp1.content)
data = pd.read_csv(text1)

data.head()

URL2 = 'https://cf-courses-data.s3.us.cloud-object-storage.appdomain.cloud/IBM-DS0321EN-SkillsNetwork/datasets/dataset_part_3.csv'
resp2 = requests.get(URL2)
text2 = io.BytesIO(resp2.content)
X = pd.read_csv(text2)

X.head(100)

"""## TASK  1

Create a NumPy array from the column <code>Class</code> in <code>data</code>, by applying the method <code>to_numpy()</code>  then
assign it  to the variable <code>Y</code>,make sure the output is a  Pandas series (only one bracket df\['name of  column']).
"""

Y = data["Class"].to_numpy()
data["Class"]

"""Standardize the data in <code>X</code> then reassign it to the variable  <code>X</code> using the transform provided below.

"""

# students get this
transform = preprocessing.StandardScaler()

"""We split the data into training and testing data using the  function  <code>train_test_split</code>.   The training data is divided into validation data, a second set used for training  data; then the models are trained and hyperparameters are selected using the function <code>GridSearchCV</code>.

Use the function train_test_split to split the data X and Y into training and test data. Set the parameter test_size to  0.2 and random_state to 2. The training data and test data should be assigned to the following labels.

<code>X_train, X_test, Y_train, Y_test</code>
"""

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size  = 0.2, random_state = 2)

"""we can see we only have 18 test samples.

"""

Y_test.shape

"""Create a logistic regression object  then create a  GridSearchCV object  <code>logreg_cv</code> with cv = 10.  Fit the object to find the best parameters from the dictionary <code>parameters</code>.

"""

parameters ={'C':[0.01,0.1,1],
             'penalty':['l2'],
             'solver':['lbfgs']}

parameters ={"C":[0.01,0.1,1],'penalty':['l2'], 'solver':['lbfgs']}# l1 lasso l2 ridge
lr=LogisticRegression()
logreg_cv = GridSearchCV(lr, parameters, cv = 10)
logreg_cv.fit(X_train, Y_train)

"""We output the <code>GridSearchCV</code> object for logistic regression. We display the best parameters using the data attribute <code>best_params\_</code> and the accuracy on the validation data using the data attribute <code>best_score\_</code>.

"""

print("tuned hpyerparameters :(best parameters) ",logreg_cv.best_params_)
print("accuracy :",logreg_cv.best_score_)

"""Calculate the accuracy on the test data using the method <code>score</code>:

"""

logreg_cv.score(X_test, Y_test)

"""Lets look at the confusion matrix:

"""

yhat=logreg_cv.predict(X_test)
plot_confusion_matrix(Y_test,yhat)

"""Examining the confusion matrix, we see that logistic regression can distinguish between the different classes.  We see that the problem is false positives.

Overview:

True Postive - 12 (True label is landed, Predicted label is also landed)

False Postive - 3 (True label is not landed, Predicted label is landed)

Create a support vector machine object then  create a  <code>GridSearchCV</code> object  <code>svm_cv</code> with cv = 10.  Fit the object to find the best parameters from the dictionary <code>parameters</code>.
"""

parameters = {
    'kernel': ['linear', 'rbf'],
    'C': [0.1, 1, 10],
    'gamma': ['scale', 'auto']
}
svm = SVC()

svm_cv = GridSearchCV(svm, parameters, cv=5, n_jobs=-1, scoring='f1')  # Adjust scoring if needed
svm_cv.fit(X_train, Y_train)

print("tuned hpyerparameters :(best parameters) ",svm_cv.best_params_)
print("accuracy :",svm_cv.best_score_)

"""Calculate the accuracy on the test data using the method <code>score</code>:

"""

svm_cv.score(X_test, Y_test)

"""We can plot the confusion matrix

"""

yhat=svm_cv.predict(X_test)
plot_confusion_matrix(Y_test,yhat)

"""Create a decision tree classifier object then  create a  <code>GridSearchCV</code> object  <code>tree_cv</code> with cv = 10.  Fit the object to find the best parameters from the dictionary <code>parameters</code>.

"""

parameters = {'criterion': ['gini', 'entropy'],
     'splitter': ['best', 'random'],
     'max_depth': [2*n for n in range(1,10)],
     'max_features': ['auto', 'sqrt'],
     'min_samples_leaf': [1, 2, 4],
     'min_samples_split': [2, 5, 10]}

tree = DecisionTreeClassifier()

tree_cv = GridSearchCV(tree, parameters, cv = 10)
tree_cv.fit(X_train, Y_train)

print("tuned hpyerparameters :(best parameters) ",tree_cv.best_params_)
print("accuracy :",tree_cv.best_score_)

"""Calculate the accuracy of tree_cv on the test data using the method <code>score</code>:

"""

tree_cv.score(X_test, Y_test)

"""We can plot the confusion matrix

"""

yhat = tree_cv.predict(X_test)
plot_confusion_matrix(Y_test,yhat)

"""Create a k nearest neighbors object then  create a  <code>GridSearchCV</code> object  <code>knn_cv</code> with cv = 10.  Fit the object to find the best parameters from the dictionary <code>parameters</code>.

"""

parameters = {'n_neighbors': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
              'algorithm': ['auto', 'ball_tree', 'kd_tree', 'brute'],
              'p': [1,2]}

KNN = KNeighborsClassifier()

knn_cv = GridSearchCV(KNN, parameters, cv = 10)

knn_cv.fit(X_train, Y_train)

print("tuned hpyerparameters :(best parameters) ",knn_cv.best_params_)
print("accuracy :",knn_cv.best_score_)

"""Calculate the accuracy of knn_cv on the test data using the method <code>score</code>:

"""

knn_cv.score(X_test, Y_test)

"""We can plot the confusion matrix

"""

yhat = knn_cv.predict(X_test)
plot_confusion_matrix(Y_test,yhat)

"""Find the method performs best:

"""

x = logreg_cv.score(X_test, Y_test)
y = svm_cv.score(X_test, Y_test)
z = tree_cv.score(X_test, Y_test)
w = knn_cv.score(X_test, Y_test)

models = ['Logistic Regression', 'SVM', 'Decision Tree', 'KNN']
scores = [x, y, z, w]
plt.bar(models, scores)

plt.xlabel('Model')
plt.ylabel('Score')
plt.title('Model Comparison')


plt.show()

# --- Section: SpaceX Dashboard ---
# Import necessary libraries
import pandas as pd
import dash
from dash import dcc, html
from dash.dependencies import Input, Output
import plotly.express as px

# Initialize the Dash app
app = dash.Dash(__name__)

# Load the SpaceX launch data
spacex_df = pd.read_csv("spacex_launch_dash.csv")
min_payload = spacex_df['Payload Mass (kg)'].min()
max_payload = spacex_df['Payload Mass (kg)'].max()

# App layout
app.layout = html.Div(children=[
    html.H1('SpaceX Launch Records Dashboard', style={'textAlign': 'center', 'color': '#503D36', 'font-size': 40}),
    
    # Dropdown for selecting launch site
    dcc.Dropdown(id='site-dropdown',
                 options=[
                     {'label': 'All Sites', 'value': 'ALL'},
                     {'label': 'CCAFS LC-40', 'value': 'CCAFS LC-40'},
                     {'label': 'CCAFS SLC-40', 'value': 'CCAFS SLC-40'},
                     {'label': 'KSC LC-39A', 'value': 'KSC LC-39A'},
                     {'label': 'VAFB SLC-4E', 'value': 'VAFB SLC-4E'}
                 ],
                 value='ALL',
                 placeholder="Select a Launch Site",
                 searchable=True
                 ),
    html.Br(),

    # Pie chart for launch success
    html.Div(dcc.Graph(id='success-pie-chart')),
    html.Br(),

    # RangeSlider for selecting payload range
    html.P("Payload range (Kg):"),
    dcc.RangeSlider(id='payload-slider',
                    min=0, max=10000, step=1000,
                    marks={0: '0', 2500: '2500', 5000: '5000', 7500: '7500', 10000: '10000'},
                    value=[min_payload, max_payload]),

    html.Br(),

    # Scatter chart for payload vs. success
    html.Div(dcc.Graph(id='success-payload-scatter-chart')),
])

# Callback function for updating the pie chart based on selected launch site
@app.callback(Output(component_id='success-pie-chart', component_property='figure'),
              Input(component_id='site-dropdown', component_property='value'))
def get_pie_chart(entered_site):
    filtered_df = spacex_df
    if entered_site == 'ALL':
        fig = px.pie(filtered_df, values='class', 
                     names='Launch Site', 
                     title='Total Success Launches by Site')
    else:
        filtered_df = spacex_df[spacex_df['Launch Site'] == entered_site]
        fig = px.pie(filtered_df, 
                     names='class', 
                     title=f'Success vs. Failed Launches for {entered_site}')
    return fig

# Callback function for updating the scatter chart based on selected site and payload range
@app.callback(
    Output(component_id='success-payload-scatter-chart', component_property='figure'),
    [Input(component_id='site-dropdown', component_property='value'),
     Input(component_id='payload-slider', component_property='value')]
)
def update_scatter_plot(entered_site, payload_range):
    filtered_df = spacex_df[(spacex_df['Payload Mass (kg)'] >= payload_range[0]) & 
                            (spacex_df['Payload Mass (kg)'] <= payload_range[1])]

    if entered_site == 'ALL':
        fig = px.scatter(filtered_df, x='Payload Mass (kg)', y='class', 
                         color='Booster Version Category', 
                         title='Correlation between Payload and Success for All Sites')
    else:
        filtered_df = filtered_df[filtered_df['Launch Site'] == entered_site]
        fig = px.scatter(filtered_df, x='Payload Mass (kg)', y='class', 
                         color='Booster Version Category',
                         title=f'Correlation between Payload and Success for {entered_site}')
    
    return fig

# Run the app
if __name__ == '__main__':
    app.run_server()

