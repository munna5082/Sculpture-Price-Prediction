# Sculpture-Price-Prediction

We have a dataset that contains information about people have bought art sculptures from various artists. It states out some of the characteristics of the sculpture and the people who bought them. We will be using this dataset to predict the cost of the sculpture a person has to pay.

Details of the project -
The project is divided into three parts - 
1. Building a machine learning model to predict the cost a person might incur when buying the sculpture - 
   So we start off with building machine learning model - we use a RandomForestRegressor model and train that model using artist sculpture datapoints. There is
   abundant information on the size and shape of the sculpture and it's shipping. In this project we will remove the outliers and select important features.
3. We build a web app on Django to mount the prediction model - 
   In this part of the project we will build the web application using the Django framework. We mount the model we build in the previous part of the project
4. We deploy the Django project on AWS - 
   In this part of the project we will deploy the Django application on AWS.
