**Machine learning model**

The dataset used for this project contains data on 10,000 customers, with 19 different variables for each customer, 
such as age, gender, education level, marital status, months on book, etc. Using this data, I created a machine learning model 
to predict if a customer is likely to leave the bank in the future. This was achieved using the RandomForestClassifier, 
leveraging the feature_importances_ attribute to identify the most significant features. The final model utilizes the 
top five variables identified as the most important.

In the second Python file, the results are displayed through an interactive dashboard. This dashboard features a 
dropdown menu where users can choose which variable to compare between customers who have left the bank and those who 
have stayed. It also presents a classification report for the model and displays the model's accuracy on the provided dataset.
