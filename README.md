**Machine learning model**

**Background**

The dataset used for this project contains data on 10,000 customers, with 19 different variables for each customer, 
such as age, gender, education level, marital status, months on book, etc. Using this data, I created a machine learning model 
to predict if a customer is likely to leave the bank in the future. This was achieved using the RandomForestClassifier, 
leveraging the feature_importances_ attribute to identify the most significant features. The final model utilizes the 
top five variables identified as the most important.

In the second Python file, the results are displayed through an interactive dashboard. This dashboard features a 
dropdown menu where users can choose which variable to compare between customers who have left the bank and those who 
have stayed. It also presents a classification report for the model and displays the model's accuracy on the provided dataset.

**Results**

The model achieved a 95% accuracy on the test data, which can be viewed as a very good result and indicates that the model is 
highly accurate. Additionally, the dashboard and its graphs can be used to conduct further analysis on the data. 

![Alt text](./LINC_Project1.png/image.png)

The classification report provides a more in-depth view of the results, offering additional insights into the model's performance.

![Alt text](./LINC_Project2.png/image.png)

**How to use**

To use the dashboard there is a url printed in the terminal, which can be copied into the browser in order to display it. 
