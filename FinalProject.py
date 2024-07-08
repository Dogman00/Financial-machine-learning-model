#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jun  1 09:43:40 2024

@author: douglaseklund
"""

from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd 
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report

#%%

file = r'credit_card_customers.xlsx'
bank = pd.read_excel(file)

#%%

left_bank = bank[bank['Attrition_Flag'] == 'Attrited Customer']
stayed_bank = bank[bank['Attrition_Flag'] == 'Existing Customer']

#print(f'Number of customers who have left: {len(left_bank)}')
#print(f'Number of customers who have not left: {len(stayed_bank)}')

#%%

left_bank_stats= {}
stayed_bank_stats= {}


titles1 = [
    'Total Transaction Amount',
    'Total Transaction Count',
    'Total Revolving Balance on the Credit Card',
    'Total no. of products held by the customer',
    'Mean age',
    'Months on book'
    ]

titles2 = [
    'Change in Transaction Count (Q4 over Q1)',
    'Average Card Utilization Ratio',
    'Change in Transaction Amount (Q4 over Q1)'
    ]

titles3 = [
    'Marital status',
    'Gender',
    'Income Category',
    'Education Level',
    'Card Category'
    ]

titles_values1 = [
    'Total_Trans_Amt',
    'Total_Trans_Ct',
    'Total_Revolving_Bal',
    'Total_Relationship_Count',
    'Customer_Age',
    'Months_on_book'
    ]
  
titles_values2 = [  
    'Total_Ct_Chng_Q4_Q1',
    'Avg_Utilization_Ratio',
    'Total_Amt_Chng_Q4_Q1'
    ]
    
titles_values3 = [
    'Marital_Status',
    'Gender',
    'Income_Category',
    'Education_Level',
    'Card_Category'
    ]

for i in range(len(titles1)):
    left_bank_stats[titles1[i]] = left_bank[titles_values1[i]].mean().round()
    stayed_bank_stats[titles1[i]] = stayed_bank[titles_values1[i]].mean().round()


for i in range(len(titles2)):
    left_bank_stats[titles2[i]] = left_bank[titles_values2[i]].mean().round(2)
    stayed_bank_stats[titles2[i]] = stayed_bank[titles_values2[i]].mean().round(2)
    
for i in range(len(titles3)):
    left_bank_stats[titles3[i]] = left_bank[titles_values3[i]].value_counts().to_dict()
    stayed_bank_stats[titles3[i]] = stayed_bank[titles_values3[i]].value_counts().to_dict()

#%%

data = bank
data = data.dropna()

#%%

X = data.drop(columns=[
    'CLIENTNUM', 
    'Attrition_Flag',
    'Naive_Bayes_Classifier_Attrition_Flag_Card_Category_Contacts_Count_12_mon_Dependent_count_Education_Level_Months_Inactive_12_mon_1',
    'Naive_Bayes_Classifier_Attrition_Flag_Card_Category_Contacts_Count_12_mon_Dependent_count_Education_Level_Months_Inactive_12_mon_2'
    ])
X = pd.get_dummies(X)

y = data['Attrition_Flag']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

random_forest = RandomForestClassifier(random_state=42)
random_forest.fit(X_train, y_train)


feature_importances = random_forest.feature_importances_
importance_df = pd.DataFrame({'Feature': X_train.columns, 'Importance': feature_importances})
importance_df = importance_df.sort_values(by='Importance', ascending=False)

#%%

X = data.drop(columns=[
    'CLIENTNUM',
    'Attrition_Flag',
    'Customer_Age',
    'Dependent_count',
    'Marital_Status',
    'Months_on_book',
    'Months_Inactive_12_mon',
    'Contacts_Count_12_mon',
    'Credit_Limit',
    'Avg_Open_To_Buy',
    'Naive_Bayes_Classifier_Attrition_Flag_Card_Category_Contacts_Count_12_mon_Dependent_count_Education_Level_Months_Inactive_12_mon_1',
    'Naive_Bayes_Classifier_Attrition_Flag_Card_Category_Contacts_Count_12_mon_Dependent_count_Education_Level_Months_Inactive_12_mon_2'
    ])

    
X = pd.get_dummies(X)

y = data['Attrition_Flag']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

#%%

random_forest = RandomForestClassifier(random_state=42)
random_forest.fit(X_train, y_train)

accuracy_rf = random_forest.score(X_test, y_test)

y_pred = random_forest.predict(X_test)
report = classification_report(y_test, y_pred)

report_values = {
    '': ['Attrited Customer', 'Existing Customer', 'accuracy', 'macro avg', 'weighted avg'],
    'Precision': [0.91, 0.96, '', 0.93, 0.95],
    'Recall': [0.79, 0.98, '', 0.89, 0.95],
    'F1-Score': [0.85, 0.97, 0.95, 0.91, 0.95],
    'Support': [327, 1699, 2026, 2026, 2026]
    }#Taken from the report variable in order to make the table nice in dashboard

report_df = pd.DataFrame(report_values)

#%%

#Not used in the dashboard
cm = confusion_matrix(y_test, y_pred)

heatmap = sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['Left', 'Stayed'], yticklabels=['Left', 'Stayed'])


plt.figure(figsize=(10, 7))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['Stayed', 'Left'], yticklabels=['Stayed', 'Left'])
plt.title('Confusion Matrix')
plt.ylabel('True outcome')
plt.xlabel('Predicted outcome')
