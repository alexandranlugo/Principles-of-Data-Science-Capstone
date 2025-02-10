#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Nov 14 14:44:23 2024

@author: alugo
"""

import random 
import numpy as np
import pandas as pd

N_number = 16745560
random.seed(N_number)
np.random.seed(N_number)

#%%
rmp_num = pd.read_csv('/Users/alugo/Downloads/rmpCapstoneNum.csv')
rmp_qual= pd.read_csv('/Users/alugo/Downloads/rmpCapstoneQual.csv')

rmp_num.head(), rmp_num.info(), rmp_qual.head(), rmp_qual.info()

#%%
#clean datasets

#rename cols
rmp_num.columns = [
    'Average Rating', 'Average Difficulty', 'Number of Ratings', 'Received Pepper','Proportion Retake',
    'Online Ratings','Male Gender','Female Gender'
    ]
rmp_qual.columns=['Major/Field','University','US State']

#drop rows w missing vals in Average Rating Col
rmp_num_clean = rmp_num.dropna(subset='Average Rating')

#set threshold for reliability of avg rating
threshold = 5
rmp_num_clean = rmp_num_clean[rmp_num_clean['Number of Ratings'] >= threshold] 

#merge num and qual datasets
merge_data = rmp_num_clean.merge(rmp_qual, left_index=True, right_index=True)
merge_data.head()

#%%
#1
import scipy.stats as stats

#separate data into male and female professors
male_ratings = merge_data[merge_data['Male Gender'] == 1]['Average Rating']
female_ratings = merge_data[merge_data['Female Gender'] == 1]['Average Rating']

#calc means for both
male_mean = round(male_ratings.mean(), 2)
female_mean = round(female_ratings.mean(), 2)

#independent t-test
t_stat, p_value = stats.ttest_ind(male_ratings, female_ratings, nan_policy='omit',equal_var=False)

print(f'average rating for male professors: {male_mean}')
print(f'average rating for female professors: {female_mean}')
print(f't-statistic: {t_stat}')
print(f'p-value: {p_value}')

#check significance at alpha 0.005
if p_value < 0.005:
    print("There's a statistically significant difference in ratings by gender.")
else:
    print("No statistically signficant difference in ratings by gender.")
    
#BOXPLOT to show the distribution of ratings for male and female professors, including medians and variability.
import seaborn as sns
import matplotlib.pyplot as plt

plt.figure(figsize=(8,6))
sns.boxplot(x=['Male'] * len(male_ratings) + ['Female'] * len(female_ratings), y=pd.concat([male_ratings, female_ratings]), palette='Set2')
plt.title('Distribution of Average Ratings by Gender')
plt.xlabel('Gender')
plt.ylabel('Average Rating')
plt.show()

#%%
#2
import statsmodels.api as sm

#remove rows w missing vals in Avg Rating and Num of Ratings
data_exp_quality = merge_data.dropna(subset=['Average Rating','Number of Ratings'])

# Cap the number of ratings at the 99th percentile
cap = data_exp_quality["Number of Ratings"].quantile(0.99)
data_capped = data_exp_quality[data_exp_quality["Number of Ratings"] <= cap]

#calc correlation between Num of Ratings and Avg Rating
corr = data_exp_quality['Number of Ratings'].corr(data_exp_quality['Average Rating'])
print(f'Correlation between number of ratings and average rating: {round(corr,2)}')

X = data_exp_quality['Number of Ratings']
y = data_exp_quality['Average Rating']

X = sm.add_constant(X)

model = sm.OLS(y, X).fit()
print(model.summary())

p_value1 = model.pvalues['Number of Ratings']
if p_value1 < 0.005:
    print('The number of ratings has a statistically significant effect on the average rating.')
else:
    print('The number of ratings does not have a statistically significant effect on the average rating.')

#scatterplot with regression line
plt.figure(figsize=(8, 6))
sns.scatterplot(x="Number of Ratings", y="Average Rating", data=data_capped, alpha=0.5)
sns.regplot(x="Number of Ratings", y="Average Rating", data=data_capped, scatter=False, color="blue")
plt.title("Relationship Between Experience (Capped Number of Ratings) and Quality")
plt.xlabel("Number of Ratings (Experience Proxy)")
plt.ylabel("Average Rating (Teaching Quality)")
plt.show()

#%%
#3
import matplotlib.pyplot as plt
import seaborn as sns

rating_difficulty = merge_data.dropna(subset=['Average Rating', 'Average Difficulty'])

#corr between Average rating and Average Difficulty
corr_rating_difficulty = rating_difficulty['Average Rating'].corr(rating_difficulty['Average Difficulty'])
print(f'Correlation between Average Rating and Average Difficulty: {round(corr_rating_difficulty,2)}')

X_1 = rating_difficulty['Average Difficulty']
y_1 = rating_difficulty['Average Rating']

X_1 = sm.add_constant(X_1)

model1 = sm.OLS(y_1, X_1).fit()
print(model1.summary())

plt.figure(figsize=(8,6))
sns.scatterplot(x='Average Difficulty', y='Average Rating', data=rating_difficulty, alpha=0.5)
sns.regplot(x='Average Difficulty', y='Average Rating', data=rating_difficulty, scatter=False, color='blue')
plt.title('Relationship betweeen Average Difficulty and Average Rating')
plt.xlabel('Average Difficulty')
plt.ylabel('Average Rating')
plt.show()

#%%
#4
#define a threshold for "a lot of online classes" - 50%?
threshold1 = 0.5
data_online_split = merge_data.dropna(subset=['Online Ratings', 'Number of Ratings','Average Rating'])
data_online_split['Proportion Online'] = data_online_split['Online Ratings'] / data_online_split['Number of Ratings']

#split data into two groups based on threshold
online_teachers = data_online_split[data_online_split['Proportion Online'] > threshold1]
not_online_teachers = data_online_split[data_online_split['Proportion Online'] <= threshold1]

#calc means
online_mean = online_teachers['Average Rating'].mean()
not_online_mean = not_online_teachers['Average Rating'].mean()

#independent t-test
t_stat1, p_val2 = stats.ttest_ind(online_teachers['Average Rating'], not_online_teachers['Average Rating'], nan_policy='omit', equal_var=False)

print(f'Average Rating for teachers teaching many online classes: {round(online_mean,2)}')
print(f'Average Rating for teachers teaching few/no online classes: {round(not_online_mean,2)}')
print(f'T-statistic: {round(t_stat1,2)}')
print(f'P-value: {round(p_val2,2)}')

if p_val2 < 0.005:
    print("There's a statistically significant difference in ratings between the two groups.")
else:    
    print("There is no statistically significant difference in ratings between the two groups.")

#new col to label groups
data_online_split['Group'] = ['Many Onlinne Classes' if x > threshold1 else 'Few/No Online Classes' for x in data_online_split['Proportion Online']]

#boxplot to compare ratings between two groups
plt.figure(figsize=(8,6))
sns.boxplot(x='Group', y='Average Rating', data=data_online_split, palette='Set2')
plt.title('Comparison of Average Ratings by Online Teaching Proportion')
plt.xlabel('Teaching Group')
plt.ylabel('Average Rating')
plt.show()

#%%
#5
rating_retake = merge_data.dropna(subset=['Average Rating','Proportion Retake'])

#calc correlation between Average Rating and Proportion Retake

corr_rating_retake = rating_retake['Average Rating'].corr(rating_retake['Proportion Retake'])
print(f'Correlation between Average Rating and Proportion Retake: {round(corr_rating_retake,2)}')

X_2 = rating_retake['Proportion Retake']
y_2 = rating_retake['Average Rating']

X_2 = sm.add_constant(X_2)

model_2 = sm.OLS(y_2, X_2).fit()
print(model_2.summary())

plt.figure(figsize=(8,6))
sns.scatterplot(x='Proportion Retake', y='Average Rating', data=rating_retake, alpha=0.5)
sns.regplot(x='Proportion Retake', y='Average Rating', data=rating_retake, scatter=False, color='blue')
plt.title('Relationship betweeen Proportion Retake and Average Rating')
plt.xlabel('Proportion Retake')
plt.ylabel('Average Rating')
plt.show()

#%%
#6
hot_ratings = merge_data[merge_data['Received Pepper'] == 1]['Average Rating']
not_hot_ratings = merge_data[merge_data['Received Pepper'] == 0]['Average Rating']

#calc means
hot_mean = hot_ratings.mean()
not_hot_mean = not_hot_ratings.mean()

#independent t-test
t_stat2, p_val3 = stats.ttest_ind(hot_ratings, not_hot_ratings, nan_policy='omit', equal_var=False)

print(f'Average Rating for Hot Teachers: {round(hot_mean,2)}')
print(f"Average Rating for Teachers who aren't Hot': {round(not_hot_mean,2)}")
print(f'T-statistic: {round(t_stat2,2)}')
print(f'P-value: {round(p_val3,2)}')

if p_val3 < 0.005:
    print("There's a statistically significant difference in ratings between hot and not hot teachers")
else:    
    print("There is no statistically significant difference in ratings between hot and not hot teachers")

#new col to label groups
merge_data['Hotness'] = merge_data['Received Pepper'].map({1:'Hot', 0:'Not Hot'})

#boxplot to compare ratings between 'hot' and 'not hot' professors
plt.figure(figsize=(8,6))
sns.boxplot(x='Hotness', y='Average Rating', data=merge_data, palette='Set2')
plt.title('Comparison of Average Ratings between "Hot" and "Not Hot" professors')
plt.xlabel('Hotness')
plt.ylabel('Average Rating')
plt.show()

#%%
#7
from sklearn.metrics import mean_squared_error

difficulty_model = merge_data.dropna(subset=['Average Rating','Average Difficulty'])

X_3 = difficulty_model['Average Difficulty']
y_3 = difficulty_model['Average Rating']

X_3 = sm.add_constant(X_3)

model_3 = sm.OLS(y_3,X_3).fit()

predictions = model_3.predict(X_3)

#calc r-sq and rmse
r_sq = model_3.rsquared
rmse = np.sqrt(mean_squared_error(y_3, predictions))

print(model_3.summary())
print("\nModel Performance:")
print(f"R-squared: {round(r_sq,2)}")
print(f"RMSE: {round(rmse,2)}")

#scatterplot with regression line
plt.figure(figsize=(8,6))
sns.scatterplot(x='Average Difficulty', y='Average Rating', data=difficulty_model, alpha=0.5)
sns.regplot(x='Average Difficulty', y='Average Rating', data=difficulty_model, scatter=False, color='blue')
plt.title('Relationship betweeen Difficulty and Rating with Regression Line')
plt.xlabel('Average Difficulty')
plt.ylabel('Average Rating')
plt.show()

#%%
#8
from statsmodels.stats.outliers_influence import variance_inflation_factor

#drop rows 
data_all_factors = merge_data.dropna(subset=["Average Rating", "Average Difficulty", "Number of Ratings", "Proportion Retake", "Online Ratings", "Male Gender", "Female Gender"])

X_4 = data_all_factors[["Average Difficulty", "Number of Ratings", "Proportion Retake", "Online Ratings", "Male Gender", "Female Gender"]]
y_4 = data_all_factors["Average Rating"]

X_4 = sm.add_constant(X_4)

model_4 = sm.OLS(y_4,X_4).fit()

predictions_2 = model_4.predict(X_4)

r_sq2 = model_4.rsquared
rmse2 = np.sqrt(mean_squared_error(y_4, predictions_2))

#variance inflation factor (VIF) for multicollinearity
vif_data = pd.DataFrame()
vif_data['Variable'] = X_4.columns
vif_data['VIF'] = [variance_inflation_factor(X_4.values, i) for i in range(X_4.shape[1])]

print(model_4.summary())
print("\nModel Performance:")
print(f"R-squared: {r_sq2}")
print(f"RMSE: {rmse2}")

print("\nVariance Inflation Factors (VIF):")
print(vif_data)

#compare 'difficulty only' model 
difficulty_model_rsq = r_sq
difficulty_model_rmse = rmse

print("\nComparison with 'Difficulty-Only' Model:")
print(f"R-squared Improvement: {r_sq2 - difficulty_model_rsq}")
print(f"RMSE Improvement: {difficulty_model_rmse - rmse2}")

#extract and standardize coefficients (excluding intercept)
coefficients = model_4.params[1:]
features = X_4.columns[1:]
standardized_coefficients = coefficients / np.std(X_4.iloc[:, 1:], axis=0)

#plot feature importance
plt.figure(figsize=(10,6))
bars = plt.bar(features, standardized_coefficients, color='skyblue')

for bar in bars:
    height = bar.get_height()
    plt.text(
        bar.get_x() + bar.get_width() / 2,  # Center the text
        height,  # Position above the bar
        f"{height:.2f}",  # Format the text with two decimals
        ha="center",  # Align horizontally
        va="bottom" if height < 0 else "top",  # Adjust vertical alignment
        fontsize=10
    )

plt.title("Feature Importance in Predicting Average Rating")
plt.xlabel("Features")
plt.ylabel("Standardized Coefficients")
plt.xticks(rotation=45)
plt.grid(axis="y", linestyle="--", alpha=0.7)
plt.tight_layout()  
plt.show()

#%%
#9
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, roc_curve, classification_report
from sklearn.utils import resample

#pre-processing
data_classification = merge_data.dropna(subset=['Average Rating','Received Pepper'])

#address class imbalance by upsampling the minority class 
pepper_yes = data_classification[data_classification['Received Pepper'] == 1]
pepper_no = data_classification[data_classification['Received Pepper'] == 0]

#upsample minority class
pepper_yes_upsampled = resample(pepper_yes, replace=True, n_samples=len(pepper_no), random_state=42)

#combine balanced class
data_balanced = pd.concat([pepper_no, pepper_yes_upsampled])

X_5 = data_balanced[['Average Rating']]
y_5 = data_balanced['Received Pepper']

#split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X_5, y_5, test_size=0.2, random_state=42, stratify=y_5)

model_5 = LogisticRegression()
model_5.fit(X_train, y_train)

y_pred_prob =   model_5.predict_proba(X_test)[:, 1]

roc_auc = roc_auc_score(y_test, y_pred_prob)
print(f'AU(ROC): {round(roc_auc,2)}')

#generate ROC Curve
fpr, tpr, thresholds = roc_curve(y_test, y_pred_prob)
plt.figure(figsize=(8,6))
plt.plot(fpr, tpr, label=f'ROC Curve (AU(ROC) = {roc_auc:.2f})')
plt.plot([0,1],[0,1],'k--',label='Random Classifier')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve')
plt.legend(loc='lower right')
plt.show()

y_pred = model_5.predict(X_test)
print('Classification Report: ')
print(classification_report(y_test, y_pred))

#%%
#10
data_classification_all = merge_data.dropna(subset=["Average Rating", "Average Difficulty", "Number of Ratings", "Proportion Retake", "Online Ratings", "Male Gender", "Female Gender", "Received Pepper"])

pepper_yes_all = data_classification_all[data_classification_all['Received Pepper'] == 1]
pepper_no_all = data_classification_all[data_classification_all['Received Pepper'] == 0]

pepper_yes_upsampled_all = resample(pepper_yes_all, replace=True, n_samples=len(pepper_no_all), random_state=42)


data_balanced_all = pd.concat([pepper_no_all, pepper_yes_upsampled_all])

X_all = data_balanced_all[["Average Rating", "Average Difficulty", "Number of Ratings", "Proportion Retake", "Online Ratings", "Male Gender", "Female Gender"]]
y_all = data_balanced_all["Received Pepper"]

X_train_all, X_test_all, y_train_all, y_test_all = train_test_split(X_all, y_all, test_size=0.2, random_state=42, stratify=y_all)

model_all = LogisticRegression(max_iter=1000)
model_all.fit(X_train_all, y_train_all)

y_pred_prob_all = model_all.predict_proba(X_test_all)[:, 1]

roc_auc_all = roc_auc_score(y_test_all, y_pred_prob_all)
print(f'AU(ROC) for all factors model: {round(roc_auc,2)}')

#generate ROC Curve
fpr_all, tpr_all, thresholds_all = roc_curve(y_test_all, y_pred_prob_all)
plt.figure(figsize=(8,6))
plt.plot(fpr_all, tpr_all, label=f"All Factors Model (AU(ROC) = {roc_auc_all:.2f}")
plt.plot([0,1],[0,1],'k--',label='Random Classifier')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve for all factors model')
plt.legend(loc='lower right')
plt.show()

y_pred_all = model_all.predict(X_test_all)
print('Classification Report for All Factors Model:')
print(classification_report(y_test_all, y_pred_all))

#compare to average rating only model
roc_auc_rating = roc_auc
print("\nComparison with 'Average Rating Only' Model:")
print(f"AU(ROC) Improvement: {roc_auc_all - roc_auc_rating}")

#%%
#EC
#link avg ratings w most common major for each university
uni_data = merge_data.groupby('University').agg({
    'Average Rating': 'mean',
    'Major/Field': lambda x: x.mode()[0]
}).reset_index()

#group by fields and calc avg prof ratings
field_ratings = uni_data.groupby('Major/Field')['Average Rating'].mean().sort_values()


#limit to top 10 and bottom 10
top_bottom_fields = pd.concat([field_ratings.head(10), field_ratings.tail(10)])

plt.figure(figsize=(10, 8))
sns.barplot(
    x=top_bottom_fields.values,
    y=top_bottom_fields.index,
    palette='coolwarm'
)
plt.title('Average Professor Ratings by Field of Study')
plt.xlabel('Average Rating')
plt.ylabel('Field of Study')
plt.tight_layout()
plt.show()





