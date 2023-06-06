#!/usr/bin/env python
# coding: utf-8

# # Prediction of The Mortality from Heart Failure

# # Background and Aim
# Heart failure is a disease greatly and negatively impact the quality of life and its prevailation and death rate is gradually increasing year by year. This project aims to predict the mortality from heart failure and extract its most important associative factors through multiple classificational algorithms.
# 

# # Data Collection
# 

# In[1]:


# import what we need here
import numpy as np
import pandas as pd
import os


# In[2]:


# the data source
# PLEASE put the corresponding csv dataset into the root directory of the colab!!! The file will be deleted everytime here!!!
# the corresponding file is available at https://www.kaggle.com/datasets/andrewmvd/heart-failure-clinical-data
df = pd.read_csv('heart_failure_clinical_records_dataset.csv')


# In[3]:


# explore data
df.head()


# In[4]:


# see the completness and more of this dataframe
df.info()
# there are only 299 records in this dataset


# In[5]:


# check the unique value of each feature
pd.set_option('display.max_rows', None) # in case if there are too many features
df.nunique()

# With the consideration of the clinical data property, it can be inferred that anaemia, diabetes, high blood pressure, sex, smoking, 
# and DEATH_EVENT are categorical variables 


# Sex - Gender of patient Male = 1, Female =0
# 
# Age - Age of patient
# 
# Diabetes - 0 = No, 1 = Yes
# 
# Anaemia - 0 = No, 1 = Yes
# 
# High_blood_pressure - 0 = No, 1 = Yes
# 
# Smoking - 0 = No, 1 = Yes
# 
# DEATH_EVENT - 0 = No, 1 = Yes
# 
# *Reference: https://www.kaggle.com/datasets/andrewmvd/heart-failure-clinical-data/discussion/181241, by Yankun Song*

# In[6]:


# check missing value
df.isnull().sum() # very lucky to have no missing value here


# In[7]:


# check duplicated record
df.duplicated().where(df.duplicated() != False).count()
# no duplication


# In[8]:


# get target variable
y = df['DEATH_EVENT']


# In[9]:


# descriptive statistics of the continuous variables
df[['age', 'creatinine_phosphokinase', 'ejection_fraction', 'platelets', 'serum_creatinine', 'serum_sodium']].describe()
# The reason why I did not include time is, the time stands for following up period
# It seems to be not sensible to say that "the longer we follow-up, the greater the possiblity that patient will pass away from HF"
# the median ejection fraction is only 38 (%), indicating that at least half of the patient are currently suffering from HF


# In[10]:


# visulization of the data
import matplotlib.pyplot as plt
import seaborn as sns


# In[11]:


_,axss = plt.subplots(3,2, figsize=[20,30]) # set canvas
sns.boxplot(x='DEATH_EVENT', y ='age', data=df, ax=axss[0][0])
sns.boxplot(x='DEATH_EVENT', y ='creatinine_phosphokinase', data=df, ax=axss[0][1])
sns.boxplot(x='DEATH_EVENT', y ='ejection_fraction', data=df, ax=axss[1][0])
sns.boxplot(x='DEATH_EVENT', y ='platelets', data=df, ax=axss[1][1])
sns.boxplot(x='DEATH_EVENT', y ='serum_creatinine', data=df, ax=axss[2][0])
sns.boxplot(x='DEATH_EVENT', y ='serum_sodium', data=df, ax=axss[2][1])



# In[12]:


# get statistic value of the numerical variables above
import scipy.stats as st

# t-test for age, and platelets
ttest_features = ['age', 'platelets']
i = 1

for feature in ttest_features:
  statistic, pvalue = st.ttest_ind(df[feature].where(df['DEATH_EVENT'] == 1).dropna(), 
                                   df[feature].where(df['DEATH_EVENT'] == 0).dropna(), equal_var=True)
  print(f'{i}. ', f'For {feature}, t = {round(statistic, 3)}, p-value = {round(pvalue, 3)}')
  print(f'{feature} is statistically significantly different between two groups\n') if pvalue < 0.05\
  else print(f'{feature} is NOT statistically significantly different between two groups\n')
  i += 1

# ranked test for other continuous vars ('creatinine_phosphokinase', 'ejection_fraction', 'serum_creatinine', 'serum_sodium')
ranked_test_features = ['creatinine_phosphokinase', 'ejection_fraction', 'serum_creatinine', 'serum_sodium']

i = 3

for feature in ranked_test_features:
  statistic, pvalue = st.mannwhitneyu(df[feature].where(df['DEATH_EVENT'] == 1).dropna(), 
                                df[feature].where(df['DEATH_EVENT'] == 0).dropna())
  print(f'{i}. ', f'For {feature}, U = {round(statistic, 3)}, p-value = {round(pvalue, 3)}')
  print(f'{feature} is statistically significantly different between two groups\n') if pvalue < 0.05\
  else print(f'{feature} is NOT statistically significantly different between two groups\n')
  i += 1

# It a bit messy, yet according to the statistic results, age, ejection_fraction, serium_creatine, and serum_sodium are possible viable
# feature for model, yet platelets and creatinine phosphokinase are possibly not

# therefore, we might drop 'platelets', 'creatinine_phosphokinase', 'time' before model training



# In[13]:


# Description of Categorical Feature (anaemia, diabetes, high blood pressure, sex, smoking)
_,axss = plt.subplots(3,2, figsize=[30,20])

sns.countplot(x='DEATH_EVENT', hue='anaemia', data=df, ax=axss[0][0])
sns.countplot(x='DEATH_EVENT', hue='diabetes', data=df, ax=axss[0][1])
sns.countplot(x='DEATH_EVENT', hue='high_blood_pressure', data=df, ax=axss[1][0])
sns.countplot(x='DEATH_EVENT', hue='sex', data=df, ax=axss[1][1])
sns.countplot(x='DEATH_EVENT', hue='smoking', data=df, ax=axss[2][0])

# Seeming that only anaemia, and hypertension are relevant to the mortality from HF


# In[14]:


# chi-square test for categorical vars
cat_vars = ['anaemia', 'diabetes','high_blood_pressure','sex','smoking']

i = 1
for feature in cat_vars:
  # establish a contigency table
  contigency = pd.crosstab(df[feature],df['DEATH_EVENT'])
  print(contigency) # see if there is anything wrong with the contigency table
  statistic, pvalue, dof, expected_freq = st.chi2_contingency(contigency)
  print(f'{i}. For {feature}, chi = {round(statistic, 3)}, p-value = {round(pvalue, 3)}')
  # statistically sig or not if alpha = 0.05
  if pvalue < 0.05:
    print(f'{feature} distribution is statistically significant between two outcomes.\n')
  else:
    print(f'{feature} distribution is NOT statistically significant between two outcomes.\n')
  i += 1

# Therefore, this project will drop 'diabetes','smoking'


# # Feature Preprocessing

# In[15]:


# Drop useless features
to_drop = ['platelets', 'creatinine_phosphokinase', 'time', 'DEATH_EVENT', 'diabetes', 'smoking'] # don't forget the outcome!!!
x = df.drop(to_drop, axis = 1)


# In[16]:


x.head()


# In[49]:


# change categorical vars into objects
cat_cols = ['anaemia', 'high_blood_pressure','sex']

for cat in cat_cols:
  x[cat] = x[cat].astype('object')

x.info()


# In[18]:


# for binary variables, ordinary encoder is enough
from sklearn.preprocessing import OrdinalEncoder

enc_oe = OrdinalEncoder()

for cat in cat_cols:
  enc_oe.fit(x[[cat]])
  x[[cat]] = enc_oe.transform(x[[cat]])

x.head()


# In[19]:


# standarize continuous data
from sklearn.preprocessing import StandardScaler
num_cols = ['age', 'ejection_fraction', 'serum_creatinine', 'serum_sodium']
scaler = StandardScaler()
scaler.fit(x[num_cols])

x[num_cols] = scaler.transform(x[num_cols])
x.head()


# # Model Training & Evaluation

# In[20]:


# Train-Test Split
from sklearn import model_selection

# Reserve 20% for testing
# stratify example:
# 100 -> y: 80 '0', 20 '1' -> 4:1
# 80% training 64: '0', 16:'1' -> 4:1
# 20% testing  16:'0', 4: '1' -> 4:1
x_train, x_test, y_train, y_test = model_selection.train_test_split(x, y, test_size=0.20, stratify = y, random_state = 1) #stratified sampling
# the dataset is relatively small, hence this project use 20% data on testing

print('training data has ' + str(x_train.shape[0]) + ' observation with ' + str(x_train.shape[1]) + ' features')
print('test data has ' + str(x_test.shape[0]) + ' observation with ' + str(x_test.shape[1]) + ' features')


# In[22]:


#@title build models
# There are three models we are going to use during this project
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier 
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import GradientBoostingClassifier

# This is for confusion matrix
from sklearn import metrics, model_selection 


# Logistic Regression
classifier_logistic = LogisticRegression()

# K Nearest Neighbors
classifier_KNN = KNeighborsClassifier()

# Random Forest
classifier_RF = RandomForestClassifier()

# Support Vector Classification
classifier_SVC = SVC(probability=True)

# GB classifier
classifier_GB = GradientBoostingClassifier()


# Logistic Regressional Classifier

# In[23]:


#@title Logistic Regressional Classifier & evaluation (by default)
classifier_logistic.fit(x_train, y_train) # train model
y_predict = classifier_logistic.predict(x_train) # predict results

# too stochastic, so I don't use point estimation to measure such a result
# res_1 = classifier_logistic.score(x_train, y_train)
# print(f'The acc for logistic classifier is {round(res_1 * 100, 3)}%')  

# cross validation
scores = model_selection.cross_val_score(classifier_logistic, x_train, y_train, cv = 10)
print(f'For Logistic Regressional Classifier, the acc is {round(scores.mean() * 100, 2)} \
  ({round(scores.mean() * 100 - scores.std() * 100 * 1.96, 2)}\
  ~ {round(scores.mean() * 100, 2) + round(scores.std() * 100 * 1.96, 2)}) %')

# Confusion Matrix
cm = metrics.confusion_matrix(y_train, y_predict)
plt.matshow(cm)
plt.colorbar()
plt.ylabel('True label')
plt.xlabel('Predicted label')
plt.show()

print(metrics.classification_report(y_train, y_predict))


# In[24]:


#@title KNN Classifier
classifier_KNN.fit(x_train, y_train) # train model
y_predict = classifier_KNN.predict(x_train) # predict results

# cross validation
scores = model_selection.cross_val_score(classifier_KNN, x_train, y_train, cv = 10)
print(f'For KNN, the acc is {round(scores.mean() * 100, 2)} \
  ({round(scores.mean() * 100 - scores.std() * 100 * 1.96, 2)}\
  ~ {round(scores.mean() * 100, 2) + round(scores.std() * 100 * 1.96, 2)}) %')

# Confusion Matrix
cm = metrics.confusion_matrix(y_train, y_predict)
plt.matshow(cm)
plt.colorbar()
plt.ylabel('True label')
plt.xlabel('Predicted label')
plt.show()

print(metrics.classification_report(y_train, y_predict))


# Random Forest

# In[25]:


#@title Random Forest
classifier_RF.fit(x_train, y_train) # train model
y_predict = classifier_RF.predict(x_train) # predict results

# cross validation
scores = model_selection.cross_val_score(classifier_RF, x_train, y_train, cv = 10)
print(f'For RF, the acc is {round(scores.mean() * 100, 2)} \
  ({round(scores.mean() * 100 - scores.std() * 100 * 1.96, 2)}\
  ~ {round(scores.mean() * 100, 2) + round(scores.std() * 100 * 1.96, 2)}) %')

# Confusion Matrix
cm = metrics.confusion_matrix(y_train, y_predict)
plt.matshow(cm)
plt.colorbar()
plt.ylabel('True label')
plt.xlabel('Predicted label')
plt.show()

print(metrics.classification_report(y_train, y_predict))

# It is all correct in training dataset, is that overfitting?


# SVC

# In[26]:


#@title SVC
classifier_SVC.fit(x_train, y_train) # train model
y_predict = classifier_SVC.predict(x_train) # predict results

# cross validation
scores = model_selection.cross_val_score(classifier_SVC, x_train, y_train, cv = 10)
print(f'For SVC, the acc is {round(scores.mean() * 100, 2)} \
  ({round(scores.mean() * 100 - scores.std() * 100 * 1.96, 2)}\
  ~ {round(scores.mean() * 100, 2) + round(scores.std() * 100 * 1.96, 2)}) %')

# Confusion Matrix
cm = metrics.confusion_matrix(y_train, y_predict)
plt.matshow(cm)
plt.colorbar()
plt.ylabel('True label')
plt.xlabel('Predicted label')
plt.show()

print(metrics.classification_report(y_train, y_predict))



# GB Classifier

# In[27]:


#@title GB Classifier
classifier_GB.fit(x_train, y_train) # train model
y_predict = classifier_GB.predict(x_train) # predict results

# cross validation
scores = model_selection.cross_val_score(classifier_GB, x_train, y_train, cv = 10)
print(f'For GB Classifier, the acc is {round(scores.mean() * 100, 2)} \
  ({round(scores.mean() * 100 - scores.std() * 100 * 1.96, 2)}\
  ~ {round(scores.mean() * 100, 2) + round(scores.std() * 100 * 1.96, 2)}) %')

# Confusion Matrix
cm = metrics.confusion_matrix(y_train, y_predict)
plt.matshow(cm)
plt.colorbar()
plt.ylabel('True label')
plt.xlabel('Predicted label')
plt.show()

print(metrics.classification_report(y_train, y_predict))


# Optimize Hyperparameters

# In[28]:


#@title Prelude
from sklearn.model_selection import GridSearchCV

# helper function for printing out grid search results 
def print_grid_search_metrics(gs):
    print ("Best score: " + str(gs.best_score_))
    print ("Best parameters set:")
    best_parameters = gs.best_params_
    for param_name in sorted(best_parameters.keys()):
        print(param_name + ':' + str(best_parameters[param_name]))


# Model 1 - Logistic Regression

# In[29]:


#@title Logistic Regression Optimization
parameters = {
    'penalty':('l2','l1'), 
    'C':(0.015, 0.016, 0.017, 0.018, 0.019, 0.02, 0.025)
}
Grid_LR = GridSearchCV(LogisticRegression(solver='liblinear'),parameters, cv = 10)
Grid_LR.fit(x_train, y_train)

# the best hyperparameter combination
# C = 1/lambda
print_grid_search_metrics(Grid_LR)   # C:(around) 0.19, penality, l2


# In[30]:


# Use the LR model with the "best" parameter
best_LR_model = Grid_LR.best_estimator_

best_LR_model.predict(x_test)

print('The test acc of the "best" model for logistic regression is', best_LR_model.score(x_test, y_test) * 100, '%')

# mapping the relationship between each parameter and the corresponding acc
LR_models = pd.DataFrame(Grid_LR.cv_results_)
res = (LR_models.pivot(index='param_penalty', columns='param_C', values='mean_test_score')
            )
_ = sns.heatmap(res, cmap='viridis')


# Model 2 - KNN Model

# In[31]:


#@title Find the optimal hyperparameter of KNN model
# Choose k and more
parameters = {
    'n_neighbors':[7,8,9,10,11,12,13,14,15],
    'weights':['uniform', 'distance'],
    'leaf_size':[1,2,3,4,5,6,7],
}
Grid_KNN = GridSearchCV(KNeighborsClassifier(),parameters, cv=10)
Grid_KNN.fit(x_train, y_train)

# the best hyperparameter combination
print_grid_search_metrics(Grid_KNN)  # n_neighbours: 13, leaf_size:1, weights:uniform


# In[32]:


best_KNN_model = Grid_KNN.best_estimator_

best_KNN_model.predict(x_test)

print('The test acc of the "best" model for KNN is', best_KNN_model.score(x_test, y_test) * 100, '%')


# Model 3 - RF

# In[33]:


#@title Find the optimal hyperparameter of RF
# Possible hyperparamter options for Random Forest
# Choose the number of trees
parameters = {
    'n_estimators' : [65,66,67,68,69,70,71,72,73,74],
    'max_depth': [11,12,13,14]
}
Grid_RF = GridSearchCV(RandomForestClassifier(),parameters, cv=5)
Grid_RF.fit(x_train, y_train)

# the best hyperparameter combination
print_grid_search_metrics(Grid_RF)  # n_estimators:70, max_depth: 11


# In[34]:


best_RF_model = Grid_RF.best_estimator_

best_RF_model.predict(x_test)

print('The test acc of the "best" model for RF is', best_RF_model.score(x_test, y_test) * 100, '%')


# Model 4 - SVC

# In[35]:


#@title Find the optimal hyperparameter of SVC
# Possible hyperparamter options for SVC
parameters = {
    'C' : [0.5, 0.6, 0.7, 0.8, 0.9, 1.0],
    'degree': [0,1,2,3,4,5,6],
}
Grid_SVC = GridSearchCV(SVC(probability = True), parameters, cv=5)
Grid_SVC.fit(x_train, y_train)

# the best hyperparameter combination
print_grid_search_metrics(Grid_SVC)  # C: 0.7, degree:0


# In[36]:


best_SVC_model = Grid_SVC.best_estimator_

best_SVC_model.predict(x_test)

print('The test acc of the "best" model for SVC is', best_SVC_model.score(x_test, y_test) * 100, '%')


# Model 5 - GB Classifier

# In[37]:


#@title Find the optimal hyperparameter of GB Classifier
# Possible hyperparamter options for GB Classifier
parameters = {
    'learning_rate' : [0.1, 0.2, 0.3],
    'n_estimators': [20, 30, 40, 50],
    'subsample': [0.7],
    'min_samples_split':[1,9, 2, 2.1]
}
Grid_GB = GridSearchCV(GradientBoostingClassifier(), parameters, cv=10)
Grid_GB.fit(x_train, y_train)

# the best hyperparameter combination
print_grid_search_metrics(Grid_GB)  # learning_rate:0.3 min_samples_split:2 n_estimators:20 subsample:0.7


# In[38]:


best_GB_model = Grid_GB.best_estimator_

best_GB_model.predict(x_test)

print('The test acc of the "best" model for GB classifier is', best_GB_model.score(x_test, y_test) * 100, '%')


# ## Model Evaluation - Confusion Matrix (Precision, Recall, Accuracy)
# **Precision**(PPV, positive predictive value): tp / (tp + fp);
# High Precision means low fp
# 
# **Recall**(sensitivity, hit rate, true positive rate): tp / (tp + fn)

# In[39]:


from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score

# calculate accuracy, precision and recall, [[tn, fp],[]]
def cal_evaluation(classifier, cm):
    tn = cm[0][0]
    fp = cm[0][1]
    fn = cm[1][0]
    tp = cm[1][1]
    accuracy  = (tp + tn) / (tp + fp + fn + tn + 0.0)
    precision = tp / (tp + fp + 0.0)
    recall = tp / (tp + fn + 0.0)
    print (classifier)
    print ("Accuracy is: " + str(accuracy))
    print ("precision is: " + str(precision))
    print ("recall is: " + str(recall))
    print ()

# print out confusion matrices
def draw_confusion_matrices(confusion_matricies):
    class_names = ['Not','Churn']
    for cm in confusion_matrices:
        classifier, cm = cm[0], cm[1]
        cal_evaluation(classifier, cm)


# In[40]:


confusion_matrices = [
    ("Random Forest", confusion_matrix(y_test,best_RF_model.predict(x_test))),
    ("Logistic Regression", confusion_matrix(y_test,best_LR_model.predict(x_test))),
    ("K nearest neighbor", confusion_matrix(y_test, best_KNN_model.predict(x_test))),
    ("SVC", confusion_matrix(y_test, best_SVC_model.predict(x_test))),
    ('GB Classifier', confusion_matrix(y_test, best_GB_model.predict(x_test)))
]

draw_confusion_matrices(confusion_matrices)


# ### Model Evaluation - ROC & AUC
# 
# **All the classifier used here have predict_prob() function, generating the corresponding prediction probability of the classification as category "1"**

# In[41]:


from sklearn.metrics import roc_curve
from sklearn import metrics
import matplotlib.pyplot as plt
from sklearn import metrics


# ROC of Random Forest

# In[42]:


# Use predict_proba to get the probability results of Random Forest
y_pred_rf = best_RF_model.predict_proba(x_test)[:, 1]
fpr_rf, tpr_rf, _ = roc_curve(y_test, y_pred_rf)

# drawing ROC curve
plt.figure(1)
plt.plot([0, 1], [0, 1], 'k--')
plt.plot(fpr_rf, tpr_rf, label='RF')
plt.xlabel('False positive rate')
plt.ylabel('True positive rate')
plt.title('ROC curve - RF model')
plt.legend(loc='best')
plt.show()

# AUC
print('The AUC of RF model is', metrics.auc(fpr_rf,tpr_rf))


# AUC for Logistic Regression Model

# In[43]:


# Use predict_proba to get the probability results of LR
y_pred_lr = best_LR_model.predict_proba(x_test)[:, 1]
fpr_lr, tpr_lr, _ = roc_curve(y_test, y_pred_lr)

# drawing ROC curve
plt.figure(1)
plt.plot([0, 1], [0, 1], 'k--')
plt.plot(fpr_lr, tpr_lr, label='LR')
plt.xlabel('False positive rate')
plt.ylabel('True positive rate')
plt.title('ROC curve - LR model')
plt.legend(loc='best')
plt.show()

# AUC
print('The AUC of LR model is', metrics.auc(fpr_lr,tpr_lr))


# AUC for KNN

# In[44]:


# Use predict_proba to get the probability results of KNN
y_pred_knn = best_KNN_model.predict_proba(x_test)[:, 1]
fpr_knn, tpr_knn, _ = roc_curve(y_test, y_pred_knn)

# drawing ROC curve
plt.figure(1)
plt.plot([0, 1], [0, 1], 'k--')
plt.plot(fpr_knn, tpr_knn, label='KNN')
plt.xlabel('False positive rate')
plt.ylabel('True positive rate')
plt.title('ROC curve - KNN model')
plt.legend(loc='best')
plt.show()

# AUC
print('The AUC of KNN model is', metrics.auc(fpr_knn,tpr_knn))


# AUC for SVC

# In[45]:


# Use predict_proba to get the probability results of SVC
y_pred_svc = best_SVC_model.predict_proba(x_test)[:, 1]
fpr_svc, tpr_svc, _ = roc_curve(y_test, y_pred_svc)

# drawing ROC curve
plt.figure(1)
plt.plot([0, 1], [0, 1], 'k--')
plt.plot(fpr_svc, tpr_svc, label='SVC')
plt.xlabel('False positive rate')
plt.ylabel('True positive rate')
plt.title('ROC curve - SVC model')
plt.legend(loc='best')
plt.show()

# AUC
print('The AUC of SVC model is', metrics.auc(fpr_svc,tpr_svc))


# AUC for GB Classifier

# In[46]:


# Use predict_proba to get the probability results of GB Classifier
y_pred_gb = best_GB_model.predict_proba(x_test)[:, 1]
fpr_gb, tpr_gb, _ = roc_curve(y_test, y_pred_gb)

# drawing ROC curve
plt.figure(1)
plt.plot([0, 1], [0, 1], 'k--')
plt.plot(fpr_gb, tpr_gb, label='GB')
plt.xlabel('False positive rate')
plt.ylabel('True positive rate')
plt.title('ROC curve - GB Classifier')
plt.legend(loc='best')
plt.show()

# AUC
print('The AUC of GB Classifier is', metrics.auc(fpr_gb,tpr_gb))


# **Despite relatively low acc, it seems that KNN performs relatively better when it comes to ROC (AUC = 0.71)**
# **Therefore, I decide to use KNN to explain the weight for each feature**

# ## RF - Feature Importance Discussion
# **Since the RF (2nd best model) can easily extract each feature's weight**, here we take it as example to see why the original author think **serum creatinine** and **ejection fraction** are the sole features to predict the mortality from the HF.

# In[56]:


importances = best_RF_model.feature_importances_

indices = np.argsort(importances)[::-1]

# Print the feature ranking
print("Feature importance ranking by RF:")
for ind in range(x.shape[1]):
  print ("{0} : {1}".format(x.columns[indices[ind]],round(importances[indices[ind]], 4)))


# ### Therefore, we can see that two features mentioned above are the most important factors contributed to the mortality of Heart Failure. 
# Apart from that, **Age** and **Serum Sodium** are also the important contributing factor to HF

# ## Discussion
# 
# 1. This project used multiple classifier models to predict the mortality from HF through limited but relatively clean data. 
# 
# 
# 2. In terms of AUC, the K-Nearest-Neighbour is the best amongst 5 models used above (Logistic Regression, Random Forest Classifier, K-nearest-neighbour, Support Vector Classification, GB Classifier). Yet it may require more data to obtain a more accurrate result and the usability of the model
# 
# 3. Clinically, 1) serum creatinine reflects kidney function, which affects blood pressure (heart workload) & heart function and may vise versa. 2) ejection fraction is the proportion of the blood with in the heart that could be pumped from the heart per blood ejection within heart, which is also a direct index to classify the heart failure grade (usually chronic) according to the New York Criteria. Apart from which listed in the original article, 3) seniority is often the direct risk factor associate with the mortality of HF, 4) high serum sodium often leads to blood pressure, leading to the high heart workload and left ventricular hypertrophy.
# 
# 4. Limitation: 1) due to the limited size of the data, including N of record and the limited fundamental representative of the feature, the accurracy of the model is limited. 2) It seems that the performance of the model other than logistic regression is not significantly higher than that of LR, this might be the reason why clinicians prefer to use traditional regression models to observe risk factors since their datasets are usually limited, 3) As a future scope, there could be several explainable classifier models worth exploring.
# 
# 

# # Insight
# 
#   **For patient with heart failure, apart from controlling the process of HF by cardiological medicine, the maintainance of kidney function and low intake of sodium is also important to prolong the expected lifespan. **

# In[ ]:




