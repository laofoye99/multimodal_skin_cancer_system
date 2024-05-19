#!/usr/bin/env python
# coding: utf-8

# # Preprocessing

# ## Import Libraries


# !pip install -q -U pandas
# !pip install -q -U matplotlib
# !pip install -q -U numpy
# !pip install -q -U seaborn
# !pip install -q -U scikit-learn
# !pip install -q -U imbalanced-learn
# !pip install -q -U Pillow
# !pip install -q -U xgboost
# !pip install -q -U lightgbm
# !pip install -q -U keras
# !pip install -q -U tensorflow
# !pip install -q -U joblib


import os
from PIL import Image

import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split

# text models
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier, StackingClassifier
from xgboost import XGBClassifier
from sklearn.neighbors import KNeighborsClassifier
from lightgbm import LGBMClassifier


# for handling imbalanced data
from collections import Counter
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import NearMiss
from imblearn.pipeline import make_pipeline
from sklearn.utils import class_weight

import joblib
import pickle

import warnings
warnings.filterwarnings('ignore', category=DeprecationWarning)


# ## Set Up

plt.rcParams['figure.figsize'] = [6,4]
cmap = mpl.colormaps['viridis']
sns.set_theme(style='whitegrid', palette='muted', font_scale=1.5)


# ## Load Data


data_path = 'G:\\HSUHK\\COM6003\\project\\archive'


# # Function

# ## Data Processing

# **three skin cancers (BCC, MEL, and SCC) and three skin disease (ACK, NEV, and SEK)**

# And we remove the "biopsed" feature, because:
# - Avoiding bias towards biopsied cases
# - Preventing data leakage from biopsy results
# - Improving model generalization to cases without biopsy data
# - Aligning the model with the intended use case of pre-biopsy diagnosis

# In[5]:


class DataProcessing:
    def __init__(self, data: pd.DataFrame, path=None):
        self.path = path
        self.data = data.copy()
        self.data = self.data.drop(columns=['biopsed'])
        self.data['patient_id'] = self.data['patient_id'].str.replace('PAT_','',regex=False).astype('int64')
        self.data = self.data.replace(['UNK','NaN'], np.nan)
        self.missing_percentage = round(self.data.isna().sum()*100/self.data.shape[0], 1)
    
    def keepNanText_6(self):
        """
        Keep the nan text in the data, 
        if the percentage of missing value is less than 10, then impute the feature using IterativeImputer,
        else if the percentage of missing value is higher than 10, then replace the nan text with 'Unknown',
        use it as a new feature.
        """
        imputer = IterativeImputer()
        enc = LabelEncoder()
        for col, percentage in self.missing_percentage.items():
            if self.data[col].dtype == 'object':
                if percentage < 10:
                    self.data[col] = enc.fit_transform(self.data[col])
                    self.data[col] = imputer.fit_transform(self.data[[col]])
                else:
                    self.data[col] = self.data[col].fillna('Unknown')
                    self.data[col] = self.data[col].astype('str')
                    self.data[col] = enc.fit_transform(self.data[col])
                    self.data[col] = self.data[col].replace(len(enc.classes_), -1).astype('int64')

                    # output the encode method
                    encoding_mapping = dict(zip(enc.classes_, range(len(enc.classes_))))
                    print(f"Feature '{col}':")
                    for category, encoding in encoding_mapping.items():
                        if category == 'Unknown':
                            print(f"{category}: -1")
                        else:
                            print(f"{category}: {encoding}")
                    print('-'*30)
            else:
                if percentage < 10:
                    self.data[col] = imputer.fit_transform(self.data[[col]])
                else:
                    self.data[col] = self.data[col].fillna(-1)
                    self.data[col] = self.data[col].astype('int64')
        
        self.data.drop(columns='img_id', inplace=True)

        return self.data
    
    def dropNanText_6(self):
        """
        Drop the rows with missing values in the data,
        if the percentage of missing value is less than 10, then impute the feature using IterativeImputer,
        else if the percentage of missing value is higher than 10, then drop the columns.
        """
        imputer = IterativeImputer()
        enc = LabelEncoder()
        for col, percentage in self.missing_percentage.items():
            if self.data[col].dtype == 'object':
                if percentage < 10:
                    self.data[col] = enc.fit_transform(self.data[col])
                    self.data[col] = imputer.fit_transform(self.data[[col]])
                else:
                    self.data.drop(columns=[col], inplace=True)
            else:
                if percentage < 10:
                    self.data[col] = imputer.fit_transform(self.data[[col]])
                else:
                    self.data.drop(columns=[col], inplace=True)

        self.data.drop(columns='img_id', inplace=True)

        return self.data
    
    def keepNanText_2(self):
        """
        Keep the nan text in the data, and change the y label to binary.
        """
        imputer = IterativeImputer()
        enc = LabelEncoder()
        for col, percentage in self.missing_percentage.items():
            if col == 'diagnostic':
                self.data['is_cancer'] = np.where(self.data['diagnostic'].isin(['BCC','MEL','SCC']),1,0)
                self.data.drop(columns=['diagnostic'], inplace=True)
            elif self.data[col].dtype == 'object':
                if percentage < 10:
                    self.data[col] = enc.fit_transform(self.data[col])
                    self.data[col] = imputer.fit_transform(self.data[[col]])
                else:
                    self.data[col] = self.data[col].fillna('Unknown')
                    self.data[col] = self.data[col].astype('str')
                    self.data[col] = enc.fit_transform(self.data[col])
                    self.data[col] = self.data[col].replace(len(enc.classes_), -1).astype('int64')

                    # output the encode method
                    encoding_mapping = dict(zip(enc.classes_, range(len(enc.classes_))))
                    print(f"Feature '{col}':")
                    for category, encoding in encoding_mapping.items():
                        if category == 'Unknown':
                            print(f"{category}: -1")
                        else:
                            print(f"{category}: {encoding}")
                    print('-'*30)
            else:
                if percentage < 10:
                    self.data[col] = imputer.fit_transform(self.data[[col]])
                else:
                    self.data[col] = self.data[col].fillna(-1)
                    self.data[col] = self.data[col].astype('int64')

        self.data.drop(columns='img_id', inplace=True)

        return self.data
    
    def dropNanText_2(self):
        """
        Drop the nan columns in the data, and change the y label to binary.
        """
        imputer = IterativeImputer()
        enc = LabelEncoder()
        for col, percentage in self.missing_percentage.items():
            if col == 'diagnostic':
                self.data['is_cancer'] = np.where(self.data['diagnostic'].isin(['BCC','MEL','SCC']),1,0)
                self.data.drop(columns=['diagnostic'], inplace=True)
            elif self.data[col].dtype == 'object':
                if percentage < 10:
                    self.data[col] = enc.fit_transform(self.data[col])
                    self.data[col] = imputer.fit_transform(self.data[[col]])
                else:
                    self.data.drop(columns=[col], inplace=True)
            else:
                if percentage < 10:
                    self.data[col] = imputer.fit_transform(self.data[[col]])
                else:
                    self.data.drop(columns=[col], inplace=True)

        self.data.drop(columns='img_id', inplace=True)

        return self.data
    
    def imageIndex2(self):
        """
        return a index of two labels of images.
        """
        for i in self.data.columns:
            if i == 'img_id':
                pass
            elif i == 'diagnostic':
                self.data['is_cancer'] = np.where(self.data['diagnostic'].isin(['BCC','MEL','SCC']),1,0)
                self.data.drop(columns='diagnostic', inplace=True)
            else:
                self.data.drop(columns=[i], inplace=True)
        
        self.data = self.data.reset_index()
        
        return self.data
    
    def imageIndex6(self):
        """
        return a index of six labels of images.
        """
        enc = LabelEncoder()
        for i in self.data.columns:
            if i == 'img_id':
                pass
            elif i == 'diagnostic':
                self.data['diagnostic'] = enc.fit_transform(self.data['diagnostic'])
                # output the encode method
                encoding_mapping = dict(zip(enc.classes_, range(len(enc.classes_))))
                print(f"Feature 'diagnostic':")
                for category, encoding in encoding_mapping.items():
                    if category == 'Unknown':
                        print(f"{category}: -1")
                    else:
                        print(f"{category}: {encoding}")
            else:
                self.data.drop(columns=[i], inplace=True)
        
        self.data = self.data.reset_index()
        
        return self.data


# ## Imbalanced Data

# In[6]:


def balance_data(X, y):
    # calculate the number of samples in each class
    class_counts = Counter(y)
    min_class_count = min(class_counts.values())

    # set the sampling strategy for both over and under sampling
    over_sample_strategy = {label: 2 * min_class_count for label in class_counts.keys() if class_counts[label] <= (2 * min_class_count)}
    under_sample_strategy = {label: 2 * min_class_count for label in class_counts.keys() if class_counts[label] > (2 * min_class_count)}

    # create a pipeline for resampling
    pipe = make_pipeline(
        SMOTE(sampling_strategy=over_sample_strategy),
        NearMiss(sampling_strategy=under_sample_strategy)
    )

    # resample the data
    X_resampled, y_resampled = pipe.fit_resample(X, y)

    # calculate the class weights
    class_weights = class_weight.compute_class_weight(class_weight='balanced',
                                                      classes=np.unique(y),
                                                      y=y)
    class_weights = dict(enumerate(class_weights))

    return X_resampled, y_resampled, class_weights


# ## Text Model

# In[7]:


def textModel(x_train, y_train, x_test, y_test, class_weights=None):
    svm = SVC(kernel='linear', random_state=42)
    svm.fit(x_train, y_train)
    y_pred = svm.predict(x_test)
    print(f'Model 1: {classification_report(y_test, y_pred)}')
    
    rf = RandomForestClassifier(n_estimators=100, class_weight=class_weights, random_state=42)
    rf.fit(x_train, y_train)
    y_pred = rf.predict(x_test)
    print(f'Model 2: {classification_report(y_test, y_pred)}')

    xgb = XGBClassifier(use_label_encoder=False, eval_metric='mlogloss', random_state=42)
    xgb.fit(x_train, y_train)
    y_pred = xgb.predict(x_test)
    print(f'Model 3: {classification_report(y_test, y_pred)}')

    knn = KNeighborsClassifier(n_neighbors=5)
    knn.fit(x_train, y_train)
    y_pred = knn.predict(x_test)
    print(f'Model 4: {classification_report(y_test, y_pred)}')

    num_class = len(np.unique(y_train))
    if num_class == 2:
        num_class = 1

    lgbm = LGBMClassifier(num_class=num_class, class_weight=class_weights, random_state=42, verbose=-1)
    lgbm.fit(x_train, y_train)
    y_pred = lgbm.predict(x_test)
    print(f'Model 5: {classification_report(y_test, y_pred)}')

    return svm, rf, xgb, knn, lgbm


# ## Load Dataset

# Explain of the features:
# - background_father: The history of any diseases or health conditions related to the patient's father, including any history of skin cancer or other diseases that may be related to skin cancer
# - background_mother: The history of any diseases or health conditions related to the patient's mother, including any history of skin cancer or other diseases that may be related to skin cancer
# - has_piped_water: Indicates whether the location or area of the patient's residence has access to piped water or not
# - has_sewage_system: Indicates whether the location or area of the patient's residence has a proper sewage system or not
# - fitspatrick: Skin tolerance to sunlight
# - itch: Whether the lesion or wound has itched or not
# - elevation: Description of the of the lesion or wound relative to the skin surface of the patient
# - biopsed: Whether the lesion or wound has been biopsied or not

# In[8]:


metadata = pd.read_csv(os.path.join(data_path, 'metadata.csv'))


# ## Data Information

# In[9]:


metadata.shape


# In[10]:


metadata.head()


# In[11]:


def count_is_null(data:pd.DataFrame):
    countNaN = data.isna().sum()
    return f'{countNaN}({countNaN*100/data.shape[0]:.1f}%)'
def count_is_null_unique(data:pd.DataFrame):
    return data.count()-data.nunique()
def data_info(data:pd.DataFrame):
    return data.agg(['count', 'nunique', count_is_null_unique, count_is_null, 'dtype']).T


# In[12]:


data_info(metadata)


# # Method One

# ## Data Processing

# In[13]:


keep_df6 = DataProcessing(metadata).keepNanText_6()


# In[14]:


data_info(keep_df6)


# In[15]:


keep_df6.drop(columns=['patient_id', 'lesion_id'], inplace=True)
keep_df6 = keep_df6.astype('int64')


# In[16]:


keep_df6.describe().T


# ## Standardize and Data Split

# In[17]:


x = keep_df6.drop(columns=['diagnostic'])
columns = x.columns
scaler = MinMaxScaler()
x = scaler.fit_transform(x)
x = pd.DataFrame(x, columns=columns)
y = keep_df6['diagnostic']

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)


# ## resample

# In[18]:


x_train, y_train, class_weights = balance_data(x_train, y_train)


# ## Model

# In[19]:


svm, rf, xgb, knn, lgbm = textModel(x_train, y_train, x_test, y_test, class_weights)


# ## Stacking

# In[20]:


# stack model
estimators1 = [('svc', svm), ('rf', rf), ('xgb', xgb), ('knn', knn), ('lgbm', lgbm)]
stack_model1 = StackingClassifier(estimators=estimators1, final_estimator=XGBClassifier())


# In[21]:


# fit the model on the training data
stack_model1.fit(x_train, y_train)
# make predictions
y_pred = stack_model1.predict(x_test)
# calculate the classification report
stack_model1_result = classification_report(y_test, y_pred)
print(f'Stack Model: {stack_model1_result}')


# # Method Two

# ## Data Processing

# In[22]:


drop_df6 = DataProcessing(metadata).dropNanText_6()


# In[23]:


drop_df6.drop(columns=['patient_id', 'lesion_id'], inplace=True)
drop_df6 = drop_df6.astype('int64')


# In[24]:


drop_df6.describe().T


# ## Standardize and Data Split

# In[25]:


x = drop_df6.drop(columns=['diagnostic'])
columns = x.columns
scaler = MinMaxScaler()
x = scaler.fit_transform(x)
x = pd.DataFrame(x, columns=columns)
y = drop_df6['diagnostic']

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)


# ## resample

# In[26]:


x_train, y_train, class_weights = balance_data(x_train, y_train)


# ## Model

# In[27]:


svm, rf, xgb, knn, lgbm = textModel(x_train, y_train, x_test, y_test, class_weights)


# ## Stacking

# In[28]:


# stack model
estimators2 = [('svc', svm), ('rf', rf), ('xgb', xgb), ('knn', knn), ('lgbm', lgbm)]
stack_model2 = StackingClassifier(estimators=estimators2, final_estimator=XGBClassifier())


# In[29]:


# fit the model on the training data
stack_model2.fit(x_train, y_train)
# make predictions
y_pred = stack_model2.predict(x_test)
# calculate the classification report
stack_model2_result = classification_report(y_test, y_pred)
print(f'Stack Model: {stack_model2_result}')


# # Method Three

# ## Data Processing

# In[30]:


keep_df2 = DataProcessing(metadata).keepNanText_2()


# In[31]:


keep_df2.drop(columns=['patient_id', 'lesion_id'], inplace=True)
keep_df2 = keep_df2.astype('int64')


# In[32]:


keep_df2.describe().T


# ## Standardize and Data Split

# In[33]:


x = keep_df2.drop(columns=['is_cancer'])
columns = x.columns
scaler = MinMaxScaler()
x = scaler.fit_transform(x)
x = pd.DataFrame(x, columns=columns)
y = keep_df2['is_cancer']

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)


# ## Model

# In[34]:


svm, rf, xgb, knn, lgbm = textModel(x_train, y_train, x_test, y_test)


# ## Stacking

# In[35]:


# stack model
estimators3 = [('svc', svm), ('rf', rf), ('xgb', xgb), ('knn', knn), ('lgbm', lgbm)]
stack_model3 = StackingClassifier(estimators=estimators3, final_estimator=LGBMClassifier(verbose=-1))


# In[36]:


# fit the model on the training data
stack_model3.fit(x_train, y_train)
# make predictions
y_pred = stack_model3.predict(x_test)
# calculate the classification report
stack_model3_result = classification_report(y_test, y_pred)
print(f'Stack Model: {stack_model3_result}')


# # Method Four

# ## Data Processing

# In[38]:


drop_df2 = DataProcessing(metadata).dropNanText_2()


# In[39]:


drop_df2.drop(columns=['patient_id', 'lesion_id'], inplace=True)
drop_df2 = drop_df2.astype('int64')


# In[40]:


drop_df2.describe().T


# ## Standardize and Data Split

# In[41]:


x = drop_df2.drop(columns=['is_cancer'])
columns = x.columns
scaler = MinMaxScaler()
x = scaler.fit_transform(x)
x = pd.DataFrame(x, columns=columns)
y = drop_df2['is_cancer']

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)


# ## Model

# In[42]:


svm, rf, xgb, knn, lgbm = textModel(x_train, y_train, x_test, y_test)


# ## Stacking

# In[43]:


# stack model
estimators4 = [('svc', svm), ('rf', rf), ('xgb', xgb), ('knn', knn), ('lgbm', lgbm)]
stack_model4 = StackingClassifier(estimators=estimators4, final_estimator=RandomForestClassifier())


# In[44]:


# fit the model on the training data
stack_model4.fit(x_train, y_train)
# make predictions
y_pred = stack_model4.predict(x_test)
# calculate the classification report
stack_model4_result = classification_report(y_test, y_pred)
print(f'Stack Model: {stack_model4_result}')


# # Conclusion

# In[45]:


print(f"Stack Model 1: {stack_model1_result}")
print('-'*30)
print(f"Stack Model 2: {stack_model2_result}")
print('-'*30)
print(f"Stack Model 3: {stack_model3_result}")
print('-'*30)
print(f"Stack Model 4: {stack_model4_result}")


# # Save the Model

# In[46]:


# save the model
joblib.dump(stack_model3, 'stack_model3.joblib')
with open('stack_model3.pkl', 'wb') as f:
    pickle.dump(stack_model3, f)


# In[47]:


image_df2 = DataProcessing(metadata).imageIndex2()
image_df2.to_csv(os.path.join(data_path, 'imageIndex2.csv'), index=False)


# In[48]:


image_df6 = DataProcessing(metadata).imageIndex6()
image_df6.to_csv(os.path.join(data_path, 'imageIndex6.csv'), index=False)

