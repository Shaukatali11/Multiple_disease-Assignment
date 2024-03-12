#!/usr/bin/env python
# coding: utf-8

# In[ ]:





# # Name :- Md Shaukat Ali
#     
# Roll :- 23CS4141
#     
# Reg  :- 23P10244

# In[ ]:


BY-KISKU SIR


# # DAY-1 LAB

# In[ ]:





# In[1]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


# In[2]:


import os 
import warnings 


# In[3]:


train=pd.read_csv("Blood_samples_dataset_balanced_2(f).csv")


# In[4]:


test= pd.read_csv("blood_samples_dataset_test.csv")


# In[5]:


train


# In[6]:


test


# # EDA on Training Data

# In[7]:


train.head()


# In[8]:


train.info()


# In[9]:


train.isnull()


# In[10]:


train.isnull().sum()


# In[11]:


train.describe()


# In[12]:


train.duplicated()


# In[13]:


train.duplicated().sum()


# In[14]:


train.shape


# In[ ]:





# In[ ]:





# # EDA on testing Data

# In[15]:


test.head()


# In[16]:


test.info()


# In[17]:


test.describe()


# In[18]:


test.isnull().sum()


# In[19]:


test.duplicated().sum()


# In[20]:


test.describe()


# In[21]:


test.shape


# In[ ]:





# # We merge the train and test dataset in 'df' 

# In[22]:


df=pd.concat([train,test],axis=0)
df.head()


# In[23]:


df.isnull().sum()


# In[24]:


df.shape


# In[25]:


df['Disease'].unique()


# In[26]:


df['Disease'].nunique()


# In[27]:


df['Disease']=df['Disease'].replace('Heart Di','Heart Disease')
df['Disease'].unique()


# In[28]:


df.columns


# In[ ]:





# In[ ]:





# # We have to classify which type of disease , The Disease feature of categorical type so we need to convert it into numerical so here we apply label encoder of different different disease

# In[29]:


from sklearn.preprocessing import LabelEncoder


# In[30]:


le=LabelEncoder()


# In[31]:


df['Disease']=le.fit_transform(df['Disease'])
df.head()


# # Split the Dataset into training and testing part:-

# In[32]:


sns.heatmap(df)


# In[34]:


y=df['Disease']
x=df.drop('Disease',axis=1)


# In[35]:


sns.countplot(x=y)
plt.show()


# In[ ]:





# In[42]:


sns.heatmap(x.corr(),square=True)
plt.show()


# In[36]:


from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test=train_test_split(x,y,test_size=0.25,random_state=42)


# In[37]:


X_train


# In[38]:


y_train = np.array(y_train).reshape(-1,1)


# In[39]:


y_train.shape


# In[40]:


X_train.shape


# In[41]:


from sklearn.metrics import accuracy_score,roc_auc_score,roc_curve


# # Imported all  the ML algorithm and make object of each model:-

# In[43]:


from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB


# In[44]:


lr=LogisticRegression(multi_class='multinomial')
dt=DecisionTreeClassifier()
rf=RandomForestClassifier()
knn=KNeighborsClassifier()
gnb=GaussianNB()


# In[45]:


def predictor(model_name):    
    print("For the {}".format(model_name)) 
    print("")
    model_name.fit(X_train,y_train)    
    y_pred_train = model_name.predict(X_train)    
    y_pred_test = model_name.predict(X_test)    
    print("The TRAIN accuracy is",accuracy_score(y_train,y_pred_train))    
    print("--"*50)    
    print("The TEST accuracy is",accuracy_score(y_test,y_pred_test))   


# # Logistic Regression

# In[46]:


predictor(lr)


# # Decision Tree

# In[47]:


predictor(dt)


# # K- NN

# In[48]:


predictor(knn)


# # Random Forest

# In[49]:


predictor(rf)


# # Naive Bayes 

# In[50]:


predictor(gnb)


# # Accuracy of different - different Model

# In[51]:


models = ['LR', 'DT', 'RF', 'KNN' , 'NB'] 
accuracies = [0.82, 1, 0.95, 0.92 , 0.81]
colors = ['skyblue', 'orange', 'green', 'red' , 'blue'] 


# In[52]:


plt.figure(figsize=(10, 6)) 
plt.bar(models, accuracies, color='skyblue')
bars = plt.bar(models, accuracies)
for i in range(len(bars)):
    bars[i].set_color(colors[i])
plt.title('Accuracy of Different Models')
plt.xlabel('Models')
plt.ylabel('Accuracy')
plt.show()


# # We apply MLP on this dataset

# In[53]:


import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Adam


# In[54]:


import tensorflow as tf
from tensorflow.keras import layers, models


# In[55]:


def create_mlp(input_shape, num_classes):
    model = models.Sequential([
        layers.Dense(128, activation='relu', input_shape=input_shape),
        layers.Dense(64, activation='relu'),
        layers.Dense(num_classes, activation='softmax')
    ])
    return model


# In[56]:


# Create MLP model
input_shape = X_train.shape[1:]
num_classes = len(np.unique(y_train))
mlp_model = create_mlp(input_shape, num_classes)


# In[57]:


# Compile MLP model
mlp_model.compile(optimizer='adam',
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])


# In[58]:


mlp_model.summary()


# In[59]:


mlp_history = mlp_model.fit(X_train, y_train, epochs=5, validation_data=(X_test, y_test), verbose=0)


# In[60]:


mlp_loss, mlp_accuracy = mlp_model.evaluate(X_test, y_test)
print("MLP Accuracy:", mlp_accuracy)


# In[61]:


from sklearn.svm import SVC


# In[62]:


# Train SVM model
svm_model = SVC(kernel='linear')
svm_model.fit(X_train, y_train)


# In[63]:


svm_accuracy = accuracy_score(y_test, svm_model.predict(X_test))
print("SVM Accuracy:", svm_accuracy)


# # Graph shows the accuracy and loss of MLP vs SVM

# In[64]:


# Plot MLP accuracy
plt.subplot(1, 2, 1)
plt.plot(mlp_history.history['accuracy'], label='MLP Training Accuracy')
plt.plot(mlp_history.history['val_accuracy'], label='MLP Validation Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.title('MLP Model Accuracy')
plt.legend()

# Plot MLP loss
plt.subplot(1, 2, 2)
plt.plot(mlp_history.history['loss'], label='MLP Training Loss')
plt.plot(mlp_history.history['val_loss'], label='MLP Validation Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('MLP Model Loss')
plt.legend()

plt.tight_layout()
plt.show()


# In[65]:


from sklearn.metrics import accuracy_score, confusion_matrix


# In[66]:


svm_pred = svm_model.predict(X_test)
svm_cm = confusion_matrix(y_test, svm_pred)

# MLP confusion matrix
mlp_pred = np.argmax(mlp_model.predict(X_test), axis=1)
mlp_cm = confusion_matrix(y_test, mlp_pred)

# Plot confusion matrices
plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
sns.heatmap(svm_cm, annot=True, fmt='d', cmap='Blues', cbar=False)
plt.title('SVM Confusion Matrix')
plt.xlabel('Predicted Label')
plt.ylabel('True Label')

plt.subplot(1, 2, 2)
sns.heatmap(mlp_cm, annot=True, fmt='d', cmap='Blues', cbar=False)
plt.title('MLP Confusion Matrix')
plt.xlabel('Predicted Label')
plt.ylabel('True Label')

plt.tight_layout()
plt.show()


# # DAY-2 LAB

# In[ ]:





# In[ ]:





# # process the dataset with Support Vector Machines (SVM). In the next step, consolidate the outcomes of MLP and SVM at decision level with logical OR and AND operators to obtain the final outcome. Finally, show both ERROR and ACCURACY curves for MLP, SVM, OR rule and AND rule in the same graph while considering loss and accuracy in the same axis.

# In[68]:


# Step 1: Obtain predictions from both models
svm_predictions = svm_model.predict(X_test)
mlp_predictions = np.argmax(mlp_model.predict(X_test), axis=-1)


# In[69]:


# Step 2: Apply logical OR and AND operators to consolidate predictions
or_predictions = np.logical_or(svm_predictions, mlp_predictions)
and_predictions = np.logical_and(svm_predictions, mlp_predictions)


# In[70]:


# Step 3: Calculate accuracy and loss for consolidated predictions
svm_accuracy = accuracy_score(y_test, svm_predictions)
mlp_accuracy = accuracy_score(y_test, mlp_predictions)
or_accuracy = accuracy_score(y_test, or_predictions)
and_accuracy = accuracy_score(y_test, and_predictions)


# In[71]:


# Step 4: Plot error and accuracy curves
plt.figure(figsize=(12, 5))

# Accuracy curve
plt.subplot(1, 2, 1)
plt.plot(mlp_history.history['accuracy'], label='MLP Accuracy')
plt.plot(mlp_history.history['val_accuracy'], label='MLP Validation Accuracy')
plt.axhline(y=svm_accuracy, color='r', linestyle='--', label='SVM Accuracy')
plt.axhline(y=or_accuracy, color='g', linestyle='-.', label='OR Rule Accuracy')
plt.axhline(y=and_accuracy, color='b', linestyle=':', label='AND Rule Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.title('Model Accuracy')
plt.legend()

# Loss curve
plt.subplot(1, 2, 2)
plt.plot(mlp_history.history['loss'], label='MLP Loss')
plt.plot(mlp_history.history['val_loss'], label='MLP Validation Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Model Loss')
plt.legend()

plt.tight_layout()
plt.show()


# In[ ]:




