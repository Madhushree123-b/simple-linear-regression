#!/usr/bin/env python
# coding: utf-8

# In[3]:


#Importing the libraries required
import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt
import statsmodels.api as sf
import statsmodels.formula.api as smf
import seaborn as sns


# In[6]:


#import the dataset
data = pd.read_csv("delivery_time.csv")
data


# In[7]:


#Checking the overall info of the dataset
data.info()


# In[8]:


#Checking if there is any null value
data.isna().sum()


# In[9]:


#EDA
data.describe()


# In[10]:


#checking for duplicated values
data[data.duplicated()]


# In[11]:


#visualizing the data
plt.hist(data["Delivery Time"])


# In[12]:


plt.hist(data["Sorting Time"])


# In[13]:


plt.plot(data["Sorting Time"],data["Delivery Time"],"ro")
plt.xlabel("Delivery Time")
plt.ylabel("Sorting Time")


# In[14]:


#trying to find outliers
data.boxplot(column=["Sorting Time"])


# In[15]:


data.boxplot(column=["Delivery Time"])


# In[16]:


#checking the distribution of the data
sns.distplot(data['Sorting Time'])


# In[17]:


sns.distplot(data['Delivery Time'])


# In[18]:


#Finding the correlation of the data
data.corr()


# In[19]:


sns.pairplot(data)


# In[20]:


#Changing the column name to make the data easier to handle (DT for Delivery time and ST for Sorting Time)
data.rename(columns={'Delivery Time': 'DT', 'Sorting Time': 'ST'}, inplace=True)
data


# In[21]:


#Building the Simple Linear Regression Model
model = smf.ols("DT~ST", data=data).fit()


# In[22]:


#Checking the model
model.summary()


# In[23]:


#Predicting the Delivery Time using the whole Sorting Time column
pred = model.predict(data["ST"])
pred


# In[24]:


#Checking the residual
model.resid


# In[25]:


#calculating the Root mean square of this model
rmse = np.sqrt(np.mean((np.array(data['ST'])-np.array(pred))**2))
rmse 


# In[26]:


#Visualizing the model and the line
plt.scatter(x=data['ST'],y=data['DT'],color='blue')
plt.plot(data['ST'],pred,color='black')
plt.xlabel('Sorting Time')
plt.ylabel('Delivery Time')


# In[27]:


#observing the residual after standardizing it
plt.plot(model.resid_pearson,'o')
plt.axhline(y=0,color='green')
plt.xlabel("Observation Number")
plt.ylabel("Standardized Residual")


# In[20]:


#Observation: r squared = 68% & rmse = 10.722


# In[28]:


#Building another model by changing trying to fit the line by transforming the independant variable
#log transformation of independant variable

model_2 = smf.ols("DT~np.log(ST)", data=data).fit()
model_2.summary()


# In[22]:


#Observation: The rsquared value has improved a bit. 


# In[29]:


#Checking the rmse for model_2
rmse_2 = np.sqrt(np.mean((np.array(data['ST'])-np.array(model_2.predict(data["ST"])))**2))
rmse_2


# In[30]:


#visualizing the model_2
plt.scatter(x=data['ST'],y=data['DT'],color='blue')
plt.plot(data['ST'],model_2.predict(data["ST"]),color='black')
plt.xlabel('Sorting Time')
plt.ylabel('Delivery Time')


# In[31]:


#observing the residual after standardizing it
plt.plot(model_2.resid_pearson,'o')
plt.axhline(y=0,color='green')
plt.xlabel("Observation Number")


# In[26]:


#observation: Even though the rsquared increased the rmse went down in model_2


# In[32]:


#Building another model by transforming the dependant variable using log to experiment with the best line fit
model_3 = smf.ols("np.log(DT)~ST", data=data).fit()
model_3.summary()


# In[28]:


#rsquared improved in this model


# In[33]:


#converting the predicted dependant variable to exponention to inverse the effect of log
pred_model_3 = np.exp(model_3.predict(data['ST']))
pred_model_3


# In[34]:


#Checking the RMSE of the model_3
rmse_3 = np.sqrt(np.mean((np.array(data['ST'])-np.array(pred_model_3))**2))
rmse_3


# In[35]:


#visualizng the model_3
plt.scatter(x=data['ST'],y=data['DT'],color='blue')
plt.plot(data['ST'],pred_model_3,color='black')
plt.xlabel('Sorting Time')
plt.ylabel('Delivery Time')


# In[36]:


#observing the residual after standardizing it
plt.plot(model_3.resid_pearson,'o')
plt.axhline(y=0,color='green')
plt.xlabel("Observation Number")


# In[37]:


#Observation: The rsquared and the rmse were better in model_3


# In[38]:


#Building another model by transforming the dependant variable to its square root
model_4 = smf.ols("np.sqrt(DT)~ST", data=data).fit()
model_4.summary()


# In[39]:


#Checking the RMSE
pred_model_4 = model_4.predict(data["ST"]) ** 2 #as we took square root of dependant variable
rmse_4 = np.sqrt(np.mean((np.array(data['ST'])-np.array(pred_model_4))**2))
rmse_4


# In[40]:


#observation: This model is not better than model_3 as rsquare and rmse is lower and higher respectively


# In[41]:


#Building another model by adding another independant variable as the square of the dependant variable
#and transforming dependant variable to log
data["ST_square"] = data.ST ** 2
model_5 = smf.ols('np.log(DT)~ST+ST_square', data=data).fit()
model_5.summary()


# In[42]:


#Calculating the RMSE
pred_model_5 = np.exp(model_5.predict(data)) #inversing the effect of log
rmse_5 = np.sqrt(np.mean((np.array(data['ST'])-np.array(pred_model_5))**2))
rmse_5


# In[43]:


#observation this is a good model as the rsquared value increased and rmse decreased


# In[44]:


#visualizing this model_5
plt.scatter(x=data['ST'],y=data['DT'],color='blue')
plt.plot(data['ST'],pred_model_5,color='black')
plt.xlabel('Sorting Time')
plt.ylabel('Delivery Time')


# In[45]:


#observing the residual after standardizing it
plt.plot(model_5.resid_pearson,'o')
plt.axhline(y=0,color='green')
plt.xlabel("Observation Number")


# In[46]:


#making a dataframe to check which model was best (rsquared should be high and rmse should be low)
model_check = pd.DataFrame({
    "Model": ["model", "model_2", "model_3", "model_4", "model_5"], 
    "R squared": [model.rsquared, model_2.rsquared, model_3.rsquared, model_4.rsquared, model_5.rsquared],
    "RMSE": [rmse, rmse_2, rmse_3, rmse_4, rmse_5]
})
model_check


# In[47]:


#Observation: model_5 is the best out of the models we made


# In[48]:


#dropping the ST_squared column that was made before and adding the prediction of model_5
data = data.drop(['ST_square'],axis=1)
data["Prediction model_5"] = pred_model_5
data


# In[ ]:





# In[ ]:




