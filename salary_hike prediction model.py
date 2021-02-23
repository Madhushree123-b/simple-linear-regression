#!/usr/bin/env python
# coding: utf-8

# In[4]:


#Importing the libraries required
import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt
import statsmodels.api as sf
import statsmodels.formula.api as smf
import seaborn as sns
import statsmodels as sm


# In[5]:


#import the dataset
salary_data = pd.read_csv("Salary_Data.csv")
salary_data


# In[7]:


#EDA(exploratary data analysis)
#Checking if there are any null value and the type of data and its shape
salary_data.info()


# In[8]:


#another way to check if there is any null value
salary_data.isna().sum()


# In[9]:


salary_data.describe()


# In[10]:


#Checking for duplicates
salary_data[salary_data.duplicated()]


# In[11]:


#visualizing the data
#Checking the distribution of the data
sns.distplot(salary_data['Salary'])


# In[12]:


#Checking the relation of the x and y variables
plt.plot(salary_data["YearsExperience"],salary_data["Salary"],"ro")
plt.xlabel("Years of Exp")
plt.ylabel("Salary")


# In[13]:


#Checking for outliers
plt.boxplot(salary_data["Salary"])


# In[14]:


plt.boxplot(salary_data["YearsExperience"])


# In[15]:


plt.hist(salary_data["Salary"])


# In[16]:


plt.hist(salary_data["YearsExperience"])


# In[17]:


#Checking the correlation of the data
salary_data.corr()


# In[18]:


sns.pairplot(salary_data)


# In[19]:


#Observation: The dependant variable is strongly correlated


# In[20]:


y = salary_data["Salary"]
x1 = salary_data["YearsExperience"]

x = sm.tools.add_constant(x1)

model_new = sm.regression.linear_model.OLS(y,x).fit()
model_new.summary()


# In[21]:


#Building the Simple Linear Regression Model
model = smf.ols("Salary~YearsExperience", data=salary_data).fit()
model.summary()


# In[22]:


#Observation: The Rsquared is very high


# In[23]:


#Predicting the Salary using the whole YearsExperience column
pred = model.predict(salary_data["YearsExperience"])
pred


# In[24]:


#Checking the residual
model.resid


# In[25]:


#calculating the Root mean square of this model
rmse = np.sqrt(np.mean((np.array(salary_data['YearsExperience'])-np.array(pred))**2))
rmse 


# In[26]:


#Visualizing the model and the line
plt.scatter(x=salary_data['YearsExperience'],y=salary_data['Salary'],color='blue')
plt.plot(salary_data['YearsExperience'],pred,color='black')
plt.xlabel('Years of experience')
plt.ylabel('Salary')


# In[27]:


#observing the residual after standardizing it
plt.plot(model.resid_pearson,'o')
plt.axhline(y=0,color='green')
plt.xlabel("Observation Number")
plt.ylabel("Standardized Residual")


# In[28]:


sns.regplot(x="YearsExperience", y="Salary", data=salary_data);


# In[29]:


#Building another model by changing trying to fit the line by transforming the independant variable
#log transformation of independant variable

model_2 = smf.ols("Salary~np.log(YearsExperience)", data=salary_data).fit()
model_2.summary()


# In[30]:


#Checking the RMSE of this model
rmse_2 = np.sqrt(np.mean((np.array(salary_data['YearsExperience'])-np.array(model_2.predict(salary_data["YearsExperience"])))**2))
rmse_2


# In[31]:


#Building another model by transforming the dependant variable using log to experiment with the best line fit
model_3 = smf.ols("np.log(Salary)~YearsExperience", data=salary_data).fit()
model_3.summary()


# In[32]:


#Checking the RMSE of this model by invering the effects of log and checking its RMSE value
pred_model_3 = np.exp(model_3.predict(salary_data['YearsExperience']))
rmse_3 = np.sqrt(np.mean((np.array(salary_data['YearsExperience'])-np.array(pred_model_3))**2))
rmse_3


# In[33]:


#Building another model by transforming the dependant variable to its square root
model_4 = smf.ols("np.sqrt(Salary)~YearsExperience", data=salary_data).fit()
model_4.summary()


# In[34]:


#Checking the RMSE of this model by invering the effects of log and checking its RMSE value
pred_model_4 = ((model_4.predict(salary_data['YearsExperience']))**2)
rmse_4 = np.sqrt(np.mean((np.array(salary_data['YearsExperience'])-np.array(pred_model_4))**2))
rmse_4


# In[35]:


#Building another model by adding another independant variable as the square of the dependant variable
#and transforming dependant variable to log
salary_data["YE_square"] = salary_data.YearsExperience ** 2
model_5 = smf.ols('np.log(Salary)~YearsExperience+YE_square', data=salary_data).fit()
model_5.summary()


# In[36]:


#Calculating the RMSE
pred_model_5 = np.exp(model_5.predict(salary_data)) #inversing the effect of log
rmse_5 = np.sqrt(np.mean((np.array(salary_data['YearsExperience'])-np.array(pred_model_5))**2))
rmse_5


# In[37]:


#making a dataframe to check which model was best (rsquared should be high and rmse should be low)import pandas as pd
import pandas as pd
model_check = pd.DataFrame({
    "Model": ["model", "model_2", "model_3", "model_4", "model_5"], 
    "R squared": [model.rsquared, model_2.rsquared, model_3.rsquared, model_4.rsquared, model_5.rsquared],
    "RMSE": [rmse, rmse_2, rmse_3, rmse_4, rmse_5]
})
model_check


# In[38]:


#Observation: the Rsquared of the first model is the highest. It looks like the best model


# In[39]:


#dropping the YE_squared column that was made before and adding the prediction of model
salary_data = salary_data.drop(['YE_square'],axis=1)
salary_data["Prediction Model 1"] = pred
salary_data


# In[ ]:




