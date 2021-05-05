#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


# In[2]:


# EDA - Exporatory Data Analysis
# TO Understan to judge the data what is happening at each steps of feature what kind of data plot at there
# understaning the while criteria



# Preprocessing - Cleaninga and modifying the data
# preprocessing so that machine can understand it


# In[3]:


d = pd.read_csv(r"C:\Users\thora\.ipynb_checkpoints\cars.csv")


# In[4]:


d


# In[5]:


d.info()


# In[6]:


d.isnull()


# In[ ]:





# In[7]:


d.describe()
# it returns all statistics of numeric columns(int,float..)


# In[8]:


d.corr()


# In[ ]:





# In[9]:


sns.scatterplot(data=d.corr(), x='width', y='price')
plt.show()


# In[10]:


plt.figure(figsize=(8,8))
sns.heatmap(d.corr(), annot=True)
plt.show()


# In[11]:


d.head(1)


# In[12]:


price_mean=d["price"].mean()
price_median=d["engine-size"].median()


# In[13]:


plt.figure(figsize=(6,6))
sns.distplot(d["price"])
# vertical line to understand skewness of data on based of mean and median
plt.axvline(price_mean,color='red')
plt.axvline(price_median,color='yellow')
plt.show()


# In[14]:


d['price'].mean()


# In[15]:


d['price'].median()


# In[16]:


d['fuel-type'].value_counts()


# In[17]:


plt.figure(figsize=(10,10))
d['fuel-type'].value_counts().plot.pie(autopct="%1.1f%%",explode=(0,0.2),)
plt.title("Fuel Type Available Cars")
plt.show()


# In[18]:


d['body-style'].value_counts()


# In[19]:


plt.figure(figsize=(7,7))
#using pands we plot barplot
d['body-style'].value_counts().plot(kind="bar")
plt.title("make counts")
plt.ylabel("count")
plt.xlabel("body-style")
plt.show()


# In[20]:


import warnings
warnings.filterwarnings('ignore')


# In[21]:


d.info()
# 8 categorical and 7 numerical data is available


# In[22]:


d.head(1)
# junk values in normalized-losses(?) and horsepower(in info it shows that there is categorical data but here its in nnumerial form)


# In[23]:


d['normalized-losses'].unique()


# In[24]:


d['normalized-losses'].replace("?", np.nan, inplace=True)
# 1 Replace junk with Nan


# In[25]:


d['normalized-losses'] = d['normalized-losses'].astype(float)
# Because Machine Leaarning  only understand  numbers


# In[26]:


d['normalized-losses'].dtype
# change to float


# In[27]:


sns.distplot(d['normalized-losses'])
plt.axvline(d['normalized-losses'].mean(), color="red")       # use of axvline when you want to draw straight line on the x-axis(we want to draw mean value)
plt.axvline(d['normalized-losses'].median(), color="yellow")
plt.show()


# In[28]:


# Not much diff in them


# In[29]:


d['normalized-losses'].mean()


# In[30]:


d['normalized-losses'].median()# lower than mean


# In[31]:


d['normalized-losses'].fillna(d['normalized-losses'].median(), inplace=True)


# In[32]:


d['normalized-losses'].unique()
# there is no junk in normalized-losses


# In[33]:


d.info()
# change to float done


# In[34]:


d['horsepower'].unique()


# In[35]:


# Now same steps for horsepower to change their data type
# Replace junk with nan


# In[36]:


d['horsepower'].replace("?", np.nan, inplace=True)


# In[37]:


# Changing datatype object to float
d['horsepower'] = d['horsepower'].astype(float)


# In[38]:


d['horsepower'].dtype


# In[39]:


sns.distplot(d["horsepower"])
plt.axvline(d["horsepower"].mean(), color="red")
plt.axvline(d["horsepower"].median(), color = "yellow")
plt.show()


# In[40]:


d['horsepower'].mean()


# In[41]:


d['horsepower'].median()# again not much difference 
# data is not normally distributes with median thats the reason we replace with the median


# In[42]:


d['horsepower'].fillna(d['horsepower'].median(), inplace=True)


# In[43]:


d.info()


# In[44]:


# Dividing Quantitave and Qualitative data
d_num = d.select_dtypes(['int64','float64'])# give list of datatype you want   for numeric data
d_cat = d.select_dtypes('object')

# Why we do this
# Because the 3rd step in EDA REGARDS WITH QUANTITAVE DATA


# In[45]:


from scipy.stats import skew
# column wise skewness


# In[46]:


# we can check correlation only for quantitative data
# if it is regression algo only then we can look for correlatiion
# and if it not regression algo then i will look for reducing the skewness directly


# In[47]:


for col in d_num:# we pickup all the data points of numeric types
    print(col)  # we get column name
    sns.distplot(d[col])      # this will give us values
    plt.show()   # this will generate distplot for each and everything
    print("skew = ",skew(d[col]))
    print("********************************************************")


# In[48]:


# BEST WAY TO CHECK COORELATION BELOW


# In[49]:


sns.heatmap(d_num.corr(), annot= True)
plt.show()
# we checking this all for price and positively skewed


# In[50]:


# What is the rule of skwness is???
#if my colomns which have positive and negative skwness but they have the positive and negative correlation
#with my target variable we will ignore those colomns


# In[51]:


np.log(-12)


# In[52]:


-10**2


# In[53]:


min(d_num['normalized-losses'])#  all no of positive numbers


# In[54]:


# if there is 0 or negative number you will not reduce the skwness


# In[55]:


np.cbrt(0)


# In[56]:


np.log(0)


# In[57]:


# here my normalized-losses is not negative skwed we do this(below)


# In[58]:


skew(np.sqrt(d_num['normalized-losses']))


# In[59]:


d_num['normalized-losses'] = np.sqrt(d_num['normalized-losses'])
# now my data is better its normally distributed now


# # LabelEncoder

# In[60]:


from sklearn.preprocessing import LabelEncoder


# In[61]:


for col in d_cat.columns:
    le = LabelEncoder()
    d_cat[col] = le.fit_transform(d_cat[col])


# # scaler

# In[62]:


d_mm = d_num.copy() #    created 2 df scaler
d_ss = d_num.copy()


# In[63]:


from sklearn.preprocessing import StandardScaler, MinMaxScaler


# In[64]:


for col in d_mm:
    mm = MinMaxScaler()
    d_mm[col] = mm.fit_transform(d_mm[[col]])#   col in 2[[]]  because it accepts 2d array


# In[65]:


d_mm.head()#    all the numbers are between 0 to 1


# In[66]:


for col in d_ss:
    ss = StandardScaler()
    d_ss[col] = ss.fit_transform(d_ss[[col]])


# In[67]:


d_ss.head()


# In[68]:


d_new = pd.concat([d_num, d_cat], axis=True)


# In[69]:


d_new.head(2)


# # outliers

# In[70]:


sns.boxplot(data=d_new, x="price")
# here you can see close to 27000 there are too many outlier


# In[71]:


# 1. now i have to identify how many productsare beyond 27k 


# In[72]:


len(d_new[d_new['price'] >= 27000])


# In[73]:


d_new.shape


# In[74]:


d_new[d_new['price'] >= 27000]


# In[75]:


d[d['price'] >= 27000]


# In[76]:


d[d['price'] >= 27000]['make'].value_counts()


# In[77]:


d[d['price'] >= 27000]['body-style'].value_counts()


# In[78]:


d['make'].value_counts()


# In[79]:


d['body-style'].value_counts()


# In[80]:


plt.figure(figsize=(10,10))
sns.boxplot(data=d, x='price', y='engine-type')
plt.show()
# there are 16 outliers in engine-type
# we have outliers at 5000 also we have outliers at diff levels also


# In[81]:


d.head(1)


# In[82]:


plt.figure(figsize=(10,10))
sns.boxplot(data=d, x='price', y='engine-location')
plt.show()


# In[83]:


plt.figure(figsize=(10,10))
sns.boxplot(data=d, x='price', y='fuel-type')
plt.show()


# In[84]:


plt.figure(figsize=(10,10))
sns.boxplot(data=d, x='price', y='body-style')
plt.show()


# In[85]:


plt.figure(figsize=(10,10))
sns.boxplot(data=d, x='price', y='make')
plt.show()


# In[86]:


d[(d['make']=='dodge') & (d['price'] > 11000)]


# In[87]:


d_new.drop(29, axis=0, inplace= True)


# In[88]:


d[(d['make']=='honda') & (d['price'] > 11000)]


# In[89]:


d_new.drop(41, axis=0, inplace= True)


# In[90]:


d[(d['make']=='isuzu') & (d['price'] > 11000)]


# In[91]:


d_new.drop(45, axis=0, inplace= True)


# In[92]:


d_new.drop(46, axis=0, inplace= True)


# # Feature engineering 

# In[93]:


d_new.corr()


# In[94]:


# To remove multicolinearity and adjust the columns


# In[95]:


d_new['size'] = d_new['width']*d_new['height']


# In[96]:


d_new['average'] = (d['city-mpg'] + d['highway-mpg'])/2
# doing this the row by row average give to use


# In[97]:


x = d_new.drop(['symboling','width', 'height', 'city-mpg', 'highway-mpg', 'price'], axis=1)

y = d_new['price']


# In[98]:


x.head()


# In[99]:


x.corr()


# In[ ]:


s.

