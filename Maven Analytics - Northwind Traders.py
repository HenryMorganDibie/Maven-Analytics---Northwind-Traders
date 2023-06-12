#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


# In[2]:


categories = pd.read_csv('categories.csv')
categories.head()


# In[3]:


categories.isnull().sum()


# In[4]:


customers = pd.read_csv('customers.csv', encoding='latin1')
customers.head()


# In[5]:


customers.isnull().sum()


# In[6]:


employees = pd.read_csv('employees.csv')
employees.head()


# In[7]:


employees.isnull().sum()


# In[8]:


# Convert 'reportsTo' column to numeric type if needed
employees['reportsTo'] = pd.to_numeric(employees['reportsTo'], errors='coerce')

# Fill missing values with -1
employees['reportsTo'].fillna(-1, inplace=True)

# Verify the changes
print(employees['reportsTo'].isnull().sum())


# In[9]:


order_details = pd.read_csv('order_details.csv')
order_details.head()


# In[10]:


order_details.isnull().sum()


# In[11]:


orders = pd.read_csv('orders.csv')
orders.head()


# In[12]:


orders.isnull().sum()


# In[13]:


# Fill missing values in 'shippedDate' column with a default date or a specific value
orders['shippedDate'].fillna('Not Shipped', inplace=True)


# In[14]:


orders.isnull().sum()


# In[15]:


products = pd.read_csv('products.csv', encoding='latin1')
products.head()


# In[16]:


products.isnull().sum()


# In[17]:


shippers = pd.read_csv('shippers.csv')
shippers.head()


# In[18]:


shippers.isnull().sum()


# In[19]:


# Print each DataFrame
print("Categories DataFrame:")
print(categories.head())

print("\nCustomers DataFrame:")
print(customers.head())

print("\nEmployees DataFrame:")
print(employees.head())

print("\nOrder Details DataFrame:")
print(order_details.head())

print("\nOrders DataFrame:")
print(orders.head())

print("\nProducts DataFrame:")
print(products.head())

print("\nShippers DataFrame:")
print(shippers.head())


# In[20]:


import pandas as pd

# Merge Categories and Products DataFrames on 'categoryID'
merged_data = pd.merge(categories, products, on='categoryID')

# Merge merged_data and Order Details DataFrames on 'productID'
merged_data = pd.merge(merged_data, order_details, on='productID')

# Merge merged_data and Orders DataFrames on 'orderID'
merged_data = pd.merge(merged_data, orders, on='orderID')

# Merge merged_data and Customers DataFrames on 'customerID'
merged_data = pd.merge(merged_data, customers, on='customerID')

# Merge merged_data and Employees DataFrames on 'employeeID'
merged_data = pd.merge(merged_data, employees, on='employeeID')

# Merge merged_data and Shippers DataFrames on 'shipperID'
merged_data = pd.merge(merged_data, shippers, on='shipperID')

# Print the merged dataset
print(merged_data)


# In[21]:


merged_data.head()


# In[22]:


from powerbiclient import QuickVisualize, get_dataset_config, Report
from powerbiclient.authentication import DeviceCodeLoginAuthentication


# In[23]:


from powerbiclient import QuickVisualize, get_dataset_config
from powerbiclient.authentication import DeviceCodeLoginAuthentication

# Define the authentication method (Device Code Login)
device_auth = DeviceCodeLoginAuthentication()

# Specify the dataset configuration for your data
dataset_config = get_dataset_config(merged_data)

# Create a Power BI report from the data
pbi_visualize = QuickVisualize(dataset_config, auth=device_auth)

pbi_visualize


# ##EDA

# In[24]:


# Count the number of products in each category
category_counts = merged_data['categoryName'].value_counts()

# Create a bar chart
plt.figure(figsize=(10, 6))  
plt.bar(category_counts.index, category_counts.values)

# Add labels and title
plt.xlabel('Category')
plt.ylabel('Number of Products')
plt.title('Product Distribution by Category')

# Rotate x-axis labels for better readability
plt.xticks(rotation=45)

# Display the chart
plt.tight_layout()
plt.show()


# In[25]:


# Calculate the percentage of products in each category
category_percentages = (merged_data['categoryName'].value_counts() / len(merged_data)) * 100

# Create a pie chart
plt.figure(figsize=(8, 8))  
plt.pie(category_percentages, labels=category_percentages.index, autopct='%1.1f%%', startangle=90)

# Add title
plt.title('Product Distribution by Category')

# Display the chart
plt.axis('equal')
plt.show()


# In[26]:


# Create a histogram of product prices
plt.figure(figsize=(10, 6))  
plt.hist(merged_data['unitPrice_x'], bins=10, edgecolor='black')

# Add labels and title
plt.xlabel('Unit Price')
plt.ylabel('Frequency')
plt.title('Distribution of Product Prices')

# Display the chart
plt.show()


# In[27]:


# Create a scatter plot of product quantity vs. price
plt.figure(figsize=(10, 6))  # Adjust the figure size if needed
plt.scatter(merged_data['quantityPerUnit'], merged_data['unitPrice_x'], alpha=0.5)

# Add labels and title
plt.xlabel('Quantity Per Unit')
plt.ylabel('Unit Price')
plt.title('Product Quantity vs. Price')

# Display the chart
plt.show()


# In[28]:


# Calculate the average unit price by category
average_price_by_category = merged_data.groupby('categoryName')['unitPrice_x'].mean()

# Create a bar chart
plt.figure(figsize=(10, 6))  # Adjust the figure size if needed
average_price_by_category.plot(kind='bar', color='steelblue')

# Add labels and title
plt.xlabel('Category')
plt.ylabel('Average Unit Price')
plt.title('Average Unit Price by Category')

# Display the chart
plt.show()


# In[29]:


# Create a box plot of product prices by category
plt.figure(figsize=(10, 6))  
sns.boxplot(x='categoryName', y='unitPrice_x', data=merged_data)

# Add labels and title
plt.xlabel('Category')
plt.ylabel('Unit Price')
plt.title('Distribution of Product Prices by Category')

# Rotate x-axis labels if needed
plt.xticks(rotation=45)

# Display the chart
plt.show()


# In[30]:


# Convert the orderDate column to datetime type
merged_data['orderDate'] = pd.to_datetime(merged_data['orderDate'])

# Group the data by orderDate and calculate total sales
total_sales_by_date = merged_data.groupby('orderDate')['quantity'].sum()

# Create a line plot of total sales over time
plt.figure(figsize=(10, 6))  # Adjust the figure size if needed
total_sales_by_date.plot(kind='line', color='steelblue')

# Add labels and title
plt.xlabel('Order Date')
plt.ylabel('Total Sales')
plt.title('Total Sales Over Time')

# Display the chart
plt.show()


# In[31]:


# Create a scatter plot of unit price vs. quantity
plt.figure(figsize=(10, 6))  
plt.scatter(merged_data['unitPrice_x'], merged_data['quantity'], color='steelblue')

# Add labels and title
plt.xlabel('Unit Price')
plt.ylabel('Quantity')
plt.title('Unit Price vs. Quantity')

# Display the chart
plt.show()


# In[32]:


# Select the numerical columns for correlation analysis
numeric_columns = ['unitPrice_x', 'quantity', 'discount', 'freight']

# Calculate the correlation matrix
correlation_matrix = merged_data[numeric_columns].corr()

# Create a heatmap of the correlation matrix
plt.figure(figsize=(10, 8)) 
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm')

# Add title
plt.title('Correlation Matrix')

# Display the chart
plt.show()


# In[33]:


# Convert the orderDate column to datetime type
merged_data['orderDate'] = pd.to_datetime(merged_data['orderDate'])

# Group the data by orderDate and calculate the total order amount
daily_order_amount = merged_data.groupby('orderDate')['unitPrice_y'].sum()

# Create a line plot of daily order amounts
plt.figure(figsize=(10, 6))
plt.plot(daily_order_amount.index, daily_order_amount.values, color='steelblue')
plt.xlabel('Date')
plt.ylabel('Total Order Amount')
plt.title('Daily Total Order Amount')
plt.show()


# In[34]:


# Perform customer segmentation
# Example: Group customers based on total purchase value
customer_segments = merged_data.groupby('customerID')['unitPrice_x'].sum()

# Create customer profiles
customer_profiles = pd.DataFrame()
customer_profiles['customerID'] = customer_segments.index
customer_profiles['totalPurchaseValue'] = customer_segments.values

# Print the customer profiles
print(customer_profiles)


# In[35]:


merged_data.columns


# In[36]:


from sklearn.cluster import KMeans

# Perform customer segmentation
customer_segments = merged_data.groupby('customerID')['unitPrice_x'].sum()

# Create customer profiles
customer_profiles = pd.DataFrame()
customer_profiles['customerID'] = customer_segments.index
customer_profiles['totalPurchaseValue'] = customer_segments.values

# Perform clustering
X = customer_profiles[['totalPurchaseValue']]
kmeans = KMeans(n_clusters=3, random_state=42)
kmeans.fit(X)
customer_profiles['segment'] = kmeans.labels_

# Visualize customer segments
plt.figure(figsize=(30, 10))
colors = ['red', 'blue', 'green']
for segment in customer_profiles['segment'].unique():
    segment_data = customer_profiles[customer_profiles['segment'] == segment]
    plt.scatter(segment_data['customerID'], segment_data['totalPurchaseValue'], color=colors[segment])
plt.xlabel('Customer ID')
plt.ylabel('Total Purchase Value')
plt.title('Customer Segmentation based on Total Purchase Value')
plt.legend(customer_profiles['segment'].unique())
plt.show()

# Print the customer profiles with segments
print(customer_profiles)


# ##Market Analysis

# In[37]:


from mlxtend.frequent_patterns import apriori
from mlxtend.frequent_patterns import association_rules

# Perform one-hot encoding on the merged_data for market basket analysis
basket = merged_data.groupby(['orderID', 'productName'])['quantity'].sum().unstack().reset_index().fillna(0).set_index('orderID')

# Convert the quantity values to binary values (0 or 1)
def encode_units(x):
    if x <= 0:
        return 0
    if x >= 1:
        return 1

basket_sets = basket.applymap(encode_units)

# Perform market basket analysis using the Apriori algorithm
frequent_itemsets = apriori(basket_sets, min_support=0.05, use_colnames=True)

# Generate association rules
rules = association_rules(frequent_itemsets, metric="lift", min_threshold=1)

# Print the frequent itemsets
print("Frequent Itemsets:")
print(frequent_itemsets)

# Print the association rules
print("\nAssociation Rules:")
print(rules)


# In[38]:


import nltk


# In[39]:


nltk.download('vader_lexicon')


# In[40]:


import nltk
from nltk.sentiment import SentimentIntensityAnalyzer

# Initialize the sentiment analyzer
sia = SentimentIntensityAnalyzer()

# Example customer reviews
customer_reviews = [
    "The product was excellent! I'm really satisfied with my purchase.",
    "The customer service was terrible. They were unhelpful and rude.",
    "I had a mixed experience with the product. It has some good features, but also some drawbacks.",
    "I love this company! They always deliver top-notch products and provide great support."
]

# Perform sentiment analysis on each customer review
sentiments = []
for review in customer_reviews:
    sentiment = sia.polarity_scores(review)
    sentiments.append(sentiment)

# Print the sentiment scores for each review
for i, sentiment in enumerate(sentiments):
    print(f"Review {i+1}: {customer_reviews[i]}")
    print(f"Sentiment: {sentiment}")
    print()



# In[45]:


# Calculate average order value
average_order_value = merged_data['unitPrice_x'].mean()

# Estimate purchase frequency
total_orders = merged_data['orderID'].nunique()
total_customers = merged_data['customerID'].nunique()
purchase_frequency = total_orders / total_customers

# Define the average customer lifespan (in months)
average_customer_lifespan = 12

# Calculate the CLV
clv = average_order_value * purchase_frequency * average_customer_lifespan

# Print the CLV
print("Estimated CLV: $", clv)


# In[ ]:





# In[ ]:




