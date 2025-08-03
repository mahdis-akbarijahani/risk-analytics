
# In[1]:


import numpy as np
import os
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
import pandas as pd
import seaborn as sns
import warnings
warnings.filterwarnings("ignore")
import itertools


# ## Importing Data

# In[2]:


desktop_path = os.path.expanduser("~/Desktop/")
application_data = pd.read_csv(os.path.join(desktop_path, "application_data.csv"))
previous_application = pd.read_csv(os.path.join(desktop_path, "previous_application.csv"))
columns_description = pd.read_csv(os.path.join(desktop_path, "columns_description.csv"))


# ## Data Dimensions

# In[3]:


# Print shapes
print("application_data     :", application_data.shape)
print("previous_application :", previous_application.shape)
print("columns_description  :", columns_description.shape)



# ## First Few rows of Data

# In[4]:


pd.set_option("display.max_rows", None, "display.max_columns", None)
display("application_data")
display(application_data.head(3))


# In[5]:


display("previous_application ")
display(previous_application.head(3))


# 
# ## Term Dictionary 

# In[6]:


# Check existing columns
print("Columns in columns_description:", columns_description.columns)

# Drop the column safely
columns_description = columns_description.drop(['1'], axis=1, errors='ignore')

# Display the modified DataFrame
display(columns_description)


# ## Percentage of Missing values in previous_application

# In[7]:


import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Calculate the percentage of missing values
miss_previous_application = pd.DataFrame((previous_application.isnull().sum()) * 100 / previous_application.shape[0]).reset_index()
miss_previous_application.columns = ['column', 'missing_percentage']  # Rename columns for clarity
miss_previous_application["type"] = "previous_application"

# Create the plot
fig = plt.figure(figsize=(18, 6))
ax = sns.pointplot(x='column', y='missing_percentage', data=miss_previous_application, hue="type")

# Customize the plot
plt.xticks(rotation=90, fontsize=7)
plt.title("Percentage of Missing Values in previous_application")
plt.ylabel("PERCENTAGE")
plt.xlabel("COLUMNS")
ax.set_facecolor("white")  # Set axes background to white
fig.set_facecolor("white")  # Set figure background to white

plt.savefig("plot11.png", format="png", dpi=300)  # Saves the current plot as a PNG file with 300 dpi resolution

plt.show()


# In[8]:


round(100*(previous_application.isnull().sum()/len(previous_application.index)),2)


# ### Removing columns with missing values more than 50%
# 
# #### key point
# As per Industrial Standard, max Threshold limit can be between 40% to 50 % depending upon the data acquired in specific sector.

# In[9]:


previous_application=previous_application.drop([ 'AMT_DOWN_PAYMENT', 'RATE_DOWN_PAYMENT', 'RATE_INTEREST_PRIMARY',
       "RATE_INTEREST_PRIVILEGED"],axis=1)


# In[ ]:





# In[10]:


import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Assuming 'previous_application' DataFrame is already defined
fig = plt.figure(figsize=(18, 6))

# Calculate the percentage of missing values
miss_previous_application = pd.DataFrame((previous_application.isnull().sum()) * 100 / previous_application.shape[0]).reset_index()
miss_previous_application.columns = ["column", "missing_percentage"]
miss_previous_application["type"] = "previous_application"

# Plot using explicit argument names
ax = sns.pointplot(x="column", y="missing_percentage", data=miss_previous_application, hue="type")

# Customize plot
plt.xticks(rotation=90, fontsize=7)
plt.title("Percentage of Missing Values in previous_application")
plt.ylabel("PERCENTAGE")
plt.xlabel("COLUMNS")

# Set background colors
ax.set_facecolor("white")  # Set axes background to white
fig.set_facecolor("white")
plt.savefig("plot12.png", format="png", dpi=300)
plt.show()


# In[11]:


round(100*(previous_application.isnull().sum()/len(previous_application.index)),2)


# ### MISSING values Suggestion

# In[12]:


print("AMT_ANNUITY NULL COUNT:" ,previous_application['AMT_ANNUITY'].isnull().sum())


# In[13]:


previous_application['AMT_ANNUITY'].describe()


# In[14]:


sns.set_style('whitegrid') 
sns.distplot(previous_application['AMT_ANNUITY']) 
plt.savefig("plot13.png", format="png", dpi=300)
plt.show()


# ### Suggestion
# We can Fill NA with 15955 i.e. Mean for this field 

# In[15]:


print("AMT_GOODS_PRICE NULL COUNT:" ,previous_application['AMT_GOODS_PRICE'].isnull().sum())


# In[16]:


previous_application['AMT_GOODS_PRICE'].describe()


# In[17]:


sns.set_style('whitegrid') 
sns.distplot(previous_application['AMT_GOODS_PRICE']) 
plt.savefig("plot14.png", format="png", dpi=300)
plt.show()


# ### Suggestion
# We can Fill NA with 112320 i.e. Median for this field 

# In[18]:


print("NAME_TYPE_SUITE NULL COUNT:" ,previous_application['NAME_TYPE_SUITE'].isnull().sum())


# In[19]:


previous_application['NAME_TYPE_SUITE'].value_counts()


# ### Suggestion
# We can Fill NA with Unaccompanied  i.e. Mode for this field 

# In[20]:


print("CNT_PAYMENT NULL COUNT:" ,previous_application['CNT_PAYMENT'].isnull().sum())


# In[21]:


previous_application['CNT_PAYMENT'].describe()


# In[22]:


sns.set_style('whitegrid') 
sns.boxplot(previous_application['CNT_PAYMENT']) 
plt.savefig("plot15.png", format="png", dpi=300)
plt.show()


# ### Suggestion
# We can Fill NA with 12 i.e. Median for this field 

# In[23]:


print("DAYS_FIRST_DRAWING :" ,previous_application['CNT_PAYMENT'].isnull().sum())


# In[24]:


previous_application['DAYS_FIRST_DRAWING'].describe()


# In[25]:


sns.set_style('whitegrid') 
sns.boxplot(previous_application['DAYS_FIRST_DRAWING']) 
plt.savefig("plot16.png", format="png", dpi=300)
plt.show()


# ### Suggestion
# We can Fill NA with 365243 i.e. Median for this field 

# In[26]:


print("DAYS_FIRST_DUE :" ,previous_application['DAYS_FIRST_DUE'].isnull().sum())


# In[27]:


previous_application['DAYS_FIRST_DUE'].describe()


# In[28]:


sns.set_style('whitegrid') 
sns.boxplot(previous_application['DAYS_FIRST_DUE']) 
plt.savefig("plot17.png", format="png", dpi=300)
plt.show()


# ### Suggestion
# We can Fill NA with -831 i.e. Median for this field 

# In[29]:


print("DAYS_LAST_DUE_1ST_VERSION :" ,previous_application['DAYS_LAST_DUE_1ST_VERSION'].isnull().sum())


# In[30]:


previous_application['DAYS_LAST_DUE_1ST_VERSION'].describe()


# In[31]:


sns.set_style('whitegrid') 
sns.boxplot(previous_application['DAYS_LAST_DUE_1ST_VERSION']) 
plt.savefig("plot18.png", format="png", dpi=300)
plt.show()


# ### Suggestion
# We can Fill NA with -361 i.e. Median for this field 

# In[32]:


print("DAYS_LAST_DUE:" ,previous_application['DAYS_LAST_DUE'].isnull().sum())


# In[33]:


previous_application['DAYS_LAST_DUE'].describe()


# In[34]:


sns.set_style('whitegrid') 
sns.boxplot(previous_application['DAYS_LAST_DUE']) 
plt.savefig("plot19.png", format="png", dpi=300)
plt.show()


# ### Suggestion
# We can Fill NA with -537 i.e. Median for this field 

# In[35]:


print("DAYS_TERMINATION :" ,previous_application['DAYS_TERMINATION'].isnull().sum())


# In[36]:


previous_application['DAYS_TERMINATION'].describe()


# In[37]:


sns.set_style('whitegrid') 
sns.boxplot(previous_application['DAYS_TERMINATION']) 
plt.savefig("plot20.png", format="png", dpi=300)
plt.show()


# ### Suggestion
# We can Fill NA with -499 i.e. Median for this field 

# In[38]:


print("NFLAG_INSURED_ON_APPROVAL:" ,previous_application['NFLAG_INSURED_ON_APPROVAL'].isnull().sum())


# In[39]:


previous_application['NFLAG_INSURED_ON_APPROVAL'].value_counts()


# ### Suggestion
# We can Fill NA with 0 i.e. Mode for this field 

# In[40]:


previous_application.isnull().sum()


# In[41]:


print("AMT_CREDIT :" ,previous_application['AMT_CREDIT'].isnull().sum())


# In[42]:


previous_application['AMT_CREDIT'].describe()


# In[43]:


sns.set_style('whitegrid') 
sns.boxplot(previous_application['AMT_CREDIT']) 
plt.savefig("plot21.png", format="png", dpi=300)
plt.show()


# ### Suggestion
# We can Fill NA with 80541 i.e. Median for this field 

# In[44]:


print("PRODUCT_COMBINATION :" ,previous_application['PRODUCT_COMBINATION'].isnull().sum())


# In[45]:


previous_application['PRODUCT_COMBINATION'].value_counts()


# ### Suggestion
# We can Fill NA with Cash i.e. Mode for this field 

# In[46]:


class color:
   PURPLE = '\033[95m'
   CYAN = '\033[96m'
   DARKCYAN = '\033[36m'
   BLUE = '\033[94m'
   GREEN = '\033[92m'
   YELLOW = '\033[93m'
   RED = '\033[91m'
   BOLD = '\033[1m'
   UNDERLINE = '\033[4m'
   END = '\033[0m'


# ### Separating numerical and categorical columns from previous_application

# In[ ]:





# In[47]:


import numpy as np

# Filter for object dtypes excluding 'type'
obj_dtypes = [i for i in previous_application.select_dtypes(include=object).columns if i not in ["type"]]

# Filter for numeric dtypes excluding 'SK_ID_CURR' and 'TARGET'
num_dtypes = [i for i in previous_application.select_dtypes(include=np.number).columns if i not in ['SK_ID_CURR', 'TARGET']]


# In[48]:


print(color.BOLD + color.PURPLE + 'Categorical Columns' + color.END, "\n")
for x in range(len(obj_dtypes)): 
    print(obj_dtypes[x])


# In[49]:


print(color.BOLD + color.PURPLE + 'Numerical' + color.END, "\n")
for x in range(len(obj_dtypes)): 
    print(obj_dtypes[x])


# 
# ## Percentage of Missing values in application_data

# In[ ]:





# In[50]:


import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Set up the figure
fig = plt.figure(figsize=(18, 6))

# Calculate the percentage of missing values
miss_application_data = pd.DataFrame((application_data.isnull().sum()) * 100 / application_data.shape[0]).reset_index()
miss_application_data.columns = ["column", "missing_percentage"]
miss_application_data["type"] = "application_data"

# Plot using explicit argument names for x, y, and data with a dark blue color for the line
ax = sns.pointplot(
    x="column", 
    y="missing_percentage", 
    data=miss_application_data, 
    hue="type", 
    palette=["#1f77b4"],  # dark blue color for the points
    markers='o',  # marker style
    dodge=True  # separate points for each hue
)

# Customize plot
plt.xticks(rotation=90, fontsize=7)
plt.title("Percentage of Missing Values in application_data")
plt.ylabel("PERCENTAGE")
plt.xlabel("COLUMNS")

# Set background colors
ax.set_facecolor("white")  # Change the axes background to light gray
fig.patch.set_facecolor("white")  # Set the figure background color to white
plt.savefig("plot22.png", format="png", dpi=300)
plt.show()


# In[51]:


round(100*(application_data.isnull().sum()/len(application_data.index)),2)


# ### Removing columns with missing values more than 40%
# 
# As per Industrial Standard, max Threshold limit can be between 40% to 50 % depending upon the data acquired in specific sector.

# In[52]:


application_data=application_data.drop([ 'EXT_SOURCE_1', 'EXT_SOURCE_2', 'EXT_SOURCE_3',
       'APARTMENTS_AVG', 'BASEMENTAREA_AVG', 'YEARS_BEGINEXPLUATATION_AVG',
       'YEARS_BUILD_AVG', 'COMMONAREA_AVG', 'ELEVATORS_AVG', 'ENTRANCES_AVG',
       'FLOORSMAX_AVG', 'FLOORSMIN_AVG', 'LANDAREA_AVG',
       'LIVINGAPARTMENTS_AVG', 'LIVINGAREA_AVG', 'NONLIVINGAPARTMENTS_AVG',
       'NONLIVINGAREA_AVG', 'APARTMENTS_MODE', 'BASEMENTAREA_MODE',
       'YEARS_BEGINEXPLUATATION_MODE', 'YEARS_BUILD_MODE', 'COMMONAREA_MODE',
       'ELEVATORS_MODE', 'ENTRANCES_MODE', 'FLOORSMAX_MODE', 'FLOORSMIN_MODE',
       'LANDAREA_MODE', 'LIVINGAPARTMENTS_MODE', 'LIVINGAREA_MODE',
       'NONLIVINGAPARTMENTS_MODE', 'NONLIVINGAREA_MODE', 'APARTMENTS_MEDI',
       'BASEMENTAREA_MEDI', 'YEARS_BEGINEXPLUATATION_MEDI', 'YEARS_BUILD_MEDI',
       'COMMONAREA_MEDI', 'ELEVATORS_MEDI', 'ENTRANCES_MEDI', 'FLOORSMAX_MEDI',
       'FLOORSMIN_MEDI', 'LANDAREA_MEDI', 'LIVINGAPARTMENTS_MEDI',
       'LIVINGAREA_MEDI', 'NONLIVINGAPARTMENTS_MEDI', 'NONLIVINGAREA_MEDI',
       'FONDKAPREMONT_MODE', 'HOUSETYPE_MODE', 'TOTALAREA_MODE',
       'WALLSMATERIAL_MODE', 'EMERGENCYSTATE_MODE',"OWN_CAR_AGE","OCCUPATION_TYPE"],axis=1)


# In[53]:


# List of columns to drop
columns_to_drop = [
    'EXT_SOURCE_1', 'EXT_SOURCE_2', 'EXT_SOURCE_3', 'APARTMENTS_AVG', 
    'BASEMENTAREA_AVG', 'YEARS_BEGINEXPLUATATION_AVG', 'YEARS_BUILD_AVG', 
    'COMMONAREA_AVG', 'ELEVATORS_AVG', 'ENTRANCES_AVG', 'FLOORSMAX_AVG', 
    'FLOORSMIN_AVG', 'LANDAREA_AVG', 'LIVINGAPARTMENTS_AVG', 'LIVINGAREA_AVG', 
    'NONLIVINGAPARTMENTS_AVG', 'NONLIVINGAREA_AVG', 'APARTMENTS_MODE', 
    'BASEMENTAREA_MODE', 'YEARS_BEGINEXPLUATATION_MODE', 'YEARS_BUILD_MODE', 
    'COMMONAREA_MODE', 'ELEVATORS_MODE', 'ENTRANCES_MODE', 'FLOORSMAX_MODE', 
    'FLOORSMIN_MODE', 'LANDAREA_MODE', 'LIVINGAPARTMENTS_MODE', 'LIVINGAREA_MODE', 
    'NONLIVINGAPARTMENTS_MODE', 'NONLIVINGAREA_MODE', 'APARTMENTS_MEDI', 
    'BASEMENTAREA_MEDI', 'YEARS_BEGINEXPLUATATION_MEDI', 'YEARS_BUILD_MEDI', 
    'COMMONAREA_MEDI', 'ELEVATORS_MEDI', 'ENTRANCES_MEDI', 'FLOORSMAX_MEDI', 
    'FLOORSMIN_MEDI', 'LANDAREA_MEDI', 'LIVINGAPARTMENTS_MEDI', 
    'LIVINGAREA_MEDI', 'NONLIVINGAPARTMENTS_MEDI', 'NONLIVINGAREA_MEDI', 
    'FONDKAPREMONT_MODE', 'HOUSETYPE_MODE', 'TOTALAREA_MODE', 
    'WALLSMATERIAL_MODE', 'EMERGENCYSTATE_MODE', 'OWN_CAR_AGE', 
    'OCCUPATION_TYPE'
]

# Check which columns actually exist in the DataFrame
columns_in_data = application_data.columns.intersection(columns_to_drop)

# Drop only the existing columns
application_data = application_data.drop(columns_in_data, axis=1)


# In[ ]:





# In[54]:


import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

# Create a figure
fig = plt.figure(figsize=(18, 6))

# Create a DataFrame for missing data percentage and reset the index
miss_application_data = pd.DataFrame((application_data.isnull().sum()) * 100 / application_data.shape[0]).reset_index()

# Rename the columns for clarity
miss_application_data.columns = ['column', 'percentage']

# Add a 'type' column to indicate the dataset
miss_application_data["type"] = "application_data"

# Plot the pointplot with dark blue color for the line and light gray for the markers
ax = sns.pointplot(
    x="column", 
    y="percentage", 
    data=miss_application_data, 
    hue="type", 
    palette=["#1f77b4"],  # dark blue for the line
    markers='o',  # marker style
    dodge=True  # separate points for each hue
)

# Rotate x-axis labels for better readability
plt.xticks(rotation=90, fontsize=7)

# Set plot title and labels
plt.title("Percentage of Missing Values in application_data")
plt.ylabel("PERCENTAGE")
plt.xlabel("COLUMNS")

# Customize the plot background colors
ax.set_facecolor("white")  # Change the face color to light gray
fig.patch.set_facecolor("white")  # Set figure background color to white
plt.savefig("plot23.png", format="png", dpi=300)
# Show the plot
plt.show()


# In[55]:


round(100*(application_data.isnull().sum()/len(application_data.index)),2)


# ### MISSING values Suggestion

# In[56]:


print("AMT_REQ_CREDIT_BUREAU_DAY NAN COUNT :" ,application_data['AMT_REQ_CREDIT_BUREAU_DAY'].isnull().sum())


# In[57]:


application_data['AMT_REQ_CREDIT_BUREAU_DAY'].describe()


# ### Suggestion
# We can Fill NA with 0 i.e. Median for this field 

# In[58]:


print("AMT_REQ_CREDIT_BUREAU_HOUR NAN COUNT :" ,application_data['AMT_REQ_CREDIT_BUREAU_HOUR'].isnull().sum())


# In[59]:


application_data['AMT_REQ_CREDIT_BUREAU_HOUR'].describe()


# ### Suggestion
# We can Fill NA with 0 i.e. Median for this field 

# In[60]:


print("AMT_REQ_CREDIT_BUREAU_MON NAN COUNT :" ,application_data['AMT_REQ_CREDIT_BUREAU_MON'].isnull().sum())


# In[61]:


application_data['AMT_REQ_CREDIT_BUREAU_MON'].describe()


# ### Suggestion
# We can Fill NA with 0 i.e. Median for this field 

# In[62]:


print("AMT_REQ_CREDIT_BUREAU_QRT NAN COUNT :" ,application_data['AMT_REQ_CREDIT_BUREAU_QRT'].isnull().sum())


# 
# ### Suggestion
# We can Fill NA with 0 i.e. Median for this field 

# In[63]:


print("AMT_REQ_CREDIT_BUREAU_WEEK NAN COUNT :" ,application_data['AMT_REQ_CREDIT_BUREAU_WEEK'].isnull().sum())


# In[64]:


application_data['AMT_REQ_CREDIT_BUREAU_WEEK'].describe()


# ### Suggestion
# We can Fill NA with 0 i.e. Median for this field 

# In[65]:


print("AMT_REQ_CREDIT_BUREAU_YEAR NAN COUNT :" ,application_data['AMT_REQ_CREDIT_BUREAU_YEAR'].isnull().sum())


# In[66]:


application_data['AMT_REQ_CREDIT_BUREAU_YEAR'].describe()


# ### Suggestion
# We can Fill NA with 0 i.e. Median for this field 

# In[67]:


print("DEF_30_CNT_SOCIAL_CIRCLE NAN COUNT :" ,application_data['DEF_30_CNT_SOCIAL_CIRCLE'].isnull().sum())


# In[68]:


application_data['DEF_30_CNT_SOCIAL_CIRCLE'].describe()


# ### Suggestion
# We can Fill NA with 0 i.e. Median for this field 

# In[69]:


print("DEF_30_CNT_SOCIAL_CIRCLE :" ,application_data['DEF_30_CNT_SOCIAL_CIRCLE'].isnull().sum())


# In[70]:


application_data['DEF_30_CNT_SOCIAL_CIRCLE'].describe()


# ### Suggestion
# We can Fill NA with 0 i.e. Median for this field 

# In[71]:


print("OBS_60_CNT_SOCIAL_CIRCLE :" ,application_data['OBS_60_CNT_SOCIAL_CIRCLE'].isnull().sum())


# In[72]:


application_data['OBS_60_CNT_SOCIAL_CIRCLE'].describe()


# ### Suggestion
# We can Fill NA with 0 i.e. Median for this field 

# In[73]:


print("DEF_60_CNT_SOCIAL_CIRCLE :" ,application_data['DEF_60_CNT_SOCIAL_CIRCLE'].isnull().sum())


# In[74]:


application_data['DEF_60_CNT_SOCIAL_CIRCLE'].describe()


# ### Suggestion
# We can Fill NA with 0 i.e. Median for this field 

# In[75]:


application_data.isnull().sum()


# In[76]:


print("AMT_ANNUITY  :" ,application_data['AMT_ANNUITY'].isnull().sum())


# In[77]:


application_data['AMT_ANNUITY'].describe()


# In[78]:


import seaborn as sns
import matplotlib.pyplot as plt

# Set style for the plot
sns.set_style('whitegrid')

# Plot distribution with dark blue for the line and muted gray for the shade
sns.distplot(
    application_data['AMT_ANNUITY'], 
    color="#1f77b4",  # dark blue for the line
    kde_kws={"shade": True, "color": "#D3D3D3"}  # light gray for the shaded area
)
plt.savefig("plot24.png", format="png", dpi=300)
# Show the plot
plt.show()


# ### Suggestion
# We can Fill NA with 0 i.e. Mean for this field as it's right skewed graph

# In[79]:


print("AMT_GOODS_PRICE   :" ,application_data['AMT_GOODS_PRICE'].isnull().sum())


# In[80]:


application_data['AMT_GOODS_PRICE'].describe()


# In[81]:


import seaborn as sns
import matplotlib.pyplot as plt

# Set style for the plot
sns.set_style('whitegrid')

# Plot distribution with dark blue for the line and muted gray for the shade
sns.distplot(
    application_data['AMT_GOODS_PRICE'], 
    color="#1f77b4",  # dark blue for the line
    kde_kws={"shade": True, "color": "#D3D3D3"}  # light gray for the shaded area
)
plt.savefig("plot25.png", format="png", dpi=300)
# Show the plot
plt.show()


# ### Suggestion
# We can Fill NA with 0 i.e. Mean for this field as it's right skewed graph

# In[82]:


print("NAME_TYPE_SUITE :" ,application_data['NAME_TYPE_SUITE'].isnull().sum())


# In[83]:


application_data['NAME_TYPE_SUITE'].value_counts()


# ### Suggestion
# We can Fill NA with "Unaccompanied" i.e. Mode for this field 

# In[84]:


print("CNT_FAM_MEMBERS :" ,application_data['CNT_FAM_MEMBERS'].isnull().sum())


# In[85]:


application_data['CNT_FAM_MEMBERS'].describe()


# In[86]:


import seaborn as sns
import matplotlib.pyplot as plt

# Set style for the plot
sns.set_style('whitegrid')

# Plot distribution with dark blue for the line and muted gray for the shade
sns.distplot(
    application_data['CNT_FAM_MEMBERS'], 
    color="#1f77b4",  # dark blue for the line
    kde_kws={"shade": True, "color": "#D3D3D3"}  # light gray for the shaded area
)
plt.savefig("plot26.png", format="png", dpi=300)
# Show the plot
plt.show()


# ### Suggestion
# We can Fill NA with 2 i.e. Median for this field, Mean is not be used as this field needs to be Whole number

# In[87]:


print("DAYS_LAST_PHONE_CHANGE :" ,application_data['DAYS_LAST_PHONE_CHANGE'].isnull().sum())


# In[88]:


application_data['DAYS_LAST_PHONE_CHANGE'].describe()


# In[89]:


import statistics 
statistics.mode(application_data['DAYS_LAST_PHONE_CHANGE'])


# ### Suggestion
# We can Fill NA with 0 i.e. Mode for this field

# ## Print the information about the attributes of application_data

# In[90]:


print(type(application_data.info()))


# # Converting negative values to absolute values
# 

# In[91]:


application_data['DAYS_BIRTH'] = abs(application_data['DAYS_BIRTH'])
application_data['DAYS_ID_PUBLISH'] = abs(application_data['DAYS_ID_PUBLISH'])
application_data['DAYS_ID_PUBLISH'] = abs(application_data['DAYS_ID_PUBLISH'])
application_data['DAYS_LAST_PHONE_CHANGE'] = abs(application_data['DAYS_LAST_PHONE_CHANGE'])



# In[92]:


display("application_data")
display(application_data.head())


# ### Separating numerical and categorical  in application_data
# 

# In[ ]:





# In[93]:


import numpy as np

# Separating numerical and categorical columns in application_data
obj_dtypes = [i for i in application_data.select_dtypes(include=object).columns if i not in ["type"]]

num_dtypes = [i for i in application_data.select_dtypes(include=np.number).columns if i not in ['SK_ID_CURR', 'TARGET']]


# In[94]:


print(color.BOLD + color.PURPLE + 'Categorical Columns' + color.END, "\n")
for x in range(len(obj_dtypes)): 
    print(obj_dtypes[x])


# In[95]:


print(color.BOLD + color.PURPLE +"Numerical Columns" + color.END, "\n")
for x in range(len(num_dtypes)): 
    print(num_dtypes[x])


# 
# ## Imbalance percentage

# In[96]:


import matplotlib.pyplot as plt

# Create a figure
fig = plt.figure(figsize=(13,6))

# First subplot
plt.subplot(121)

# Plot pie chart with muted blue and gray colors
application_data["CODE_GENDER"].value_counts().plot.pie(
    autopct="%1.0f%%",
    colors=["#A1C6EA", "#D3D3D3"],  # muted light blue and light gray (nude tones)
    startangle=60,
    wedgeprops={"linewidth": 2, "edgecolor": "k"},
    explode=[0.05, 0, 0],  # explode the first slice slightly
    shadow=True
)

# Set title
plt.title("Distribution of Gender")
plt.savefig("plot27.png", format="png", dpi=300)
# Show the plot
plt.show()


# ### Point to infer from the graph
# It's non balanced data

# 
# ## Distribution of Target variable

# TARGET :Target variable (1 - client with payment difficulties: he/she had late payment more than X days on at least one of the first Y installments of the loan in sample, 0 - all other cases)
# 

# In[97]:


import matplotlib.pyplot as plt

# Create a figure
plt.figure(figsize=(14,7))

# First subplot: Pie chart for the "TARGET" distribution
plt.subplot(121)
application_data["TARGET"].value_counts().plot.pie(
    autopct="%1.0f%%",
    colors=["#A1C6EA", "#D3D3D3"],  # muted light blue and light gray (nude tones)
    startangle=60,
    labels=["repayer", "defaulter"],
    wedgeprops={"linewidth": 2, "edgecolor": "k"},
    explode=[0.1, 0],  # slightly explode the first slice
    shadow=True
)

# Set title for the pie chart
plt.title("Distribution of Target Variable")

# Second subplot: Horizontal bar chart for the "TARGET" distribution
plt.subplot(122)
ax = application_data["TARGET"].value_counts().plot(kind="barh", color=["#A1C6EA", "#D3D3D3"])  # Apply the same muted colors

# Add text annotations to the bar chart
for i, j in enumerate(application_data["TARGET"].value_counts().values):
    ax.text(0.7, i, j, weight="bold", fontsize=20)

# Set title for the bar chart
plt.title("Count of Target Variable")
plt.savefig("plot28.png", format="png", dpi=300)
# Show the plot
plt.show()


# ### Point to infer from the graph
# 8% out of total client population have difficulties in repaying loans.

# ### Concatenating application_data and previous_application

# In[98]:


application_data_x = application_data[[x for x in application_data.columns if x not in ["TARGET"]]]
previous_application_x = previous_application[[x for x in previous_application.columns if x not in ["TARGET"]]]
application_data_x["type"] = "application_data"
previous_application_x["type"] = "previous_application"
data = pd.concat([application_data_x,previous_application_x],axis=0) 


# ## Distribution in Contract types in application_data 
# 
# NAME_CONTRACT_TYPE : Identification if loan is cash , consumer or revolving
# 

# In[99]:


import matplotlib.pyplot as plt

# Create the figure and subplot
fig = plt.figure(figsize=(13, 6))
plt.subplot(121)

# Get value counts for pie chart data based on contract types
contract_counts = data[data["type"] == "application_data"]["NAME_CONTRACT_TYPE"].value_counts()

# Dynamically set explode to match the number of categories
explode_values = [0.05] + [0] * (len(contract_counts) - 1)  # Explodes the first segment only

# Generate the pie chart
contract_counts.plot.pie(
    autopct="%1.0f%%",
    colors=["#A1C6EA", "#D3D3D3", "#FFC107"],  # Colors for different contract types
    startangle=60,
    wedgeprops={"linewidth": 2, "edgecolor": "k"},
    explode=explode_values,  # Dynamically adjusted explode list
    shadow=True
)

# Set the title
plt.title("Distribution of Contract Types in Application Data")
plt.savefig("plot29.png", format="png", dpi=300)
# Display the plot
plt.show()


# ### Point to infer from the graph
# 
# The percentage of revolving loans and cash loans are 10% & 90%.

# ## Gender Distribution in application_data 

# In[100]:


import matplotlib.pyplot as plt

# Create the figure and subplot
fig = plt.figure(figsize=(13, 6))
plt.subplot(121)

# Get value counts of gender in application data
gender_counts = data[data["type"] == "application_data"]["CODE_GENDER"].value_counts()

# Determine explode list based on the number of categories
explode = [0.05] + [0] * (len(gender_counts) - 1)  # Explode first slice, no others

# Generate the pie chart with the new colors
gender_counts.plot.pie(
    autopct="%1.0f%%",
    colors=["#A1C6EA", "#D3D3D3"],  # Make sure to match the number of categories
    startangle=60,
    wedgeprops={"linewidth": 2, "edgecolor": "k"},
    explode=explode,  # Use the dynamically created explode list
    shadow=True
)

# Set the title
plt.title("Distribution of Gender in Application Data")
plt.savefig("plot30.png", format="png", dpi=300)
# Display the plot
plt.show()


# ### Point to infer from the graph
# 
# Female : 66% 
# 
# Male : 34% 

# ## Distribution of Contract type by gender

# In[101]:


import matplotlib.pyplot as plt
import seaborn as sns

# Create the figure and subplot
fig = plt.figure(figsize=(13, 6))
plt.subplot(121)

# Filter the data and create the count plot with the new colors
ax = sns.countplot(
    x="NAME_CONTRACT_TYPE",  # Specify x using the x keyword argument
    hue="CODE_GENDER",
    data=data[data["type"] == "application_data"],
    palette=["#A1C6EA", "#D3D3D3"]  # Updated color palette
)

# Set the background color and title
ax.set_facecolor("lightgrey")
ax.set_title("Distribution of Contract Type by Gender - Application Data")
plt.savefig("plot31.png", format="png", dpi=300)
# Display the plot
plt.show()


# ### Point to infer from the graph
# 
# Cash loans is always prefered over Revolving loans by both genders

# ## Distribution of client owning a car and by gender
# 
# FLAG_OWN_CAR Flag if the client owns a car .

# In[102]:


import matplotlib.pyplot as plt

# Create the figure and define the size
fig = plt.figure(figsize=(10, 6))

# First subplot: Distribution of clients owning a car
plt.subplot(121)
data["FLAG_OWN_CAR"].value_counts().plot.pie(
    autopct="%1.0f%%",
    colors=["#A1C6EA", "#D3D3D3"],  # Updated colors
    startangle=60,
    wedgeprops={"linewidth": 2, "edgecolor": "k"},
    explode=[0.05, 0],  # Explode the first slice slightly
    shadow=True
)
plt.title("Distribution of Clients Owning a Car")

# Second subplot: Distribution of clients owning a car by gender
plt.subplot(122)
# Get value counts for the 'CODE_GENDER' for clients who own a car
gender_counts = data[data["FLAG_OWN_CAR"] == "Y"]["CODE_GENDER"].value_counts()

# Check the number of unique values in gender_counts
num_categories = len(gender_counts)

# Create the pie chart
gender_counts.plot.pie(
    autopct="%1.0f%%",
    colors=["#A1C6EA", "#D3D3D3"][:num_categories],  # Ensure colors match the categories
    startangle=90,
    wedgeprops={"linewidth": 2, "edgecolor": "k"},
    explode=[0.05] + [0] * (num_categories - 1),  # Explode the first slice, others stay
    shadow=True
)
plt.title("Distribution of Clients Owning a Car by Gender")
plt.savefig("plot32.png", format="png", dpi=300)
# Display the plots
plt.show()


# ### Point to infer from the graph
# 
# SUBPLOT 1 : Distribution of client owning a car. 34% of clients own a car .
# 
# SUBPLOT 2 : Distribution of client owning a car by gender. Out of total clients who own car 57% are male and 43% are female.

# ## Distribution of client owning a house or flat and by gender
# 
# FLAG_OWN_REALTY - Flag if client owns a house or flat

# In[103]:


import matplotlib.pyplot as plt

# Create the figure and define the size
plt.figure(figsize=(13, 6))

# First subplot: Distribution of clients owning a house or flat
plt.subplot(121)
# Get value counts for FLAG_OWN_REALTY
realty_counts = data["FLAG_OWN_REALTY"].value_counts()
realty_counts.plot.pie(
    autopct="%1.0f%%",
    colors=["#A1C6EA", "#D3D3D3"],  # Updated colors
    startangle=90,
    wedgeprops={"linewidth": 2, "edgecolor": "k"},
    explode=[0.05] * len(realty_counts),  # Explode all slices slightly
    shadow=True
)
plt.title("Distribution of Clients Owning a House or Flat")

# Second subplot: Distribution of clients owning a house or flat by gender
plt.subplot(122)
gender_counts = data[data["FLAG_OWN_REALTY"] == "Y"]["CODE_GENDER"].value_counts()
gender_counts.plot.pie(
    autopct="%1.0f%%",
    colors=["#A1C6EA", "#D3D3D3"],  # Updated colors
    startangle=90,
    wedgeprops={"linewidth": 2, "edgecolor": "k"},
    explode=[0.05] * len(gender_counts),  # Explode all slices slightly
    shadow=True
)
plt.title("Distribution of Clients Owning a House or Flat by Gender")
plt.savefig("plot33.png", format="png", dpi=300)
# Display the plots
plt.show()


# ### Point to infer from the graph
# 
# SUBPLOT 1 : Distribution of client owning a house or flat . 69% of clients own a flat or house .
# 
# SUBPLOT 2 : Distribution of client owning a house or flat by gender . Out of total clients who own house 67% are female and 33% are male.

# ## Distribution of Number of children and family members of client by repayment status.
# 
# CNT_CHILDREN - Number of children the client has.
# 
# CNT_FAM_MEMBERS - How many family members does client have.

# In[104]:


import matplotlib.pyplot as plt
import seaborn as sns

# Create the figure and define the size
fig = plt.figure(figsize=(12, 10))
fig.set_facecolor("lightblue")  # Set the figure background color

# First subplot: Distribution of number of children by repayment status
plt.subplot(211)
sns.countplot(x="CNT_CHILDREN", hue="TARGET", data=application_data, palette=["#A1C6EA", "#D3D3D3"])
plt.legend(loc="upper center")
plt.title("Distribution of Number of Children Client Has by Repayment Status")

# Second subplot: Distribution of number of family members by repayment status
plt.subplot(212)
sns.countplot(x="CNT_FAM_MEMBERS", hue="TARGET", data=application_data, palette=["#A1C6EA", "#D3D3D3"])
plt.legend(loc="upper center")
plt.title("Distribution of Number of Family Members Client Has by Repayment Status")

# Display the plots
plt.tight_layout()  # Adjusts the layout to prevent overlap
plt.savefig("plot34.png", format="png", dpi=300)
plt.show()


# 
# ## Distribution of contract type ,gender ,own car ,own house with respect to Repayment status(Target variable)

# In[105]:


import matplotlib.pyplot as plt
import seaborn as sns
import itertools

# Filtering data for default and non-default
default = application_data[application_data["TARGET"] == 1][['NAME_CONTRACT_TYPE', 'CODE_GENDER', 'FLAG_OWN_CAR', 'FLAG_OWN_REALTY']]
non_default = application_data[application_data["TARGET"] == 0][['NAME_CONTRACT_TYPE', 'CODE_GENDER', 'FLAG_OWN_CAR', 'FLAG_OWN_REALTY']]

# Column names and length
d_cols = ['NAME_CONTRACT_TYPE', 'CODE_GENDER', 'FLAG_OWN_CAR', 'FLAG_OWN_REALTY']
d_length = len(d_cols)

# Save individual plots for default clients
for i, col in enumerate(d_cols):
    fig, ax = plt.subplots(figsize=(4, 4))
    ax.pie(
        default[col].value_counts(),
        autopct="%1.0f%%",
        colors=["#A1C6EA", "#D3D3D3"],  # Light blue and gray
        startangle=90,
        wedgeprops={"linewidth": 1, "edgecolor": "white"},
        shadow=True
    )
    circ = plt.Circle((0, 0), 0.7, color="white")
    ax.add_artist(circ)
    ax.set_ylabel("")
    ax.set_title(f"{col} - Defaulter")
    fig.savefig(f"{col}_defaulter.png", format="png", dpi=300)
    plt.close(fig)

# Save individual plots for non-default clients
for i, col in enumerate(d_cols):
    fig, ax = plt.subplots(figsize=(4, 4))
    ax.pie(
        non_default[col].value_counts(),
        autopct="%1.0f%%",
        colors=["#1F4E79", "#D3D3D3"],  # Dark blue and gray
        startangle=90,
        wedgeprops={"linewidth": 1, "edgecolor": "white"},
        shadow=True
    )
    circ = plt.Circle((0, 0), 0.7, color="white")
    ax.add_artist(circ)
    ax.set_ylabel("")
    ax.set_title(f"{col} - Repayer")
    fig.savefig(f"{col}_repayer.png", format="png", dpi=300)
    plt.close(fig)
plt.savefig("plot36.png", format="png", dpi=300)
# Optional: Display the plots (if needed)
plt.show()


# ### Point to infer from the graph
# 
# Percentage of males is 10% more in defaults than non defaulters.
# 
# Percentage of Cash Loans is 4% more in defaults than Revolving Loans.

# ### Distribution of amount data
# 
# AMT_INCOME_TOTAL - Income of the client
# 
# AMT_CREDIT - Credit amount of the loan
# 
# AMT_ANNUITY - Loan annuity
# 
# AMT_GOODS_PRICE - For consumer loans it is the price of the goods for which the loan is given

# In[106]:


import matplotlib.pyplot as plt
import seaborn as sns
import itertools

# Define columns and colors for the plots
cols = ['AMT_INCOME_TOTAL', 'AMT_CREDIT', 'AMT_ANNUITY', 'AMT_GOODS_PRICE']
length = len(cols)

# Define the updated color scheme (light blue, dark blue, dark gray, light gray)
cs = ["#A1C6EA", "#003366", "#505050", "#D3D3D3"]  # Light blue, dark blue, dark gray, light gray

# Create a figure with a light grey background
fig = plt.figure(figsize=(18, 18))
fig.set_facecolor("lightgrey")

# Loop through the columns, creating subplots
for i, j, k in itertools.zip_longest(cols, range(length), cs):
    plt.subplot(2, 2, j + 1)
    sns.histplot(data[data[i].notnull()][i], color=k, kde=True)  # Using histplot for better visuals
    plt.axvline(data[i].mean(), label="mean", linestyle="dashed", color="k")
    plt.legend(loc="best")
    plt.title(i)
    plt.subplots_adjust(hspace=0.2)
plt.savefig("plot37.png", format="png", dpi=300)
# Display the plot
plt.show()


# 
# ## Comparing summary statistics between defaulters and non - defaulters for loan amounts.
# 
# 

# In[107]:


import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import itertools

# Assuming application_data and cols are already defined

# Grouping and preparing the data
df = application_data.groupby("TARGET")[cols].describe().transpose().reset_index()
df = df[df["level_1"].isin([ 'mean', 'std', 'min', 'max'])] 
df_x = df[["level_0", "level_1", 0]]
df_y = df[["level_0", "level_1", 1]]
df_x = df_x.rename(columns={'level_0':"amount_type", 'level_1':"statistic", 0:"amount"})
df_x["type"] = "REPAYER"
df_y = df_y.rename(columns={'level_0':"amount_type", 'level_1':"statistic", 1:"amount"})
df_y["type"] = "DEFAULTER"
df_new = pd.concat([df_x, df_y], axis=0)

# List of unique statistics
stat = df_new["statistic"].unique().tolist()
length = len(stat)

# Create the plots
plt.figure(figsize=(13, 15))

# Define custom color palette
custom_palette = {"REPAYER": "#A1C6EA", "DEFAULTER": "#D3D3D3"}

# Loop to generate subplots
for i, j in itertools.zip_longest(stat, range(length)):
    plt.subplot(2, 2, j + 1)
    # Corrected the syntax for sns.barplot
    fig = sns.barplot(
        x="amount_type", 
        y="amount", 
        hue="type", 
        data=df_new[df_new["statistic"] == i],
        palette=custom_palette
    )
    plt.title(f"{i} -- Defaulters vs Non-defaulters")
    plt.subplots_adjust(hspace=0.4)
    fig.set_facecolor("white")  # Set background color for each subplot
plt.savefig("plot37.png", format="png", dpi=300)
plt.show()


# ### Point to infer from the graph
# 
# #### Income of client -
# 
# 1 . Average income of clients who default and who do not are almost same.
# 
# 2 . Standard deviation in income of client who default is very high compared to who do not default.
# 
# 3 . Clients who default also has maximum income earnings
# 
# #### Credit amount of the loan ,Loan annuity,Amount goods price -
# 
# 1 . Statistics between credit amounts,Loan annuity and Amount goods price given to cilents who default and who dont are almost similar.

# ## Average Income,credit,annuity & goods_price by gender
# 

# In[108]:


import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

# Assuming `data` is already defined and contains the necessary columns

cols = ['AMT_INCOME_TOTAL', 'AMT_CREDIT', 'AMT_ANNUITY', 'AMT_GOODS_PRICE']

# Group by gender and calculate the mean for the selected columns
df1 = data.groupby("CODE_GENDER")[cols].mean().transpose().reset_index()

# Prepare data for females
df_f = df1[["index", "F"]]
df_f = df_f.rename(columns={'index': "amt_type", 'F': "amount"})
df_f["gender"] = "FEMALE"

# Prepare data for males
df_m = df1[["index", "M"]]
df_m = df_m.rename(columns={'index': "amt_type", 'M': "amount"})
df_m["gender"] = "MALE"

# Prepare data for "XNA" gender (if applicable)
df_xna = df1[["index", "XNA"]]
df_xna = df_xna.rename(columns={'index': "amt_type", 'XNA': "amount"})
df_xna["gender"] = "XNA"

# Combine all the data
df_gen = pd.concat([df_m, df_f, df_xna], axis=0)

# Define custom color palette
custom_palette = {"MALE": "#A1C6EA", "FEMALE": "#D3D3D3", "XNA": "#B0B0B0"}  # Add a neutral color for "XNA" if needed

# Create the bar plot
plt.figure(figsize=(12, 5))

# Correctly specify the arguments for sns.barplot
ax = sns.barplot(data=df_gen, x="amt_type", y="amount", hue="gender", palette=custom_palette)

# Set the title for the plot
plt.title("Average Income, Credit, Annuity & Goods Price by Gender")
plt.savefig("plot38.png", format="png", dpi=300)
# Show the plot
plt.show()


# ## Scatter plot between credit amount and annuity amount

# In[109]:


fig = plt.figure(figsize=(10,8))
plt.scatter(application_data[application_data["TARGET"]==0]['AMT_ANNUITY'],application_data[application_data["TARGET"]==0]['AMT_CREDIT'],s=35,
            color="#A1C6EA",alpha=.5,label="REPAYER",linewidth=.5,edgecolor="k")
plt.scatter(application_data[application_data["TARGET"]==1]['AMT_ANNUITY'],application_data[application_data["TARGET"]==1]['AMT_CREDIT'],s=35,
            color="r",alpha=.2,label="DEFAULTER",linewidth=.5,edgecolor="k")
plt.legend(loc="best",prop={"size":15})
plt.xlabel("AMT_ANNUITY")
plt.ylabel("AMT_CREDIT")
plt.title("Scatter plot between credit amount and annuity amount")
plt.savefig("plot39.png", format="png", dpi=300)
plt.show()


# ## Pair Plot between amount variables
# 
# AMT_INCOME_TOTAL - Income of the client
# 
# AMT_CREDIT - Credit amount of the loan
# 
# AMT_ANNUITY - Loan annuity
# 
# AMT_GOODS_PRICE - For consumer loans it is the price of the goods for which the loan is given

# In[110]:


amt = application_data[[ 'AMT_INCOME_TOTAL','AMT_CREDIT',
                         'AMT_ANNUITY', 'AMT_GOODS_PRICE',"TARGET"]]
amt = amt[(amt["AMT_GOODS_PRICE"].notnull()) & (amt["AMT_ANNUITY"].notnull())]
sns.pairplot(amt,hue="TARGET",palette=["#A1C6EA","r"])
plt.savefig("plot40.png", format="png", dpi=300)
plt.show()


# # Distribution of Suite type
# 
# NAME_TYPE_SUITE - Who was accompanying client when he was applying for the loan.

# In[111]:


import matplotlib.pyplot as plt
import seaborn as sns

# Assuming `data` is already defined and contains the necessary columns

# Define shades of blue for different gender categories, including "XNA"
blue_palette = {"M": "#A1C6EA", "F": "#6397B1", "XNA": "#3B5998"}  # Use different blue shades for each category

plt.figure(figsize=(18, 12))

# First subplot - Distribution of Suite type (general)
plt.subplot(121)
sns.countplot(y=data["NAME_TYPE_SUITE"],
              palette=["#A1C6EA", "#6397B1", "#3B5998"],  # Shades of blue
              order=data["NAME_TYPE_SUITE"].value_counts().index[:5])
plt.title("Distribution of Suite Type")
plt.xlabel("Count")
plt.ylabel("Suite Type")

# Second subplot - Distribution of Suite type by gender
plt.subplot(122)
sns.countplot(y=data["NAME_TYPE_SUITE"],
              hue=data["CODE_GENDER"],
              palette=blue_palette,  # Use the custom blue palette for gender
              order=data["NAME_TYPE_SUITE"].value_counts().index[:5])
plt.title("Distribution of Suite Type by Gender")
plt.xlabel("Count")
plt.ylabel("")  # Remove the y-axis label for the second plot

# Adjust layout

plt.subplots_adjust(wspace=0.4)
plt.savefig("plot41.png", format="png", dpi=300)
plt.show()


# ### Distribution of client income type
# 
# NAME_INCOME_TYPE Clients income type (businessman, working, maternity leave,…)

# In[113]:


import matplotlib.pyplot as plt
import seaborn as sns

# Assuming `data` is already defined and contains the necessary columns

# Define shades of blue for different gender categories, including "XNA"
blue_palette = {"M": "#A1C6EA", "F": "#6397B1", "XNA": "#3B5998"}  # Shades of blue for each category

plt.figure(figsize=(18, 12))

# First subplot - Distribution of client income type (general)
plt.subplot(121)
sns.countplot(y=data["NAME_INCOME_TYPE"],
              palette=["#A1C6EA", "#6397B1", "#3B5998"],  # Different blue shades
              order=data["NAME_INCOME_TYPE"].value_counts().index[:4])
plt.title("Distribution of Client Income Type")
plt.xlabel("Count")
plt.ylabel("Income Type")

# Second subplot - Distribution of client income type by gender
plt.subplot(122)
sns.countplot(y=data["NAME_INCOME_TYPE"],
              hue=data["CODE_GENDER"],
              palette=blue_palette,  # Apply custom blue palette for gender categories
              order=data["NAME_INCOME_TYPE"].value_counts().index[:4])
plt.title("Distribution of Client Income Type by Gender")
plt.xlabel("Count")
plt.ylabel("")  # Remove y-axis label for the second plot

# Adjust layout
plt.subplots_adjust(wspace=0.4)
plt.savefig("plot42.png", format="png", dpi=300)
plt.show()


# 
# ### Distribution of Education type by loan repayment status
# 
# NAME_EDUCATION_TYPE Level of highest education the client achieved..
# 
# 

# In[116]:


import matplotlib.pyplot as plt
import seaborn as sns

# Assuming `application_data` is already defined and contains the necessary columns

# Define a blue color palette for pie chart sections
blue_shades = ["#A1C6EA", "#6397B1", "#3B5998", "#87AFC7", "#2A6478"]  # Extended shades of blue for each slice

plt.figure(figsize=(25, 25))

# First subplot - Education type distribution for Repayers
plt.subplot(121)
application_data[application_data["TARGET"] == 0]["NAME_EDUCATION_TYPE"].value_counts().plot.pie(
    fontsize=12,
    autopct="%1.0f%%",
    colors=blue_shades,  # Apply the blue shades to the pie chart
    wedgeprops={"linewidth": 2, "edgecolor": "white"},
    shadow=True
)
# Add white circle in the center for a donut effect
circ = plt.Circle((0, 0), 0.7, color="white")
plt.gca().add_artist(circ)
plt.title("Distribution of Education Type for Repayers", color="black")

# Second subplot - Education type distribution for Defaulters
plt.subplot(122)
application_data[application_data["TARGET"] == 1]["NAME_EDUCATION_TYPE"].value_counts().plot.pie(
    fontsize=12,
    autopct="%1.0f%%",
    colors=blue_shades,  # Apply the blue shades to the pie chart
    wedgeprops={"linewidth": 2, "edgecolor": "white"},
    shadow=True
)
# Add white circle in the center for a donut effect
circ = plt.Circle((0, 0), 0.7, color="white")
plt.gca().add_artist(circ)
plt.title("Distribution of Education Type for Defaulters", color="b")
plt.ylabel("")  # Remove the y-axis label
plt.savefig("plot43.png", format="png", dpi=300 )
plt.show()


# ### Point to infer from the graph
# 
# Clients who default have proportionally 9% less higher education compared to clients who do not default.

# ### Average Earnings by different professions and education types

# In[117]:


import matplotlib.pyplot as plt
import seaborn as sns

# Assuming `data` is already defined and contains the necessary columns

# Group data and calculate average income for each education and income type
edu = data.groupby(['NAME_EDUCATION_TYPE', 'NAME_INCOME_TYPE'])['AMT_INCOME_TOTAL'].mean().reset_index()
edu = edu.sort_values(by='AMT_INCOME_TOTAL', ascending=False)

# Create a custom color palette with fallback colors for missing categories
unique_education_types = edu['NAME_EDUCATION_TYPE'].unique()
education_palette = {"Higher education": "#A1C6EA", "Secondary / secondary special": "#6397B1", "Incomplete higher": "#3B5998"}

# Assign colors for any additional unique education types
for edu_type in unique_education_types:
    if edu_type not in education_palette:
        education_palette[edu_type] = sns.color_palette("husl", len(unique_education_types)).as_hex()[unique_education_types.tolist().index(edu_type)]

# Plot
fig = plt.figure(figsize=(13, 7))
ax = sns.barplot(x='NAME_INCOME_TYPE', y='AMT_INCOME_TOTAL', data=edu, hue='NAME_EDUCATION_TYPE', palette=education_palette)
ax.set_facecolor("white")  # Set background color to black
plt.title("Average Earnings by Different Professions and Education Types")
plt.savefig("plot44.png", format="png", dpi=300 )
plt.show()


# 
# ### Distribution of Education type by loan repayment status
# 
# NAME_FAMILY_STATUS - Family status of the client

# In[119]:


import matplotlib.pyplot as plt
import seaborn as sns

# Define custom colors
custom_colors = ["#A1C6EA", "#6397B1", "#3B5998", "#A1C6EA", "#6397B1", "#3B5998"]

plt.figure(figsize=(16, 8))

# Plot for Repayers
plt.subplot(121)
application_data[application_data["TARGET"] == 0]["NAME_FAMILY_STATUS"].value_counts().plot.pie(
    autopct="%1.0f%%",
    startangle=120,
    colors=custom_colors,  # Apply custom colors
    wedgeprops={"linewidth": 2, "edgecolor": "white"},
    shadow=True,
    explode=[0, 0.07, 0, 0, 0, 0]
)
plt.title("Distribution of Family Status for Repayers", color="b")

# Plot for Defaulters
plt.subplot(122)
application_data[application_data["TARGET"] == 1]["NAME_FAMILY_STATUS"].value_counts().plot.pie(
    autopct="%1.0f%%",
    startangle=120,
    colors=custom_colors,  # Apply the same custom colors
    wedgeprops={"linewidth": 2, "edgecolor": "white"},
    shadow=True,
    explode=[0, 0.07, 0, 0, 0]
)
plt.title("Distribution of Family Status for Defaulters", color="b")
plt.ylabel("")
plt.savefig("plot45.png", format="png", dpi=300 )
plt.show()


# ### Point to infer from the graph
# 
# Percentage of single people are more in defaulters than non defaulters.

# 
# ### Distribution of Housing type by loan repayment status
# 
# NAME_HOUSING_TYPE - What is the housing situation of the client (renting, living with parents, ...)

# In[ ]:





# 
# ### Distribution normalized population of region where client lives by loan repayment status
# 
# REGION_POPULATION_RELATIVE - Normalized population of region where client lives (higher number means the client lives in more populated region).
# 
# 

# In[121]:


import matplotlib.pyplot as plt
import seaborn as sns

# Define custom palette with dark blue and light blue
custom_palette = ["#1f77b4", "#A1C6EA"]  # Dark blue and light blue

fig = plt.figure(figsize=(13, 8))

# Plot for Non-Default loans
plt.subplot(121)
sns.violinplot(
    y=application_data[application_data["TARGET"] == 0]["REGION_POPULATION_RELATIVE"],
    x=application_data[application_data["TARGET"] == 0]["NAME_CONTRACT_TYPE"],
    palette=custom_palette
)
plt.title("Distribution of Region Population for Non-Default Loans", color="b")

# Plot for Default loans
plt.subplot(122)
sns.violinplot(
    y=application_data[application_data["TARGET"] == 1]["REGION_POPULATION_RELATIVE"],
    x=application_data[application_data["TARGET"] == 1]["NAME_CONTRACT_TYPE"],
    palette=custom_palette
)
plt.title("Distribution of Region Population for Default Loans", color="b")

plt.subplots_adjust(wspace=0.2)
fig.set_facecolor("white")
plt.savefig("plot46.png", format="png", dpi=300 )
plt.show()


# ### Point to infer from the graph
# 
# In High population density regions people are less likely to default on loans.

# 
# ### Client's age
# 
# DAYS_BIRTH - Client's age in days at the time of application.
# 
# 

# In[123]:


import matplotlib.pyplot as plt 
import seaborn as sns

# Set the figure size
fig = plt.figure(figsize=(13, 15))

# Plot for repayers (TARGET == 0)
plt.subplot(221)
sns.histplot(application_data[application_data["TARGET"] == 0]["DAYS_BIRTH"], kde=True, color="#A1C6EA", bins=30)
plt.title("Age Distribution of Repayers")

# Plot for defaulters (TARGET == 1)
plt.subplot(222)
sns.histplot(application_data[application_data["TARGET"] == 1]["DAYS_BIRTH"], kde=True, color="#6397B1", bins=30)
plt.title("Age Distribution of Defaulters")

# Plot for age vs loan repayment status (hue by gender)
plt.subplot(223)
sns.boxplot(x=application_data["TARGET"], y=application_data["DAYS_BIRTH"], hue=application_data["CODE_GENDER"], palette=["#A1C6EA", "#6397B1", "#3B5998"])
plt.axhline(application_data["DAYS_BIRTH"].mean(), linestyle="dashed", color="k", label="Average Age of Clients")
plt.legend(loc="lower right")
plt.title("Client Age vs Loan Repayment Status (hue=Gender)")

# Plot for age vs loan repayment status (hue by contract type)
plt.subplot(224)
sns.boxplot(x=application_data["TARGET"], y=application_data["DAYS_BIRTH"], hue=application_data["NAME_CONTRACT_TYPE"], palette=["#A1C6EA", "#6397B1"])
plt.axhline(application_data["DAYS_BIRTH"].mean(), linestyle="dashed", color="k", label="Average Age of Clients")
plt.legend(loc="lower right")
plt.title("Client Age vs Loan Repayment Status (hue=Contract Type)")

# Adjust space between subplots
plt.subplots_adjust(wspace=0.2, hspace=0.3)

# Set the background color for the entire figure
fig.set_facecolor("white")
plt.savefig("plot47.png", format="png", dpi=300 )
# Display the plot
plt.show()


# ### Point to infer from the graph
# 
# Average clients age is comparatively less in non repayers than repayers in every aspect.
# 
# Younger people tend to default more than elder people.

# ### Distribution of days employed for target variable.
# 
# DAYS_EMPLOYED - How many days before the application for target variable the person started current employment

# In[124]:


fig = plt.figure(figsize=(13,5))

plt.subplot(121)
sns.distplot(application_data[application_data["TARGET"]==0]["DAYS_EMPLOYED"],color="b")
plt.title("days employed distribution of repayers")

plt.subplot(122)
sns.distplot(application_data[application_data["TARGET"]==1]["DAYS_EMPLOYED"],color="#A1C6EA")
plt.title("days employed distribution of defaulters")

fig.set_facecolor("white")
plt.savefig("plot48.png", format="png", dpi=300 )


# ### Distribution of registration days for target variable.
# 
# DAYS_REGISTRATION How many days before the application did client change his registration

# In[125]:


fig = plt.figure(figsize=(13,5))

plt.subplot(121)
sns.distplot(application_data[application_data["TARGET"]==0]["DAYS_REGISTRATION"],color="b")
plt.title("registration days distribution of repayers")

plt.subplot(122)
sns.distplot(application_data[application_data["TARGET"]==1]["DAYS_REGISTRATION"],color="#A1C6EA")
plt.title("registration days distribution of defaulter")

fig.set_facecolor("white")
plt.savefig("plot49.png", format="png", dpi=300 )


# ### Distribution in contact information provided by client
# 
# FLAG_MOBIL - Did client provide mobile phone (1=YES, 0=NO)
# 
# FLAG_EMP_PHONE - Did client provide work phone (1=YES, 0=NO)
# 
# FLAG_WORK_PHONE - Did client provide home phone (1=YES, 0=NO)
# 
# FLAG_CONT_MOBILE - Was mobile phone reachable (1=YES, 0=NO)
# 
# FLAG_PHONE - Did client provide home phone (1=YES, 0=NO)
# 
# FLAG_EMAIL - Did client provide email (1=YES, 0=NO)

# In[126]:


import matplotlib.pyplot as plt
import seaborn as sns
import itertools

# Assume `application_data` is already loaded and preprocessed as per the description.

# Define the relevant columns
x = application_data[['FLAG_MOBIL', 'FLAG_EMP_PHONE', 'FLAG_WORK_PHONE', 'FLAG_CONT_MOBILE', 
                      'FLAG_PHONE', 'FLAG_EMAIL', 'TARGET']]

# Replace 0/1 with "YES"/"NO" and replace target 0/1 with "repayers"/"defaulters"
x["TARGET"] = x["TARGET"].replace({0: "repayers", 1: "defaulters"})
x = x.replace({1: "YES", 0: "NO"})

# Columns to be plotted
cols = ['FLAG_MOBIL', 'FLAG_EMP_PHONE', 'FLAG_WORK_PHONE', 'FLAG_CONT_MOBILE', 'FLAG_PHONE', 'FLAG_EMAIL']
length = len(cols)

# Create figure
fig = plt.figure(figsize=(15, 12))
fig.set_facecolor("white")

# Loop to create subplots
for i, j in zip(cols, range(length)):
    plt.subplot(2, 3, j+1)
    sns.countplot(x=i, hue="TARGET", data=x, palette=["#A1C6EA", "#3B5998"])  # Pass column as string
    plt.title(i, color="b")  # Set title for each subplot

# Adjust layout and show plot
plt.subplots_adjust(wspace=0.2, hspace=0.3)
plt.savefig("plot50.png", format="png", dpi=300 )
plt.show()


# 
# ### Distribution of registration days for target variable.
# 
# REGION_RATING_CLIENT - Home credit rating of the region where client lives (1,2,3).
# 
# REGION_RATING_CLIENT_W_CITY - Home credit rating of the region where client lives with taking city into account (1,2,3). Percentage of defaulters are less in 1-rated regions compared to repayers.

# In[127]:


import matplotlib.pyplot as plt
import seaborn as sns

# Define custom color palette
custom_colors = ["#A1C6EA", "#6397B1", "#3B5998", "#A1C6EA", "#6397B1", "#3B5998"]

fig = plt.figure(figsize=(13, 13))

# Plot for Repayers (Region Rating)
plt.subplot(221)
application_data[application_data["TARGET"] == 0]["REGION_RATING_CLIENT"].value_counts().plot.pie(
    autopct="%1.0f%%", fontsize=12,
    colors=custom_colors,  # Apply custom color palette
    wedgeprops={"linewidth": 2, "edgecolor": "white"}, shadow=True
)
plt.title("Distribution of region rating for Repayers", color="b")

# Plot for Defaulters (Region Rating)
plt.subplot(222)
application_data[application_data["TARGET"] == 1]["REGION_RATING_CLIENT"].value_counts().plot.pie(
    autopct="%1.0f%%", fontsize=12,
    colors=custom_colors,  # Apply custom color palette
    wedgeprops={"linewidth": 2, "edgecolor": "white"}, shadow=True
)
plt.title("Distribution of region rating for Defaulters", color="b")
plt.ylabel("")  # Hide ylabel for clarity

# Plot for Repayers (City Region Rating)
plt.subplot(223)
application_data[application_data["TARGET"] == 0]["REGION_RATING_CLIENT_W_CITY"].value_counts().plot.pie(
    autopct="%1.0f%%", fontsize=12,
    colors=custom_colors,  # Apply custom color palette
    wedgeprops={"linewidth": 2, "edgecolor": "white"}, shadow=True
)
plt.title("Distribution of city region rating for Repayers", color="b")

# Plot for Defaulters (City Region Rating)
plt.subplot(224)
application_data[application_data["TARGET"] == 1]["REGION_RATING_CLIENT_W_CITY"].value_counts().plot.pie(
    autopct="%1.0f%%", fontsize=12,
    colors=custom_colors,  # Apply custom color palette
    wedgeprops={"linewidth": 2, "edgecolor": "white"}, shadow=True
)
plt.title("Distribution of city region rating for Defaulters", color="b")
plt.ylabel("")  # Hide ylabel for clarity

# Set the background color for the entire figure
fig.set_facecolor("white")
plt.savefig("plot51.png", format="png", dpi=300 )
# Show the plot
plt.show()


# ### Point to infer from the graph
# 
# Percentage of defaulters are less in 1-rated regions compared to repayers.
# 
# Percentage of defaulters are more in 3-rated regions compared to repayers.

# ### Peak days and hours for applying loans (defaulters vs repayers)
# 
# WEEKDAY_APPR_PROCESS_START - On which day of the week did the client apply for the loan.
# 
# HOUR_APPR_PROCESS_START - Approximately at what hour did the client apply for the loan.
# 
# 

# In[128]:


import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd

# Define custom color palette
custom_palette = ["#A1C6EA", "#6397B1", "#3B5998", "#A1C6EA", "#6397B1"]

# Create day and hr groupings with aggregation
day = application_data.groupby("TARGET").agg({"WEEKDAY_APPR_PROCESS_START": "value_counts"})
day = day.rename(columns={"WEEKDAY_APPR_PROCESS_START": "value_counts"})
day = day.reset_index()

# Split data for Repayers and Defaulters
day_0 = day[:7]
day_1 = day[7:]

# Calculate percentages
day_0["percentage"] = day_0["value_counts"] * 100 / day_0["value_counts"].sum()
day_1["percentage"] = day_1["value_counts"] * 100 / day_1["value_counts"].sum()

# Concatenate the data
days = pd.concat([day_0, day_1], axis=0)

# Replace target values
days["TARGET"] = days["TARGET"].replace({1: "defaulters", 0: "repayers"})

# Plotting the Peak days
fig = plt.figure(figsize=(13, 15))

# Plot for Peak days (Repayers vs Defaulters)
plt.subplot(211)
order = ['SUNDAY', 'MONDAY', 'TUESDAY', 'WEDNESDAY', 'THURSDAY', 'FRIDAY', 'SATURDAY']
ax = sns.barplot(data=days, x="WEEKDAY_APPR_PROCESS_START", y="percentage", 
                 hue="TARGET", order=order, palette=custom_palette)
ax.set_facecolor("white")  # Set the background color to white
ax.set_title("Peak days for applying loans (defaulters vs repayers)")

# Create hour aggregation
hr = application_data.groupby("TARGET").agg({"HOUR_APPR_PROCESS_START": "value_counts"})
hr = hr.rename(columns={"HOUR_APPR_PROCESS_START": "value_counts"}).reset_index()

# Split data for Repayers and Defaulters
hr_0 = hr[hr["TARGET"] == 0]
hr_1 = hr[hr["TARGET"] == 1]

# Calculate percentages
hr_0["percentage"] = hr_0["value_counts"] * 100 / hr_0["value_counts"].sum()
hr_1["percentage"] = hr_1["value_counts"] * 100 / hr_1["value_counts"].sum()

# Concatenate hour data
hrs = pd.concat([hr_0, hr_1], axis=0)

# Replace target values
hrs["TARGET"] = hrs["TARGET"].replace({1: "defaulters", 0: "repayers"})

# Sort by hour
hrs = hrs.sort_values(by="HOUR_APPR_PROCESS_START", ascending=True)

# Plotting the Peak hours
plt.subplot(212)
ax1 = sns.pointplot(data=hrs, x="HOUR_APPR_PROCESS_START", y="percentage", 
                    hue="TARGET", palette=custom_palette)
ax1.set_facecolor("white")  # Set the background color to white
ax1.set_title("Peak hours for applying loans (defaulters vs repayers)")

# Set the figure background color
fig.set_facecolor("white")
plt.savefig("plot52.png", format="png", dpi=300 )
# Show the plot
plt.show()


# ### Point to infer from the graph
# 
# On tuesdays , percentage of defaulters applying for loans is greater than that of repayers.
# 
# From morning 4'O clock to 9'O clock percentage of defaulters applying for loans is greater than that of repayers.

# 
# ### Distribution in organization types for repayers and defaulters
# 
# ORGANIZATION_TYPE - Type of organization where client works.
# 
# 

# In[129]:


import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Grouping by TARGET and calculating value counts for ORGANIZATION_TYPE
org = application_data.groupby("TARGET")["ORGANIZATION_TYPE"].value_counts().reset_index(name="value_counts")

# Splitting into repayers (TARGET=0) and defaulters (TARGET=1)
org_0 = org[org["TARGET"] == 0]
org_1 = org[org["TARGET"] == 1]

# Calculating percentage for each type
org_0["percentage"] = org_0["value_counts"] * 100 / org_0["value_counts"].sum()
org_1["percentage"] = org_1["value_counts"] * 100 / org_1["value_counts"].sum()

# Concatenating and organizing data
organization = pd.concat([org_0, org_1], axis=0)
organization = organization.sort_values(by="ORGANIZATION_TYPE", ascending=True)
organization["TARGET"] = organization["TARGET"].replace({0: "repayers", 1: "defaulters"})

# Plotting
plt.figure(figsize=(13, 7))
ax = sns.pointplot(x="ORGANIZATION_TYPE", y="percentage", data=organization, hue="TARGET", palette=["b", "#6397B1"])
plt.xticks(rotation=90)
plt.grid(True, alpha=.3)

# Set the background color to white
plt.gca().set_facecolor("white")
plt.gcf().patch.set_facecolor("white")  # also set the figure background to white

ax.set_title("Distribution in organization types for repayers and defaulters")
plt.savefig("plot53.png", format="png", dpi=300 )
plt.show()


# ### Point to infer from the graph
# 
# Organizations like Business Entity Type 3,Construction,Self-employed percentage of defaulters are higher than repayers.

# ### Distribution client's social surroundings with observed and defaulted 30 DPD (days past due)
# 
# OBS_30_CNT_SOCIAL_CIRCLE- How many observation of client's social surroundings with observable 30 DPD (days past due) default.
# 
# DEF_30_CNT_SOCIAL_CIRCLE-How many observation of client's social surroundings defaulted on 30 DPD (days past due) .
# 
# OBS_60_CNT_SOCIAL_CIRCLE - How many observation of client's social surroundings with observable 60 DPD (days past due) default.
# 
# DEF_60_CNT_SOCIAL_CIRCLE - How many observation of client's social surroundings defaulted on 60 (days past due) DPD.

# In[130]:


fig = plt.figure(figsize=(20,20))
plt.subplot(421)
sns.boxplot(data=application_data,x='TARGET',y='OBS_30_CNT_SOCIAL_CIRCLE',
            hue="TARGET", palette="Set3")
plt.title("Client's social surroundings with observable 30 DPD (days past due) def",color="b")
plt.subplot(422)
sns.boxplot(data=application_data,x='TARGET',y='DEF_30_CNT_SOCIAL_CIRCLE',
            hue="TARGET", palette="Set3")
plt.title("Client's social surroundings defaulted on 30 DPD (days past due)",color="b")
plt.subplot(423)
sns.boxplot(data=application_data,x='TARGET',y='OBS_60_CNT_SOCIAL_CIRCLE',
            hue="TARGET", palette="Set3")
plt.title("Client's social surroundings with observable 60 DPD (days past due) default",color="b")
plt.subplot(424)
sns.boxplot(data=application_data,x='TARGET',y='DEF_60_CNT_SOCIAL_CIRCLE',
            hue="TARGET", palette="Set3")
plt.title("Client's social surroundings defaulted on 60 DPD (days past due)",color="b")
fig.set_facecolor("white")
plt.savefig("plot53.png", format="png", dpi=300 )


# ### Number of days before application client changed phone .
# 
# DAYS_LAST_PHONE_CHANGE - How many days before application did client change phone.
# 
# 

# In[131]:


import matplotlib.pyplot as plt
import seaborn as sns

# Set figure size and background color
plt.figure(figsize=(13, 7))
plt.gcf().patch.set_facecolor("white")  # Set the entire figure background to white

# Plotting violin plot on the left
plt.subplot(121)
ax = sns.violinplot(x="TARGET", y="DAYS_LAST_PHONE_CHANGE", data=application_data, palette=["b", "#6397B1"])
ax.set_facecolor("white")  # Set the axes background to white
ax.set_title("Days before application client changed phone - Violin Plot")

# Plotting box plot on the right
plt.subplot(122)
ax1 = sns.boxplot(x="TARGET", y="DAYS_LAST_PHONE_CHANGE", data=application_data, palette=["b", "#6397B1"])
ax1.set_facecolor("white")  # Set the axes background to white
ax1.set_ylabel("")
ax1.set_title("Days before application client changed phone - Box Plot")

# Adjust layout
plt.subplots_adjust(wspace=0.2)
plt.savefig("plot54.png", format="png", dpi=300 )
plt.show()


# ### Point to infer from the graph
# 
# Average days of defaulters phone change is less than average days of repayers phone change.

# ### Documents provided by the clients.
# 
# FLAG_DOCUMENT - Did client provide documents.(1,0)

# In[132]:


import matplotlib.pyplot as plt
import seaborn as sns
import itertools

# Define the columns and create a subset DataFrame with "TARGET"
cols = [ 'FLAG_DOCUMENT_2', 'FLAG_DOCUMENT_3', 'FLAG_DOCUMENT_4', 'FLAG_DOCUMENT_5', 'FLAG_DOCUMENT_6',
         'FLAG_DOCUMENT_7', 'FLAG_DOCUMENT_8', 'FLAG_DOCUMENT_9', 'FLAG_DOCUMENT_10', 'FLAG_DOCUMENT_11', 
         'FLAG_DOCUMENT_12', 'FLAG_DOCUMENT_13', 'FLAG_DOCUMENT_14', 'FLAG_DOCUMENT_15', 'FLAG_DOCUMENT_16', 
         'FLAG_DOCUMENT_17', 'FLAG_DOCUMENT_18', 'FLAG_DOCUMENT_19', 'FLAG_DOCUMENT_20', 'FLAG_DOCUMENT_21']

df_flag = application_data[cols + ["TARGET"]]
length = len(cols)

# Replace target values with readable labels
df_flag["TARGET"] = df_flag["TARGET"].replace({1: "defaulter", 0: "repayer"})

# Set up the figure with a light grey background
fig = plt.figure(figsize=(13, 24))
fig.set_facecolor("white")

# Loop through each column to create a count plot
for i, j in zip(cols, range(length)):
    plt.subplot(5, 4, j + 1)
    ax = sns.countplot(x=df_flag[i], hue=df_flag["TARGET"], palette=["#6397B1", "b"])
    plt.yticks(fontsize=5)
    plt.xlabel("")
    plt.title(i)
    ax.set_facecolor("white")  # Set the plot background to white

# Adjust spacing between subplots
plt.subplots_adjust(wspace=0.4, hspace=0.6)
plt.savefig("plot55.png", format="png", dpi=300 )
plt.show()


# ### Equiries to Credit Bureau about the client before application.
# 
# AMT_REQ_CREDIT_BUREAU_HOUR - Number of enquiries to Credit Bureau about the client one hour before application.
# 
# AMT_REQ_CREDIT_BUREAU_DAY - Number of enquiries to Credit Bureau about the client one day before application (excluding one hour before application).
# 
# AMT_REQ_CREDIT_BUREAU_WEEK - Number of enquiries to Credit Bureau about the client one week before application (excluding one day before application).
# 
# AMT_REQ_CREDIT_BUREAU_MON - Number of enquiries to Credit Bureau about the client one month before application (excluding one week before application).
# 
# AMT_REQ_CREDIT_BUREAU_QRT - Number of enquiries to Credit Bureau about the client 3 month before application (excluding one month before application).
# 
# AMT_REQ_CREDIT_BUREAU_YEAR - Number of enquiries to Credit Bureau about the client one day year (excluding last 3 months before application).

# In[134]:


import matplotlib.pyplot as plt

cols = ['AMT_REQ_CREDIT_BUREAU_HOUR', 'AMT_REQ_CREDIT_BUREAU_DAY',
        'AMT_REQ_CREDIT_BUREAU_WEEK', 'AMT_REQ_CREDIT_BUREAU_MON',
        'AMT_REQ_CREDIT_BUREAU_QRT', 'AMT_REQ_CREDIT_BUREAU_YEAR']

# Define color palette for repayers and defaulters
colors = ["#6397B1", "b"]

# Maximum inquiries plot
application_data.groupby("TARGET")[cols].max().transpose().plot(kind="barh",
                                                                figsize=(10, 5), width=0.8,
                                                                color=colors)
plt.title("Maximum inquiries made by defaulters and repayers")
plt.xlabel("Number of Inquiries")
plt.legend(["Repayers", "Defaulters"], loc="best")
plt.savefig("max_inquiries_plot.png", format="png", dpi=300)  # Save the first plot
plt.show()

# Average inquiries plot
application_data.groupby("TARGET")[cols].mean().transpose().plot(kind="barh",
                                                                 figsize=(10, 5), width=0.8,
                                                                 color=colors)
plt.title("Average inquiries made by defaulters and repayers")
plt.xlabel("Average Number of Inquiries")
plt.legend(["Repayers", "Defaulters"], loc="best")
plt.savefig("avg_inquiries_plot.png", format="png", dpi=300)  # Save the second plot
plt.show()

# Standard deviation in inquiries plot
application_data.groupby("TARGET")[cols].std().transpose().plot(kind="barh",
                                                                figsize=(10, 5), width=0.8,
                                                                color=colors)
plt.title("Standard deviation in inquiries made by defaulters and repayers")
plt.xlabel("Standard Deviation of Inquiries")
plt.legend(["Repayers", "Defaulters"], loc="best")
plt.savefig("std_inquiries_plot.png", format="png", dpi=300)  # Save the third plot
plt.show()


# 
# ### Current loan id having previous loan applications.
# 
# SK_ID_PREV - ID of previous credit in Home credit related to loan in our sample. (One loan in our sample can have 0,1,2 or more previous loan applications in Home Credit, previous application could, but not necessarily have to lead to credit).
# 
# SK_ID_CURR ID of loan in our sample.
# 

# In[135]:


import matplotlib.pyplot as plt
import seaborn as sns

# Group by SK_ID_CURR to count previous loan applications
x = previous_application.groupby("SK_ID_CURR")["SK_ID_PREV"].count().reset_index()

# Set figure size and background color
plt.figure(figsize=(13, 7))
plt.gcf().patch.set_facecolor("white")  # Set figure background to white

# Distribution plot with custom color
ax = sns.histplot(x["SK_ID_PREV"], color="#6397B1", kde=True)  # Replaced distplot with histplot since distplot is deprecated

# Add dashed lines for mean, standard deviation, and max
plt.axvline(x["SK_ID_PREV"].mean(), linestyle="dashed", color="r", label="Average")
plt.axvline(x["SK_ID_PREV"].std(), linestyle="dashed", color="b", label="Standard Deviation")
plt.axvline(x["SK_ID_PREV"].max(), linestyle="dashed", color="g", label="Maximum")

# Customizations
plt.legend(loc="best")
plt.title("Current loan id having previous loan applications")
ax.set_facecolor("white")  # Set the plot background to white
plt.savefig("plot57.png", format="png", dpi=300 )
plt.show()


# ### Point to infer from the graph
# On average current loan ids have 4 to 5 loan applications previously

# 
# ### Contract types in previous applications
# 
# NAME_CONTRACT_TYPE Contract product type (Cash loan, consumer loan [POS] ,...) of the previous application.
# 

# In[136]:


get_ipython().system('pip install squarify')


# In[137]:


import squarify
import matplotlib.pyplot as plt

# Get value counts for NAME_CONTRACT_TYPE
cnts = previous_application["NAME_CONTRACT_TYPE"].value_counts()

# Define custom colors
custom_colors = ["#A1C6EA", "#6397B1", "#3B5998", "#A1C6EA", "#6397B1", "#3B5998"]

# Plotting the treemap
plt.figure(figsize=(8, 6))
squarify.plot(
    sizes=cnts.values,
    label=cnts.index,
    value=cnts.values,
    linewidth=2,
    edgecolor="k",
    alpha=0.8,
    color=custom_colors[:len(cnts)]  # Ensure colors match the number of categories
)
plt.axis("off")
plt.title("Contract types in previous applications")
plt.savefig("plot58.png", format="png", dpi=300 )
plt.show()


# ### Point to infer from the graph
# 
# 
# Cash loan applications are maximum followed by consumer loan applications.

# ### Previous loan amounts applied and loan amounts credited.
# 
# AMT_APPLICATION-For how much credit did client ask on the previous application.
# 
# AMT_CREDIT-Final credit amount on the previous application. This differs from AMT_APPLICATION in a way that the AMT_APPLICATION is the amount for which the client initially applied for, but during our approval process he could have received different amount - AMT_CREDIT.

# In[139]:


import matplotlib.pyplot as plt
import seaborn as sns

# Set the figure background color to white
plt.figure(figsize=(20, 20))
plt.gcf().patch.set_facecolor("white")

# First subplot
plt.subplot(211)
ax = sns.kdeplot(previous_application["AMT_APPLICATION"], color="#6397B1", linewidth=3)
ax = sns.kdeplot(previous_application[previous_application["AMT_CREDIT"].notnull()]["AMT_CREDIT"], color="#3B5998", linewidth=3)
plt.axvline(previous_application[previous_application["AMT_CREDIT"].notnull()]["AMT_CREDIT"].mean(), color="#3B5998", linestyle="dashed", label="AMT_CREDIT_MEAN")
plt.axvline(previous_application["AMT_APPLICATION"].mean(), color="#6397B1", linestyle="dashed", label="AMT_APPLICATION_MEAN")
plt.legend(loc="best")
plt.title("Previous loan amounts applied and loan amounts credited.")
ax.set_facecolor("white")  # Set background of plot to white

# Second subplot
plt.subplot(212)
diff = (previous_application["AMT_CREDIT"] - previous_application["AMT_APPLICATION"]).reset_index()
diff = diff[diff[0].notnull()]
ax1 = sns.kdeplot(diff[0], color="#A1C6EA", linewidth=3, label="Difference in amount requested by client and amount credited")
plt.axvline(diff[0].mean(), color="white", linestyle="dashed", label="Mean")
plt.title("Difference in amount requested by client and amount credited")
ax1.legend(loc="best")
ax1.set_facecolor("white")  # Set background of plot to white
plt.savefig("plot59.png", format="png", dpi=300)
# Show the plots
plt.show()


# ### Total and average amounts applied and credited in previous applications
# 
# AMT_APPLICATION-For how much credit did client ask on the previous application. >AMT_CREDIT-Final credit amount on the previous application. This differs from AMT_APPLICATION in a way that the AMT_APPLICATION is the amount for which the client.

# In[ ]:





# ### Annuity of previous application
# 
# AMT_ANNUITY - Annuity of previous application

# In[145]:


plt.figure(figsize=(14,5))
plt.subplot(121)
previous_application.groupby("NAME_CONTRACT_TYPE")["AMT_ANNUITY"].sum().plot(kind="bar")
plt.xticks(rotation=0)
plt.title("Total annuity amount by contract types in previous applications")
plt.subplot(122)
previous_application.groupby("NAME_CONTRACT_TYPE")["AMT_ANNUITY"].mean().plot(kind="bar")
plt.title("average annuity amount by contract types in previous applications")
plt.xticks(rotation=0)
plt.savefig("plot60.png", format="png", dpi=300)
plt.show()


# 
# ### Count of application status by application type.
# 
# NAME_CONTRACT_TYPE -Contract product type (Cash loan, consumer loan [POS] ,...) of the previous application.
# 
# NAME_CONTRACT_STATUS -Contract status (approved, cancelled, ...) of previous application.
# 
# 

# In[146]:


# Define custom colors
custom_colors = ["#A1C6EA", "#6397B1", "#3B5998", "#A1C6EA", "#6397B1", "#3B5998"]

# Plot with the updated color palette
ax = pd.crosstab(
    previous_application["NAME_CONTRACT_TYPE"],
    previous_application["NAME_CONTRACT_STATUS"]
).plot(
    kind="barh", 
    figsize=(10, 7), 
    stacked=True, 
    color=custom_colors
)

# Adjust labels and title
plt.xticks(rotation=0)
plt.ylabel("Count")
plt.title("Count of Application Status by Application Type")

# Set the background color for the axes
ax.set_facecolor("white")
plt.savefig("plot61.png", format="png", dpi=300)
# Show the plot
plt.show()


# ### Point to infer from the graph
# 
# Consumer loan applications are most approved loans and cash loans are most cancelled and refused loans.

# ### Contract status by weekdays
# 
# WEEKDAY_APPR_PROCESS_START - On which day of the week did the client apply for previous application

# In[147]:


# Prepare the data
hr = pd.crosstab(
    previous_application["WEEKDAY_APPR_PROCESS_START"],
    previous_application["NAME_CONTRACT_STATUS"]
).stack().reset_index()
hr.rename(columns={0: "count"}, inplace=True)  # Rename the stacked column for clarity

# Define custom colors
custom_colors = ["#A1C6EA", "#6397B1", "#3B5998", "#FFA500"]

# Plot the pointplot
plt.figure(figsize=(12, 8))
ax = sns.pointplot(
    x="WEEKDAY_APPR_PROCESS_START",
    y="count",
    hue="NAME_CONTRACT_STATUS",
    data=hr,
    palette=custom_colors,  # Use custom color palette
    scale=1
)

# Adjust the appearance of the plot
ax.set_facecolor("white")  # Set the axes background color to white
ax.set_ylabel("Count")
ax.set_title("Contract Status by Weekdays")
plt.grid(True, alpha=0.2)
plt.savefig("plot62.png", format="png", dpi=300)
# Show the plot
plt.show()


# ### Contract status by hour of the day
# 
# HOUR_APPR_PROCESS_START - Approximately at what day hour did the client apply for the previous application.
# 
# 

# In[148]:


# Prepare the data
hr = pd.crosstab(
    previous_application["HOUR_APPR_PROCESS_START"],
    previous_application["NAME_CONTRACT_STATUS"]
).stack().reset_index()
hr.rename(columns={0: "count"}, inplace=True)  # Rename the stacked column for clarity

# Define custom colors
custom_colors = ["#A1C6EA", "#6397B1", "#3B5998", "#FFA500"]

# Plot the pointplot
plt.figure(figsize=(12, 8))
ax = sns.pointplot(
    x="HOUR_APPR_PROCESS_START",
    y="count",
    hue="NAME_CONTRACT_STATUS",
    data=hr,
    palette=custom_colors,  # Use the updated color palette
    scale=1
)

# Adjust the appearance of the plot
ax.set_facecolor("white")  # Set the axes background color to white
ax.set_ylabel("Count")
ax.set_title("Contract Status by Day Hours")
plt.grid(True, alpha=0.2)
plt.savefig("plot63.png", format="png", dpi=300)
# Show the plot
plt.show()


# ### Point to infer from the graph
# 
# Morning 11'o clock have maximum number of approvals.
# 
# Morning 10'o clock have maximum number of refused and cancelled contracts.

# ### Peak hours for week days for applying loans.

# In[149]:


# Prepare the data
hr = pd.crosstab(
    previous_application["HOUR_APPR_PROCESS_START"],
    previous_application["WEEKDAY_APPR_PROCESS_START"]
).stack().reset_index()
hr.rename(columns={0: "count"}, inplace=True)  # Rename the stacked column for clarity

# Define custom colors
custom_colors = ["#A1C6EA", "#6397B1", "#3B5998", "#FFA500"]

# Plot the pointplot
plt.figure(figsize=(12, 8))
ax = sns.pointplot(
    x="HOUR_APPR_PROCESS_START",
    y="count",
    hue="WEEKDAY_APPR_PROCESS_START",
    data=hr,
    palette=custom_colors,  # Use the updated color palette
    scale=1
)

# Adjust the appearance of the plot
ax.set_facecolor("white")  # Set the axes background color to white
ax.set_ylabel("Count")
ax.set_title("Peak Hours for Weekdays")
plt.grid(True, alpha=0.2)
plt.savefig("plot65.png", format="png", dpi=300)
# Show the plot
plt.show()


# 
# ### Percentage of applications accepted,cancelled,refused and unused for different loan purposes.
# 
# NAME_CASH_LOAN_PURPOSE - Purpose of the cash loan.
# 
# NAME_CONTRACT_STATUS - Contract status (approved, cancelled, ...) of previous application.
# 
# 

# In[151]:


import itertools  # Ensure itertools is imported if not already

# Calculate percentages
purpose = pd.crosstab(
    previous_application["NAME_CASH_LOAN_PURPOSE"],
    previous_application["NAME_CONTRACT_STATUS"]
)
total = purpose.sum(axis=1)  # Total across columns for each row
purpose["a"] = (purpose["Approved"] * 100) / total
purpose["c"] = (purpose["Canceled"] * 100) / total
purpose["r"] = (purpose["Refused"] * 100) / total
purpose["u"] = (purpose["Unused offer"] * 100) / total

# Prepare the data for plotting
purpose_new = purpose[["a", "c", "r", "u"]].stack().reset_index()
purpose_new.columns = ["NAME_CASH_LOAN_PURPOSE", "NAME_CONTRACT_STATUS", "percentage"]
purpose_new["NAME_CONTRACT_STATUS"] = purpose_new["NAME_CONTRACT_STATUS"].replace({
    "a": "accepted_percentage",
    "c": "cancelled_percentage",
    "r": "refused_percentage",
    "u": "unused_percentage"
})

# Unique statuses and colors
statuses = purpose_new["NAME_CONTRACT_STATUS"].unique().tolist()
colors = ["#A1C6EA", "#6397B1", "#3B5998", "#FFA500"]  # Updated color palette

# Create the figure
fig = plt.figure(figsize=(14, 18))
fig.set_facecolor("white")

# Loop through each status
for i, (status, color) in enumerate(zip(statuses, colors)):
    plt.subplot(2, 2, i + 1)
    data = purpose_new[purpose_new["NAME_CONTRACT_STATUS"] == status]
    sns.barplot(
        x="percentage",
        y="NAME_CASH_LOAN_PURPOSE",
        data=data.sort_values(by="percentage", ascending=False),
        color=color
    )
    plt.ylabel("")
    plt.xlabel("Percentage")
    plt.title(f"{status.capitalize()} by Purpose")
    plt.subplots_adjust(wspace=0.7)
    plt.savefig("plot66.png", format="png", dpi=300)


# ### Point to infer from the graph
# Purposes like XAP ,electronic eqipment ,everey day expences and education have maximum loan acceptance.
# 
# Loan puposes like payment of other loans ,refusal to name goal ,buying new home or car have most refusals.
# 
# 40% of XNA purpose loans are cancalled.

# ### Contract status relative to decision made about previous application.
# 
# DAYS_DECISION - Relative to current application when was the decision about previous application made.
# 
# 

# In[152]:


plt.figure(figsize=(13, 6))

# Updated palette
sns.violinplot(
    y=previous_application["DAYS_DECISION"],
    x=previous_application["NAME_CONTRACT_STATUS"],
    palette=["#A1C6EA", "#6397B1", "#3B5998", "#A1C6EA"]
)

# Horizontal lines for averages with updated colors
plt.axhline(
    previous_application[previous_application["NAME_CONTRACT_STATUS"] == "Approved"]["DAYS_DECISION"].mean(),
    color="#A1C6EA", linestyle="dashed", label="accepted_average"
)
plt.axhline(
    previous_application[previous_application["NAME_CONTRACT_STATUS"] == "Refused"]["DAYS_DECISION"].mean(),
    color="#6397B1", linestyle="dashed", label="refused_average"
)
plt.axhline(
    previous_application[previous_application["NAME_CONTRACT_STATUS"] == "Cancelled"]["DAYS_DECISION"].mean(),
    color="#3B5998", linestyle="dashed", label="cancelled_average"
)
plt.axhline(
    previous_application[previous_application["NAME_CONTRACT_STATUS"] == "Unused offer"]["DAYS_DECISION"].mean(),
    color="#A1C6EA", linestyle="dashed", label="unused_average"
)

# Legend, title, and display
plt.legend(loc="best")
plt.title("Contract Status Relative to Decision Made About Previous Application")
plt.savefig("plot67.png", format="png", dpi=300)
plt.show()


# ### Point to infer from the graph
# 
# On average approved contract types have higher number of decision days compared to cancelled and refused contracts.

# 
# ### Client payment methods & reasons for application rejections
# 
# NAME_PAYMENT_TYPE - Payment method that client chose to pay for the previous application.
# 
# CODE_REJECT_REASON - Why was the previous application rejected.
# 
# 

# In[153]:


import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

# Plot figure with two subplots
plt.figure(figsize=(8, 12))

# First subplot: Reasons for application rejections
plt.subplot(211)
rej = previous_application["CODE_REJECT_REASON"].value_counts().reset_index()
rej.columns = ['CODE_REJECT_REASON', 'Count']  # Rename columns for clarity

# Plot the barplot for rejection reasons
ax = sns.barplot(
    x="CODE_REJECT_REASON",  # Correct keyword argument for x-axis
    y="Count",  # Correct keyword argument for y-axis (the counts column)
    data=rej[:6],  # Data for the barplot
    palette=["#A1C6EA", "#6397B1", "#3B5998", "#A1C6EA", "#6397B1", "#3B5998"]  # Updated colors
)

# Annotate the percentages on the bars
for i, j in enumerate(np.around((rej["Count"][:6].values * 100) / (rej["Count"][:6].sum()))):
    ax.text(0.7, i, j, weight="bold")  # Display percentage on bars

plt.xlabel("Top as percentage & Bottom as Count")
plt.ylabel("CODE_REJECT_REASON")
plt.title("Reasons for application rejections")

# Second subplot: Clients' payment methods
plt.subplot(212)
pay = previous_application["NAME_PAYMENT_TYPE"].value_counts().reset_index()
pay.columns = ['NAME_PAYMENT_TYPE', 'Count']  # Rename columns for clarity

# Plot the barplot for payment methods
ax1 = sns.barplot(
    x="NAME_PAYMENT_TYPE",  # Correct keyword argument for x-axis
    y="Count",  # Correct keyword argument for y-axis (the counts column)
    data=pay,  # Data for the barplot
    palette=["#A1C6EA", "#6397B1", "#3B5998"]  # Updated colors
)

# Annotate the percentages on the bars
for i, j in enumerate(np.around((pay["Count"].values * 100) / (pay["Count"].sum()))):
    ax1.text(0.7, i, j, weight="bold")  # Display percentage on bars

plt.xlabel("Top as percentage & Bottom as Count")
plt.ylabel("NAME_PAYMENT_TYPE")
plt.title("Clients payment methods")

# Adjust space between subplots
plt.subplots_adjust(hspace=0.3)
plt.savefig("plot68.png", format="png", dpi=300)
# Show plot
plt.show()


# ### Point to infer from the graph
# 
# Around 81% of rejected applications the reason is XAP.
# 
# 62% of chose to pay through cash by bank for previous applications.

# 
# #### Distribution in Client suite type & client type.
# 
# NAME_TYPE_SUITE - Who accompanied client when applying for the previous application.
# 
# NAME_CLIENT_TYPE - Was the client old or new client when applying for the previous application.
# 
# 

# In[154]:


import matplotlib.pyplot as plt
import seaborn as sns

# Define the custom color palette
custom_palette = ["#A1C6EA", "#6397B1", "#3B5998"]

# Create the figure with subplots
plt.figure(figsize=(20, 20))

# First subplot: Pie chart for 'NAME_TYPE_SUITE'
plt.subplot(121)
previous_application["NAME_TYPE_SUITE"].value_counts().plot.pie(
    autopct="%1.1f%%",  # Show percentage on pie chart
    fontsize=12,
    colors=custom_palette,  # Use custom color palette
    wedgeprops={"linewidth": 2, "edgecolor": "white"},  # Add white edge to slices
    shadow=True  # Add shadow effect
)

# Add a circle in the center to make the pie chart look like a donut
circ = plt.Circle((0, 0), .7, color="white")
plt.gca().add_artist(circ)
plt.title("NAME_TYPE_SUITE")

# Second subplot: Pie chart for 'NAME_CLIENT_TYPE'
plt.subplot(122)
previous_application["NAME_CLIENT_TYPE"].value_counts().plot.pie(
    autopct="%1.1f%%",  # Show percentage on pie chart
    fontsize=12,
    colors=custom_palette,  # Use custom color palette
    wedgeprops={"linewidth": 2, "edgecolor": "white"},  # Add white edge to slices
    shadow=True  # Add shadow effect
)

# Add a circle in the center to make the pie chart look like a donut
circ = plt.Circle((0, 0), .7, color="white")
plt.gca().add_artist(circ)
plt.title("NAME_CLIENT_TYPE")
plt.savefig("plot69.png", format="png", dpi=300)
# Show the plot
plt.show()


# ### Point to infer from the graph
# About 60% clients are un-accompained when applying for loans.
# 
# 73% clients are old clients

# 
# ## Popular goods for applying loans
# 
# NAME_GOODS_CATEGORY - What kind of goods did the client apply for in the previous application.
# 
# 

# In[156]:


# Get the value counts for 'NAME_GOODS_CATEGORY' and reset the index
goods = previous_application["NAME_GOODS_CATEGORY"].value_counts().reset_index()

# Rename columns for clarity
goods.columns = ['NAME_GOODS_CATEGORY', 'count']

# Convert 'count' to numeric (in case it's treated as a string)
goods['count'] = pd.to_numeric(goods['count'], errors='coerce')

# Calculate percentage of each category
goods["percentage"] = round(goods["count"] * 100 / goods["count"].sum(), 2)

# Create the plot
fig = plt.figure(figsize=(12, 5))

# Create a pointplot to display the data
# Change x to 'NAME_GOODS_CATEGORY' and y to 'percentage'
ax = sns.pointplot(x="NAME_GOODS_CATEGORY", y="percentage", data=goods, color="b")

# Rotate the x-axis labels for better readability
plt.xticks(rotation=80)

# Set labels and title
plt.xlabel("NAME_GOODS_CATEGORY")
plt.ylabel("Percentage")
plt.title("Popular goods for applying loans")

# Set background color for the plot and figure
ax.set_facecolor("white")
fig.set_facecolor('white')
plt.savefig("plot70.png", format="png", dpi=300)
# Show the plot
plt.show()


# ### Point to infer from the graph
# XNA ,Mobiles ,Computers and consumer electronics are popular goods for applying loans

# 
# ### Previous applications portfolio and product types
# 
# NAME_PORTFOLIO - Was the previous application for CASH, POS, CAR, …
# 
# NAME_PRODUCT_TYPE - Was the previous application x-sell o walk-in.

# In[157]:


import matplotlib.pyplot as plt
import seaborn as sns

# Custom color palette as provided
custom_colors = ["#A1C6EA", "#6397B1", "#3B5998", "#A1C6EA", "#6397B1", "#3B5998"]

# Create the figure
plt.figure(figsize=(20, 20))

# Plot for 'NAME_PORTFOLIO' in the first subplot
plt.subplot(121)
previous_application["NAME_PORTFOLIO"].value_counts().plot.pie(
    autopct="%1.1f%%", 
    fontsize=12,
    colors=custom_colors,  # Use custom colors
    wedgeprops={"linewidth": 2, "edgecolor": "white"},
    shadow=True
)
plt.title("Previous Applications Portfolio")

# Plot for 'NAME_PRODUCT_TYPE' in the second subplot
plt.subplot(122)
previous_application["NAME_PRODUCT_TYPE"].value_counts().plot.pie(
    autopct="%1.1f%%", 
    fontsize=12,
    colors=custom_colors[:3],  # Use the first 3 colors from the palette
    wedgeprops={"linewidth": 2, "edgecolor": "white"},
    shadow=True
)
plt.title("Previous Applications Product Types")
plt.savefig("plot71.png", format="png", dpi=300)
# Show the plot
plt.show()


# 
# ### Approval,canceled and refusal rates by channel types.
# 
# CHANNEL_TYPE - Through which channel we acquired the client on the previous application.
# 
# NAME_CONTRACT_STATUS- Contract status (approved, cancelled, ...) of previous application.
# 
# 

# In[158]:


import matplotlib.pyplot as plt
import pandas as pd

# Custom color palette as provided
custom_colors = ["#A1C6EA", "#6397B1", "#3B5998"]

# Create the crosstab
app = pd.crosstab(previous_application["CHANNEL_TYPE"], previous_application["NAME_CONTRACT_STATUS"])

# Calculate the approval, refused, and canceled rates
app1 = app
app1["approval_rate"] = app1["Approved"] * 100 / (app1["Approved"] + app1["Refused"] + app1["Canceled"])
app1["refused_rate"] = app1["Refused"] * 100 / (app1["Approved"] + app1["Refused"] + app1["Canceled"])
app1["canceled_rate"] = app1["Canceled"] * 100 / (app1["Approved"] + app1["Refused"] + app1["Canceled"])

# Select the rate columns for plotting
app2 = app[["approval_rate", "refused_rate", "canceled_rate"]]

# Create the horizontal bar plot with stacked bars
ax = app2.plot(kind="barh", stacked=True, figsize=(10, 7), color=custom_colors)

# Set the plot's appearance
ax.set_facecolor("white")
ax.set_xlabel("Percentage")
ax.set_title("Approval, Cancel and Refusal Rates by Channel Types")
plt.savefig("plot72.png", format="png", dpi=300)
# Show the plot
plt.show()


# ### Point to infer from the graph
# 
# Channel types like Stone ,regional and country-wide have maximum approval rates.
# 
# Channel of coorporate sales have maximum refusal rate.
# 
# Credit-cash centres and Contact centres have maximum cancellation rates.

# ### Highest amount credited seller areas and industries.
# 
# SELLERPLACE_AREA - Selling area of seller place of the previous application.
# 
# NAME_SELLER_INDUSTRY - The industry of the seller.

# In[160]:


import matplotlib.pyplot as plt
import seaborn as sns

# Custom color palette as provided
custom_colors = ["#A1C6EA", "#6397B1", "#3B5998"]

# Create the figure with subplots
fig = plt.figure(figsize=(13, 5))

# First subplot: Highest amount credited seller place areas
plt.subplot(121)
are = previous_application.groupby("SELLERPLACE_AREA")["AMT_CREDIT"].sum().reset_index()
are = are.sort_values(by="AMT_CREDIT", ascending=False)
ax = sns.barplot(y="AMT_CREDIT", x="SELLERPLACE_AREA", data=are[:15], color=custom_colors[0])  # Apply first color
ax.set_facecolor("white")
ax.set_title("Highest Amount Credited Seller Place Areas")

# Second subplot: Highest amount credited seller industries
plt.subplot(122)
sell = previous_application.groupby("NAME_SELLER_INDUSTRY")["AMT_CREDIT"].sum().reset_index().sort_values(by="AMT_CREDIT", ascending=False)
ax1 = sns.barplot(y="AMT_CREDIT", x="NAME_SELLER_INDUSTRY", data=sell, color=custom_colors[1])  # Apply second color
ax1.set_facecolor("white")
ax1.set_title("Highest Amount Credited Seller Industries")
plt.xticks(rotation=90)

# Adjust subplot spacing
plt.subplots_adjust(wspace=0.5)

# Set the figure background color
fig.set_facecolor("white")
plt.savefig("plot73.png", format="png", dpi=300)
# Show the plot
plt.show()


# 
# ### Popular terms of previous credit at application.
# 
# CNT_PAYMENT - Term of previous credit at application of the previous application.
# 
# 

# In[161]:


plt.figure(figsize=(13, 5))

# Define custom colors
custom_palette = ["#A1C6EA", "#6397B1", "#3B5998", "#A1C6EA", "#6397B1", "#3B5998"]

# Use the 'x' argument to specify the column and 'order' to reorder based on counts
ax = sns.countplot(x="CNT_PAYMENT", data=previous_application, palette=custom_palette, 
                   order=previous_application["CNT_PAYMENT"].value_counts().index)

# Set the background color and other styling
ax.set_facecolor("white")
plt.xticks(rotation=90)  # Rotate x-axis labels for better readability
plt.title("Popular Terms of Previous Credit at Application")
plt.savefig("plot74.png", format="png", dpi=300)
# Show the plot
plt.show()


# ### Point to infer from the graph
# 
# Popular term of previous credit are 6months ,10months ,1year ,2years & 3 years.

# ### Detailed product combination of the previous application

# In[162]:


plt.figure(figsize=(10, 8))

# Define a list of distinct blue colors
distinct_blue_palette = ["#A1C6EA", "#6397B1", "#3B5998", "#4B81A3", "#3C6B8B", "#5A86A0"]

# Create the count plot with the distinct blue color palette
sns.countplot(y = previous_application["PRODUCT_COMBINATION"], 
              order = previous_application["PRODUCT_COMBINATION"].value_counts().index, 
              palette=distinct_blue_palette)

# Set the title and show the plot
plt.title("Detailed Product Combination of the Previous Application - Count")
plt.savefig("plot75.png", format="png", dpi=300)
plt.show()


# 
# ### Frequency distribution of intrest rates and client insurance requests
# 
# NAME_YIELD_GROUP - Grouped interest rate into small medium and high of the previous application.
# 
# NFLAG_INSURED_ON_APPROVAL - Did the client requested insurance during the previous application.

# In[163]:


plt.figure(figsize=(12,6))

# Subplot 1: Client requesting insurance
plt.subplot(121)
previous_application["NFLAG_INSURED_ON_APPROVAL"].value_counts().plot.pie(
    autopct="%1.1f%%", 
    fontsize=8,
    colors=["#A1C6EA", "#6397B1", "#3B5998", "#4B81A3"],  # Custom blue color palette
    wedgeprops={"linewidth":2, "edgecolor":"white"},
    shadow=True
)
circ = plt.Circle((0, 0), .7, color="white")
plt.gca().add_artist(circ)
plt.title("Client Requesting Insurance")

# Subplot 2: Interest Rates
plt.subplot(122)
previous_application["NAME_YIELD_GROUP"].value_counts().plot.pie(
    autopct="%1.1f%%", 
    fontsize=8,
    colors=["#A1C6EA", "#6397B1", "#3B5998", "#4B81A3"],  # Custom blue color palette
    wedgeprops={"linewidth":2, "edgecolor":"white"},
    shadow=True
)
circ = plt.Circle((0, 0), .7, color="white")
plt.gca().add_artist(circ)
plt.title("Interest Rates")
plt.savefig("plot76.png", format="png", dpi=300)
# Show the plot
plt.show()


# ### Days variables - Relative to application date of current application
# 
# DAYS_FIRST_DRAWING - Relative to application date of current application when was the first disbursement of the previous application.
# 
# DAYS_FIRST_DUE - Relative to application date of current application when was the first due supposed to be of the previous application.
# 
# DAYS_LAST_DUE_1ST_VERSION - Relative to application date of current application when was the first due of the previous application.
# 
# DAYS_LAST_DUE -Relative to application date of current application when was the last due date of the previous application.
# 
# DAYS_TERMINATION - Relative to application date of current application when was the expected termination of the previous application.

# In[164]:


cols = ['DAYS_FIRST_DRAWING', 'DAYS_FIRST_DUE', 'DAYS_LAST_DUE_1ST_VERSION','DAYS_LAST_DUE', 'DAYS_TERMINATION']
plt.figure(figsize=(12,6))
sns.heatmap(previous_application[cols].describe()[1:].transpose(),
            annot=True, linewidths=2, linecolor="b", cmap="Blues")  # Using 'Blues' colormap
plt.savefig("plot77.png", format="png", dpi=300)
plt.show()


# # Corelation between variables

# 
# ### Application Data

# In[165]:


# Select only the numeric columns from the application_data DataFrame
numeric_data = application_data.select_dtypes(include=['number'])

# Calculate the correlation matrix for numeric columns
corrmat = numeric_data.corr()

# Create a heatmap
f, ax = plt.subplots(figsize=(8, 8))
sns.heatmap(corrmat, ax=ax, cmap="Blues")
plt.savefig("plot78.png", format="png", dpi=300)
plt.show()


# 
# # Previous Application 

# In[166]:


# Select only the numeric columns from the previous_application DataFrame
numeric_data = previous_application.select_dtypes(include=['number'])

# Calculate the correlation matrix for numeric columns
corrmat = numeric_data.corr()

# Create a heatmap
f, ax = plt.subplots(figsize=(8, 8))
sns.heatmap(corrmat, ax=ax, cmap="Blues")
plt.savefig("plot79.png", format="png", dpi=300)
plt.show()


# In[167]:


# Select only the numeric columns from the previous_application DataFrame
numeric_data = previous_application.select_dtypes(include=['number'])

# Calculate the correlation matrix for numeric columns
corrmat = numeric_data.corr()

# Create a dataframe with the upper triangular part of the correlation matrix
corrdf = corrmat.where(np.triu(np.ones(corrmat.shape), k=1).astype(bool))

# Unstack the correlation matrix and reset index
corrdf = corrdf.unstack().reset_index()

# Rename columns for clarity
corrdf.columns = ['Var1', 'Var2', 'Correlation']

# Drop rows with missing correlation values (NaNs)
corrdf.dropna(subset=['Correlation'], inplace=True)

# Round the correlation values to 2 decimal places and take the absolute value
corrdf['Correlation'] = corrdf['Correlation'].round(2)
corrdf['Correlation'] = abs(corrdf['Correlation'])

# Sort the correlations by the absolute value in descending order and display the top 10
top_correlations = corrdf.sort_values(by='Correlation', ascending=False).head(10)

# Show the result
top_correlations


# # Application Data

# 
# ## Top 10 Correlation Fields for Repayer

# In[183]:


df_repayer = application_data[application_data['TARGET'] == 0]
df_defaulter = application_data[application_data['TARGET'] == 1]


# In[185]:


# Step 1: Select only the numeric columns
numeric_data = df_repayer.select_dtypes(include=['number'])

# Step 2: Calculate the correlation matrix for numeric columns
corrmat = numeric_data.corr()

# Step 3: Get the upper triangle of the correlation matrix (excluding diagonal)
corrdf = corrmat.where(np.triu(np.ones(corrmat.shape), k=1).astype(bool))

# Step 4: Unstack and reset the index to create a dataframe
corrdf = corrdf.unstack().reset_index()

# Step 5: Rename the columns for clarity
corrdf.columns = ['Var1', 'Var2', 'Correlation']

# Step 6: Drop rows with missing correlation values
corrdf.dropna(subset=['Correlation'], inplace=True)

# Step 7: Round the correlation values to 2 decimal places and take the absolute value
corrdf['Correlation'] = round(corrdf['Correlation'], 2)
corrdf['Correlation'] = abs(corrdf['Correlation'])

# Step 8: Sort the correlations by the absolute value in descending order and display the top 10
top_correlations = corrdf.sort_values(by='Correlation', ascending=False).head(10)

# Show the result
top_correlations


# ## Top 10 Correlation Fields for Defaulter

# In[187]:


# Step 1: Select only the numeric columns from the dataframe
numeric_data = df_defaulter.select_dtypes(include=['number'])

# Step 2: Calculate the correlation matrix for the numeric columns
corrmat = numeric_data.corr()

# Step 3: Get the upper triangle of the correlation matrix (excluding diagonal)
corrdf = corrmat.where(np.triu(np.ones(corrmat.shape), k=1).astype(bool))

# Step 4: Unstack the correlation matrix and reset the index to create a dataframe
corrdf = corrdf.unstack().reset_index()

# Step 5: Rename the columns for clarity
corrdf.columns = ['Var1', 'Var2', 'Correlation']

# Step 6: Drop rows with missing correlation values
corrdf.dropna(subset=['Correlation'], inplace=True)

# Step 7: Round the correlation values to 2 decimal places and take the absolute value
corrdf['Correlation'] = round(corrdf['Correlation'], 2)
corrdf['Correlation'] = abs(corrdf['Correlation'])

# Step 8: Sort the correlations by the absolute value in descending order and display the top 10
top_correlations = corrdf.sort_values(by='Correlation', ascending=False).head(10)

# Show the result
top_correlations


# In[ ]:


mergeddf =  pd.merge(application_data,previous_application,on='SK_ID_CURR')
mergeddf.head()


# In[ ]:


y = mergeddf.groupby('SK_ID_CURR').size()
dfA = mergeddf.groupby('SK_ID_CURR').agg({'TARGET': np.sum})
dfA['count'] = y
display(dfA.head(10))


# In[157]:


dfA.sort_values(by = 'count',ascending=False).head(10)


# In[158]:


df_repayer = dfA[dfA['TARGET'] == 0]
df_defaulter = dfA[dfA['TARGET'] == 1]


# ### Repayers' Borrowing History 

# In[159]:


df_repayer.sort_values(by = 'count',ascending=False).head(10)


# ### Defaulters' Borrowing History 

# In[160]:


df_defaulter.sort_values(by = 'count',ascending=False).head(10)


# 

# In[ ]:




