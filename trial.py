# # Data Science research Methods Report-2

# ## **Introduction**
# The PAMAP2 Physical Activity Monitoring dataset (available here) contains data from 9 participants who participated in 18 various physical activities (such as walking, cycling, and soccer) while wearing three inertial measurement units (IMUs) and a heart rate monitor. This information is saved in separate text files for each subject. The goal is to build hardware and/or software that can determine the amount and type of physical activity performed by an individual by using insights derived from analysing the given dataset. 

# %%HTML
# <style type="text/css">
# table.dataframe td, table.dataframe th {
#     border: 1px  black solid !important;
#   color: black !important;
# }
# </style>


import os
from IPython.display import display
from matplotlib import rcParams
import numpy as np
import seaborn as sns
os.chdir("/home/sahil/Downloads/PAMAP2_Dataset/Protocol")
import csv
import pandas as pd
import matplotlib.pyplot as plt

# ## Data Cleaning
# For tidying up the data :
# - We load the data of various subjects and give relevant column names
#   for various features. 
# - The data for all subjects are then stacked together to form one table.
# - We remove the 'Orientation' columns because it was mentioned 
#   in the data report that it is invalid in this data collection.
# - Similarly, the rows with Activity ID "0" are also removed as
#   it does not relate to any specific activity.
# - The missing values are filled up using the mean for that feature.



"""
Given below are functions to give relevant names to the columns and create a
single table containing data for all subjects
"""
def gen_activity_names():
    # Using this function all the activity names are mapped to their ids
    act_name = {}
    act_name[0] = 'transient'
    act_name[1] = 'lying'
    act_name[2] = 'sitting'
    act_name[3] = 'standing'
    act_name[4] = 'walking'
    act_name[5] = 'running'
    act_name[6] = 'cycling'
    act_name[7] = 'Nordic_walking'
    act_name[9] = 'watching_TV'
    act_name[10] = 'computer_work'
    act_name[11] = 'car driving'
    act_name[12] = 'ascending_stairs'
    act_name[13] = 'descending_stairs'
    act_name[16] = 'vacuum_cleaning'
    act_name[17] = 'ironing'
    act_name[18] = 'folding_laundry'
    act_name[19] = 'house_cleaning'
    act_name[20] = 'playing_soccer'
    act_name[24] = 'rope_jumping'
    return act_name
def generate_three_IMU(name):
    x = name +'_x'
    y = name +'_y'
    z = name +'_z'
    return [x,y,z]
def generate_four_IMU(name):
    x = name +'_x'
    y = name +'_y'
    z = name +'_z'
    w = name +'_w'
    return [x,y,z,w]
def generate_cols_IMU(name):
    # temp
    temp = name+'_temperature'
    output = [temp]
    # acceleration 16
    acceleration16 = name+'_3D_acceleration_16'
    acceleration16 = generate_three_IMU(acceleration16)
    output.extend(acceleration16)
    # acceleration 6
    acceleration6 = name+'_3D_acceleration_6'
    acceleration6 = generate_three_IMU(acceleration6)
    output.extend(acceleration6)
    # gyroscope
    gyroscope = name+'_3D_gyroscope'
    gyroscope = generate_three_IMU(gyroscope)
    output.extend(gyroscope)
    # magnometer
    magnometer = name+'_3D_magnetometer'
    magnometer = generate_three_IMU(magnometer)
    output.extend(magnometer)
    # oreintation
    oreintation = name+'_4D_orientation'
    oreintation = generate_four_IMU(oreintation)
    output.extend(oreintation)
    return output
def load_IMU():
    output = ['time_stamp','activity_id', 'heart_rate']
    hand = 'hand'
    hand = generate_cols_IMU(hand)
    output.extend(hand)
    chest = 'chest'
    chest = generate_cols_IMU(chest)
    output.extend(chest)
    ankle = 'ankle'
    ankle = generate_cols_IMU(ankle)
    output.extend(ankle)
    return output
def load_subjects(root='/home/sahil/Downloads/PAMAP2_Dataset/Protocol/subject'):
    output = pd.DataFrame()
    cols = load_IMU()
    
    for i in range(101,110,2):
        path = root + str(i) +'.dat'
        subject = pd.read_table(path, header=None, sep='\s+')
        subject.columns = cols 
        subject['id'] = i
        output = output.append(subject, ignore_index=True)
    output.reset_index(drop=True, inplace=True)
    return output
data = load_subjects()# Add your own location for the data here to replicate the code
# for eg data = load_subjects('filepath')
data = data.drop(data[data['activity_id']==0].index)# Removing rows with activity id of 0
act = gen_activity_names()
data['activity_name'] = data.activity_id.apply(lambda x:act[x])
display(data.head())

# **Note**: The procedure to replace missing values using the feature mean is performed
# after hypothesis testing and EDA as filling up the missing values would lead to us getting
# incorrect sample sizes for hypotheses testing.For Hypotheses testing the blank rows of an 
# attribute will simply be ignored.


def clean_data(data): # Function for extracting clean data
    #data = data.interpolate()
    # fill all the NaN values in a coulmn with the mean values of the column
    for colName in data.columns:
        data[colName] = data[colName].fillna(data[colName].mean())
    activity_mean = data.groupby(['activity_id']).mean().reset_index()
    return data


# ## Exploratory Data Analysis
# After labeling the data appropriately, it is randomly split into training and testing sets. 
# In the training set, we perform Exploratory Data Analysis and come up with potential hypotheses. 
# We then test those hypotheses on the testing set.
# 60% of data is used for training in this case(Exploratory data analysis) and the rest for testing

def train_test_split(data,split_size):
    np.random.seed(5)
    msk = np.random.rand(len(data)) < split_size # This code implies 80% of the values will be True
    train = data[msk] # Generating training data
    test = data[~msk] # generating testing data  
    return train,test
train,test = train_test_split(data,0.55)

# ### Data Visualizations

# * Boxplot of heart rate grouped by activity 

rcParams['figure.figsize'] = 40,25 # Setting the figure dimensions 
rcParams['font.size'] = 35 # Setting the text and number font size
ax=sns.boxplot(x="activity_name",y="heart_rate",data=train)
ax.set_xticklabels(ax.get_xticklabels(),rotation=45)# Rotating Text
plt.show()

#   1. From the boxplot we can notice that activities like running and rope jumping have higher average heart rate than other activities
#   2. 'Nordic_walking' and 'running' have a lot of outliers on the lower side
#   3.  Activities like 'lying','sitting' and standing have a lot of outliers on the upper side.

# * Boxplot of hand temperature grouped by activity

ax=sns.boxplot(x="activity_name",y="hand_temperature",data=train)
ax.set_xticklabels(ax.get_xticklabels(),rotation=45)# Rotating Text
plt.show()

# 1. "Ironing" and "vacuum_cleaning" may have higher average hand temperatures compared to other activitiies
# 2. "Lying" and "standing" have outliers on the upper side while "ascending_stairs" has it on the lower side

# * Boxplot of ankle temperature grouped by activity

ax=sns.boxplot(x="activity_name",y="ankle_temperature",data=train)
ax.set_xticklabels(ax.get_xticklabels(),rotation=45) # Rotating Text
plt.show()

# 1. Interestly, we see that ankle_temperature might be lower on average while lying.
# 2. Outliers are mostly present in "rope_jumping" and "vacuum_cleaning" on the lower side. 

# * Boxplot of chest temperature grouped by activity

ax=sns.boxplot(x="activity_name",y="chest_temperature",data=train)
ax.set_xticklabels(ax.get_xticklabels(),rotation=45) # Rotating Text
plt.show()

# 1. Just like ankle temperature,the mean of chest temperature seems to be lower while lying and 
#    even "running" seems to have lower average although the data is more widely distributed.
# 2. The outliers are only present in "lying" and they are on the higher side.

# * A joint plot trying to investigate possibility of correlation between heart rate 
#   and chest temperature.

plt.clf()
rcParams['font.size'] = 20 # Setting the text and number font size
g = sns.JointGrid(data=train, x="heart_rate", y="chest_temperature",
                  height=10,ratio=3)
g.plot_joint(sns.scatterplot,palette='colorblind')
g.plot_marginals(sns.histplot)
# ax.set_xticklabels(ax.get_xticklabels(),rotation=45) # Rotating Text
plt.show()

# 1. From the scatter plot, we see that there does not seem to be a correlation between
#    the two variables.
# 2. The respective histograms indicate that both the features considered have 
#    a multi-modal distribution

# ### Decriptive Statistics
# Mean of heart rate and temperatures for each activity

display(train.groupby(by='activity_name')[['heart_rate','chest_temperature','hand_temperature',
    'ankle_temperature']].mean())
discard = ['activity_id','activity','time_stamp']# Columns to exclude from descriptove statistics

# Descriptive info only for columns choosen
display(train[[i for i in train.columns if i not in discard]].describe()) 