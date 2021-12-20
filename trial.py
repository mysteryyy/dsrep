# # Data Science Research Methods Report-2

# ## **Introduction**
# The PAMAP2 Physical Activity Monitoring dataset (available here) contains data from 9 participants who participated in 18 various physical activities (such as walking, cycling, and soccer) while wearing three inertial measurement units (IMUs) and a heart rate monitor. This information is saved in separate text files for each subject. The goal is to build hardware and/or software that can determine the amount and type of physical activity performed by an individual by using insights derived from analysing the given dataset. 


import os
import pdb
import tabula
from IPython.display import display
from matplotlib import rcParams
from scipy.stats import ranksums,ttest_ind
import numpy as np
import random
import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt
os.chdir("/home/sahil/Downloads/PAMAP2_Dataset/") # Setting up working directory
import warnings
warnings.filterwarnings("ignore")

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
# - Added a new feature, 'BMI' or Body Mass Index for the 'subject_detail' table
# - Additional feature, 'Activity Type' is added to the data which classifies activities 
#   into 3 classes, 'Light' activity,'Moderate' activity and 'Intense' activity.
#   1. Lying,sitting,ironing and standing are labelled as 'light' activities.
#   2. Vacuum cleaning,descending stairs,normal walking,Nordic walking and cycling are
#      considered as 'Moderate' activities
#   3. Ascending stairs,running and rope jumping are labelled as 'Intense' activities.  
#   This classification makes it easier to perform hypothesis testing between pair of attributes.



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

def load_subjects(root1='/home/sahil/Downloads/PAMAP2_Dataset/Protocol/subject',
        root2 ='/home/sahil/Downloads/PAMAP2_Dataset/Optional/subject' ):
    cols = load_IMU()
    output = pd.DataFrame()
    for i in range(101,110):
        path1 = root1 + str(i) + '.dat'
        subject= pd.DataFrame()
         
        subject_prot = pd.read_table(path1, header=None, sep='\s+') # subject data from 
         # protocol activities
        subject = subject.append(subject_prot)

        subject.columns = cols 
        subject = subject.sort_values(by='time_stamp') # Arranging all measurements according to
         # time
        subject['id'] = i
        output = output.append(subject, ignore_index=True)
    return output
data = load_subjects()# Add your own location for the data here to replicate the code
# for eg data = load_subjects('filepath')
data = data.drop(data[data['activity_id']==0].index)# Removing rows with activity id of 0
act = gen_activity_names()
data['activity_name'] = data.activity_id.apply(lambda x:act[x])
data = data.drop([i for i in data.columns if 'orientation' in i], axis=1)  # Dropping Orientation  columns
cols_6g = [i for i in data.columns if '_6_' in i] # 6g acceleration data columns
data =  data.drop(cols_6g,axis=1) # dropping 6g acceleration columns
display(data.head())

# Saaving transformed data in pickle format becuse it has the fastest read time compared
# to all other formats
data.to_pickle("activity_data.pkl")  # Saving transformed data for future use


# **Note**: The procedure to replace missing values using the feature mean is performed
# after hypothesis testing and EDA as filling up the missing values would lead to us getting
# incorrect sample sizes for hypotheses testing. For Hypotheses testing the blank rows of an 
# attribute will simply be ignored.


def clean_data(data): # Function for extracting clean data
    #data = data.interpolate()
    # fill all the NaN values in a column with the mean values of the column
    for colName in data.columns:
        data[colName] = data[colName].fillna(data[colName].mean())
    return data


# ## Exploratory Data Analysis
# After labelling the data appropriately, we have selected 4 subjects for training set and 
# 4 subjects for testing set such that the training and testing set have approximately equal size.
# In the training set, we perform Exploratory Data Analysis and come up with potential hypotheses. 
# We then test those hypotheses on the testing set.
# 50% of data is used for training in this case(Exploratory data analysis) and the rest for testing.

def train_test_split(data,split_size):
    np.random.seed(5)
    msk = np.random.rand(len(data)) < split_size # This code implies (split_size*100)% of the values will be True
    train = data[msk] # Generating training data
    test = data[~msk] # generating testing data  
    return train,test

def train_test_split_by_subjects(data): # splitting by subjects
    subjects = [i for i in range(101,109)] # Eliminate subject 109  due to less activities
    train_subjects = [101,103,104,105]
    test_subjects = [i for i in subjects if i not in train_subjects]
    train = data[data.id.isin(train_subjects)] # Generating training data
    test = data[data.id.isin(test_subjects)] # generating testing data  
    return train,test

def split_by_activities(data):
   light = ['lying','sitting','standing','ironing'] 
   moderate = ['vacuum_cleaning','descending_stairs','normal_walking',
           'nordic_walking','cycling']
   intense = ['ascending_stairs','running','rope_jumping']
   def split(activity): #  method for returning activity labels for activities
       if activity in light:
           return 'light'
       elif activity in moderate:
           return 'moderate'
       else:
           return 'intense'
   data['activity_type'] = data.activity_name.apply(lambda x:split(x))
   return data



def random_subset(data,subset_frac): # For selecting a random subset of data
    np.random.seed(8)
    msk = np.random.rand(len(data)) < subset_frac # This code implies (split_size*100)% of the values will be True
    subset = data[msk] # Generating subset
    return subset



data = pd.read_pickle("activity_data.pkl")
data = split_by_activities(data)
train,test = train_test_split_by_subjects(data)
subj_det = tabula.read_pdf("subjectInformation.pdf",pages=1) # loading subject detail table from pdf file
# Eliminating unnecessary columns and fixing the column alignment of the table
sd = subj_det[0]
new_cols = list(sd.columns)[1:9]
sd = sd[sd.columns[0:8]]
sd.columns = new_cols 
subj_det=sd

# Calculating BMI of the subjects
height_in_metres = subj_det['Height (cm)']/100
weight_in_kg = subj_det['Weight (kg)']
subj_det['BMI'] =  weight_in_kg/(height_in_metres)**2


# ### Data Visualizations


# * Bar chart for frequency of activities.

rcParams['figure.figsize'] = 40,25
ax=sns.countplot(x="activity_name",data=train)
ax.set_xticklabels(ax.get_xticklabels(),rotation=45)# Rotating Text
plt.show()

# * 3D scatter plot of coordinates for lying 

plt.clf()
train_running = train[train.activity_name=='lying']
fig = plt.figure()
ax = fig.add_subplot(projection='3d')
x = train_running["chest_3D_acceleration_16_x"]
y = train_running["chest_3D_acceleration_16_y"]
z = train_running["chest_3D_acceleration_16_z"]
ax.scatter(x,y,z)
ax.set_xlabel('X Axis')
ax.set_ylabel('Y Axis')
ax.set_zlabel('Z Axis')
plt.show()


# * 3D scatter plot of chest acceleraation coordinates for running

plt.clf()
train_running = train[train.activity_name=='running']
fig = plt.figure()
ax = fig.add_subplot(projection='3d')
x = train_running["chest_3D_acceleration_16_x"]
y = train_running["chest_3D_acceleration_16_y"]
z = train_running["chest_3D_acceleration_16_z"]
ax.scatter(x,y,z)
ax.set_xlabel('X Axis')
ax.set_ylabel('Y Axis')
ax.set_zlabel('Z Axis')
plt.show()


# * Time series plot of x axis chest acceleration

plt.clf()
random.seed(4)
train1 = train[train.id==random.choice(train.id.unique())]
sns.lineplot(x='time_stamp',y='chest_3D_acceleration_16_z',hue='activity_name',data=train1)
plt.show()


# * 3D scatter plot of coordinates of all coordinates of chest acceleration

plt.clf()
fig = plt.figure()
ax = fig.add_subplot(projection='3d')
train_subset = random_subset(train,0.1)
x = train_subset["chest_3D_acceleration_16_x"]
y = train_subset["chest_3D_acceleration_16_y"]
z = train_subset["chest_3D_acceleration_16_z"]
ax.scatter(x,y,z)
ax.set_xlabel('X Axis')
ax.set_ylabel('Y Axis')
ax.set_zlabel('Z Axis')
plt.show()


# * Boxplot of rolling mean of vertical chest acceleration

train['rolling_mean'] = train['chest_3D_acceleration_16_z'].rolling(256).mean()
ax=sns.boxplot(x="activity_name",y="rolling_mean",data=train)
ax.set_xticklabels(ax.get_xticklabels(),rotation=45)# Rotating Text
plt.show()

# * Boxplot of rolling mean of horizontal ankle acceleration along x axis

train['rolling_mean'] = train['ankle_3D_acceleration_16_x'].rolling(256).mean()
ax=sns.boxplot(x="activity_name",y="rolling_mean",data=train)
ax.set_xticklabels(ax.get_xticklabels(),rotation=45)# Rotating Text
plt.show()

# * Boxplot of rolling mean of horizontal ankle acceleration along y axis 

train['rolling_mean'] = train['ankle_3D_acceleration_16_y'].rolling(256).mean()
ax=sns.boxplot(x="activity_name",y="rolling_mean",data=train)
ax.set_xticklabels(ax.get_xticklabels(),rotation=45)# Rotating Text
plt.show()

# * Boxplot of heart rate grouped by activity type. 

rcParams['figure.figsize'] = 15,10
ax=sns.boxplot(x="activity_type",y="heart_rate",data=train)
ax.set_xticklabels(ax.get_xticklabels(),rotation=0)# Rotating Text
plt.show()

#  1. We observe that moderate and intense activities have higher heart rate than
#     light activities as expected.
#  2. There doesn.t seem to be much seperation between moderate and intesne activity
#     heart rate.



# * Boxplot of heart rate grouped by activity. 

rcParams['figure.figsize'] = 40,25
ax=sns.boxplot(x="activity_name",y="heart_rate",data=train)
ax.set_xticklabels(ax.get_xticklabels(),rotation=45)# Rotating Text
plt.show()

#   1.  Most of the activities have a skewed distribution for heart rate.
#   2. 'Nordic_walking','running' and 'cycling' have a lot of outliers on the lower side.
#   3.  Activities like 'lying','sitting' and 'standing' have a lot of outliers on the upper side.

# * Boxplot of hand temperature grouped by activity type.


rcParams['figure.figsize'] = 15,10
ax=sns.boxplot(x="activity_type",y="hand_temperature",data=train)
ax.set_xticklabels(ax.get_xticklabels(),rotation=0)
plt.show()

# 1. Hand temperature of moderate activitie have a lot of outliers on the lower side.
# 2. There doesn't seem to be much difference in temperatures between activities.

# * Boxplot of hand temperature grouped by activity.



rcParams['figure.figsize'] = 40,25
ax=sns.boxplot(x="activity_name",y="hand_temperature",data=train)
ax.set_xticklabels(ax.get_xticklabels(),rotation=45)# Rotating Text
plt.show()

# 1. Hand temperature data of 'playing_soccer' seems to have a very pronounced positive skew.
# 2. "car_driving" and "watching_tv" have the least dispersion in hand temperature.

# * Boxplot of ankle temperature grouped by activity_type


rcParams['figure.figsize'] = 15,10
ax=sns.boxplot(x="activity_type",y="ankle_temperature",data=train)
ax.set_xticklabels(ax.get_xticklabels(),rotation=0) 
plt.show()

# 1. Ankle temperature of light and moderate activitie have  outliers on the lower side.
# 2. There doesn't seem to be much difference in temperatures between activities.

# * Boxplot of ankle temperature grouped by activity

rcParams['figure.figsize'] = 40,25
ax=sns.boxplot(x="activity_name",y="ankle_temperature",data=train)
ax.set_xticklabels(ax.get_xticklabels(),rotation=45) # Rotating Text
plt.show()

# 1. For ankle temperature, 'playing_soccer' has the least dispersed distribution.
# 2. Outliers are mostly present in 'vacuum_cleaning' on the lower side. 

# * Boxplot of chest temperature grouped by activity_type


rcParams['figure.figsize'] = 15,10
ax=sns.boxplot(x="activity_type",y="chest_temperature",data=train)
ax.set_xticklabels(ax.get_xticklabels(),rotation=0) 
plt.show()

# 1. For chest temperatures, only the 'intense' activity type has an outlier.
# 2. For this feature as well, there doesn't seem to be much difference between 
#    temperatures.

# * Boxplot of chest temperature grouped by activity.

rcParams['figure.figsize'] = 40,25
ax=sns.boxplot(x="activity_name",y="chest_temperature",data=train)
ax.set_xticklabels(ax.get_xticklabels(),rotation=45) # Rotating Text
plt.show()

# 1. Most of the activities seem to have a skewed distribution for chest temperature.
# 2. 'car_driving' and 'watching_tv' seem to have the least dispersed distribution.

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
#    a multi-modal distribution.


# ### Descriptive Statistics
# Subject Details

display(subj_det)

# Mean of heart rate and temperatures for each activity
display(train.groupby(by='activity_name')[['heart_rate','chest_temperature','hand_temperature',
    'ankle_temperature']].mean())
discard = ['activity_id','activity','time_stamp','id']# Columns to exclude from descriptive statistics

# Creating table with only relevant columns
train_trimmed = train[[i for i in train.columns if i not in discard]]

# Descriptive info of relevant feature

display(train_trimmed.describe())

# Correlation table of relevant features

display(train_trimmed.corr()) 

# Variance of each axis of acceleration grouped by activities

coordinates = [i for i in train.columns if 'acceleration' in i]
display(train.groupby(by='activity_name')[coordinates].var())

# ## Hypothesis Testing  

# Based on the exploratory data analysis carried out, the following hypotheses are tested on  
# the test set:
# - Hand temperature is higher during 'ironing' and 'vacuum_cleaning' compared
#   to other activities.
# - Ankle temperature is lower than other activities while lying.
# - Chest temperature is lower while lying compared to other activities. 
# - Rolling mean of vertical chest acceleration is higher for lying than other activities


# Based on the EDA  we performed, it does not seem that the data is normally distributed. It is 
# for this reason that Wilcoxon rank sum test was used to test the above hypothesis instead of the usual t-test which assumes that the samples follow a normal distribution.
# We test the above hypothesis using the confidence level of 5%.

# $H_0$(Null) : The hand temperature while ironing and while doing other activities are not significantly  different.
# $H_1$(Alternate) : The hand temperature while ironing is likely to be higher compared to other activities.

test1 = test[test.activity_name=='ironing'].hand_temperature.dropna()# Hand temperature while ironing
test2 = test[test.activity_name!='ironing'].hand_temperature.dropna()# hand temperature while not ironing.Nan values dropped to get the actual size of samples.
print(ranksums(test1,test2,alternative='greater'))

# Since we get a p-value of 0 which is lower than 0.05 we reject the null hypothesis and accept
# the alternate hypothesis. 

# $H_0$(Null) : The hand temperature while 'vacuum_cleaning' and while doing other activities are not significantly  different.
# $H_1$(Alternate) : The hand temperature while 'vacuum_cleaning' is likely to be higher compared to other activities.

test1 = test[test.activity_name=='vacuum_cleaning'].hand_temperature.dropna()
test2 = test[test.activity_name!='vacuum_cleaning'].hand_temperature.dropna()
print(ranksums(test1,test2,alternative='greater'))

# Since we get a p-value of 0 which is lower than 0.05 we reject the null hypothesis and accept
# the alternate hypothesis. 

# $H_0$(Null) : The ankle temperature while lying and while doing other activities are not significantly  different.
# $H_1$(Alternate) : The ankle temperature while lying is likely to be lower compared to other activities.

test1 = test[test.activity_name=='lying'].ankle_temperature.dropna()
test2 = test[test.activity_name!='lying'].ankle_temperature.dropna()
print(ranksums(test1,test2,alternative='less'))

# Since we get a p-value of 0 which is lower than 0.05 we reject the null hypothesis and accept
# the alternate hypothesis. 

# $H_0$(Null) : The chest temperature while lying is likely to be equal or greater to teh chest temperature of other activities.
# $H_1$(Alternate) : The chest temperature while lying is likely to be lower compared to other activities.

test1 = test[test.activity_name=='lying'].chest_temperature.dropna()
test2 = test[test.activity_name!='lying'].chest_temperature.dropna()
print(ranksums(test1,test2,alternative='less'))

# Since we get a p-value of 0 which is lower than 0.05 we reject the null hypothesis and accept
# the alternate hypothesis. 

# $H_0$(Null) : The rolling mean for vertical chest temperature is same as other activities.
# $H_1$(Alternate) : The rolling mean for vertical chest acceleration is likely to be higher compared to other activities.

test['rolling_mean'] = test['chest_3D_acceleration_16_z'].rolling(256).mean()
test1 = test[test.activity_name=='lying'].rolling_mean.dropna()
test2 = test[test.activity_name!='lying'].rolling_mean.dropna()
print(ranksums(test1,test2,alternative='greater'))

# Since we get a p-value of 0 which is lower than 0.05 we reject the null hypothesis and accept
# the alternate hypothesis. 

