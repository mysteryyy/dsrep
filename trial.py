# # Data Science Research Methods Report-2

# ## **Introduction**
# The PAMAP2 Physical Activity Monitoring dataset (available here) contains data from 9 participants who participated in 18 various physical activities (such as walking, cycling, and soccer) while wearing three inertial measurement units (IMUs) and a heart rate monitor. This information is saved in separate text files for each subject. The goal is to build hardware and/or software that can determine the amount and type of physical activity performed by an individual by using insights derived from analysing the given dataset. 


import os
import pdb
import tabula
import itertools
import antropy as ant
from IPython.display import display
from matplotlib import rcParams
from numpy.fft import rfft
from scipy.stats import ranksums,ttest_ind
import numpy as np
from numpy.lib.stride_tricks import sliding_window_view
import random
import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import  LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import OneHotEncoder
from sklearn.decomposition import PCA
from sklearn.metrics import classification_report,accuracy_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report
from sklearn import preprocessing
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
# - The missing values are filled up using the linear interpolation method.
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
# Saving transformed data in pickle format becuse it has the fastest read time compared
# to all other formats
data.to_pickle("activity_data.pkl")  # Saving transformed data for future use




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

def split_by_activities(data):
   main = ['lying','sitting','standing','walking','running','cycling'] 
   others = [i for i in data.activity_name.unqiue() if i not in main]
   sit_stand = ['sitting','standing']
   if activity in main:
       if activity in sit_stand:
           return 'sitting/standing'
       else:
           return activity
   else:
       return 'others'




def random_subset(data,subset_frac): # For selecting a random subset of data
    np.random.seed(8)
    msk = np.random.rand(len(data)) < subset_frac # This code implies (split_size*100)% of the values will be True
    subset = data[msk] # Generating subset
    return subset

# Loading data and doing the train-test split for EDA and Hypothesis testing.
data = pd.read_pickle("activity_data.pkl")
data = split_by_activities(data)
train,test = train_test_split_by_subjects(data) # train and test data for EDA and hypothesis testing respectively.
subj_det = tabula.read_pdf("subjectInformation.pdf",pages=1) # loading subject detail table from pdf file.
# Eliminating unnecessary columns and fixing the column alignment of the table.
sd = subj_det[0]
new_cols = list(sd.columns)[1:9]
sd = sd[sd.columns[0:8]]
sd.columns = new_cols 
subj_det=sd

# Create clean data for use in modelling
eliminate = ['activity_id','activity_name','time_stamp','id'] # Columns not meant to be cleaned
features = [i for i in data.columns if i not in eliminate]
clean_data = data
clean_data[features] =clean_data[features].ffill()
display(clean_data.head())

# After linear interpolation, the first four values of heart rate are still missing. So we fill that using back fill method.
clean_data = clean_data.dropna()
display(clean_data.head())

# Finally, save the clean data for future use in model prediction
clean_data.to_pickle("clean_act_data.pkl")

# ## Exploratory Data Analysis
# After labelling the data appropriately, we have selected 4 subjects for training set and 
# 4 subjects for testing set such that the training and testing set have approximately equal size.
# In the training set, we perform Exploratory Data Analysis and come up with potential hypotheses. 
# We then test those hypotheses on the testing set.
# 50% of data is used for training in this case(Exploratory data analysis) and the rest for testing.

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


# * 3D scatter plot of chest acceleration coordinates for lying
#
#   It is expected that vertical chest acceleration will be more while lying due to the
#   movements involved and an attempt is made to check this visually over here.

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

#    As we see, there seems to be more variance along the z axis(vertical direction) than the
#    x and y axis.


# * 3D scatter plot of chest acceleration coordinates for running
#
#   Since running involves mostly horizontal movements for the chest, we expect
#   most of chest acceleration data to lie on the horizontal x amd y axis.

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

#   As we expected, for running, most of the points lie along the x and y axis.

# * Time series plot of x axis chest acceleration

plt.clf()
random.seed(4)
train1 = train[train.id==random.choice(train.id.unique())]
sns.lineplot(x='time_stamp',y='chest_3D_acceleration_16_z',hue='activity_name',data=train1)
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
#  2. There doesn't seem to be much seperation between moderate and intesne activity
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
# - Heart rate of moderate activities are greater than heart rate of light activities.
# - Heart rate of intense activities are greater than heart rate of light activities.
# - Chest acceleration along z axis is greater than that along x axis during lying.
# - Chest acceleration along x axis is greater than that along z axis during running.


# Based on the EDA  we performed, it does not seem that the data is normally distributed. It is 
# for this reason that Wilcoxon rank sum test was used to test the above hypothesis instead of the usual t-test which assumes that the samples follow a normal distribution.
# We test the above hypothesis using the confidence level of 5%.

# ### Hypothesis 1
# $H_0$(Null) : The heart rate during  moderate activities are the same or lower than that of light activities. 
# $H_1$(Alternate) : The heart rate during moderate activities are likely to be higher during lying compared to light activities.  

test1 = test[test.activity_type=='moderate'].heart_rate.dropna()# Heart rate of moderate activities with nan values dropped
test2 = test[test.activity_type=='light'].heart_rate.dropna()# Heart rate of light activities with nan values dropped
print(ranksums(test1,test2,alternative='greater'))

# ### Hypothesis 2
# $H_0$(Null) : The heart rate during intense activities are the same or lower than that of light activities. 
# $H_1$(Alternate) : The heart rate during intense activities are likely to be higher during than during lower activities.  

test1 = test[test.activity_type=='intense'].heart_rate.dropna()# Heart rate of moderate activities with nan values dropped
test2 = test[test.activity_type=='light'].heart_rate.dropna()# Heart rate of light activities with nan values dropped
print(ranksums(test1,test2,alternative='greater'))


# Since we get a p-value of 0 which is lower than 0.05 we reject the null hypothesis and accept
# the alternate hypothesis. 


# ### Hypothesis 3
# $H_0$(Null) : The z axis chest acceleration during lying is lower or same as the x axis acceleration. 
# $H_1$(Alternate) :The z axis chest acceleration during lying is higher than the x axis acceleration. 


test_l = test[test.activity_name=='lying']
feature1='chest_3D_acceleration_16_z'
feature2='chest_3D_acceleration_16_x'
test1 = test_l[feature1]
test2 = test_l[feature2]
print(ranksums(test1,test2,alternative='greater'))

# Since we get a p-value of 0 which is lower than 0.05 we reject the null hypothesis and accept
# the alternate hypothesis. 

# ### Hypothesis 4
# $H_0$(Null) : The x axis chest acceleration during running is lower or same as the z axis acceleration. 
# $H_1$(Alternate) :The x axis chest acceleration during lying is higher than the z axis acceleration. 


test_l = test[test.activity_name=='running']
feature1='chest_3D_acceleration_16_x'
feature2='chest_3D_acceleration_16_z'
test1 = test_l[feature1]
test2 = test_l[feature2]
print(ranksums(test1,test2,alternative='greater'))

# Since we get a p-value of 0 which is lower than 0.05 we reject the null hypothesis and accept
# the alternate hypothesis. 

# ## Model Prediction

clean_data = pd.read_pickle("clean_act_data.pkl")
train_subjects = [101,103,104,105]
discard = ['activity_id','activity','activity_name','time_stamp','id','activity_type']# Columns to exclude from descriptive stat
def determine_axis(feat):
    if 'x' in feat:
        return 'x'
    elif 'y' in feat:
        return 'y'
    else:
        return 'z'
def spectral_centroid(signal):
    spectrum = np.abs(np.fft.rfft(signal))
    normalized_spectrum = spectrum / np.sum(spectrum)  # like a probability mass function
    normalized_frequencies = np.linspace(0, 1, len(spectrum))
    spectral_centroid = np.sum(normalized_frequencies * normalized_spectrum)
    return spectral_centroid
def peak_freq(data):
    dt=0.01
    n = len(data)
    fhat = np.fft.fft(data,n)                     # Compute the FFT
    PSD = fhat * np.conj(fhat) / n             # Power spectrum (power per freq)
    freq = (1/(dt*n)) * np.arange(n)           # Create x-axis of frequencies in Hz
    return freq[np.argmax(fhat[0:n//2])]
    #L = np.arange(1,np.floor(n/2),dtype='int') # Only consider the first half
def peak_freq2(data):
    dt=0.01
    n = len(data)
    fhat = np.fft.fft(data,n)                     # compute the fft
    psd = fhat * np.conj(fhat) / n             # power spectrum (power per freq)
    freq = (1/(dt*n)) * np.arange(n)           # create x-axis of frequencies in hz
    return freq[np.argmax(psd)]
    #l = np.arange(1,np.floor(n/2),dtype='int') # only consider the first half

def sliding_window_feats(data,feats,win_len,step):
    final=[]
    i=0
    for i in range(0,len(data),100):
        if((i+256) > len(data)):
            break
        temp = data.iloc[i:i+256]
        temp1 = pd.DataFrame()
        for feat in feats:
            temp1[f'{feat}_roll_mean'] = [temp[feat].mean()]
            temp1[f'{feat}_roll_median'] = [temp[feat].median()]
            temp1[f'{feat}_roll_var'] = [temp[feat].var()]
            temp1[f'{feat}_spectral_centroid'] = [spectral_centroid(temp[feat])]
        temp1['time_stamp'] = [list(temp.time_stamp.values)[-1]]
        temp1[feats] = [temp[feats].iloc[-1]]
        temp1['activity_name'] = [temp['activity_name'].iloc[-1]]
        temp1['activity_type'] = [temp['activity_type'].iloc[-1]]
        final.append(temp1)
    final_data = pd.concat(final)
    return final_data


def create_sliding_window_corr(data,win_len):
   hand_coords = [f'hand_3D_acceleration_16_{i}' for i in ['x','y','z']] 
   chest_coords = [f'chest_3D_acceleration_16_{i}' for i in ['x','y','z']] 
   ankle_coords = [f'ankle_3D_acceleration_16_{i}' for i in ['x','y','z']] 
   def calc_corr(coords):
      for i in itertools.combinations(coords,2):
        ax1 = determine_axis(i[0])
        ax2 = determine_axis(i[1])
        coord_type = coords[0][0:5]
        if '_' in coord_type:
            coord_type = coord_type[:-1]
        data[f'{coord_type}_{ax1}_{ax2}_corr']=data[
        i[0]].rolling(win_len).corr(data[i[1]])
      return data
   data = calc_corr(hand_coords)
   data = calc_corr(chest_coords)
   data = calc_corr(ankle_coords) 
   return data
def create_sliding_window_feats(data,feats,win_len):
   for feat in feats:
       data[f'{feat}_roll_mean'] = data[feat].rolling(win_len).mean()
       data[f'{feat}_roll_median'] = data[feat].rolling(win_len).median()
       data[f'{feat}_roll_var'] = data[feat].rolling(win_len).var()
#        data = data.dropna()
   return data
def train_test_split_acttype(clean_data,features):
    clean_data = clean_data.dropna()
    train = clean_data[clean_data.id.isin(train_subjects)]
    val = clean_data[clean_data.id.isin([102,106])]
    test = clean_data[clean_data.id.isin([107,108])]
    x_train = train[features]
    x_val = val[features]
    x_test = test[features]
    y_train = le.fit_transform(train.activity_type)
    y_val = le.fit_transform(val.activity_type)
    y_test = le.fit_transform(test.activity_type)
    return x_train,x_val,x_test,y_train,y_val,y_test
def train_test_split_actname(clean_data,features):
    le = preprocessing.LabelEncoder()
    clean_data = clean_data.dropna()
    train = clean_data[clean_data.id.isin(train_subjects)]
    val = clean_data[clean_data.id.isin([102,106])]
    test = clean_data[clean_data.id.isin([107,108])]
    x_train = train[features]
    x_val = val[features]
    x_test = test[features]
    y_train = le.fit_transform(train.activity_name)
    y_val = le.fit_transform(val.activity_name)
    y_test = le.fit_transform(test.activity_name)
    return x_train,x_val,x_test,y_train,y_val,y_test 


feats = [i for i in clean_data.columns if i not in discard]
final=[]
for i in clean_data.id.unique():
   temp = clean_data[clean_data.id==i]
   temp = sliding_window_feats(temp,feats,256,100)
   temp['id'] = [i]*len(temp)
   final.append(temp)
clean_data_feats = pd.concat(final) 
clean_data_feats.to_pickle("activity_short_data.pkl")
# print(clean_data[[i for i in clean_data.columns if 'roll' in i]].head())
clean_data_feats = pd.read_pickle("activity_short_data.pkl")
features = [i for i in clean_data_feats.columns if i not in discard]
x_train,x_val,x_test,y_train,y_val,y_test = train_test_split_actname(clean_data_feats,features)
rf = RandomForestClassifier(min_samples_split=6,max_depth=130)
rf.fit(x_train,y_train)
print("accuracy: ")
print(accuracy_score(rf.predict(x_val),y_val))
print("Thank You")




