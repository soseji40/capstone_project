import pandas as pd
import numpy as np
import datetime

# load 1990 - 2016 data made by Troy Harper
data = pd.read_csv('../data/mlb-game-data-1990-2016.csv')

#drop the unneeded columns
data.drop(['at', 'winning_pitcher', 'losing_pitcher','save','ten_game','box'],axis=1,inplace=True)

#Filling NaNs in the double_header games by using different data set. 
nan_dh_df = data[(data["attendance"].isnull()) & (data["double_header"] == 1)].sort_values('date')

#most double header games only included attendance in the first game
for x in list(nan_dh_df.index):
    data['attendance'][x] = data.iloc[x + 1]['attendance']

#both missing 2015 values are missing in other datasets, must be a collection issue. We will drop thoese rows (4)
no_nan_att_data = data[data['attendance'].notnull()]

#inings that show NaN is the full 9 inings or 9 inings with no home team batting, fill NaN with 9.0
no_nan_att_data['innings'].fillna(9.0,inplace=True)

#There's one NaN value in streak feature, it's a opening game. Therefore, we fill it with 0
no_nan_att_data[no_nan_att_data['streak'].isnull()]
no_nan_att_data['streak'].fillna(0,inplace=True)

#Win or Lose has more than w, lose or tie. Agrregate them to win, lose, tied. 
no_nan_att_data["w_or_l"].replace(to_replace = ['W-wo','W &V;','W &X;', 'W &H;'], value = ['W','W','W','W'], inplace= True)
no_nan_att_data["w_or_l"].replace(to_replace = ['L &H;','L &V;'], value =['L','L'], inplace=True)

#Record is win-lose foramt, convert it to winning%
test_rec = no_nan_att_data.record.values
temp_rec_list = []
for x in test_rec:
    temp_rec_list.append( x.split("-"))

w_percent = []
for x, y in temp_rec_list:
    if (int(x) + int (y)) != 0:
        w_percent.append(int(x)/(int(x) + int (y)))
    if (int(x) + int (y)) == 0:
        w_percent.append(0)

no_nan_att_data.record = w_percent

#games back has up + int for leading teams and postive int for trailing teams. Make negative value for trailing team.

new_gb_t = []
for x in no_nan_att_data['gb']:
    if 'up' in x:
        new_gb_t.append(x.replace('up',''))
    if 'up' not in x and 'Tied' not in x:
        new_gb_t.append('-' + x )  
    if 'Tied' in x:
        new_gb_t.append('0')

#negative sign and the space like '- 4.0' has to be changed to '-4.0' and some 0s have negative sign 

new_gb_str=[]
for x in new_gb_t:
    if x == '-0':
        new_gb_str.append(0)        
    elif '- ' in x:
        new_gb_str.append(x.replace('- ','-'))
    else:
        new_gb_str.append(x)

#set everything to float
gb_float_list = []
for x in new_gb_str:
    gb_float_list.append(float(x))

no_nan_att_data['gb']=gb_float_list

#timestamp both date and time
no_nan_att_data['date'] = pd.to_datetime(no_nan_att_data.date)
no_nan_att_data['time'] = pd.to_datetime(no_nan_att_data.time)

#convert time to minutes format
minutes_list=[]
for x in no_nan_att_data['time']:
    minutes_list.append(x.hour * 60 + x.minute)
no_nan_att_data['time']=minutes_list

#year dummies
year_dummy = pd.get_dummies(no_nan_att_data['date'].dt.year)
year_dummy.columns = list(set(no_nan_att_data['date'].dt.year))
year_dummy[2017] = [0 for x in range(len(year_dummy.index))]

#month dummies
month_dummy = pd.get_dummies(no_nan_att_data['date'].dt.month)
month_dummy.columns = ['march','april', 'may','june','july','aug','sep','oct']

#weekday dummies
weekday_dummy = pd.get_dummies(no_nan_att_data['date'].dt.weekday)
weekday_dummy.columns=['M', 'T', 'W', 'TH', 'F', 'SA', 'S']

#day or night game dummies
day_dummy = pd.get_dummies(no_nan_att_data['d_or_n'],prefix='time')

#convert streak to scalar value
streak_value = []
for x in list(no_nan_att_data['streak'].values):
    if type(x) != int:
        if '+' in x:
            streak_value.append(len(x))
        if '-' in x:
            streak_value.append(-len(x)) 
        if x == 0 or x == '0':
            streak_value.append(0)
    else:
        streak_value.append(0)
no_nan_att_data['streak'] = streak_value

#win or lose result dummy
win_dummy = pd.get_dummies(no_nan_att_data['w_or_l'],prefix='result')

#attendace has comma, convert it to int
no_nan_att_data['attendance'] =[x.replace(',', '') for x in list(no_nan_att_data['attendance'].values)]
no_nan_att_data['attendance']= no_nan_att_data['attendance'].astype(int)

#putting data together

final_data = pd.concat([no_nan_att_data, win_dummy,year_dummy, month_dummy,weekday_dummy, day_dummy],axis=1)

#column name year, month, weekday to create groupby and EDA
year_list = no_nan_att_data['date'].dt.year
final_data['year'] = year_list 
month_list = no_nan_att_data['date'].dt.month
final_data['month'] = month_list 
weekday_list = no_nan_att_data['date'].dt.weekday
final_data['weekday'] = weekday_list 

#drop dummied features
final_drop = final_data.drop(['w_or_l','d_or_n'],axis=1)

num_cols = ['runs', 'runs_allowed', 'innings', 'record',
       'div_rank', 'gb', 'time', 'attendance','runs_pg',
       'runs_ma', 'runs_allowed_ma','last_attendance','streak']

cate_cols = ['double_header','opening_day', 'result_L', 'result_T', 'result_W', 'march',
       'april', 'may', 'june', 'july', 'aug', 'sep', 'oct', 'M', 'T', 'W',
       'TH', 'F', 'SA', 'S', 'time_D', 'time_N','rival']
