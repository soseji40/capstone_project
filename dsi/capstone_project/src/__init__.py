from scipy import stats
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import datetime
from IPython import get_ipython
from IPython import display
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, BaggingRegressor
from sklearn.svm import SVR
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.metrics import r2_score, fbeta_score, f1_score
from sklearn.decomposition import PCA
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import normalize, StandardScaler, PolynomialFeatures, Normalizer
from statsmodels.stats.outliers_influence import variance_inflation_factor
from sklearn.feature_selection import SelectKBest, f_regression
from sklearn.svm import LinearSVR

get_ipython().run_line_magic('matplotlib', 'inline')

np.random.seed(7)

def team_data_clean(dataframe):
    #create a time, run data frame to calculate the 5 game moving average
    temp_run_ma_df = dataframe[['date','runs']].sort_values('date',axis=0)
    temp_ma = pd.rolling_mean(temp_run_ma_df['runs'],window=5,min_periods=0)
    dataframe['runs_ma'] = temp_ma
    
    #shift stats
    previous_date = ['attendance','runs', 'runs_allowed', 'innings', 'time', 'result_L', 'result_T', 'result_W']
    
    #swift the stats by the year 
    shifted_data = dataframe[previous_date].shift(1,axis=0)
    
    #drop the NaNs in opening game and fill it with 0.0 
    shifted_data.fillna(value = 0.0, axis=0, inplace=True)
    shifted_data.rename(columns={"attendance":"last_attendance"}, inplace=True)
    
    #fill previous date, excluding attendance with 0 for opening games
    previous_date_stats = ['runs', 'runs_allowed', 'innings', 'time', 'result_L', 'result_T', 'result_W']
    
    dataframe.loc[dataframe[dataframe['opening_day'] == 1][previous_date_stats].index, previous_date_stats]  = 0
    
    dataframe.drop(previous_date_stats, axis=1, inplace= True)
    team_df = pd.concat([dataframe, shifted_data], axis=1)
    
    return team_df

#Performs clean up for test data
def test_data_clean(dataframe):
    #create a time, run data frame to calculate the 5 game moving average
    temp_run_ma_df = dataframe[['date','runs']].sort_values('date',axis=0)
    temp_ma = pd.rolling_mean(temp_run_ma_df['runs'],window=5,min_periods=0)
    dataframe['runs_ma'] = temp_ma
    
    #shift stats
    previous_date = ['attendance','runs', 'runs_allowed', 'innings', 'time', 'result_L', 'result_W']
    
    #swift the stats by the year 
    shifted_data = dataframe[previous_date].shift(1,axis=0)
    
    #drop the NaNs in opening game and fill it with 0.0 
    shifted_data.fillna(value = 0.0, axis=0, inplace=True)
    shifted_data.rename(columns={"attendance":"last_attendance"}, inplace=True)
    
    #fill previous date, excluding attendance with 0 for opening games
    previous_date_stats = ['runs', 'runs_allowed', 'innings', 'time', 'result_L', 'result_W']
    
    dataframe.loc[dataframe[dataframe['opening_day'] == 1][previous_date_stats].index, previous_date_stats]  = 0
    
    dataframe.drop(previous_date_stats, axis=1, inplace= True)
    team_df = pd.concat([dataframe, shifted_data], axis=1)
    team_df['result_T'] = [0 for x in range(len(team_df.index))]
    team_df['march'] = [0 for x in range(len(team_df.index))]
    
    return team_df

#dictionary of rival teams 
team_rival={
'LAA'       : ['LAD', 'TEX'],
'HOU'       : ['TEX','OAK','STL'],
'OAK'       : ['SFG','LAA'],
'TOR'       : ['NYY','BOS'],
'ATL'       : ['NYM','WSN'],
'MIL'       : ['CHC','STL'],
'STL'       : ['CHC','CIN'],
'CHC'       : ['STL','CHW'],
'ARI'       : ['LAD', 'COL'],
'LAD'	    : ['SFG','LAA'],
'SFG'	    : ['LAD','OAK'],
'CLE'	    : ['DET','CIN'],
'SEA'	    : ['OAK','TEX'], 
'MIA'	    : ['ATL','NYM'],
'NYM'		: ['NYY','PHI'],
'WSN'       : ['BAL','ATL'],
'BAL'	    : ['NYY', 'WSN'],
'SDP'	    : ['LAD','ARI'],
'PHI'	    : ['NYM','ATL'],
'PIT'	    : ['STL','PHI'],
'TEX'	    : ['LAA','HOU'],
'TBR'		: ['NYY','BOS'],
'BOS'	    : ['NYY', 'TBR','TBD'],
'CIN'		: ['STL','CLE'],
'COL'	    : ['LAD','ARI'],
'KCR'	    : ['STL','DET'],
'DET'	    : ['CLE','CHW'],
'MIN'		: ['CHW','DET'],
'CHW'	    : ['CHC','MIN'],
'NYY'	    : ['BOS','NYM']	
}

#add rival boolean to the team data
def add_rival(team, dataframe):
    bool_list =[]
    
    for opponents in list(dataframe['opponent']):
        for teams in team_rival[team]:
            temp_list=[]
            if opponents == teams:
                temp_list.append(1)
            else:
                temp_list.append(0)

        bool_list.append(sum(temp_list))
    dataframe['rival'] = bool_list
    return dataframe  

#convert dataframe with new index to pickle
def pickle_able(data, filename):
    data['index'] = range(len(data.index))
    data.set_index('index')
    data.to_pickle(filename)
    return None

#numerical and categorical features

num_cols = ['runs', 'runs_allowed', 'innings', 'record',
       'div_rank', 'gb', 'time', 'attendance','runs_pg',
       'runs_ma', 'runs_allowed_ma','last_attendance','streak']

num_no_att_cols = ['runs', 'runs_allowed', 'innings', 'record',
       'div_rank', 'gb', 'time', 'runs_pg',
       'runs_ma', 'runs_allowed_ma','last_attendance','streak']

cate_cols = ['double_header','opening_day', 'result_L', 'result_T', 'result_W', 'march',
       'april', 'may', 'june', 'july', 'aug', 'sep', 'oct', 'M', 'T', 'W',
       'TH', 'F', 'SA', 'S', 'time_D', 'time_N','rival']

#performs numeric edas 
def eda (team_df):    
    #basic statistics 
    team_stats = team_df[num_cols].describe().T
    team_skew_values = list()
    for num_col in team_stats.index:
        num_col_skew = stats.skew(team_df[num_col])
        team_skew_values.append(num_col_skew)
    team_stats['skew'] = team_skew_values
    
    #heatmap
    fig = plt.figure(figsize=(20,10))
    sns.heatmap(team_df[num_cols].corr(), annot=True)
    
    #distribution plot
    fig = plt.figure(figsize=(30,40))
    for i, num_col in enumerate(team_stats.index):
        fig.add_subplot(7,2,1+i)
        sns.distplot(team_df[num_col])
    
        mean_value = team_df[num_col].mean()
        plt.axvline(mean_value, c='red')
    
        median_value = team_df[num_col].median()
        plt.axvline(median_value, c='green')
    
    #attendance box plot by year
    fig = plt.figure(figsize=(20,10))
    order = set(team_df['year'].values)
    sns.boxplot(x='year', y='attendance', data=team_df, order=order)
    
    #scatterplot with attendance
    fig = plt.figure(figsize=(20,20))
    for i, num_col in enumerate(num_cols):
        fig.add_subplot(7,2,1+i)
        plt.scatter(team_df[num_col], team_df['attendance'])
    
    #residualplot
    fig = plt.figure(figsize=(20,20))
    for i, num_col in enumerate(num_cols):
        fig.add_subplot(7,2,1+i)
        sns.residplot(x=num_col, y='attendance', data=team_df)
        
    #swarmplot
    fig = plt.figure(figsize=(15,5))
    sns.swarmplot(x='month', y="attendance", hue='weekday',data= team_df)
    fig = plt.figure(figsize=(15,5))
    sns.swarmplot(x='weekday', y="attendance", hue='month',data= team_df)
    
    #pointplot
    fig = plt.figure(figsize=(10,5))
    sns.pointplot(x="weekday", y="attendance", hue="month", data=team_df)
    fig = plt.figure(figsize=(10,5))
    sns.pointplot(x="month", y="attendance", hue="weekday", data=team_df)

    return team_stats 

def model_fit_ready(team_df):
    try:
        feature = team_df.drop(['date', 'team','opponent','year','month','weekday', 'attendance'], axis=1)
    except:
        feature = team_df.drop(['date', 'team','opponent','attendance'], axis=1)
    target = team_df['attendance']
    return team_df, feature, target 

def annual_mean_r2(team_df):
    aml = []

    for year in set(team_df['year'].values):
        counter = 0
        while counter < len(team_df[team_df['year'] == year]['attendance']):
            annual_mean = team_df[team_df['year'] == year]['attendance'].sum()/len(team_df[team_df['year'] == year]['attendance'])
            aml.append(annual_mean)
            counter += 1  
    return r2_score(team_df['attendance'], aml)

#Performs model fit and scoring on naive models. 
def model_fit_score(feature, target):
    
    X_train, X_test, y_train, y_test = train_test_split(feature, target, test_size=0.2,random_state=7)

    score_table = pd.DataFrame(index=['Bagging_r2','Decision_tree_r2','Random_forest_r2','Gradient_boost_r2'], columns=['train_score','test_score'])
    train_score = []
    test_score = []
    
    model_list = [BaggingRegressor(), DecisionTreeRegressor(), RandomForestRegressor(), GradientBoostingRegressor()]
    for model in model_list:
        model.fit(X_train, y_train)
        train_score.append(model.score(X_train, y_train))
        test_score.append(model.score(X_test, y_test))
    score_table['train_score'] = train_score
    score_table['test_score'] = test_score 

    return score_table


def grid_score(feature, target, benchmark_table):    
    #GridSearch for Bagging Regressor
    br_grid_steps = (
        ("scaler", StandardScaler()),
        ("model", BaggingRegressor(n_jobs=-1)),
        )
    
    br_grid_pipe = Pipeline(br_grid_steps)
    
    br_param_grid = {
        "model__n_estimators": range(15,30),
        "model__max_samples": np.linspace(0.1,1.0,10, dtype=float),
        }
    
    br_grid = GridSearchCV(br_grid_pipe, param_grid = br_param_grid)
    
    #Decision Tree Grid Search Pipeline
    tree_steps = (
        ("scaler", StandardScaler()),
        ("model", DecisionTreeRegressor()),
        )
    
    tree_pipe = Pipeline(tree_steps)  
    
    tree_param_grid = {
        "model__splitter": ["random", "best"],
        "model__max_depth": [None, 3, 5, 7, 9],
        "model__max_features" : ['auto', 'sqrt','log2']
        }
    
    tree_grid = GridSearchCV(tree_pipe, param_grid = tree_param_grid)  
    
    #Random Forest Grid Search Pipeline
    rf_grid_steps = (
        ("scaler", StandardScaler()),
        ("model", RandomForestRegressor(n_jobs=-1)),
        )
    
    rf_grid_pipe = Pipeline(rf_grid_steps)    
    
    rf_param_grid = {
        "model__n_estimators": range(15,20),
        "model__max_features" : ['auto', 'sqrt','log2'],
        "model__min_samples_leaf": [10,30,50,70],
        "model__max_depth": [None, 3, 5, 7, 9]
        }
    
    rf_grid = GridSearchCV(rf_grid_pipe, param_grid = rf_param_grid)
    
    #Gradient Boosting Grid Search Pipeline
    gb_grid_steps = (
        ("scaler", StandardScaler()),
        ("model", GradientBoostingRegressor()),
        )
    
    gb_grid_pipe = Pipeline(gb_grid_steps)
    
    gb_param_grid = {
        "model__loss": ['ls', 'lad', 'huber', 'quantile'],
        "model__n_estimators": [ 100, 150, 200],
        "model__max_features" : ['auto', 'sqrt','log2'],
        "model__max_depth": [None, 3,  9]
        }
    
    gb_grid = GridSearchCV(gb_grid_pipe, param_grid = gb_param_grid)

    #Train Test Split
    X_train, X_test, y_train, y_test = train_test_split(feature, target, test_size=0.2,random_state=7)

    #Dataframe to store scores 
    score_table = pd.DataFrame(index=['Bagging_r2','Decision_tree_r2','Random_forest_r2','Gradient_boost_r2'], columns=['P&O_train_score','P&O_test_score'])
    train_score = []
    test_score = []

    model_list = [br_grid, tree_grid, rf_grid, gb_grid]
    for model in model_list:
        model.fit(X_train, y_train)
        train_score.append(model.score(X_train, y_train))
        test_score.append(model.score(X_test, y_test))
        
    score_table['P&O_train_score'] = train_score
    score_table['P&O_test_score'] = test_score 

    score_table = pd.concat([benchmark_table, score_table], axis=1)
    
    for model in [tree_grid.best_estimator_ , rf_grid.best_estimator_, gb_grid.best_estimator_]:
        importances = model.named_steps["model"].feature_importances_
        indices = np.argsort(importances)[::-1]
        names = np.sort(importances)[::-1]

        # Print the feature ranking
        print(str(model)) 
        print("Feature top 10 ranking:")

        for f in range(10):
            print("%d. %s (%f)" % (f + 1, feature.columns[indices[f]], importances[indices[f]]))

        # Plot the feature importances
        plt.figure(figsize=(10,5))
        plt.title("Feature importances")
        plt.bar(range(10),importances[indices][:10], color="r", align="center")
        plt.xticks(range(10), feature.columns[indices][:10])
        plt.xlim([-1, 10])
        plt.show()
    
    bg_im = np.sort(np.mean([tree.feature_importances_ for tree in br_grid.best_estimator_.named_steps["model"].estimators_],axis=0))[::-1]
    
 
    bg_id = np.argsort(bg_im)[::-1]
    
    # Print the feature ranking
    print(str(br_grid.best_estimator_)) 
    print("Feature top 10 ranking:")

    for f in range(10):
        print("%d. %s (%f)" % (f + 1, feature.columns[bg_id[f]], bg_im[bg_id[f]]))

    # Plot the feature importances
    plt.figure(figsize=(10,5))
    plt.title("Feature importances")
    plt.bar(range(10),bg_im[bg_id][:10], color="r", align="center")
    plt.xticks(range(10), feature.columns[bg_id][:10])
    plt.xlim([-1, 10])
    plt.show()
    
    return score_table
  
def score_feature_importance(feature, target):    
    #GridSearch for Bagging Regressor
    br_grid_steps = (
        ("scaler", StandardScaler()),
        ("model", BaggingRegressor(n_jobs=-1)),
        )
    
    bagging_regressor = Pipeline(br_grid_steps)
    
    #Decision Tree Grid Search Pipeline
    tree_steps = (
        ("scaler", StandardScaler()),
        ("model", DecisionTreeRegressor()),
        )
    
    decision_tree = Pipeline(tree_steps)  
    
    #Random Forest Grid Search Pipeline
    rf_grid_steps = (
        ("scaler", StandardScaler()),
        ("model", RandomForestRegressor(n_jobs=-1)),
        )
    
    random_forest = Pipeline(rf_grid_steps)    
    
    #Gradient Boosting Grid Search Pipeline
    gb_grid_steps = (
        ("scaler", StandardScaler()),
        ("model", GradientBoostingRegressor()),
        )
    
    gradient_boosting = Pipeline(gb_grid_steps)

    #Train Test Split
    X_train, X_test, y_train, y_test = train_test_split(feature, target, test_size=0.2,random_state=7)

    #Dataframe to store scores 
    score_table = pd.DataFrame(index=['Bagging_r2','Decision_tree_r2','Random_forest_r2','Gradient_boost_r2'], columns=['preprocessed_train_score','preprocessed_test_score'])
    train_score = []
    test_score = []

    model_list = [bagging_regressor , decision_tree , random_forest, gradient_boosting]
    for model in model_list:
        model.fit(X_train, y_train)
        train_score.append(model.score(X_train, y_train))
        test_score.append(model.score(X_test, y_test))
        
    score_table['preprocessed_train_score'] = train_score
    score_table['preprocessed_test_score'] = test_score 
    
    for model in [decision_tree , random_forest, gradient_boosting]:
        importances = model.named_steps["model"].feature_importances_
        indices = np.argsort(importances)[::-1]
        names = np.sort(importances)[::-1]

        # Print the feature ranking
        print(str(model)) 
        print("Feature top 10 ranking:")

        for f in range(10):
            print("%d. %s (%f)" % (f + 1, feature.columns[indices[f]], importances[indices[f]]))

        # Plot the feature importances
        plt.figure(figsize=(10,5))
        plt.title("Feature importances")
        plt.bar(range(10),importances[indices][:10], color="r", align="center")
        plt.xticks(range(10), feature.columns[indices][:10])
        plt.xlim([-1, 10])
        plt.show()
    
    bg_im = np.sort(np.mean([tree.feature_importances_ for tree in bagging_regressor.named_steps["model"].estimators_],axis=0))[::-1]
    
 
    bg_id = np.argsort(bg_im)[::-1]
    
    # Print the feature ranking
    print(str(bagging_regressor)) 
    print("Feature top 10 ranking:")

    for f in range(10):
        print("%d. %s (%f)" % (f + 1, feature.columns[bg_id[f]], bg_im[bg_id[f]]))

    # Plot the feature importances
    plt.figure(figsize=(10,5))
    plt.title("Feature importances")
    plt.bar(range(10),bg_im[bg_id][:10], color="r", align="center")
    plt.xticks(range(10), feature.columns[bg_id][:10])
    plt.xlim([-1, 10])
    plt.show()
    
    return score_table

def interaction_feature(feature, target): 
    poly = PolynomialFeatures(degree=2, interaction_only=True)
    
    record =pd.DataFrame(poly.fit_transform(feature[['div_rank','record','streak','gb']]),index=feature.index, columns=['bias','div_rank','record','streak','gb','div_rank*record','div_rank*streak','div_rank*gb','record*streak','record*record','streak*record'])

    record_int = record[['div_rank*record','div_rank*streak','div_rank*gb','record*streak','record*record','streak*record']]

    run = pd.DataFrame(poly.fit_transform(feature[['runs','runs_ma','runs_pg']]),index=feature.index, columns=['bias', 'runs', 'runs_ma', 'runs_pg', 'runs*runs_ma', 'runs*runs_pg', 'runs_ma*runs_pg'])
    runs_int = run[['bias','runs*runs_ma', 'runs*runs_pg', 'runs_ma*runs_pg']]

    time = pd.DataFrame(poly.fit_transform(feature[['time','innings','runs_allowed']]), index=feature.index, columns=['1', 'time', 'innings', 'runs_allowed', 'time*innings', 'time*runs_allowed', 'innings*runs_allowed'])
    time_int = time[['time*innings', 'time*runs_allowed', 'innings*runs_allowed']]

    new_features = pd.concat([feature,time_int,runs_int,record_int],axis=1)

    feature_p = pd.DataFrame(index= feature.columns,columns=['f_score','p_value'])
    feature_p['f_score']=f_regression(feature, target)[0]
    feature_p['p_value']=f_regression(feature, target)[1]

    kept_features = feature_p[feature_p['p_value'] < 0.05].index
    unkept_features = feature_p[feature_p['p_value'] > 0.05]

    return new_features, new_features[kept_features], unkept_features

def non_para_model_fit (feature, target, pca_n_component):

    #Logistic Regression Grid Search Pipeline
    lr_steps = (
        ("scaler", StandardScaler()),
        ("norm", Normalizer()),
        ("pca", PCA(n_components=pca_n_component)),
        ("lr", LogisticRegression()),
        )
    
    lr_pipe = Pipeline(lr_steps)  
    
    lr_param_grid = {
        'lr__C':np.logspace(-3,3,7),
        'lr__penalty':['l2','l1']
        }
    
    lr_grid = GridSearchCV(lr_pipe, param_grid = lr_param_grid)  

    #Support Vector Machine Grid Search Pipeline

    svm_steps =(
        ("scaler", StandardScaler()),
        ("norm", Normalizer()),
        ("svm", LinearSVR()),
        )
    svm_pipe = Pipeline(svm_steps)

    svm_param_grid = {
        'svm__C':np.logspace(-3,3,7),
        'svm__loss':['epsilon_insensitive', 'squared_epsilon_insensitive']
        }

    svm_grid = GridSearchCV(svm_pipe, param_grid=svm_param_grid)

    X_train, X_test, y_train, y_test = train_test_split(feature, target, test_size=0.2,random_state=7)

    #Dataframe to store scores 
    score_table = pd.DataFrame(index=['Logistic_Regression','Support Vector Machine'], columns=['P&O_train_score','P&O_test_score'])
    train_score = []
    test_score = []

    model_list = [lr_grid, svm_grid]
    for model in model_list:
        model.fit(X_train, y_train)
        train_score.append(model.score(X_train, y_train))
        test_score.append(model.score(X_test, y_test))
        
    score_table['P&O_train_score'] = train_score
    score_table['P&O_test_score'] = test_score 



    return score_table
