import time
from datetime import datetime

import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.cluster import MiniBatchKMeans
from sklearn.model_selection import train_test_split


# from google.cloud import storage
# bucket = storage.Client().bucket('nyc_taxi_data_zsc')
# for i in ['train.csv','test.csv','weather.csv','NYC_2016Holidays.csv','fastest_routes_train_part_1.csv','fastest_routes_train_part_2.csv','fastest_routes_test.csv']:
#     blob = bucket.blob(i)
#     blob.download_to_filename(i)


# compute haversine distance
def haversine_array(lat1, lng1, lat2, lng2):
    lat1, lng1, lat2, lng2 = map(np.radians, (lat1, lng1, lat2, lng2))
    AVG_EARTH_RADIUS = 6371  # in km
    lat = lat2 - lat1
    lng = lng2 - lng1
    d = np.sin(lat * 0.5) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(lng * 0.5) ** 2
    h = 2 * AVG_EARTH_RADIUS * np.arcsin(np.sqrt(d))
    return h


# compute manhattan distance
def dummy_manhattan_distance(lat1, lng1, lat2, lng2):
    a = haversine_array(lat1, lng1, lat1, lng2)
    b = haversine_array(lat1, lng1, lat2, lng1)
    return a + b


# compute directions
def bearing_array(lat1, lng1, lat2, lng2):
    AVG_EARTH_RADIUS = 6371  # in km
    lng_delta_rad = np.radians(lng2 - lng1)
    lat1, lng1, lat2, lng2 = map(np.radians, (lat1, lng1, lat2, lng2))
    y = np.sin(lng_delta_rad) * np.cos(lat2)
    x = np.cos(lat1) * np.sin(lat2) - np.sin(lat1) * np.cos(lat2) * np.cos(lng_delta_rad)
    return np.degrees(np.arctan2(y, x))


# compute if it is weekend today
def is_weekend(d):
    return int(d.isoweekday() in (6, 7))


# compute if it is a rest day today(weekend or public holiday)
def is_rest(d):
    return int(any([d.isoweekday() in (6, 7), d in holidays]))


# ----BEGIN---- reading data ----BEGIN----
# original training data
train = pd.read_csv('train.csv', parse_dates=['pickup_datetime'])
# original test data
test = pd.read_csv('test.csv', parse_dates=['pickup_datetime'])

# reading weather data
weather = pd.read_csv('weather.csv', parse_dates=['Time'])
# reading holiday data
holiday = pd.read_csv('NYC_2016Holidays.csv', sep=';')

# additional OSRM training data
osrm_1 = pd.read_csv('fastest_routes_train_part_1.csv', usecols=['id', 'total_distance', 'total_travel_time', 'number_of_steps'])
osrm_2 = pd.read_csv('fastest_routes_train_part_2.csv', usecols=['id', 'total_distance', 'total_travel_time', 'number_of_steps'])
# additional OSRM test data
osrm_test = pd.read_csv('fastest_routes_test.csv', usecols=['id', 'total_distance', 'total_travel_time', 'number_of_steps'])

# combine OSRM data since they are too large
osrm_train = pd.concat((osrm_1, osrm_2))
# merge train and test data respectively
train = train.merge(osrm_train, how='left', on='id')
test = test.merge(osrm_test, how='left', on='id')
# ----END---- reading data ----END----


# ----BEGIN---- decompose date ----BEGIN----
for df in [train, test]:
    df['year'] = df['pickup_datetime'].dt.year
    df['month'] = df['pickup_datetime'].dt.month
    df['day'] = df['pickup_datetime'].dt.day
    df['hr'] = df['pickup_datetime'].dt.hour
    df['minute'] = df['pickup_datetime'].dt.minute
    df['dayofweek'] = df['pickup_datetime'].dt.dayofweek
    df['store_and_fwd_flag'] = 1 * (df.store_and_fwd_flag.values == 'Y')
    df['time'] = df['hr'] + df['minute'] / 60
# ----END---- decompose date ----END----


# ----BEGIN---- weather data analyzing ----BEGIN----
# extract weather of 2016
weather = weather.loc[weather['Time'].dt.year == 2016]

# note snow day
weather['snow'] = (weather['Events'] == 'Snow').map(int) + (weather['Events'] == 'Fog\n\t,\nSnow').map(int)
weather['year'] = weather['Time'].dt.year
weather['month'] = weather['Time'].dt.month
weather['day'] = weather['Time'].dt.day
weather['hr'] = weather['Time'].dt.hour

weather = weather[['month', 'day', 'hr', 'Temp.', 'Precip', 'snow', 'Visibility']]
train = pd.merge(train, weather, on=['month', 'day', 'hr'], how='left')
test = pd.merge(test, weather, on=['month', 'day', 'hr'], how='left')
# ----END---- weather data analyzing ----END----


# ----BEGIN---- holiday data ----BEGIN----
# add year to holiday data
holidays = set(holiday['Date'].map(lambda x: datetime.strptime(x + ' 2016', '%B %d %Y').date()).to_list())

# if day is weekend or rest day
test['is_weekend'] = test['pickup_datetime'].dt.date.map(is_weekend)
test['is_rest'] = test['pickup_datetime'].dt.date.map(is_rest)
train['is_weekend'] = train['pickup_datetime'].dt.date.map(is_weekend)
train['is_rest'] = train['pickup_datetime'].dt.date.map(is_rest)
# ----END---- holiday data ----END----


# ----BEGIN---- handle error data ----BEGIN----
mean = np.mean(train['trip_duration'])
std = np.std(train['trip_duration'])
train = train[train['trip_duration'] <= mean + 2 * std]
train = train[train['trip_duration'] >= 60]

# New York City range
city_long_border = (-74.05, -73.70)
city_lat_border = (40.54, 40.92)

# get rid of locations out of the range of New York
train = train[train['pickup_longitude'] <= city_long_border[1]]
train = train[train['pickup_longitude'] >= city_long_border[0]]
train = train[train['pickup_latitude'] <= city_lat_border[1]]
train = train[train['pickup_latitude'] >= city_lat_border[0]]
train = train[train['dropoff_longitude'] <= city_long_border[1]]
train = train[train['dropoff_longitude'] >= city_long_border[0]]
train = train[train['dropoff_latitude'] <= city_lat_border[1]]
train = train[train['dropoff_latitude'] >= city_lat_border[0]]
# ----END---- handle error data ----END----


# ----BEGIN---- k-means ----BEGIN----
# combine pickup and dropoff data 
coords = np.vstack((train[['pickup_latitude', 'pickup_longitude']].values,
                    train[['dropoff_latitude', 'dropoff_longitude']].values))
# train k-mean
print(time.strftime('%H:%M:%S', time.localtime(time.time())))
print('training k-mean ... ...')
kmeans = MiniBatchKMeans(n_clusters=100, batch_size=100000).fit(coords)
# kmeans = KMeans(n_clusters=100).fit(coords)
print(time.strftime('%H:%M:%S', time.localtime(time.time())))
# apply k-means to training data
train.loc[:, 'pickup_cluster'] = kmeans.predict(train[['pickup_latitude', 'pickup_longitude']])
train.loc[:, 'dropoff_cluster'] = kmeans.predict(train[['dropoff_latitude', 'dropoff_longitude']])

# apply k-means to test data
test.loc[:, 'pickup_cluster'] = kmeans.predict(test[['pickup_latitude', 'pickup_longitude']])
test.loc[:, 'dropoff_cluster'] = kmeans.predict(test[['dropoff_latitude', 'dropoff_longitude']])
# ----END---- k-means ----END----


# compute straight line
train.loc[:, 'distance_haversine'] = haversine_array(train['pickup_latitude'].values, train['pickup_longitude'].values,
                                                     train['dropoff_latitude'].values, train['dropoff_longitude'].values)
test.loc[:, 'distance_haversine'] = haversine_array(test['pickup_latitude'].values, test['pickup_longitude'].values,
                                                    test['dropoff_latitude'].values, test['dropoff_longitude'].values)

# compute manhattan distance
train.loc[:, 'distance_dummy_manhattan'] = dummy_manhattan_distance(train['pickup_latitude'].values,
                                                                    train['pickup_longitude'].values,
                                                                    train['dropoff_latitude'].values,
                                                                    train['dropoff_longitude'].values)
test.loc[:, 'distance_dummy_manhattan'] = dummy_manhattan_distance(test['pickup_latitude'].values,
                                                                   test['pickup_longitude'].values,
                                                                   test['dropoff_latitude'].values,
                                                                   test['dropoff_longitude'].values)
# compute directions
train.loc[:, 'direction'] = bearing_array(train['pickup_latitude'].values, train['pickup_longitude'].values,
                                          train['dropoff_latitude'].values, train['dropoff_longitude'].values)
test.loc[:, 'direction'] = bearing_array(test['pickup_latitude'].values, test['pickup_longitude'].values,
                                         test['dropoff_latitude'].values, test['dropoff_longitude'].values)

# ----BEGIN---- one-hot areas ----BEGIN----
# a = pd.Series(range(9))

cluster_pickup_train = pd.get_dummies(train['pickup_cluster'], prefix='pick', prefix_sep='_')
cluster_dropoff_train = pd.get_dummies(train['dropoff_cluster'], prefix='drop', prefix_sep='_')
# passenger_count_train = pd.get_dummies(pd.concat([train['passenger_count'],a],ignore_index=True), prefix='pc', prefix_sep='_')[:len(train['passenger_count'])]
vendor_train = pd.get_dummies(train['vendor_id'], prefix='vi', prefix_sep='_')
store_and_fwd_flag_train = pd.get_dummies(train['store_and_fwd_flag'], prefix='sf', prefix_sep='_')

cluster_pickup_test = pd.get_dummies(test['pickup_cluster'], prefix='pick', prefix_sep='_')
cluster_dropoff_test = pd.get_dummies(test['dropoff_cluster'], prefix='drop', prefix_sep='_')
# passenger_count_test = pd.get_dummies(pd.concat([test['passenger_count'],a],ignore_index=True), prefix='pc', prefix_sep='_')[:len(test['passenger_count'])]
vendor_test = pd.get_dummies(test['vendor_id'], prefix='vi', prefix_sep='_')
store_and_fwd_flag_test = pd.get_dummies(test['store_and_fwd_flag'], prefix='sf', prefix_sep='_')

train = pd.concat([train,
                   cluster_pickup_train,
                   cluster_dropoff_train,
                   vendor_train,
                   store_and_fwd_flag_train
                   ], axis=1)
test = pd.concat([test,
                  cluster_pickup_test,
                  cluster_dropoff_test,
                  vendor_test,
                  store_and_fwd_flag_test
                  ], axis=1)
# ----END---- one-hot areas ----END----


# transfer trip duration into log representation
Z = np.log(train['trip_duration'] + 1)

train_X = train.drop(
    ['id', 'pickup_datetime', 'dropoff_datetime', 'trip_duration', 'pickup_cluster', 'dropoff_cluster', 'year', 'month', 'day',
     'hr', 'minute', 'vendor_id', 'store_and_fwd_flag'],
    axis=1)
Test_id = test['id']
test_X = test.drop(
    ['id', 'pickup_datetime', 'pickup_cluster', 'dropoff_cluster', 'year', 'month', 'day', 'hr', 'minute', 'vendor_id',
     'store_and_fwd_flag'], axis=1)

# ----BEGIN---- train the model ----BEGIN----
# split train data into 80/20 to avoid over-fitting
Xtrain, Xval, Ztrain, Zval = train_test_split(train_X, Z, test_size=0.2, random_state=0)
data_tr = xgb.DMatrix(Xtrain, label=Ztrain)
data_test = xgb.DMatrix(test_X)
data_val = xgb.DMatrix(Xval, label=Zval)
evallist = [(data_tr, 'train'), (data_val, 'valid')]

# original parameters for xgboost model
# if running BO, this para should be commented
# start comment
parms = {'max_depth': 20,  # maximum depth of a tree
         'objective': 'reg:squarederror',
         'eta': 0.15,
         'subsample': 0.85,  # SGD will use this percentage of data
         'lambda ': 3,  # L2 regularization term,>1 more conservative
         'colsample_bytree ': 0.9,
         'colsample_bylevel': 1,
         'min_child_weight': 15
         }

print(time.strftime('%H:%M:%S', time.localtime(time.time())))
print('training ... ...')
model = xgb.train(parms, data_tr, num_boost_round=10, evals=evallist,
                  early_stopping_rounds=2, maximize=False,
                  verbose_eval=10)
print(time.strftime('%H:%M:%S', time.localtime(time.time())))
# end commented code
# ----END--- training model ----END----


# ----BEGIN---- find optimal parameters ----BEGIN----
# def xgb_bo(max_depth, eta, subsample, lam, min_child_weight):
#     parms = {'max_depth': int(max_depth),  # maximum depth of a tree
#              'objective': 'reg:squarederror',
#              'eta': eta,
#              'subsample': subsample,  # SGD will use this percentage of data
#              'lambda ': lam,  # L2 regularization term,>1 more conservative
#              'colsample_bytree ': 0.9,
#              'colsample_bylevel': 1,
#              'min_child_weight': int(min_child_weight)
#              }
#
#     model = xgb.train(parms, data_tr, num_boost_round=50, evals=evallist,
#                       early_stopping_rounds=10, maximize=False,
#                       verbose_eval=False)
#     score = model.best_score
#     return -score
#
# bo = BayesianOptimization(xgb_bo,
#                           {
#                               "max_depth": (6, 15),
#                               "eta": (0.05, 0.15),
#                               "subsample": (0.65, 0.85),
#                               "lam": (0.5, 3),
#                               "min_child_weight": (10, 100)
#                           })
# num_iter = 30
# init_points = 50
# bo.maximize(init_points=init_points, n_iter=num_iter)
# ----END---- find optimal parameters ----END----


# ----BEGIN---- predict the results ----BEGIN----
# predict the results and output it into submission files in 
# order to submit it to Kaggle
sub_name = 'submission.csv'
pred = model.predict(data_test)
pred = np.exp(pred) - 1
submission = pd.concat([Test_id, pd.DataFrame(pred)], axis=1)
submission.columns = ['id', 'trip_duration']
submission['trip_duration'] = submission.apply(lambda x: 1 if (x['trip_duration'] <= 0) else x['trip_duration'], axis=1)
submission.to_csv(sub_name, index=False)
# ----END---- predict the results ----END----
