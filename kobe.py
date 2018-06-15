import numpy as np
import pandas as pd
import scipy as sp
import matplotlib.pyplot as plt
import matplotlib.cm as cm

from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import confusion_matrix
import time

from sklearn.model_selection import KFold
# in 0.20. Use sklearn.model_selection.KFold instead.

# Read data and a glimpse
file_name = "./input/data.csv"
raw = pd.read_csv(file_name)

raw.head()

# Dropping nans and plot
nona = raw[pd.notnull(raw['shot_made_flag'])]

alpha = 0.2
plt.figure(figsize=(10, 10))

# location_x, location_y
plt.subplot(121)
plt.scatter(nona.loc_x, nona.loc_y, color='blue', alpha=alpha)
plt.title("loc_x and loc_y")

# latitude and longitude
plt.subplot(122)
plt.scatter(nona.lon, nona.lat, color='green', alpha=alpha)
plt.title('lat and lon')

# plt.show()

# 把X-Y坐标系转换成极坐标polar coordinate
# 多加一列数据dist,记录离框的距离
raw['dist'] = np.sqrt(raw['loc_x'] ** 2 + raw['loc_y'] ** 2)

# 多加一列数据angle,记录角度
loc_x_zero = raw['loc_x'] == 0
raw['angle'] = np.array([0] * len(raw))
raw['angle'][~loc_x_zero] = np.arctan(raw['loc_y'][~loc_x_zero] / raw['loc_x'][~loc_x_zero])
# Since some of loc_x values cause an error by zero-division,
# we set just np.pi / 2 to the corresponding rows.
raw['angle'][loc_x_zero] = np.pi / 2

# minutes_remaining and seconds_remaining
raw['remaining_time'] = raw['minutes_remaining'] * 60 + raw['seconds_remaining']

# action_type, combined_shot_type, shot_type
print(nona.action_type.unique())
print(nona.combined_shot_type.unique())
print(nona.shot_type.unique())

# Season
nona['season'].unique()

raw['season'] = raw['season'].apply(lambda x: int(x.split('-')[0]))
raw['season'].unique()

# opponent, matchup
pd.DataFrame({'matchup': nona.matchup, 'opponent': nona.opponent})

# Shot distance
plt.figure(figsize=(5, 5))
plt.scatter(raw.dist, raw.shot_distance, color='blue')
plt.title('dist and short_distance')

# shot_zone_area, shot_zone_basic, shot_zone_range
plt.figure(figsize=(20, 10))


def scatter_plot_by_category(feat):
    alpha = 0.1
    gs = nona.groupby(feat)
    cs = cm.rainbow(np.linspace(0, 1, len(gs)))
    for g, c in zip(gs, cs):
        plt.scatter(g[1].loc_x, g[1].loc_y, color=c, alpha=alpha)


# shot_zone_area
plt.subplot(131)
scatter_plot_by_category('shot_zone_area')
plt.title('shot_zone_area')

# shot_zone_basic
plt.subplot(132)
scatter_plot_by_category('shot_zone_basic')
plt.title('shot_zone_basic')

# shot_zone_range
plt.subplot(133)
scatter_plot_by_category('shot_zone_range')
plt.title('shot_zone_range')

# plt.show()

drops = ['shot_id', 'team_id', 'team_name', 'shot_zone_area', 'shot_zone_range', 'shot_zone_basic',
         'matchup', 'lon', 'lat', 'seconds_remaining', 'minutes_remaining',
         'shot_distance', 'loc_x', 'loc_y', 'game_event_id', 'game_id', 'game_date']
for drop in drops:
    raw = raw.drop(drop, 1)

# turn categorical variables into dummy variables
categorical_vars = ['action_type', 'combined_shot_type', 'shot_type', 'opponent', 'period', 'season']
for var in categorical_vars:
    raw = pd.concat([raw, pd.get_dummies(raw[var], prefix=var)], 1)
    raw = raw.drop(var, 1)

# separating data for training and submission
df = raw[pd.notnull(raw['shot_made_flag'])]
submission = raw[pd.isnull(raw['shot_made_flag'])]
submission = submission.drop('shot_made_flag', 1)

# separate df into explanatory and response variables
train = df.drop('shot_made_flag', 1)
train_y = df['shot_made_flag']


# log-loss
def logloss(act, pred):
    epsilon = 1e-15
    pred = sp.maximum(epsilon, pred)
    pred = sp.minimum(1 - epsilon, pred)
    ll = sum(act * sp.log(pred) + sp.subtract(1, act) * sp.log(sp.subtract(1, pred)))
    ll = ll * -1.0 / len(act)
    return ll


# find the best n_estimators for RandomForestClassifier
print('Finding best n_estimators for RandomForestClassifier...')
min_score = 100000
best_n = 0
scores_n = []
range_n = np.logspace(0, 2, num=3).astype(int)
for n in range_n:
    print("the number of trees : {0}".format(n))
    t1 = time.time()

    rfc_score = 0.
    rfc = RandomForestClassifier(n_estimators=n)
    for train_k, test_k in KFold(len(train), n_folds=10, shuffle=True):
        rfc.fit(train.iloc[train_k], train_y.iloc[train_k])
        # rfc_score += rfc.score(train.iloc[test_k], train_y.iloc[test_k])/10
        pred = rfc.predict(train.iloc[test_k])
        rfc_score += logloss(train_y.iloc[test_k], pred) / 10
    scores_n.append(rfc_score)
    if rfc_score < min_score:
        min_score = rfc_score
        best_n = n

    t2 = time.time()
    print('Done processing {0} trees ({1:.3f}sec)'.format(n, t2 - t1))
print(best_n, min_score)

# find best max_depth for RandomForestClassifier
# n_estimators:显然这个参数使随机森林的重要参数之一,它表示的是森林里树的个数,理论上越大越好
# max_features：随机选择特征集合的子集和的个数，如果个数越少，方差就会减少得越快。
# 通常情况我们的max_features取默认值，但是如果样本的特征数非常多，我们就让max_featrues=sqrt(n_features)。

print('Finding best max_depth for RandomForestClassifier...')
min_score = 100000
best_m = 0
scores_m = []
# astype实现类型转换
range_m = np.logspace(0, 2, num=3).astype(int)
for m in range_m:
    print("the max depth : {0}".format(m))
    t1 = time.time()

    rfc_score = 0.
    rfc = RandomForestClassifier(max_depth=m, n_estimators=best_n)
    for train_k, test_k in KFold(len(train), n_folds=10, shuffle=True):
        rfc.fit(train.iloc[train_k], train_y.iloc[train_k])
        # rfc_score += rfc.score(train.iloc[test_k], train_y.iloc[test_k])/10
        pred = rfc.predict(train.iloc[test_k])
        rfc_score += logloss(train_y.iloc[test_k], pred) / 10
    scores_m.append(rfc_score)
    if rfc_score < min_score:
        min_score = rfc_score
        best_m = m

    t2 = time.time()
    print('Done processing {0} trees ({1:.3f}sec)'.format(m, t2 - t1))
print(best_m, min_score)

plt.figure(figsize=(10, 5))
plt.subplot(121)
plt.plot(range_n, scores_n)
plt.ylabel('score')
plt.xlabel('number of trees')

plt.subplot(122)
plt.plot(range_m, scores_m)
plt.ylabel('score')
plt.xlabel('max depth')

plt.show()

print("Making prediction...")
model = RandomForestClassifier(n_estimators=best_n, max_depth=best_m)
model.fit(train, train_y)
pred = model.predict_proba(submission)

print("Write to file")
sub = pd.read_csv("./input/sample_submission.csv")
sub['shot_made_flag'] = pred
sub.to_csv("./output/real_submission.csv", index=False)
print("Training is done")
