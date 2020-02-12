# 랜덤 포레스트 적용
# age 부분 최빈값으로 대체
# fare 부분 이상치 제거 및 최빈값으로 NaN 대체
# deck 부분 컬럼 삭제


import pandas as pd


df = pd.read_csv("d:\\data\\tat\\train.csv")
df2 = pd.read_csv("d:\\data\\tat\\test.csv")

pd.set_option('display.max_columns', 15)

# 
mask1 = (df['Age'] < 10) | (df['Sex'] == 'female')
mask2 = (df2['Age'] < 10) | (df2['Sex'] == 'female')

df['child_women'] = mask1.astype(int)
df2['child_women'] = mask2.astype(int)

print(df.columns)
print(df2.columns)

# =============================================================================
# print(df.info())
# print(df2.info())
# print('\n')
# =============================================================================

print(df.isnull().sum(axis=0), '\n')
print(df2.isnull().sum(axis=0), '\n')

train_test_data = [df, df2] # combining train and test dataset

for dataset in train_test_data:
    dataset['Title'] = dataset['Name'].str.extract(' ([A-Za-z]+)\.', expand=False)



print(df['Title'].value_counts())
print(df2['Title'].value_counts())

title_mapping = {"Mr": 0, "Miss": 1, "Mrs": 2, 
                 "Master": 3, "Dr": 3, "Rev": 3, "Col": 3, "Major": 3, "Mlle": 3,"Countess": 3,
                 "Ms": 3, "Lady": 3, "Jonkheer": 3, "Don": 3, "Dona" : 3, "Mme": 3,"Capt": 3,"Sir": 3 }

for dataset in train_test_data:
    dataset['Title'] = dataset['Title'].map(title_mapping)

print(df.head())
print(df2.head())

df.drop(['Name'], axis=1, inplace=True)
df2.drop(['Name'], axis=1, inplace=True)

print(df.isnull().sum(axis=0), '\n')
print(df2.isnull().sum(axis=0), '\n')

for dataset in train_test_data:
    dataset['Cabin'] = dataset['Cabin'].str[:1]

    
cabin_mapping = {"A": 0, "B": 0.4, "C": 0.8, "D": 1.2, "E": 1.6, "F": 2, "G": 2.4, "T": 2.8}
for dataset in train_test_data:
    dataset['Cabin'] = dataset['Cabin'].map(cabin_mapping)
    
df["Cabin"].fillna(df.groupby("Pclass")["Cabin"].transform("median"), inplace=True)
df2["Cabin"].fillna(df2.groupby("Pclass")["Cabin"].transform("median"), inplace=True)    

# Cabin 칼럼 날려버리기
rdf = df.drop(['Ticket'], axis =1)
rdf2 = df2.drop(['Ticket'], axis =1)

print(rdf.columns.values, '\n')
print(rdf2.columns.values, '\n')


# =============================================================================
# # 나이값 Age컬럼의 결측치를 최빈값으로 대체
# age_freq1 = rdf['Age'].value_counts(dropna=True).idxmax()
# age_freq2 = rdf2['Age'].value_counts(dropna=True).idxmax()
# 
# print(age_freq1) # 24.0
# print(age_freq2) # 24.0
# =============================================================================
age_mean1 = rdf['Age'].median()
age_mean2 = rdf2['Age'].median()

rdf['Age'].fillna(age_mean1, inplace = True)
rdf2['Age'].fillna(age_mean2, inplace = True)

train_test_data2 = [rdf,rdf2]
for dataset in train_test_data2:
    dataset.loc[ dataset['Age'] <= 16, 'Age'] = 0,
    dataset.loc[(dataset['Age'] > 16) & (dataset['Age'] <= 26), 'Age'] = 1,
    dataset.loc[(dataset['Age'] > 26) & (dataset['Age'] <= 36), 'Age'] = 2,
    dataset.loc[(dataset['Age'] > 36) & (dataset['Age'] <= 62), 'Age'] = 3,
    dataset.loc[ dataset['Age'] > 62, 'Age'] = 4
    
# Embarked 결측치 값을 최빈값으로 대체
# test.csv 에는 Embarked에 결측치 값이 없으므로 과정 생략.

print(rdf.isnull().sum(axis=0), '\n')
print(rdf2.isnull().sum(axis=0), '\n')

most_freq1 = rdf['Embarked'].value_counts(dropna=True).idxmax()


# most_freq2 = rdf2['Embarked'].value_counts(dropna=True).idxmax() 을 생략

rdf['Embarked'].fillna(most_freq1, inplace = True)

# rdf2['Embarked'].fillna(most_freq2, inplace = True) 을 생략

print(rdf.isnull().sum(axis=0), '\n')
print(rdf2.isnull().sum(axis=0), '\n')



# ndf : train.csv // ndf2 : test.csv

ndf = rdf[['Survived','Pclass','Sex','Age','SibSp','Parch','Embarked', 'child_women', 'Fare', 'Title', 'Cabin']]
ndf2 = rdf2[['Pclass','Sex','Age','SibSp','Parch','Embarked', 'child_women', 'Fare', 'Title', 'Cabin']]

print(ndf.head(), '\n')
print(ndf2.head(), '\n')

# 더미변수를 이용하여 숫자형 변수로 만들기!
gender = pd.get_dummies(ndf['Sex'])
gender2 = pd.get_dummies(ndf2['Sex'])

ndf = pd.concat([ndf,gender], axis = 1)
ndf2 = pd.concat([ndf2,gender2], axis = 1)

onehot_embarked = pd.get_dummies(ndf['Embarked'], prefix = 'town') # 접두사를 정함.
onehot_embarked2 = pd.get_dummies(ndf2['Embarked'], prefix = 'town') # 접두사를 정함.

ndf = pd.concat([ndf, onehot_embarked], axis = 1)
ndf2 = pd.concat([ndf2, onehot_embarked2], axis = 1)

ndf.drop(['Sex', 'Embarked'], axis = 1 , inplace = True)
ndf2.drop(['Sex', 'Embarked'], axis = 1 , inplace = True)

print(ndf.isnull().sum(axis=0), '\n')
print(ndf2.isnull().sum(axis=0), '\n')


# 운임값(fare)의 이상치를 제거 (훈련데이터만 제거 할 것)
local_std1 = ndf['Fare'].std() * 5

# 이상치 값들이 무엇이 있는지 확인!
result1 = ndf['Fare'][ndf['Fare'] > local_std1]
print(result1)

ndf = ndf[:][ndf['Fare'] < local_std1]


print(ndf.isnull().sum(axis=0), '\n')
print(ndf2.isnull().sum(axis=0), '\n')


# 운임값(fare)의 결측치를 최빈값으로 채움.
fare_freq1 = ndf['Fare'].value_counts(dropna=True).idxmax()
fare_freq2 = ndf2['Fare'].value_counts(dropna=True).idxmax()


ndf['Fare'].fillna(fare_freq1, inplace = True)
ndf2['Fare'].fillna(fare_freq2, inplace = True)

print(ndf.isnull().sum(axis=0), '\n')
print(ndf2.isnull().sum(axis=0), '\n')


print(ndf.head())
print('\n')

print(ndf2.head())
print('\n')

X = ndf[['Pclass','Age','SibSp','Parch','female','male','town_C','town_Q','town_S', 'child_women', 'Fare', 'Title', 'Cabin']] # 독립변수
y = ndf['Survived'] # 종속변수

test = ndf2[['Pclass','Age','SibSp','Parch','female','male','town_C','town_Q','town_S', 'child_women', 'Fare', 'Title', 'Cabin']]

from sklearn import preprocessing

X = preprocessing.StandardScaler().fit(X).transform(X)
test = preprocessing.StandardScaler().fit(test).transform(test)

print(X)
print(test)


# 5단계 . 데이터셋을 훈련데이터와 테스트 데이터로 나눈다.

"""
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3, random_state = 10)

# 설명 : test_size = 0.3 에 의해서 7:3 비율로 훈련과 테스트를 나누고 
# random_state = 10 에 의해서 나중에 split할 때도 항상 일정하게 split 할 수 있게 한다.
"""

# 6단계. 머신러닝 모델을 생성한다.(랜덤 포레스트를 사용)

from sklearn import svm

model=svm.SVC(kernel='rbf',C=1 ,gamma=0.1)
model.fit(X,y)
y_hat=model.predict(X)

# =============================================================================
# # oob 평가
# 
# print(tree_model.oob_score_)
# print('\n')
# 
# =============================================================================



print(y_hat[0:10])
print(y.values[0:10])


# =============================================================================
# # confusion matrix에 대한 변수 지정 (tn,fp,fn,tp)
# from sklearn import metrics
# tn, fp, fn, tp = metrics.confusion_matrix( y, y_hat ).ravel()
# print(tn,fp,fn,tp)
# print('\n')
# 
# =============================================================================

# 9단계 정확도 확인
from sklearn.metrics import accuracy_score
accuracy = accuracy_score(y, y_hat)
print(accuracy)


# submission.csv 을 생성할때 활용 코드


# Prediction
y_pred = model.predict(test)

sample_submission=pd.read_csv('d:\\data\\tat\\gender_submission.csv', index_col=0)


# Submission
submission = pd.DataFrame(data = y_pred, columns = sample_submission.columns, index = sample_submission.index)
submission.to_csv('d:\\data\\tat\\submission_svm04.csv', index=True)

