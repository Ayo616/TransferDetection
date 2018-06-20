import pandas as pd
from sklearn.utils import shuffle


features = ['f1','f2','f3','f4','f5','f6','f7','f8']
path = '../dataset/data.csv'
content = pd.read_csv(path)
# print(content.head())
print(content['label'].value_counts())

# pick normal data
normal_data = content[content['label'] ==0]
# pick unnormal data
oulier_data = content[content['label'] == 1]

# constract the training data
#  3000(100)
train_normal = normal_data.sample(n = 3000)
train_outlier = oulier_data.sample(n = 100)
traindata = pd.concat([train_normal,train_outlier],axis=0)
traindata = shuffle(traindata).reset_index(drop=True)


# constract the training data
#  1500(10)
test_normal = normal_data.drop_duplicates(traindata).sample(n = 1500)
test_outlier = oulier_data.drop_duplicates(traindata).sample(n = 10)
testdata = pd.concat([normal_data,oulier_data],axis=0)
testdata = shuffle(testdata).reset_index(drop=True)


# save the data file
traindata.to_csv('../dataset/traindata.csv',index=False)
testdata.to_csv('../dataset/testdata.csv',index=False)