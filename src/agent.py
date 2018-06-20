from sklearn.ensemble import IsolationForest
import pandas as pd


def MyAgent():
    content = pd.read_csv('../dataset/traindata.csv')
    c = content.iloc[:, list(range(8))]
    clf = IsolationForest()
    clf.fit(c)
    result = clf.predict(c)
    normal = []
    outlier = []
    data = result.tolist()
    for index, each in enumerate(data):
        if each == 1:
            normal.append(c.iloc[index])
        if each == -1:
            outlier.append(c.iloc[index])
        # print(each)

    return normal,outlier


if __name__ == '__main__':
    a,b = MyAgent()
    ex = [81.6328125, 38.82288933, 0.606775649, 1.898487468, 0.89632107, 11.50102307, 15.46231231, 273.1602382]
    # print(type(a))
    for i in a:
        # print(i.tolist())
        if i.tolist() == ex:
            # print(True)
            print()
