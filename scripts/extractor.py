import pandas as pd
import numpy as np
from sklearn import neighbors
import matplotlib.pyplot as plt
import seaborn as sn
from sklearn.metrics import ConfusionMatrixDisplay, confusion_matrix
from sklearn.preprocessing import MinMaxScaler, StandardScaler

def replace_string_with_int(x):
    if x == 'rest':
        return 0
    elif x == 'transport':
        return 1
    elif x == 'flying':
        return 2

TRAIN_TEST_SPLIT = 0.8

header = ["accx", "accy", "accz", "gyrx", "gyry", "gyrz", "no"]

def split_dataframe(df, chunk_size = 30): 
    chunks = list()
    num_chunks = len(df) // chunk_size + 1
    for i in range(num_chunks):
        chunks.append(df[i*chunk_size:(i+1)*chunk_size])
    return chunks

'''
def confusion_matrix(results, labels):
    data = {'y_Actual':results[:,0].astype(np.uint8), 'y_Predicted':labels}
    df = pd.DataFrame(data, columns=['y_Actual', 'y_Predicted'])
    cm = pd.crosstab(df['y_Actual'], df['y_Predicted'],
    rownames=['Actual'], colnames=['Predicted'])
    cm = cm.astype(np.float32) / cm.sum(axis=1)[:,np.newaxis]*100.0
    sn.heatmap(cm, annot=True, fmt='.2f', annot_kws={"size":8})
    plt.title('Normalized Confusion Matrix in %')
    plt.show()
'''
df_test = pd.DataFrame(columns=header)
df_train = pd.DataFrame(columns=header)

df = pd.read_csv("../data/flying_all.csv", names=header, skiprows=1000, nrows=30000)
df = df.drop(columns=['no'])
df["label"] = 'flying'
# Split into training and test-set
dfs = np.split(df, [int(len(df)*TRAIN_TEST_SPLIT)], axis=0)


#dfs[0].to_csv("train_flying1.csv")  
#dfs[1].to_csv("test_flying1.csv")
df_train = pd.concat([dfs[0], df_train],axis=0)
df_test = pd.concat([dfs[1], df_test],axis=0)
df = pd.read_csv("../data/measure_transport.csv", names=header, skiprows=1000, nrows=30000)
df = df.drop(columns=["no"])
df["label"] = 'transport'
# Split into training and test-set
dfs = np.split(df, [int(len(df)*TRAIN_TEST_SPLIT)], axis=0)


#dfs[0].to_csv("train_flying1.csv") 
#dfs[1].to_csv("test_flying1.csv")
df_train = pd.concat([dfs[0], df_train],axis=0)
df_test = pd.concat([dfs[1], df_test],axis=0)

df = pd.read_csv("../data/rest_horizontal.csv", names=header, skiprows=1000, nrows=10000)
df = df.drop(columns=["no"])
df["label"] = 'rest'
# Split into training and test-set
dfs = np.split(df, [int(len(df)*TRAIN_TEST_SPLIT)], axis=0)

#dfs[0].to_csv("train_flying1.csv") 
#dfs[1].to_csv("test_flying1.csv")
df_train = pd.concat([dfs[0], df_train],axis=0)
df_test = pd.concat([dfs[1], df_test],axis=0)

df = pd.read_csv("../data/rest_vertical_left.csv", names=header, skiprows=1000, nrows=10000)
df = df.drop(columns=["no"])
df["label"] = 'rest'
# Split into training and test-set
dfs = np.split(df, [int(len(df)*TRAIN_TEST_SPLIT)], axis=0)

#dfs[0].to_csv("train_flying1.csv") 
#dfs[1].to_csv("test_flying1.csv")
df_train = pd.concat([dfs[0], df_train],axis=0)
df_test = pd.concat([dfs[1], df_test],axis=0)

df = pd.read_csv("../data/rest_vertical_right.csv", names=header, skiprows=1000, nrows=10000)
df = df.drop(columns=["no"])
df["label"] = 'rest'
# Split into training and test-set
dfs = np.split(df, [int(len(df)*TRAIN_TEST_SPLIT)], axis=0)

#dfs[0].to_csv("train_flying1.csv")  
#dfs[1].to_csv("test_flying1.csv")
df_train = pd.concat([dfs[0], df_train],axis=0)
df_test = pd.concat([dfs[1], df_test],axis=0)



k = 10

x_train = df_train.iloc[:, 0:-2].to_numpy()
y_train = df_train.iloc[:, -2:-1].to_numpy()

scaler = MinMaxScaler()
x_train = scaler.fit_transform(x_train);
print(x_train)


clf = neighbors.KNeighborsClassifier(n_neighbors = k)
clf.fit(x_train.astype(float), y_train.ravel())


x_test = df_test.iloc[:,:-2].to_numpy()
y_test = df_test.iloc[:,-2:-1].to_numpy()

x_test = scaler.transform(x_test);
print(x_test)

y_predict = clf.predict(x_test.astype(float))





y_test = y_test.ravel().tolist()
y_predict = y_predict.tolist()
y_test = list(map(replace_string_with_int ,y_test))
y_predict = list(map(replace_string_with_int ,y_predict))

accuracy = np.mean(np.array(y_test) == np.array(y_predict))
print("Accuracy:", np.array(accuracy), "\nTrue Class:", np.array(y_test[:]), "\npredicted Class:", np.array(y_predict[:]))

cm = confusion_matrix(y_test, y_predict)
labels = ['rest', 'transport', 'flying']
cm = confusion_matrix(y_test, y_predict)
cmd = ConfusionMatrixDisplay(cm, display_labels=labels)
cmd.plot()
plt.show()

