import pandas as pd
import numpy as np
from sklearn import neighbors
import matplotlib.pyplot as plt
import seaborn as sn
from sklearn.metrics import ConfusionMatrixDisplay, confusion_matrix
from sklearn.preprocessing import MinMaxScaler, StandardScaler
import joblib

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

def gen_mean(df):
        ret_df = pd.DataFrame()
        ret_df = df[["accx", "accy", "accz", "gyrx", "gyry", "gyrz"]].mean(axis=0)
        ret_df = ret_df.to_frame().T
        ret_df.rename(columns={"accx":"accx_mean", 
                                        "accy":"accy_mean", 
                                        "accz":"accz_mean", 
                                        "gyrx":"gyrx_mean", 
                                        "gyry":"gyry_mean", 
                                        "gyrz":"gyrz_mean"},inplace=True)
        return ret_df

def gen_std(df):
        ret_df = pd.DataFrame()
        ret_df = df[["accx", "accy", "accz", "gyrx", "gyry", "gyrz"]].std(axis=0)
        ret_df = ret_df.to_frame().T
        ret_df.rename(columns={"accx":"accx_std", 
                                        "accy":"accy_std", 
                                        "accz":"accz_std", 
                                        "gyrx":"gyrx_std", 
                                        "gyry":"gyry_std", 
                                        "gyrz":"gyrz_std"},inplace=True)
        return ret_df

def gen_max(df):
        ret_df = pd.DataFrame()
        ret_df = df[["accx", "accy", "accz", "gyrx", "gyry", "gyrz"]].max(axis=0)
        ret_df = ret_df.to_frame().T
        ret_df.rename(columns={"accx":"accx_max", 
                                        "accy":"accy_max", 
                                        "accz":"accz_max", 
                                        "gyrx":"gyrx_max", 
                                        "gyry":"gyry_max", 
                                        "gyrz":"gyrz_max"},inplace=True)
        return ret_df

def gen_min(df):
        ret_df = pd.DataFrame()
        ret_df = df[["accx", "accy", "accz", "gyrx", "gyry", "gyrz"]].min(axis=0)
        ret_df = ret_df.to_frame().T
        ret_df.rename(columns={"accx":"accx_min", 
                                        "accy":"accy_min", 
                                        "accz":"accz_min", 
                                        "gyrx":"gyrx_min", 
                                        "gyry":"gyry_min", 
                                        "gyrz":"gyrz_min"},inplace=True)
        return ret_df

def gen_features(df):
    df_list = split_dataframe(df)
    label = df["label"].iloc[0]
    complete_df = pd.DataFrame()
    for i in df_list[:-1]:
        mean_df = gen_mean(i)
        std_df = gen_std(i)
        min_df = gen_min(i)
        max_df = gen_max(i)
        temp_df = pd.concat([std_df, max_df, min_df, mean_df],axis=1)
        complete_df = pd.concat([complete_df, temp_df],axis=0)
    complete_df["label"] = label
    complete_df.reset_index()
    return complete_df

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
df_test = pd.DataFrame()
df_train = pd.DataFrame()

df = pd.read_csv("../data/flying_all.csv", names=header, skiprows=1000, nrows=30000)
df = df.iloc[:,:-1]
df["label"] = 'flying'
df = gen_features(df)

# Split into training and test-set
dfs = np.split(df, [int(len(df)*TRAIN_TEST_SPLIT)], axis=0)


#dfs[0].to_csv("train_flying1.csv")  
#dfs[1].to_csv("test_flying1.csv")
df_train = pd.concat([dfs[0], df_train],axis=0)
df_test = pd.concat([dfs[1], df_test],axis=0)
df = pd.read_csv("../data/measure_transport.csv", names=header, skiprows=1000, nrows=30000)
df = df.iloc[:,:-1]
df["label"] = 'transport'
df = gen_features(df)
# Split into training and test-set
dfs = np.split(df, [int(len(df)*TRAIN_TEST_SPLIT)], axis=0)


#dfs[0].to_csv("train_flying1.csv") 
#dfs[1].to_csv("test_flying1.csv")
df_train = pd.concat([dfs[0], df_train],axis=0)
df_test = pd.concat([dfs[1], df_test],axis=0)

df = pd.read_csv("../data/rest_horizontal.csv", names=header, skiprows=1000, nrows=10000)
df = df.iloc[:,:-1]
df["label"] = 'rest'
df = gen_features(df)
# Split into training and test-set
dfs = np.split(df, [int(len(df)*TRAIN_TEST_SPLIT)], axis=0)

#dfs[0].to_csv("train_flying1.csv") 
#dfs[1].to_csv("test_flying1.csv")
df_train = pd.concat([dfs[0], df_train],axis=0)
df_test = pd.concat([dfs[1], df_test],axis=0)

df = pd.read_csv("../data/rest_vertical_left.csv", names=header, skiprows=1000, nrows=10000)
df = df.iloc[:,:-1]
df["label"] = 'rest'
df = gen_features(df)
# Split into training and test-set
dfs = np.split(df, [int(len(df)*TRAIN_TEST_SPLIT)], axis=0)

#dfs[0].to_csv("train_flying1.csv") 
#dfs[1].to_csv("test_flying1.csv")
df_train = pd.concat([dfs[0], df_train],axis=0)
df_test = pd.concat([dfs[1], df_test],axis=0)

df = pd.read_csv("../data/rest_vertical_right.csv", names=header, skiprows=1000, nrows=10000)
df = df.iloc[:,:-1]
df["label"] = 'rest'
df = gen_features(df)
print(df)
# Split into training and test-set
dfs = np.split(df, [int(len(df)*TRAIN_TEST_SPLIT)], axis=0)

#dfs[0].to_csv("train_flying1.csv")  
#dfs[1].to_csv("test_flying1.csv")
df_train = pd.concat([dfs[0], df_train],axis=0)
df_test = pd.concat([dfs[1], df_test],axis=0)


df_train_split = split_dataframe(df_train, 30)
print(df_train_split[0])
print(df_train_split[1])

k = 10
print(df_train.iloc[:, 0:-1].to_numpy())
x_train = df_train.iloc[:, 0:-1].to_numpy()
y_train = df_train.iloc[:, -1:].to_numpy()

scaler = MinMaxScaler()
x_train = scaler.fit_transform(x_train);
print(x_train)


clf = neighbors.KNeighborsClassifier(n_neighbors = k)
clf.fit(x_train.astype(float), y_train.ravel())


x_test = df_test.iloc[:,:-1].to_numpy()
y_test = df_test.iloc[:,-1:].to_numpy()

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


joblib.dump(scaler, 'scaler.gz')
joblib.dump(clf, "model.gz")

