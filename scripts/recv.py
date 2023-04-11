import serial
import pandas as pd
import joblib
from sklearn import neighbors
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from multiprocessing import Process, Queue

PORT = "COM15"
BAUDRATE = "115200"
SENDCOUNTER = 30
scaler = None
model = None
header = ["accx", "accy", "accz", "gyrx", "gyry", "gyrz","no"]

def replace_int_with_string(x):
    if x == 0:
        return 'rest'
    elif x == 1:
        return 'transport'
    elif x == 2:
        return 'flying'

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
    temp_df = pd.DataFrame()
    mean_df = gen_mean(df)
    std_df = gen_std(df)
    min_df = gen_min(df)
    max_df = gen_max(df)
    temp_df = pd.concat([temp_df, std_df],axis=1)
    temp_df = pd.concat([temp_df, max_df],axis=1)
    temp_df = pd.concat([temp_df, min_df], axis=1)
    temp_df = pd.concat([temp_df, mean_df], axis=1)
    #print(temp_df)
    return temp_df


def do_serial(q):
    dev = serial.Serial()
    dev.baudrate = BAUDRATE
    dev.port = PORT
    dev.open()
    data = []
    datalist = []
    msg = ""
    cnt = 0
    c = dev.read_until(b'\r\n')
    while True:
        c = dev.read_until(b'\r\n')
        
        msg += c[:-2].decode()   
        #print("Lineread: Counter" + str(cnt) + "\n")
        data = msg.split(",")
        data = [float(i) for i in data]
        datalist.append(data)
        msg = ""
        cnt += 1

                

        if cnt == SENDCOUNTER:
            q.put(datalist)
            datalist = []
            cnt = 0
def main(q):

    print("main")



def start():
    q = Queue()
    p = Process(target=do_serial, args=(q,))
    p.start()
    
    while True:
        # read queued data from serial port
        if not q.empty():
            data = q.get()
            df_in = pd.DataFrame(data, columns=header)
            df_in = df_in.iloc[:,:-1]
            df_test = gen_features(df_in);
            x_test = df_test.to_numpy()
            x_test = scaler.transform(x_test);
            prediction = model.predict(x_test)
            print(prediction + "\n")
    p.join()

if __name__ == '__main__':
    scaler = joblib.load('scaler.gz')
    model = joblib.load('model.gz')
    start()






