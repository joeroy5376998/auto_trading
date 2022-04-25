from locale import normalize


class Trader():

    # constructor
    def __init__(self):
        self.action_list = []
        self.hold = [0]
        self.predict_price = [0]

    def normalize(self,data):
        self.scalar = MinMaxScaler(feature_range=(-1,1))
        data = data['open']
        data = data.values.reshape(-1,1)
        data = self.scalar.fit_transform(data)
        return data

    # train
    def train(self, dataset):
        # normalization
        dataset = self.normalize(dataset)

        # split train and validation dataset
        X = dataset[:-1]
        y = dataset[1:]
        n = len(dataset)
        train_points = int(0.9*n)+1
        X_train, X_val = X[:train_points], X[train_points:]
        y_train, y_val = y[:train_points], y[train_points:]

        X_train_avg = []
        y_train_avg = []

        # moving average
        avg = 3
        for i in range(avg,len(X_train)):
            X_train_avg.append([X_train[i-avg:i].mean()])
            y_train_avg.append([y_train[i-avg:i].mean()])

        # 轉成 numpt array
        X_train = np.array(X_train_avg)
        y_train = np.array(y_train_avg)
        # 轉成輸入 LSTM 三維陣列
        X_train = X_train.reshape(X_train.shape[0], X_train.shape[1], 1)
        X_val = X_val.reshape(X_val.shape[0], X_val.shape[1], 1)

        # model
        epoch = 100
        self.model = Sequential()
        self.model.add(LSTM(32, activation = 'relu', return_sequences = True,input_shape=(1, 1)))
        self.model.add(LSTM(32, activation = 'relu'))
        self.model.add(Dense(1))
        self.model.compile(loss = 'mean_squared_error', optimizer = 'adam')
        self.model.summary()
        callback = EarlyStopping(monitor="val_loss", patience=10, verbose=1, mode="auto")
        self.model.fit(X_train, y_train, epochs=epoch, batch_size=1, callbacks=[callback], validation_data=(X_val, y_val))


    def action(self,tomarrow,today):
        if tomarrow < today: # 預測隔天會跌
            if self.hold[i] == 1: # 若有持股則賣出
                act = -1
                self.hold.append(0)
            elif self.hold[i] == -1: #持續作空
                act = 0 
                self.hold.append(-1)
            else: # 若沒有持股則放空
                act = -1
                self.hold.append(-1)
        else: # 預測隔天會漲
            if self.hold[i] == 0: # 若沒有持股則買進
                act = 1
                self.hold.append(1)
            elif self.hold[i] == 1: # 繼續持有
                act = 0 
                self.hold.append(1)
            else: # 回補空單
                act = 1 
                self.hold.append(0)
        self.action_list.append(act)
        return act

    # predict
    def predict_action(self, data):
        # normalization
        _data = np.array([data])
        _data = _data.reshape(_data.shape[0],_data.shape[1],1)
        value = self.model.predict(_data) # 預測隔天開盤價
        if value >= 0:
            value -= 0.05
        else:
            value += 0.05

        self.predict_price.append(value[0]) # 紀錄預測結果
        # 轉成 list 再回傳以便寫入 csv
        return [self.action(value[0],data[0])]


def load_data(file_name):
    df = pd.read_csv(file_name, header = None)
    col = ['open','high','low','close']
    df.columns = col
    return df

# You can write code above the if-main block.

if __name__ == "__main__":
    # You should not modify this part.
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--training", default="training_data.csv", help="input training data file name")
    parser.add_argument("--testing", default="testing_data.csv", help="input testing data file name")
    parser.add_argument("--output", default="output.csv", help="output file name")
    args = parser.parse_args()

    # The following part is an example.
    # You can modify it at will.
    import pandas as pd
    import numpy as np
    import matplotlib.pyplot as plt
    import csv
    # make result predictable
    from numpy.random import seed
    seed(1)
    import tensorflow as tf
    tf.random.set_seed(1)
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import Dense, LSTM
    from tensorflow.keras.callbacks import EarlyStopping
    from sklearn.preprocessing import MinMaxScaler


    training_data = load_data(args.training)
    trader = Trader()
    trader.train(training_data)

    testing_data = load_data(args.testing)
    testing_data = trader.normalize(testing_data) # normalize
    n = len(testing_data)
    with open(args.output, "w", newline = '') as output_file:
        writer = csv.writer(output_file)
        for i in range(n-1):
            # We will perform your action as the open price in the next day.
            action = trader.predict_action(testing_data[i]) # 參數為當天的開高低收
            writer.writerow(action)

            # this is your option, you can leave it empty.
            # trader.re_training()
