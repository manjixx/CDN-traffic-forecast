import tensorflow as tf
import os
import pandas
import csv
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
os.environ['CUDA_VISIBLE_DEVICES'] = "1"
import numpy as np


class MNISTLoader():
    def __init__(self):
        table = dict()
        tt = pandas.read_csv(r'data/test_vote.csv')
        for t in range(tt.shape[0]):
            table[tt.iloc[t, 0]] = 1
        data_name_mean = np.load("data/bertmean_try.npy")
        data_name_cls = np.load("data/bertcls_try.npy")
        data_all = np.load("data/bertall_try.npy")
        label_file=pandas.read_csv(r'data/name_input_0.csv')
        train_data=[]
        test_data=[]
        train_label=[]
        test_label=[]
        test_name=[]
        test_label_id=[]
        test_table = dict()
        for i in range(len(data_name_mean)):
            temp=[0]*70
            temp[label_file.iloc[i,3]]=1
            data=[]
            data.append(data_name_mean[i])
            data.append(data_name_mean[i])
            data.append(data_name_cls[i])
            for j in range(5):
                data.append(data_all[i * 5 + j])
            if i not in table:
                train_data.append(data)
                train_label.append(temp)
            else:
                test_data.append(data)
                test_label.append(temp)
                test_name.append(label_file.iloc[i, 1])
                test_label_id.append(label_file.iloc[i, 3])
                if label_file.iloc[i, 3] not in test_table:
                    test_table[label_file.iloc[i, 3]] = 1
                else:
                    test_table[label_file.iloc[i, 3]] += 1
        self.train_data = np.array(train_data)
        self.test_data = np.array(test_data)
        self.train_label=np.array(train_label)
        self.test_label =np.array(test_label)
        self.test_name = test_name
        self.test_label_id=test_label_id
        self.test_table=test_table
        self.num_train_data, self.num_test_data = self.train_data.shape[0], self.test_data.shape[0]

    def get_batch(self, batch_size):
        index = np.random.randint(0, self.num_train_data, batch_size)
        return self.train_data[index, :], self.train_label[index]

class S_ATT(tf.keras.Model):
    def __init__(self):
        super().__init__()
        self.dense_squeeze1 = tf.keras.layers.Dense(units=256, activation=tf.nn.leaky_relu)
        self.dense_squeeze2 = tf.keras.layers.Dense(units=256, activation=tf.nn.leaky_relu)
        self.dense_squeeze3 = tf.keras.layers.Dense(units=128, activation=tf.nn.leaky_relu)
        self.dense_squeeze4 = tf.keras.layers.Dense(units=128, activation=tf.nn.leaky_relu)
        self.dense_squeeze5 = tf.keras.layers.Dense(units=128, activation=tf.nn.leaky_relu)
        self.dense_squeeze6 = tf.keras.layers.Dense(units=128, activation=tf.nn.leaky_relu)
        self.dense_squeeze7 = tf.keras.layers.Dense(units=256, activation=tf.nn.leaky_relu)
        self.dense_squeeze8 = tf.keras.layers.Dense(units=128, activation=tf.nn.leaky_relu)

        self.dense_query1 = tf.keras.layers.Dense(units=1024, activation=tf.nn.leaky_relu)
        self.dense_query2 = tf.keras.layers.Dense(units=1024, activation=tf.nn.leaky_relu)
        self.dense_query3 = tf.keras.layers.Dense(units=1024, activation=tf.nn.leaky_relu)
        self.dense_query4 = tf.keras.layers.Dense(units=1024, activation=tf.nn.leaky_relu)
        self.dense_query5 = tf.keras.layers.Dense(units=1024, activation=tf.nn.leaky_relu)
        self.dense_query6 = tf.keras.layers.Dense(units=1024, activation=tf.nn.leaky_relu)
        self.dense_query7 = tf.keras.layers.Dense(units=1024, activation=tf.nn.leaky_relu)
        self.dense_query8 = tf.keras.layers.Dense(units=1024, activation=tf.nn.leaky_relu)

        self.lstm = tf.keras.layers.LSTM(units=1024, activation=tf.nn.leaky_relu,return_sequences=True)

        self.dense1 = tf.keras.layers.Dense(units=4096, activation=tf.nn.leaky_relu)
        self.dense2 = tf.keras.layers.Dense(units=2048, activation=tf.nn.leaky_relu)
        self.dense3 = tf.keras.layers.Dense(units=1024, activation=tf.nn.leaky_relu)
        self.dense4 = tf.keras.layers.Dense(units=512, activation=tf.nn.leaky_relu)
        self.dense5 = tf.keras.layers.Dense(units=256, activation=tf.nn.leaky_relu)
        self.dense6 = tf.keras.layers.Dense(units=128, activation=tf.nn.leaky_relu)
        self.dense7 = tf.keras.layers.Dense(units=70, activation=tf.nn.leaky_relu)

    def ATT_train(self, inputs):
        x1 = tf.reshape(inputs[:, 0, :], [-1, 1 * 768])
        x2 = tf.reshape(inputs[:, 1, :], [-1, 1 * 768])
        x3 = tf.reshape(inputs[:, 2, :], [-1, 1 * 768])
        x4 = tf.reshape(inputs[:, 3, :], [-1, 1 * 768])
        x5 = tf.reshape(inputs[:, 4, :], [-1, 1 * 768])
        x6 = tf.reshape(inputs[:, 5, :], [-1, 1 * 768])
        x7 = tf.reshape(inputs[:, 6, :], [-1, 1 * 768])
        x8 = tf.reshape(inputs[:, 7, :], [-1, 1 * 768])
        x1 = tf.nn.dropout(x1, 0.5)
        x1 = self.dense_squeeze1(x1)
        x2 = tf.nn.dropout(x2, 0.5)
        x2 = self.dense_squeeze2(x2)
        x3 = tf.nn.dropout(x3, 0.5)
        x3 = self.dense_squeeze3(x3)
        x4 = tf.nn.dropout(x4, 0.5)
        x4 = self.dense_squeeze4(x4)
        x5 = tf.nn.dropout(x5, 0.5)
        x5 = self.dense_squeeze5(x5)
        x6 = tf.nn.dropout(x6, 0.5)
        x6 = self.dense_squeeze6(x6)
        x7 = tf.nn.dropout(x7, 0.5)
        x7 = self.dense_squeeze7(x7)
        x8 = tf.nn.dropout(x8, 0.5)
        x8 = self.dense_squeeze8(x8)
        x = tf.concat([x1, x2, x3, x4, x5, x6, x7, x8], 1)

        q1 = self.dense_query1(x)
        q2 = self.dense_query2(x)
        q3 = self.dense_query3(x)
        q4 = self.dense_query4(x)
        q5 = self.dense_query5(x)
        q6 = self.dense_query6(x)
        q7 = self.dense_query7(x)
        q8 = self.dense_query8(x)
        query = tf.stack([q1, q2, q3, q4, q5, q6, q7, q8], 1)
        print(query.shape)
        lstm=self.lstm(query)
        dense=tf.reshape(lstm,[-1,8*1024])

        dense = tf.nn.dropout(dense, 0.5)
        dense=self.dense1(dense)
        dense = tf.nn.dropout(dense, 0.5)
        dense=self.dense2(dense)
        dense = tf.nn.dropout(dense, 0.5)
        dense = self.dense3(dense)
        dense = tf.nn.dropout(dense, 0.5)
        dense = self.dense4(dense)
        dense = tf.nn.dropout(dense, 0.5)
        dense = self.dense5(dense)
        dense = tf.nn.dropout(dense, 0.5)
        dense = self.dense6(dense)
        dense = tf.nn.dropout(dense, 0.5)
        dense = self.dense7(dense)
        output = tf.nn.softmax(dense)
        return output

    def ATT_test(self, inputs):
        x1 = tf.reshape(inputs[:, 0, :], [-1, 1 * 768])
        x2 = tf.reshape(inputs[:, 1, :], [-1, 1 * 768])
        x3 = tf.reshape(inputs[:, 2, :], [-1, 1 * 768])
        x4 = tf.reshape(inputs[:, 3, :], [-1, 1 * 768])
        x5 = tf.reshape(inputs[:, 4, :], [-1, 1 * 768])
        x6 = tf.reshape(inputs[:, 5, :], [-1, 1 * 768])
        x7 = tf.reshape(inputs[:, 6, :], [-1, 1 * 768])
        x8 = tf.reshape(inputs[:, 7, :], [-1, 1 * 768])
        x1 = self.dense_squeeze1(x1)
        x2 = self.dense_squeeze2(x2)
        x3 = self.dense_squeeze3(x3)
        x4 = self.dense_squeeze4(x4)
        x5 = self.dense_squeeze5(x5)
        x6 = self.dense_squeeze6(x6)
        x7 = self.dense_squeeze7(x7)
        x8 = self.dense_squeeze8(x8)
        x = tf.concat([x1, x2, x3, x4, x5, x6, x7, x8], 1)

        q1 = self.dense_query1(x)
        q2 = self.dense_query2(x)
        q3 = self.dense_query3(x)
        q4 = self.dense_query4(x)
        q5 = self.dense_query5(x)
        q6 = self.dense_query6(x)
        q7 = self.dense_query7(x)
        q8 = self.dense_query8(x)
        query = tf.stack([q1, q2, q3, q4, q5, q6, q7, q8], 1)
        lstm = self.lstm(query)

        dense = tf.reshape(lstm, [-1, 8 * 1024])
        dense = self.dense1(dense)
        dense = self.dense2(dense)
        dense = self.dense3(dense)
        dense = self.dense4(dense)
        dense = self.dense5(dense)
        dense = self.dense6(dense)
        dense = self.dense7(dense)
        output = tf.nn.softmax(dense)
        return output

def train():
    optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate,decay=0.00035)
    checkpoint = tf.train.Checkpoint(classifier=model)
    num_batches = int(data_loader.num_train_data // batch_size * num_epochs)
    for batch_index in range(num_batches):
        X, y = data_loader.get_batch(batch_size)
        with tf.GradientTape() as tape:
            y_pred = model.ATT_train(X)
            loss = tf.keras.losses.categorical_crossentropy(y_true=y, y_pred=y_pred)
            loss = tf.reduce_mean(loss)
            if batch_index % 200 ==0:
                print("batch %d: loss %f" % (batch_index, loss.numpy()))
        grads = tape.gradient(loss, model.variables)
        optimizer.apply_gradients(grads_and_vars=zip(grads, model.variables))
    path = checkpoint.save('save/model_classifier.ckpt')
    print("model saved to %s" % path)

def test():
    lstm_pro=[]
    checkpoint = tf.train.Checkpoint(classifier=model)
    checkpoint.restore('save/model_classifier.ckpt-1')
    num_batches = int(data_loader.num_test_data // batch_size)
    predict_table=dict()
    test_table=data_loader.test_table
    print(data_loader.num_test_data)
    for batch_index in range(num_batches+1):
        if batch_index<num_batches:
            start_index, end_index = batch_index * batch_size, (batch_index + 1) * batch_size
            output= model.ATT_test(data_loader.test_data[start_index: end_index])
        else:
            start_index, end_index = batch_index * batch_size, data_loader.num_test_data
            output = model.ATT_test(data_loader.test_data[end_index-batch_size: end_index])[batch_index * batch_size - end_index: ]
        name=data_loader.test_name[start_index: end_index]
        y_pred=tf.argmax(output,1)
        output=np.array(output)
        for index,label in enumerate(data_loader.test_label[start_index: end_index]):
            lstm_pro.append(output[index])
            pred = int(y_pred[index])
            # with open(r'data/result_test_mlp.csv', 'a', encoding='utf-8') as fs:
            #     datalist = [name[index],pred]
            #     csv_write = csv.writer(fs)
            #     csv_write.writerow(datalist)
            if pred not in predict_table:
                predict_table[pred] = [1, 0]
            else:
                predict_table[pred][0] += 1
            if label[y_pred[index]]==1:
                predict_table[pred][1] += 1
            else:
                pass
    accuracy=0
    for t in predict_table:
        accuracy += test_table[t] / data_loader.num_test_data * (predict_table[t][1] / predict_table[t][0])
    print("test accuracy: %f" % accuracy)
    np.save('data/lstm_pro.npy', lstm_pro)


if __name__ =='__main__':
    num_epochs = 195
    batch_size = 768
    learning_rate = 0.001
    model = S_ATT()
    data_loader = MNISTLoader()
    train()
    test()
