#coding=utf-8

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np 
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
import time
import random
import math

#双层LSTM网络 流量非线性映射关系  需要较高时间粒度的数据 目前小时单位数据无法准确描述  

#读取数据
def data_Upsample():
    data_path="D:/desktop/pywork/LsTM/Dataset.xlsx"
    data=pd.read_excel(data_path,index_col="Time")
    #print(data.info())
    #print(data.shape[0])
    #重新设置时间index 原始数据会出现59:59:999的问题！！
    timeindex=pd.date_range(start="2017-01-01 00:00:00",periods=data.shape[0],freq="1H")
    data.index=pd.to_datetime(timeindex)
    #print(data.loc["2017-04-10 00:00":"2017-04-11 00:00", :])
    #样本数据处理
    data=data.replace(0,np.nan).fillna(method="pad",axis=0)
    #样本升采样
    data=data.resample('15min').interpolate("time",axis=0)
    data.to_excel("dataset2.xlsx")


data_path = "D:/desktop/pywork/LsTM/dataset.xlsx"
data = pd.read_excel(data_path, index_col="Time")
#print(data.info())
#数据预处理 归一化
ymin=np.min(data)[-1]
ymax=np.max(data)[-1]
ymean=np.mean(data)[-1]
ystd=np.std(data)[-1]
#normolize_data=(data-np.min(data))/(np.max(data)-np.min(data))
normolize_data=(data-np.mean(data))/np.std(data)
#print(ymin,ymax,ymean,ystd)
#分割训练测试集和验证集
valid_x,valid_y=normolize_data.iloc[-10000:,:-1],normolize_data.iloc[-10000:,-1]
X_data,Y_data=normolize_data.iloc[:-10000,:-1],normolize_data.iloc[:-10000,-1]

train_x,test_x,train_y,test_y=train_test_split(X_data,Y_data,test_size=0.4,random_state=0)

#print(np.shape(train_x),np.shape(test_x),np.shape(valid_x))

def LSTM_Train(train_x,train_y,model,time_steps=8,num_units=2,batch_size=512,input_size=2,output_size=1,lr=0.01,epoch_size=200,save=True):

    '''
    :param train_x:
    :param train_y:
    :param time_steps:序列段长度
    :param num_units:隐藏层节点数目
    :param batch_size:序列段批处理数目
    :param input_size:输入维度
    :param output_size:输出维度
    :param lr:学习率
    :param epoch_size:
    :param save:
    :return:loss
    '''
    #初始化

    weight={ 'in':tf.Variable(tf.random_normal([input_size,num_units])),
        'out':tf.Variable(tf.random_normal([num_units,1]))}

    biases={ 'in':tf.Variable(tf.constant(1.0,shape=[num_units,])),
        'out':tf.Variable(tf.constant(1.0,shape=[1,])) }
    #图占位

    x=tf.placeholder("float",[None,time_steps,input_size],name='x')
    y=tf.placeholder("float",[None,output_size])

    #定义网络--------------------------------
    input=tf.reshape(x,[-1,input_size])
    input_rnn1=tf.matmul(input,weight["in"])+biases["in"]
    input_rnn=tf.reshape(input_rnn1,[-1,time_steps,input_size])

    #lstmCell=tf.nn.rnn_cell.BasicLSTMCell(num_units[0])
    Cell=tf.nn.rnn_cell.BasicLSTMCell(num_units)
    h0=Cell.zero_state(tf.shape(input_rnn)[0],dtype=tf.float32)
    output_rnn,final_states=tf.nn.dynamic_rnn(Cell,input_rnn,initial_state=h0,dtype=tf.float32)
    output= tf.transpose(output_rnn,[1,0,2])
    #一个全连接层输出
    pred=tf.add(tf.matmul(output[-1],weight['out']),biases['out'],name='pred')
    #定义损失函数
    loss=tf.reduce_mean(tf.square(tf.reshape(pred,[-1])-tf.reshape(y,[-1])))
    #定义优化器
    train_op=tf.train.AdamOptimizer(lr).minimize(loss)
    saver=tf.train.Saver(tf.global_variables())
    loss_list=[]
    with tf.Session() as sess:#开始会话
        sess.run(tf.global_variables_initializer())
        #start_time=time.time()
        #保存图 可视化
        # if save:
        #     tenboard_dir='./tenborad/lstmtest/'
        #     writer=tf.summary.FileWriter(tenboard_dir+'graph' )
        #     writer.add_graph(sess.graph)
        i=0#iteration
        e=0#epoch
        while(i<10000000000):
            start=0
            end=start+batch_size
            while (end<=len(train_x)):#batch循环
                train_x_batch=train_x.values.reshape(-1,time_steps,input_size)[start:end,...]
                train_y_batch=train_y.values.reshape([-1,output_size])[start:end,...]

                _,loss_=sess.run([train_op,loss],feed_dict={x:train_x_batch,y:train_y_batch})
                i+=1
                start+=batch_size
                end+=batch_size
            loss_list.append(loss_)  # epoch完成
            e += 1
            print('Num of epoch:', e, 'loss:', loss_list[-1])
            if e==1:
                if save:
                    saver.save(sess, './model/lstm_model'+str(model)+'.ckpt')
            elif e > 1 and loss_list[-2] > loss_list[-1]:
                if save:
                    saver.save(sess, './model/lstm_model'+str(model)+'.ckpt')
            if e>epoch_size:
                break
        #end_time=time.time()
        print('The Train has finished!\n num of iteration:%d num of epoch:%d loss end: %f'%(i,e,loss_list[-1]))
        return loss_list


def prediction(x_input,model):
    #saver=tf.train.Saver(tf.global_variables())
    saver = tf.train.import_meta_graph('./model/lstm_model0.ckpt.meta')
    with tf.Session() as sess:
        saver.restore(sess,tf.train.latest_checkpoint('./model/'))
        x=tf.get_default_graph().get_tensor_by_name('x:0')
        pred=tf.get_default_graph().get_tensor_by_name('pred:0')
        predict=[]
        for i in range(np.shape(x_input)[0]):
            next_seq=sess.run(pred,feed_dict={x:x_input.values.reshape(-1,x.shape[1],x.shape[2])[i:i+1,...]})
            predict.append(next_seq)
        return predict
        # plt.figure()
        # plt.plot(list(range(len(test_y))),test_y*ystd+ymean,color='b')
        # plt.plot(list(range(len(test_y))),np.array([value for value in np.reshape(predict,[-1])])*ystd+ymean,color='r')
        # plt.show()
def LSTM_Valid( model,time_step,num_unit,batch_size,lr,epoch_size):
    tf.reset_default_graph()
    loss=LSTM_Train(train_x.iloc[:,-time_step:],train_y,model,time_steps=time_step,num_units=num_unit,batch_size=batch_size,lr=lr,epoch_size=epoch_size)
    if loss[-1]!=np.nan:
        pre_y=prediction(valid_x.iloc[:,-time_step:],model)
        r2=r2_score(valid_y,[value for value in np.reshape(pre_y,[-1])])
    else:
        r2=0
    return r2
def Param_random(Param):
    num_unit_range=[ 2**i   for i in  range(1,10)]
    num_layer_range=[2,3,]
    num_batch_range=range(1000,5000,500)
    num_lr_range=[ 10**(-i) for i in  range(1,6)]
    num_epoch_range = range(5, 100, 5)
    num_timestep_range=range(3,9)
    random.choice(num_unit_range)

    Param["num_unit"] = random.choice(num_timestep_range)
    Param["num_unit"]=[random.choice(num_unit_range) for i in range(random.choice(num_layer_range))]
    Param["batch"]=random.choice(num_batch_range)
    Param["lr"]=random.choice(num_lr_range)
    Param["epoch"]=random.choice(num_epoch_range)

    return Param
def SA_Param_Choose(Tmax=30,MaxStep=2,alpha=0.9):

    param_init={"time_step":3,"num_unit":[16,16],"batch":100,"lr":0.1,"epoch":5}
    f1 = LSTM_Valid(0,param_init["time_step"],param_init["num_unit"], param_init["batch"], param_init["lr"], param_init["epoch"])
    param=param_init.copy()
    T = Tmax
    s=1
    Tmin = 1e-8
    fitness_list=[]
    while T > Tmin:
        step = 1
        while step < MaxStep:
            randomParam=Param_random(param)
            f2 = LSTM_Valid(str(s),randomParam["time_step"],randomParam["num_unit"], randomParam["batch"], randomParam["lr"], randomParam["epoch"])
            df = f2 - f1
            if df > 0:
                f1 = f2
                param=randomParam.copy()
            elif f2 != 0:  # 不接受破坏约束的解
                if ((math.e) ** (df / T) > np.random.rand()):
                    f1 = f2
                    param=randomParam.copy()
            step += 1
            s+=1
            print("T now :%f,r2 now :%f" %(T,f1))
        T = T * alpha
        fitness_list.append(f1)
        print(param)
    pd.Series(fitness_list).to_excel("SA_result.xlsx")

#SA_Param_Choose()
predict_lag=3
timestep=3
#loss=LSTM_Train(train_x.iloc[:,-2*(predict_lag+timestep):-2*predict_lag],train_y,model=0,time_steps=timestep,num_units=2,batch_size=1500,lr=0.01,epoch_size=10000)
pre_y=prediction(X_data.iloc[:,-2*(predict_lag+timestep):-2*predict_lag],0)
r2=r2_score(Y_data,[value for value in np.reshape(pre_y,[-1])])

pd.DataFrame({
    "pre_y":pd.Series(np.array([value for value in np.reshape(pre_y,[-1])])*ystd+ymean),
    "test_y":pd.Series((Y_data*ystd+ymean).values),
}).to_excel("TestResult1.xlsx")

print(r2)
