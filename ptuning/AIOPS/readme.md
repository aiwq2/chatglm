## 这里讲一下各个不同版本的训练和验证json文件的特征

train_jianjie
train_jianjie2:目前已经跑通过一次的版本，里面的数据依照时间间隔3之差值来构造，低于1.0的差值直接忽略，label为某个时刻的label
train_jinanjie3：里面的数据依照时间间隔6的差值来构造，低于？的差值直接忽略，label为某个时刻的label某个节点的label，并且训练集和验证集不能随机划分必须按照一定的顺序，也就是说相当于6个要为1组