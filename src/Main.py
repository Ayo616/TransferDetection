

import torch
import torch.nn as nn
from src.DataFactory import Factory
from src.Loss import linear_mmd2, Euclidean, CombinedLoss
from src.Network import TOLN
from torch.utils import model_zoo
from torch.autograd import Variable
import torch.optim as optimizer
from tqdm import tqdm
import numpy as np

from src.agent import MyAgent


def train(epoch):
    # 参数
    total_loss_train = 0
    # 损失函数
    criterion = nn.MSELoss()
    # 加载数据
    f = Factory()
    traindata = f.get_train_data()
    testdadta = f.get_test_data()
    # 加载网络
    model = TOLN()
    # 优化
    optimizer = torch.optim.SGD(model.parameters(), lr = 0.001, momentum=0.9)

    # 代理结果
    normal,outlier = MyAgent()

    for (batch_id, (data, target)),(batch_ids, (datas, targets)) in zip(enumerate(traindata),enumerate(testdadta)):

        # 从输出得到输出
        source_data, source_label = Variable(data),Variable(target),
        source_data = source_data.float()

        target_data, target_label = Variable(datas),Variable(targets),
        target_data = target_data.float()

        # 放入模型中
        # input 原来网络的output
        # result 新网络的output
        output,result = model(source_data,target_data)



    # 根据代理结果判断，该批次的label
        final = []
        bool =  []
        for each in data.numpy().tolist():
            for i in normal:
                if i.tolist() == each:
                    final.append(1)
                    bool.append(True)
            for i in outlier:
                if i.tolist() == each:
                    final.append(-1)
                    bool.append(False)

        final = torch.from_numpy(np.array(final))

        # 构建loss1,计算原来网络中输入输出值之间的误差
        loss = criterion(output,final.float().reshape(5,1))

        # 构建MMD误差，计算两个领域的整体分布误差
        loss2 = linear_mmd2(data,datas)

        # 构建特定点距离loss
        data = data.numpy()#转换格式
        datas = datas.numpy()#转换格式

        average_distance = Euclidean(data,datas)

        for index,item in enumerate(average_distance):
            if bool[index] == False:
                # average_distance[index] = 1/item
                average_distance[index] = -np.log2(item)
                # print('False')
            if bool[index] == True:
                average_distance[index] = np.log2(item)
                # print('True')
        loss3  = sum(average_distance)

        # allLoss = CombinedLoss(loss,loss2,loss3)

        allLoss = (loss.float() + loss2.float() +torch.from_numpy(np.array(loss3)).float())

        # 清空、反向、更新权重
        optimizer.zero_grad()
        allLoss.backward()
        optimizer.step()

        if batch_id % 50 == 0:
            print('Epoch: ', epoch, '| train loss: %.4f' % loss.data.numpy())

    torch.save(model, 'model_epoch30.pkl')
    return model




def test(model):
    model.eval()
    # 加载数据
    f = Factory()
    test_loss = 0
    correct = 0
    testdadta = f.get_extra_test_data()
    for batch_id, (data, target) in enumerate(testdadta):
        data, target = Variable(data), Variable(target)
        data,target = data.float(),target.float()
        out, _ = model(data,data)
        # sum up batch loss
        print(out)
        # test_loss += torch.nn.L1Loss(out, target.long()).data[0]
        # get the index of the max log-probability
        pred = out.data.max(1, keepdim=True)[1]
        correct += pred.eq(target.data.view_as(pred).long()).cpu().sum()
        print('aaa',type(correct))

    test_loss =1

    print( {
        'epoch': e,
        'average_loss': test_loss,
        'correct': correct,
        'total': len(testdadta),
        'accuracy': 100. * correct / len(testdadta)
    })



if __name__ == '__main__':

    epochs = 30
    # model = None
    # for e in tqdm(range(epochs)):
    #     model = train(e)

    model = torch.load('model_epoch30.pkl')

    test(model)
