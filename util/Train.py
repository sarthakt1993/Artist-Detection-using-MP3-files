import numpy as np
import torch
import torch.nn  as nn
from torch.autograd import Variable 

def load_data(X,y,batch_size):
    data =[]
    for i in range(len(X)):
        data.append([np.swapaxes(X[i],0,2), y[i]])
    dataloader = torch.utils.data.DataLoader(data, shuffle=True, batch_size=batch_size)
    return dataloader

from ignite.metrics import Accuracy, Precision, Recall, Fbeta
from torch.autograd import Variable 

def train_predict(dataloader_train,dataloader_val,model,epochs,learning_rate,use_cuda):
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    
    if use_cuda:
        model = model.cuda()
    model = model.train()
    
    start.record()
    train_loss_list=[]
    val_loss_list=[]
    train_f1=[]
    val_f1=[]
    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)
    
    precision = Precision()
    recall = Recall()
    f1 = Fbeta(beta=1.0, average=True, precision=precision, recall=recall)

    
    for epoch in range(epochs):
        print("Epoch: {}".format(epoch+1))
        for i,(img, label) in enumerate(dataloader_train):
            img, label = Variable(img),Variable(label)
            if use_cuda:
                img = img.cuda()
                label = label.cuda()
            optimizer.zero_grad()
            pred = model.forward(img)
            _,my_label = torch.max(label, dim=1)
            loss = loss_fn(pred,my_label)
            if i == len(dataloader_train)-1:
                train_loss_list.append(loss.item())
            loss.backward()
            optimizer.step()
            precision.update((pred, my_label))
            recall.update((pred, my_label))
            f1.update((pred, my_label))
        print("\tTrain loss: {:0.2f}".format(train_loss_list[-1]))
        precision.compute()
        recall.compute()
        train_f1.append(f1.compute()*100)
        print("\tTrain F1 Score: {:0.2f}%".format(train_f1[-1]))
        
        precision = Precision()
        recall = Recall()
        f1 = Fbeta(beta=1.0, average=True, precision=precision, recall=recall)
        
        with torch.no_grad():
            for i,(img, label) in enumerate(dataloader_val):
                img, labels = Variable(img),Variable(label)
                if use_cuda:
                    img = img.cuda()
                    label = label.cuda()
                pred = model(img)
                _,my_label = torch.max(label, dim=1)
                loss = loss_fn(pred,my_label)
                if i == len(dataloader_val)-1:
                    val_loss_list.append(loss.item())
                precision.update((pred, my_label))
                recall.update((pred, my_label))
                f1.update((pred, my_label))
        print("\n\tVal loss: {:0.2f}".format(val_loss_list[-1]))
        precision.compute()
        recall.compute()
        val_f1.append(f1.compute()*100)
        print("\tVal F1 Score: {:0.2f}%".format(val_f1[-1]))
    
    end.record()
    torch.cuda.synchronize()
    time = start.elapsed_time(end)
    return (train_loss_list,val_loss_list,train_f1,val_f1,time,model)

def test_predict(model,dataloader_test,use_cuda):
    if use_cuda:
        model = model.cuda()
    
    precision = Precision()
    recall = Recall()
    f1 = Fbeta(beta=1.0, average=True, precision=precision, recall=recall)
    
    for i,(img, label) in enumerate(dataloader_test):
        img, labels = Variable(img),Variable(label)
        if use_cuda:
            img = img.cuda()
            label = label.cuda()
            pred = model(img)
            _,my_label = torch.max(label, dim=1)
            precision.update((pred, my_label))
            recall.update((pred, my_label))
            f1.update((pred, my_label))
            
    precision.compute()
    recall.compute()
    print("\tF1 Score: {:0.2f}".format(f1.compute()*100))

def plot_trn_val_loss(train,val):
    plt.figure(figsize=(12,6))
    plt.plot(train,label="Training Dataset")
    plt.plot(val,label="Vallidation Dataset")
    plt.xlabel("Number of epochs")
    plt.ylabel("Cross Entropy Loss")
    plt.xticks(range(0,len(train),2))
    plt.legend(loc='upper right')
    plt.show()
    
def plot_trn_val_f1(train,val):
    plt.figure(figsize=(12,6))
    plt.plot(train,label="Training Dataset")
    plt.plot(val,label="Vallidation Dataset")
    plt.xlabel("Number of epochs")
    plt.ylabel("F1 score")
    plt.xticks(range(0,len(train),2))
    plt.legend(loc='upper right')
    plt.show()