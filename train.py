import time
import matplotlib.pylab as plt
import torch
from torchvision import datasets
import torchvision.transforms.v2 as transforms

import models

ds_transfrom=transforms.Compose([
    transforms.ToImage(),
    transforms.ToDtype(torch.float32,scale=True)
])
ds_train=datasets.FashionMNIST(
    root='data',
    train=True,
    download=True,
    transform=ds_transfrom
)
ds_test=datasets.FashionMNIST(
    root='data',
    train=False,
    download=True,
    transform=ds_transfrom
)
batch_size=64
dataloader_train=torch.utils.data.DataLoader(
    ds_train,
    batch_size=batch_size,
    shuffle=True
)
dataloader_test=torch.utils.data.DataLoader(
    ds_test,
    batch_size=batch_size
)
for imge_batch,label_bach in dataloader_test:
    #print(imge_batch.shape)
    #print(label_bach.shape)
    
    model=models.MyModel()

    loss_fn = torch.nn.CrossEntropyLoss()
    learning_rate= 1e-3
    optimizer= torch.optim.SGD(model.parameters(),lr=learning_rate)
    n_epochs=20
    val_acc_log=[]
    train_loss_log=[]
    val_loss_log=[]
    train_acc_log=[]
for epoch in range(n_epochs):
        time_start=time.time()
        print(f'epoch{epoch+1}/{n_epochs}')
        train_loss=models.train(model,dataloader_train,loss_fn,optimizer)
        time_end=time.time()
        print(f'training loss:{train_loss}')
        train_loss_log.append(train_loss)

        val_acc=models.test_accuracy(model,dataloader_test)
        print(f'validation accuracy:{val_acc*100:.3f}%')
        val_acc_log.append(val_acc)

        val_loss=models.test(model,dataloader_test,loss_fn)
        print(f'validation loss:{val_loss}')
        val_loss_log.append(val_loss)

        train_acc=models.test_accuracy(model,dataloader_train)
        print(f'training accuracy:{train_acc*100:.3f}%')
        train_acc_log.append(train_acc)
plt.subplot(1,2,1)
plt.plot(range(1, n_epochs+1), train_loss_log, label='train')
plt.plot(range(1, n_epochs+1), val_loss_log, label='validation')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.xticks(range(1, n_epochs+1))
plt.grid()
plt.legend()

plt.subplot(1,2,2)
plt.plot(range(1, n_epochs+1), train_acc_log, label='train')
plt.plot(range(1, n_epochs+1), val_acc_log, label='validation')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.xticks(range(1, n_epochs+1))
plt.grid()
plt.legend()

plt.show()
