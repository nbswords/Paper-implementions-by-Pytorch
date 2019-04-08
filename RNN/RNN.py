import torch
import torch.nn as nn
import torchvision.transforms as transforms
import torchvision.datasets as dsets
from torch.autograd import Variable
import torch.nn.functional as F

train_dataset = dsets.MNIST(
    root = './data',
    train = True,
    transform = transforms.ToTensor(),
    download = True
)
test_dataset = dsets.MNIST(
    root = './data',
    train = False,
    transform = transforms.ToTensor()
)

print('train_dataset :', train_dataset.train_data.size())
train_dataset : torch.Size([60000, 28, 28])

print('train_dataset :', train_dataset.train_labels.size())
train_dataset : torch.Size([60000])

print('test_dataset :', test_dataset.test_data.size())
test_dataset : torch.Size([10000, 28, 28])

print('test_dataset :', test_dataset.test_labels.size())
test_dataset : torch.Size([10000])

batch_size = 100
n_iters = 1000
num_epochs = n_iters / (len(train_dataset) / batch_size)
num_epochs = int(num_epochs)

train_loader = torch.utils.data.DataLoader(dataset = train_dataset,
                                           batch_size = batch_size,
                                           shuffle = True)
test_loader = torch.utils.data.DataLoader(dataset = test_dataset,
                                           batch_size = batch_size,
                                           shuffle = False)

class RNNModel(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(RNNModel, self).__init__()   
        
        self.rnn = nn.RNN(
            input_size = input_dim,
            hidden_size = 100,
            num_layers = 1,
            batch_first=True,
            nonlinearity='relu')
        
        self.fc = nn.Linear(100, 10)
        
    def forward(self, x):
        # (layer_dim, batch_size, hidden_dim)
        h0 = None
        # another hidden state example :
        # h0 = Variable(torch.zeros(1, x.size(0), 100))
        
        out, hn = self.rnn(x, h0)
        # "out" dim : (100, 28, 100)
        # "-1" means the last time step
        
        out = self.fc(out[:, -1, :])  # (100, 100)
        # "out" dim : (100, 10)
        
        return out
input_dim = 28
output_dim = 10
model = RNNModel(input_dim, output_dim)
In [73]:
cost_fun = nn.CrossEntropyLoss()
In [74]:
lr = 0.1
opt = torch.optim.SGD(model.parameters(), lr = lr)
iters = 0
for epoch in range(num_epochs):
    for i, (images, labels) in enumerate(train_loader):
        # (100, 1, 28, 28) > (100, 28, 28)
        images = Variable(images.view(-1, 28, 28))
        labels = Variable(labels)
        
        opt.zero_grad()
        
        outputs = model(images)
        
        loss = cost_fun(outputs, labels)
        loss.backward()
        opt.step()
        
        iters += 1
        
        if iters % 5 == 0:
            correct = 0
            total = 0
            for images, labels in test_loader:
                images = Variable(images.view(-1, 28, 28))
                outputs = model(images)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                
                correct += (predicted == labels).sum()
            acc = 100 * correct / total
            print('Iteration: {}. Loss: {}. Acc: {}'.format(iters, loss.data[0], acc))