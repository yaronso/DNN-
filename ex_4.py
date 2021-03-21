import torch
import torch.nn as nn
import torch.optim as opt
import torch.nn.functional as func
from torchvision import transforms
from torchvision import datasets
import numpy as np


# Model A: with two hidden layers: (SGD optimizer)
class A(nn.Module):
    def __init__(self, image_size=784):
        super(A, self).__init__()
        self.image_size = image_size
        self.fc0 = nn.Linear(image_size, 100)
        self.fc1 = nn.Linear(100, 50)
        self.fc2 = nn.Linear(50, 10)

    def forward(self, x):
        x = x.view(-1, self.image_size)
        x = func.relu(self.fc0(x))
        x = func.relu(self.fc1(x))
        x = self.fc2(x)
        return func.log_softmax(x)


# Model B: with two hidden layers: (ADAM optimizer)
class B(nn.Module):
    def __init__(self, image_size=784):
        super(B, self).__init__()
        self.image_size = image_size
        self.fc0 = nn.Linear(image_size, 100)
        self.fc1 = nn.Linear(100, 50)
        self.fc2 = nn.Linear(50, 10)

    def forward(self, x):
        x = x.view(-1, self.image_size)
        x = func.relu(self.fc0(x))
        x = func.relu(self.fc1(x))
        x = self.fc2(x)
        return func.log_softmax(x)


# Model C: (add dropout layers to model A)
class C(A):
    # adding dropout on the output of the hidden layers
    def forward(self, x):
        x = x.view(-1, self.image_size)
        x = func.relu(self.fc0(x))
        x = func.dropout(x, p=0.5)
        x = func.relu(self.fc1(x))
        x = func.dropout(x, p=0.5)
        x = self.fc2(x)
        return func.log_softmax(x)


# Model D: (add Batch Normalization layers to model A)
class D(nn.Module):
    def __init__(self, image_size=784):
        super(D, self).__init__()
        self.image_size = image_size
        self.fc0 = nn.Linear(image_size, 100)
        self.norm1 = nn.BatchNorm1d(image_size)
        self.fc1 = nn.Linear(100, 50)
        self.norm2 = nn.BatchNorm1d(100)
        self.fc2 = nn.Linear(50, 10)
        self.norm3 = nn.BatchNorm1d(50)

    def forward(self, x):
        x = x.view(-1, self.image_size)
        x = self.norm1(x)
        x = func.relu(self.fc0(x))
        x = self.norm2(x)
        x = func.relu(self.fc1(x))
        x = self.fc2(x)
        return func.log_softmax(x)


# Model E: five hidden layers using ReLU activation function
class E(nn.Module):
    def __init__(self, image_size=784):
        super(E, self).__init__()
        self.image_size = image_size
        self.fc0 = nn.Linear(image_size, 128)
        self.fc1 = nn.Linear(128, 64)
        self.fc2 = nn.Linear(64, 10)
        self.fc3 = nn.Linear(10, 10)
        self.fc4 = nn.Linear(10, 10)
        self.fc5 = nn.Linear(10, 10)

    def forward(self, x):
        x = x.view(-1, self.image_size)
        x = func.relu(self.fc0(x))
        x = func.relu(self.fc1(x))
        x = func.relu(self.fc2(x))
        x = func.relu(self.fc3(x))
        x = func.relu(self.fc4(x))
        x = self.fc5(x)
        return func.log_softmax(x)


# Model F: five hidden layers using Sigmoid activation function
class F(nn.Module):
    def __init__(self, image_size=784):
        super(F, self).__init__()
        self.image_size = image_size
        self.fc0 = nn.Linear(image_size, 128)
        self.fc1 = nn.Linear(128, 64)
        self.fc2 = nn.Linear(64, 10)
        self.fc3 = nn.Linear(10, 10)
        self.fc4 = nn.Linear(10, 10)
        self.fc5 = nn.Linear(10, 10)

    def forward(self, x):
        x = x.view(-1, self.image_size)
        x = func.sigmoid(self.fc0(x))
        x = func.sigmoid(self.fc1(x))
        x = func.sigmoid(self.fc2(x))
        x = func.sigmoid(self.fc3(x))
        x = func.sigmoid(self.fc4(x))
        x = self.fc5(x)
        return func.log_softmax(x)


# training the model
def train(model, optimizer, train_loader):
    model.train()
    for batch_idx, (data, labels) in enumerate(train_loader):
        # reset the gradient
        optimizer.zero_grad()
        output = model(data)
        loss = func.nll_loss(output, labels)
        loss.backward()
        optimizer.step()


# testing the model
def test(model, loader):
    model.eval()
    test_loss = 0
    correct = 0
    loader_len = len(loader.dataset)
    # with torch.no_grad():
    for data, target in loader:
        output = model(data)
        # sum up the batch loss
        test_loss += func.nll_loss(output, target, size_average=False).item()
        # get the index of the max log probability
        pred = output.max(1, keepdim=True)[1]
        correct += pred.eq(target.view_as(pred)).cpu().sum()
    test_loss /= loader_len
    rate = 100. * correct / loader_len
    return test_loss, rate, correct


def create_file(model, test_x):
    file = open("test_y", 'w+')
    for test in test_x:
        output = model(test)
        pred = output.max(1, keepdim=True)[1].item()
        pred_str = str(pred) + '\n'
        file.write(pred_str)
    file.close()


def find_max_acc(acc_lst):
    count = 0
    max_acc = max(acc_lst)
    for acc in acc_lst:
        if acc == max_acc:
            return count
        count += 1


def fcnn_flow():
    # normalize the data set to values between 0 and 1.
    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize([0.], [1])])
    train_set = datasets.FashionMNIST('/.dataset', train=True, download=True, transform=transform)
    test_set = datasets.FashionMNIST('/.dataset', train=False, transform=transform)
    # split the training set to 80% train set and 20% validation set
    train_loader, validation_loader = torch.utils.data.random_split(train_set, [int(len(train_set) * 0.8),
                                                                               int(len(train_set) * 0.2)])
    # The final train and validation loaders
    train_loader = torch.utils.data.DataLoader(train_loader, batch_size=64, shuffle=True)
    validation_loader = torch.utils.data.DataLoader(validation_loader, batch_size=64, shuffle=True)
    # The final test loader
    test_loader = torch.utils.data.DataLoader(test_set, batch_size=1)
    # load the text_x file using numpy load text function()
    test_x = np.loadtxt("test_x")
    # normalize the test x values
    test_x /= 255.0
    test_x = transform(test_x)[0].float()
    my_models = []
    my_opts = []
    acc_lst = []
    EPOCHS = 10
    a = A()
    b = B()
    c = C()
    d = D()
    e = E()
    f = F()
    my_models.append(a)
    my_models.append(b)
    my_models.append(c)
    my_models.append(d)
    my_models.append(e)
    my_models.append(f)
    for model in my_models:
        if model is a or model is c or model is d:
            my_opts.append(opt.SGD(model.parameters(), lr=0.1))
        else:
            my_opts.append(opt.Adam(model.parameters(), lr=0.001))
    len_models = len(my_models)
    # train each model with 10 epochs using pytorch train loader
    for i in range(len_models):
        for epoch in range(EPOCHS):
            train(my_models[i], my_opts[i], train_loader)
        # Test the current model with pytorch test loader
        test_loss, acc, correct = test(my_models[i], test_loader)
        print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(test_loss, correct,
                                                                                     len(test_loader.dataset),
                                                                                     100. * correct / len(
                                                                                         test_loader.dataset)))
        curr_model_acc = float(100. * correct / len(test_loader.dataset))
        acc_lst.append(curr_model_acc)
    # end of models loop
    max = find_max_acc(acc_lst)
    # create the test_y predictions file by using the model with the max accuracy
    create_file(my_models[max], test_x)


if __name__ == '__main__':
    fcnn_flow()
