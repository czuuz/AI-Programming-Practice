import numpy as np
import os
import torch
from LeNet import LeNet
from colored_mnist import ColoredMNIST
from torchvision import transforms
from torch.nn import CrossEntropyLoss
from torch.optim import SGD
from torch.utils.data import DataLoader
from torchvision.datasets import mnist
from torchvision.transforms import ToTensor
import torch.nn.functional as F
from torch.autograd import grad
import torch.optim as optim
from ConvNet import ConvNet


def LeNettrain(data_enhence = 'on'):
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")

    kwargs = {'num_workers': 1, 'pin_memory': True} if use_cuda else {}
    batch_size = 256
    jit = transforms.ColorJitter(brightness=0.5, contrast=0.5, saturation=0.5, hue=0.5)  # 随机颜色数据增强
    jit_origin = transforms.ColorJitter()
    if data_enhence == 'off':
        jit = jit_origin
    train1_loader = torch.utils.data.DataLoader(
        ColoredMNIST(root='./data', env='train1',
                     transform=transforms.Compose([
                         jit,
                         transforms.ToTensor(),
                         transforms.Normalize((0.1307, 0.1307, 0.), (0.3081, 0.3081, 0.3081))
                     ])),
        batch_size=2000, shuffle=True, **kwargs)

    train2_loader = torch.utils.data.DataLoader(
        ColoredMNIST(root='./data', env='train2',
                     transform=transforms.Compose([
                         jit,
                         transforms.ToTensor(),
                         transforms.Normalize((0.1307, 0.1307, 0.), (0.3081, 0.3081, 0.3081))
                     ])),
        batch_size=2000, shuffle=True, **kwargs)

    test_loader = torch.utils.data.DataLoader(
        ColoredMNIST(root='./data', env='test', transform=transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307, 0.1307, 0.), (0.3081, 0.3081, 0.3081))
        ])),
        batch_size=1000, shuffle=True, **kwargs)


    model = LeNet().to(device)
    sgd = SGD(model.parameters(), lr=1e-1)
    # loss_fn = CrossEntropyLoss()
    all_epoch = 100
    prev_acc = 0
    for epoch in range(all_epoch):
        model.train()
        train_loader = [iter(x) for x in train1_loader]
        for idx, (train_x, train_label) in enumerate(train_loader):
            train_x = train_x.to(device)
            train_label = train_label.to(device).float()
            sgd.zero_grad()
            predict_y = model(train_x)
            loss = F.binary_cross_entropy_with_logits(predict_y, train_label)
            loss.backward()
            sgd.step()

        model.eval()

        acc = test_model(model, device, test_loader)
        print('accuracy: {:.3f}'.format(acc), flush=True)
        if not os.path.isdir("models"):
            os.mkdir("models")
        torch.save(model, 'models/mnist_{:.3f}.pkl'.format(acc))
        if np.abs(acc - prev_acc) < 1e-4:
            break
        prev_acc = acc
    print("Model finished training")


def compute_irm_penalty(losses, dummy):
  g1 = grad(losses[0::2].mean(), dummy, create_graph=True)[0]
  g2 = grad(losses[1::2].mean(), dummy, create_graph=True)[0]

  return (g1*g2).sum()


def irm_train(model, device, train_loaders, optimizer, epoch):
    model.train()

    train_loaders = [iter(x) for x in train_loaders]

    dummy_w = torch.nn.Parameter(torch.Tensor([1.0])).to(device)
    batch_idx = 0
    # penalty_multiplier = epoch ** 1.6
    penalty_multiplier = 1004
    print(f'Using penalty multiplier {penalty_multiplier}')
    # reg = 1e-3
    while True:
        optimizer.zero_grad()
        error = 0
        penalty = 0
        for loader in train_loaders:
            data, target = next(loader, (None, None))
            if data is None:
                return
            data, target = data.to(device), target.to(device).float()
            output = model(data)
            loss_erm = F.binary_cross_entropy_with_logits(output * dummy_w, target, reduction='none')
            penalty += compute_irm_penalty(loss_erm, dummy_w)
            error += loss_erm.mean()
        (error + penalty_multiplier * penalty).backward()
        # (reg*error + (1-reg) * penalty).backward()
        optimizer.step()
        if batch_idx % 2 == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tERM loss: {:.6f}\tGrad penalty: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loaders) * len(data),
                       100. * batch_idx / len(train_loaders[0]), error.item(), penalty.item()))
            print('First 20 logits', output.data.cpu().numpy()[:20])

        batch_idx += 1


def rex_train(model, device, train_loaders, optimizer, epoch):
    model.train()

    train_loaders = [iter(x) for x in train_loaders]

    batch_idx = 0
    # penalty_multiplier = epoch ** 1.3
    penalty_multiplier = 500.0
    print(f'Using penalty multiplier {penalty_multiplier}')
    while True:
        optimizer.zero_grad()
        error = 0
        penalty = 0
        loss = torch.zeros(len(train_loaders))
        for i, loader in enumerate(train_loaders):
            data, target = next(loader, (None, None))
            if data is None:
                return
            data, target = data.to(device), target.to(device).float()
            output = model(data)
            loss_erm = F.binary_cross_entropy_with_logits(output, target, reduction='none')
            # error += loss_erm.mean()
            loss[i] = loss_erm.mean()
        mean = loss.mean()
        penalty = ((loss-mean)**2).mean()
        (mean + penalty_multiplier * penalty).backward()
        optimizer.step()
        if batch_idx % 2 == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tERM loss: {:.6f}\tGrad penalty: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loaders) * len(data),
                       100. * batch_idx / len(train_loaders[0]), mean.item(), penalty.item()))
            print('First 20 logits', output.data.cpu().numpy()[:20])

        batch_idx += 1


def test_model(model, device, test_loader, set_name="test set"):
  model.eval()
  test_loss = 0
  correct = 0
  with torch.no_grad():
    for data, target in test_loader:
      data, target = data.to(device), target.to(device).float()
      output = model(data)
      test_loss += F.binary_cross_entropy_with_logits(output, target, reduction='sum').item()
      pred = torch.where(torch.gt(output, torch.Tensor([0.0]).to(device)),
                         torch.Tensor([1.0]).to(device),
                         torch.Tensor([0.0]).to(device))  # get the index of the max log-probability
      correct += pred.eq(target.view_as(pred)).sum().item()

  test_loss /= len(test_loader.dataset)

  print('\nPerformance on {}: Average loss: {:.4f}, Accuracy: {}/{} ({:.2f}%)\n'.format(
    set_name, test_loss, correct, len(test_loader.dataset),
    100. * correct / len(test_loader.dataset)))

  return 100. * correct / len(test_loader.dataset)


def train(type, data_enhence):
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")

    kwargs = {'num_workers': 1, 'pin_memory': True} if use_cuda else {}
    batch_size = 256
    jit = transforms.ColorJitter(brightness=0.5, contrast=0.5, saturation=0.5, hue=0.5)#随机颜色数据增强
    jit_origin = transforms.ColorJitter()
    if data_enhence=='off':
        jit = jit_origin
    train1_loader = torch.utils.data.DataLoader(
        ColoredMNIST(root='./data', env='train1',
                     transform=transforms.Compose([
                         jit,
                         transforms.ToTensor(),
                         transforms.Normalize((0.1307, 0.1307, 0.), (0.3081, 0.3081, 0.3081))
                     ])),
        batch_size=2000, shuffle=True, **kwargs)

    train2_loader = torch.utils.data.DataLoader(
        ColoredMNIST(root='./data', env='train2',
                     transform=transforms.Compose([
                         jit,
                         transforms.ToTensor(),
                         transforms.Normalize((0.1307, 0.1307, 0.), (0.3081, 0.3081, 0.3081))
                     ])),
        batch_size=2000, shuffle=True, **kwargs)

    test_loader = torch.utils.data.DataLoader(
        ColoredMNIST(root='./data', env='test', transform=transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307, 0.1307, 0.), (0.3081, 0.3081, 0.3081))
        ])),
        batch_size=1000, shuffle=True, **kwargs)
    model = LeNet().to(device)
    optimizer = optim.Adam(model.parameters(), lr=0.0005)
    #VREX用500的lambda,optim用Adam
    #IRM
    for epoch in range(1, 100):
        if type=='IRM':
            irm_train(model, device, [train1_loader, train2_loader], optimizer, epoch)
        elif type=='VREx':
            rex_train(model, device, [train1_loader, train2_loader], optimizer, epoch)
        train1_acc = test_model(model, device, train1_loader, set_name='train1 set')
        train2_acc = test_model(model, device, train2_loader, set_name='train2 set')
        test_acc = test_model(model, device, test_loader)
        if train1_acc > 70 and train2_acc > 70 and test_acc > 60:
            print('found acceptable values. stopping training.')
            return





if __name__=='__main__':
     train('IRM', 'off')
    # LeNettrain('off')


