from __future__ import print_function, division
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
import numpy as np
from torchvision import transforms
import time
from model.SteelDataset import SteelDataset
import json

from model.residual_attention_network import ResidualAttentionModel_448input as ResidualAttentionModel
model_file = 'model_92_sgd.pkl'


device = ('cuda' if torch.cuda.is_available() else 'cpu')

# for test
def test(model, test_loader, btrain=False, model_file='model_92.pkl'):
    # Test
    if not btrain:
        model.load_state_dict(torch.load(model_file))
    model.eval()

    correct = 0
    total = 0
    #
    class_correct = list(0. for o in range(7))
    class_total = list(0. for o in range(7))

    for images, labels in test_loader:
        images, labels = images.to(device), labels.to(device)

        outputs = model(images)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels.data).sum()
        #
        c = (predicted == labels.data).squeeze()
        for i in range(labels.size(0)):
            label = labels.data[i]
            class_correct[label] += c[i]
            class_total[label] += 1

    print('Accuracy of the model on the test images: %d %%' % (100 * float(correct) / total))
    print('Accuracy of the model on the test images:', float(correct)/total)
    for i in range(7):
        if class_total[i] != 0:
            print('Accuracy of %5s : %2d %%' % (
                classes[i], 100 * class_correct[i] / class_total[i]))
        else:
            print('No instances of %5s in the test set.\n' % (classes[i]))
    return correct / total


if __name__ == '__main__':
    
    with open('config.json', 'r') as config_file:
        config = json.load(config_file)
    
    train_parameter = config['train_parameters']
    model_parameters = config['model_parameters']
    train_data = config['data_paths']['train_data']


    transform = transforms.Compose([
            transforms.Resize(size=(150,2300)),
            transforms.Grayscale(1),
            transforms.ToTensor(),
            transforms.Normalize([0.5], [0.222])])  

    test_transform = transforms.Compose([
        transforms.ToTensor()
    ])


    dataset = SteelDataset(train_data, transform=transform)
    train_size = int(0.8 * len(dataset))
    test_size = len(dataset) - train_size
    train_dataset, test_dataset = random_split(dataset, [train_size, test_size])

    batch_size = train_parameter['batch_size'] 
    #
    train_loader = DataLoader(dataset=train_dataset,
                              batch_size=batch_size, 
                              shuffle=True, num_workers=8)
    test_loader = DataLoader(dataset=test_dataset,
                             batch_size=2,
                             shuffle=False)

    

    classes = ('broken', 'cracks', 'dents', 'depressions', 'hairs', 'incrustations', 'irregular overlaps')
    
    model = ResidualAttentionModel(len(classes)).to(device)

    lr = train_parameter['learning_rate']  # 0.1
    momentum = train_parameter['momentum']  # 0.9
    weight_decay = train_parameter['weight_decay']  # 0.0001
    nesterov = train_parameter['nesterov']  # True
    criterion = nn.CrossEntropyLoss()

    optimizer = optim.SGD(model.parameters(), lr=lr, momentum=momentum, nesterov=nesterov, weight_decay=weight_decay)

    is_train = True
    is_pretrain = False
    acc_best = 0
    total_epoch = train_parameter['epochs'] 

    if is_train is True:
        if is_pretrain == True:
            model.load_state_dict((torch.load(model_file)))

        # Training
        for epoch in range(total_epoch):
            model.train()
            tims = time.time()
            for i, (images, labels) in enumerate(train_loader):
                images, labels = images.to(device), labels.to(device)

                # Forward + Backward + Optimize
                optimizer.zero_grad()
                outputs = model(images)

                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()
                if (i + 1) % 20 == 0:
                    print("Epoch [%d/%d], Iter [%d/%d] Loss: %.4f" % (
                    epoch + 1, total_epoch, i + 1, len(train_loader), loss.item()))

            print('This epoch takes time:', time.time() - tims)
            print('\nEvaluate test set:')
            acc = test(model, test_loader, btrain=True)
            if acc > acc_best:
                acc_best = acc
                print('current best acc,', acc_best)
                torch.save(model.state_dict(), model_file)

            # Decaying Learning Rate
            if (epoch + 1) / float(total_epoch) == 0.3 or (epoch + 1) / float(total_epoch) == 0.6 or (
                    epoch + 1) / float(total_epoch) == 0.9:
                lr /= 10
                print('reset learning rate to:', lr)
                for param_group in optimizer.param_groups:
                    param_group['lr'] = lr
                    print(param_group['lr'])
                optimizer = torch.optim.Adam(model.parameters(), lr=lr)
                optim.SGD(model.parameters(), lr=lr, momentum=momentum, nesterov=nesterov, weight_decay=weight_decay)
        # Save the Model
        torch.save(model.state_dict(), 'last_model_92_sgd.pkl')

    else:
        test(model, test_loader, btrain=False)
