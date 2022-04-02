# Main function of training and testing of different popular attention mechanisms on MedMNIST v2 and Chaoyang dataset
# Implemented by Yuchuan Li

import argparse
import os

import torch
import torch.nn as nn
import torchvision.transforms as transforms
from torch import optim
from torch.utils.data import DataLoader
from tqdm import tqdm
from sklearn.metrics import roc_auc_score

from dataset.Chaoyang import CHAOYANG
import medmnist
from medmnist import INFO, Evaluator

from model.models import resnet18_attention, resnet34_attention, resnet50_attention, SimpleNet


def train(args):
    # main function of training procedure
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # dataloader preparation for MedMNIST and Chaoyang dataset respectively
    if args.dataset == 'MedMNIST':
        # data_flag for sub-dataset selection
        data_flag = 'dermamnist'
        info = INFO[data_flag]
        num_class = len(info['label'])
        DataClass = getattr(medmnist, info['python_class'])

        train_dataset = DataClass(split='train', transform=transforms.Compose(
            [transforms.ToTensor(), transforms.Normalize(mean=[.5], std=[.5])]), download=True)
        test_dataset = DataClass(split='test', transform=transforms.Compose(
            [transforms.ToTensor()]), download=True)

        train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=args.batch_size, shuffle=True)
        test_loader = torch.utils.data.DataLoader(dataset=test_dataset, batch_size=args.batch_size, shuffle=False)

    if args.dataset == 'Chaoyang':
        train_dataset = CHAOYANG(root="../dataset/Chaoyang/",
                                 json_name="train.json",
                                 train=True,
                                 transform=transforms.Compose(
                                     [transforms.RandomHorizontalFlip(), transforms.Resize((256, 256)),
                                      transforms.ToTensor()])
                                 )
        test_dataset = CHAOYANG(root="../dataset/Chaoyang/",
                                json_name="test.json",
                                train=False,
                                transform=transforms.Compose([transforms.ToTensor()])
                                )

        train_loader = torch.utils.data.DataLoader(dataset=train_dataset,
                                                   batch_size=args.batch_size,
                                                   num_workers=0,
                                                   drop_last=False,
                                                   shuffle=True)
        test_loader = torch.utils.data.DataLoader(dataset=test_dataset,
                                                  batch_size=int(args.batch_size / 2),
                                                  num_workers=0,
                                                  drop_last=False,
                                                  shuffle=False)

        num_class = 4

    # setup model for MedMNIST and Chaoyang dataset respectively
    if args.dataset == 'MedMNIST':
        model = SimpleNet(num_classes=num_class, att_choice=args.att)
    else:
        model = resnet18_attention(pretrained=args.pretrained, n_classes=num_class, att_choice=args.att)
    model.to(device)

    # setup loss function
    criterion = nn.CrossEntropyLoss()

    # setup optimizer for MedMNIST and Chaoyang dataset respectively
    if args.dataset == 'MedMNIST':
        optimizer = optim.SGD(model.parameters(), lr=args.lr)
    else:
        optimizer = optim.Adam(model.parameters(), lr=args.lr)

    # print the summary information of training procedure for future reference
    print(f'''Starting training:
            Epochs:          {args.epochs}
            Batch size:      {args.batch_size}
            Learning rate:   {args.lr}
            Dataset type:    {args.dataset}
            Pre-trained:     {args.pretrained}
            Attention type:  {args.att}
            Device:          {device.type}
        ''')

    auc_train_best = 0
    auc_test_best = 0
    acc_train_best = 0
    acc_test_best = 0

    for epoch in range(args.epochs):
        model.train()
        for inputs, targets in tqdm(train_loader):
            # forward + backward + optimize
            inputs = inputs.to(device=device)
            targets = targets.to(device=device)
            outputs = model(inputs)

            targets = targets.squeeze().long()
            loss = criterion(outputs, targets)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        # evaluation of accuracy and AUC
        if args.dataset == 'MedMNIST':
            metrics = test_MedMNIST('train', model, train_loader, data_flag, device)
            acc_train = metrics.ACC
            auc_train = metrics.AUC
            metrics2 = test_MedMNIST('test', model, test_loader, data_flag, device)
            acc_test = metrics2.ACC
            auc_test = metrics2.AUC
        else:
            acc_train, auc_train = test_chaoyang(model, train_loader, epoch, device)
            acc_test, auc_test = test_chaoyang(model, test_loader, epoch, device)

        # function for saving the best epoch depending on previous best performance
        if acc_train >= acc_train_best and auc_train >= auc_train_best and auc_test >= auc_test_best and acc_test >= acc_test_best:
            acc_train_best = acc_train
            auc_train_best = auc_train
            auc_test_best = auc_test
            acc_test_best = acc_test

            # directory selection
            path = './log/' + str(args.dataset)
            if not os.path.exists(path):
                os.mkdir(path)
            path = './log/' + str(args.dataset) + '/' + str(args.att)
            if not os.path.exists(path):
                os.mkdir(path)
            torch.save(model.state_dict(), str(path + '/' + 'best.pth'))
            print('New Best Checkpoint saved!')
            print('###############################################################################')

        # save checkpoint at the end of each epoch
        torch.save(model.state_dict(), str(path + '/' + 'epoch_{}.pth'.format(epoch + 1)))
        print(str('checkpoint of epoch {} saved!'.format(epoch + 1)))
        print('###############################################################################')


def test_chaoyang(model, data_loader, epoch, device):
    # evaluation function for Chaoyang dataset
    model.eval()
    prob_all = []
    label_all = []

    y_true = torch.tensor([]).to(device)
    y_score = torch.tensor([]).to(device)

    data_loader = data_loader

    for inputs, targets in tqdm(data_loader):
        with torch.no_grad():
            inputs = inputs.to(device=device)
            if len(list(targets.size())) > 1:
                targets = targets.long()
            else:
                targets = targets.long().unsqueeze(1)
            outputs = model(inputs)

            # one_hot transformation
            one_hot = torch.zeros(outputs.shape[0], outputs.shape[1]).scatter_(1, targets, 1)

            # concat the prediction results from each batch
            prob_all.extend(outputs.cpu().numpy())
            label_all.extend(one_hot.numpy())

            y_true = torch.cat((y_true, targets.to(device)), 0)
            y_score = torch.cat((y_score, outputs), 0)

    # calculation of accuracy and AUC at the end of epoch
    pred_y = torch.max(y_score, 1)[1].data.unsqueeze(1)
    accuracy = sum(pred_y == y_true) / y_true.size(0)
    auc = roc_auc_score(label_all, prob_all)
    print('epoch:{}, AUC:{:.4f}, accuracy:{:.4f}'.format(epoch, auc, accuracy.item()))

    return accuracy, auc


def test_MedMNIST(split, model, data_loader, data_flag, device):
    # evaluation function for MedMNIST dataset
    model.eval()
    y_true = torch.tensor([]).to(device=device)
    y_score = torch.tensor([]).to(device=device)

    data_loader = data_loader

    with torch.no_grad():
        for inputs, targets in data_loader:
            inputs = inputs.to(device=device)
            targets = targets.to(device=device)
            outputs = model(inputs)

            targets = targets.squeeze().long()
            outputs = outputs.softmax(dim=-1)
            targets = targets.float().resize_(len(targets), 1)

            y_true = torch.cat((y_true, targets), 0)
            y_score = torch.cat((y_score, outputs), 0)

        # calculation of accuracy and AUC at the end of epoch
        y_true = y_true.cpu().numpy()
        y_score = y_score.detach().cpu().numpy()

        # evaluator adopted from https://github.com/MedMNIST/MedMNIST
        evaluator = Evaluator(data_flag, split)
        metrics = evaluator.evaluate(y_score)

        print('%s performance, AUC: %.4f, accuracy:%.4f' % (split, *metrics))

        return metrics


def get_args():
    # args function
    parser = argparse.ArgumentParser(description='Explore different attention on Medical Image Classification')
    parser.add_argument('--epochs', metavar='E', type=int, default=30, help='Number of epochs')
    parser.add_argument('--batch-size', dest='batch_size', metavar='B', type=int, default=64, help='Batch size')
    parser.add_argument('--learning-rate', metavar='LR', type=float, default=1e-5, help='Learning rate', dest='lr')
    parser.add_argument('--att', type=str, default=None, help='Choice of attention mechanism')
    parser.add_argument('--dataset', type=str, default='MedMNIST', help='Choice of dataset')
    parser.add_argument('--pretrained', action='store_true', default=False, help='Use pretrained checkpoints, only '
                                                                                 'applicable to Chaoyang dataset')

    return parser.parse_args()


if __name__ == '__main__':
    args = get_args()
    train(args)
