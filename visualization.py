# Visualization function of attention maps on MedMNIST v2 and Chaoyang dataset
# Implemented by Yuchuan Li

from visualizer import get_local
get_local.activate()

import argparse
import torch
import torchvision.transforms as transforms

import cv2
import matplotlib.pyplot as plt

from torch.utils.data import DataLoader
from tqdm import tqdm

from dataset.Chaoyang import CHAOYANG
import medmnist
from medmnist import INFO

from model.models import resnet18_attention, resnet34_attention, resnet50_attention, SimpleNet


def visualize(args):
    index = args.index
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # dataloader preparation for MedMNIST and Chaoyang dataset respectively
    if args.dataset == 'MedMNIST':
        # data_flag for sub-dataset selection
        data_flag = 'dermamnist'
        info = INFO[data_flag]
        num_class = len(info['label'])
        DataClass = getattr(medmnist, info['python_class'])

        test_dataset = DataClass(split='test', transform=transforms.Compose(
            [transforms.ToTensor()]), download=True)

        test_loader = torch.utils.data.DataLoader(dataset=test_dataset, batch_size=1, shuffle=False)

    if args.dataset == 'Chaoyang':
        test_dataset = CHAOYANG(root="../dataset/Chaoyang/",
                                json_name="test.json",
                                train=False,
                                transform=transforms.Compose([transforms.ToTensor()])
                                )

        test_loader = torch.utils.data.DataLoader(dataset=test_dataset,
                                                  batch_size=1,
                                                  num_workers=0,
                                                  drop_last=False,
                                                  shuffle=False)

        num_class = 4

    # setup model for MedMNIST and Chaoyang dataset respectively
    if args.dataset == 'MedMNIST':
        model = SimpleNet(num_classes=num_class, att_choice=args.att)
    else:
        model = resnet18_attention(pretrained=False, n_classes=num_class, att_choice=args.att)
    model.load_state_dict(torch.load(args.load, map_location=device))
    model.to(device)
    model.eval()

    # flag of the desired index of visualization
    flag = 1
    for inputs, targets in tqdm(test_loader):
        with torch.no_grad():
            if flag == index:
                inputs = inputs.to(device=device)
                _ = model(inputs)

                # extract the original image from tensor format
                image = inputs.squeeze().cpu().numpy().transpose(1, 2, 0)

                # extract the cache of attention activation
                # adopted from https://github.com/luo3300612/Visualizer
                cache = get_local.cache
                if args.dataset == 'MedMNIST':
                    # selection of attention map and mask based on network
                    att_map = cache['SimpleNet.forward']
                    mask1 = att_map[0][0, 0, :, :].squeeze()
                else:
                    # selection of attention map and mask based on network
                    att_map = cache['BasicBlock.forward']
                    mask1 = att_map[6][0, 0, :, :].squeeze()

                # preparation of plot visualization
                fig, ax = plt.subplots(1, 3, figsize=(10, 7))
                fig.tight_layout()

                # show the original input image
                ax[0].imshow(image, alpha=1)
                ax[0].axis('off')

                # normalize the attention map
                mask1 = (mask1-mask1.min())/(mask1.max()-mask1.min())
                # resize for visualization
                mask = cv2.resize(mask1, (image.shape[0], image.shape[1]))
                # format change for uint8
                normed_mask = mask / mask.max()
                normed_mask = (normed_mask * 255).astype('uint8')
                ax[1].imshow(normed_mask, alpha=1, interpolation='nearest', cmap=plt.cm.jet)
                ax[1].axis('off')

                # show the overlay of attention map and original input image
                ax[2].imshow(image, alpha=1)
                ax[2].axis('off')
                ax[2].imshow(normed_mask, alpha=0.4, interpolation='nearest', cmap=plt.cm.jet)
                ax[2].axis('off')
                plt.show()

                # break after the visualization
                break

            else:
                flag = flag + 1


def get_args():
    # args function
    parser = argparse.ArgumentParser(description='Visualization')
    parser.add_argument('--load', type=str, default=None, help='Load model from a .pth file')
    parser.add_argument('--att', type=str, default=None, help='Choice of attention mechanism')
    parser.add_argument('--dataset', type=str, default='MedMNIST', help='Choice of the dataset')
    parser.add_argument('--index', type=int, default=1, help='Choice of the index of input image')

    return parser.parse_args()


if __name__ == '__main__':
    args = get_args()
    visualize(args)
