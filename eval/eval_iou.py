# Code to calculate IoU (mean and per-class) in a dataset
# Nov 2017
# Eduardo Romera
#######################

import numpy as np
import torch
import torch.nn.functional as F
import os
import importlib
import time

from PIL import Image
from argparse import ArgumentParser

from torch.autograd import Variable
from torch.utils.data import DataLoader
from torchvision.transforms import Compose, CenterCrop, Normalize, Resize
from torchvision.transforms import ToTensor, ToPILImage

from dataset import cityscapes
from erfnet2 import ERFNet as ERFNet_original
from erfnet import ERFNet
from bisenetv2 import BiSeNetv2
from enet import ENet
from transform import Relabel, ToLabel, Colorize
from iouEval import iouEval, getColorEntry

NUM_CHANNELS = 3
NUM_CLASSES = 20
NUM_CLASSES_TASK4 = 19

image_transform = ToPILImage()
input_transform_cityscapes = Compose([
    Resize(512, Image.BILINEAR),
    ToTensor(),
])
target_transform_cityscapes = Compose([
    Resize(512, Image.NEAREST),
    ToLabel(),
    Relabel(255, 19),   #ignore label to 19
])

def main(args):

    modelpath = args.loadDir + args.loadModel
    weightspath = args.loadDir + args.loadWeights

    print ("Loading model: " + modelpath)
    print ("Loading weights: " + weightspath)

    # input validation
    assert (args.model in ["Erfnet", "BisenetV2", "Enet"]), "Invalid model"

    if args.task == 4:
        num_classes = NUM_CLASSES_TASK4
    else:
        num_classes = NUM_CLASSES

    # model management
    if args.model == "Erfnet":
        if args.task == 2:
            model = ERFNet_original(num_classes)
        else:
            model = ERFNet(num_classes)
    elif args.model == "BisenetV2":
        model = BiSeNetv2(num_class=num_classes)
    elif args.model == "Enet":
        model = ENet(num_class=num_classes)
    else:
        raise ValueError("Invalid model")

    #model = torch.nn.DataParallel(model)
    if (not args.cpu):
        model = torch.nn.DataParallel(model).cuda()

    def load_my_state_dict(model, state_dict):  #custom function to load model when not all dict elements
        own_state = model.state_dict()
        for name, param in state_dict.items():
            if name not in own_state:
                if name.startswith("module."):
                    own_state[name.split("module.")[-1]].copy_(param)
                else:
                    print(name, " not loaded")
                    continue
            else:
                own_state[name].copy_(param)
        return model

    def task4_load_my_state_dict(model, state_dict):
        own_state = model.state_dict()
        for name, param in state_dict["state_dict"].items():
            own_state["module." + name].copy_(param)
        return model

    if args.task == 4 or args.task == 3:
        model = task4_load_my_state_dict(model, torch.load(weightspath, map_location=lambda storage, loc: storage))
    else:
        model = load_my_state_dict(model, torch.load(weightspath, map_location=lambda storage, loc: storage))

    print ("Model and weights LOADED successfully")

    model.eval()

    if(not os.path.exists(args.datadir)):
        print ("Error: datadir could not be loaded")

    ignore_index = 19   # we want to ignore label 19

    city = cityscapes(args.datadir, input_transform_cityscapes, target_transform_cityscapes, subset=args.subset)
    loader = DataLoader(city, num_workers=args.num_workers, batch_size=args.batch_size, shuffle=False)
    iouEvalVal = iouEval(NUM_CLASSES, ignoreIndex=ignore_index)

    start = time.time()

    for step, (images, labels, filename, filenameGt) in enumerate(loader):
        if (not args.cpu):
            images = images.cuda()
            labels = labels.cuda()

        inputs = Variable(images)
        with torch.no_grad():
            outputs = model(inputs)

        iouEvalVal.addBatch(outputs.max(1)[1].unsqueeze(1).data, labels)

        filenameSave = filename[0].split("leftImg8bit/")[1] 

        if args.verbose:
            print (step, filenameSave)

    iouVal, iou_classes = iouEvalVal.getIoU()

    iou_classes_str = []
    for i in range(iou_classes.size(0)):
        iouStr = getColorEntry(iou_classes[i])+'{:0.4f}'.format(iou_classes[i]*100) + '\033[0m'
        iou_classes_str.append(iouStr)

    print("---------------------------------------")
    print("Took ", time.time()-start, "seconds")
    print("=======================================")
    #print("TOTAL IOU: ", iou * 100, "%")
    print("Per-Class IoU:")
    print(iou_classes_str[0], "Road")
    print(iou_classes_str[1], "sidewalk")
    print(iou_classes_str[2], "building")
    print(iou_classes_str[3], "wall")
    print(iou_classes_str[4], "fence")
    print(iou_classes_str[5], "pole")
    print(iou_classes_str[6], "traffic light")
    print(iou_classes_str[7], "traffic sign")
    print(iou_classes_str[8], "vegetation")
    print(iou_classes_str[9], "terrain")
    print(iou_classes_str[10], "sky")
    print(iou_classes_str[11], "person")
    print(iou_classes_str[12], "rider")
    print(iou_classes_str[13], "car")
    print(iou_classes_str[14], "truck")
    print(iou_classes_str[15], "bus")
    print(iou_classes_str[16], "train")
    print(iou_classes_str[17], "motorcycle")
    print(iou_classes_str[18], "bicycle")
    print("=======================================")
    iouStr = getColorEntry(iouVal)+'{:0.4f}'.format(iouVal*100) + '\033[0m'
    print ("MEAN IoU: ", iouStr, "%")

if __name__ == '__main__':
    parser = ArgumentParser()

    parser.add_argument('--state')

    parser.add_argument('--loadDir',default="../")
    parser.add_argument('--loadWeights', default="trained_models/erfnet_pretrained.pth")
    parser.add_argument('--loadModel', default="eval/erfnet2.py")
    parser.add_argument('--subset', default="val")  #can be val or train (must have labels)
    parser.add_argument('--datadir', default="/home/shyam/ViT-Adapter/segmentation/data/cityscapes/")
    parser.add_argument('--num-workers', type=int, default=4)
    parser.add_argument('--batch-size', type=int, default=1)
    parser.add_argument('--cpu', action='store_true')
    parser.add_argument('--model', default="Erfnet")            # Erfnet, BisenetV2, Enet
    parser.add_argument('--task', type=int, default=2)
    parser.add_argument('--verbose', action='store_true')

    main(parser.parse_args())