# Copyright (c) OpenMMLab. All rights reserved.
import os
import glob
import torch
import random
from PIL import Image
import numpy as np
from erfnet import ERFNet
from erfnet2 import ERFNet as ERFNet_original
from bisenetv2 import BiSeNetv2
from enet import ENet
import torch.nn.functional as F
from argparse import ArgumentParser
from ood_metrics import fpr_at_95_tpr, calc_metrics, plot_roc, plot_pr, plot_barcode
from sklearn.metrics import roc_auc_score, roc_curve, auc, precision_recall_curve, average_precision_score

seed = 42

# general reproducibility
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)

NUM_CHANNELS = 3
NUM_CLASSES_TASK4 = 19
NUM_CLASSES = 20
# gpu training specific
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = True


def main():
    parser = ArgumentParser()
    parser.add_argument(
        "--input",
        default="/home/shyam/Mask2Former/unk-eval/RoadObsticle21/images/*.webp",
        nargs="+",
        help="A list of space separated input images; "
             "or a single glob pattern such as 'directory/*.jpg'",
    )
    parser.add_argument('--loadDir', default="../")
    parser.add_argument('--loadWeights', default="trained_models/erfnet_pretrained.pth")
    parser.add_argument('--loadModel', default="eval/erfnet_original.py")
    parser.add_argument('--subset', default="val")  # can be val or train (must have labels)
    parser.add_argument('--datadir', default="/home/shyam/ViT-Adapter/segmentation/data/cityscapes/")
    parser.add_argument('--num-workers', type=int, default=4)
    parser.add_argument('--batch-size', type=int, default=1)
    parser.add_argument('--cpu', action='store_true')
    parser.add_argument('--isColab', action='store_true')
    parser.add_argument('--method', default="MaxLogit")  # MaxLogit, MSP, MaxEntropy
    parser.add_argument('--withT', type=float, default=None)  # Temperature scaling
    parser.add_argument('--model', default="Erfnet")  # Erfnet, BisenetV2, Enet
    parser.add_argument('--task', type=int, default=2)

    args = parser.parse_args()
    anomaly_score_list = []
    ood_gts_list = []

    print("Task: ", args.task)
    print("Method: ", args.method)
    print("Dataset: ", args.input[0].replace("\\", "/").split("/")[
        args.input[0].replace("\\", "/").split("/").index("Validation_Dataset") + 1])

    # input validation
    assert (args.task in [2, 3, 4]), "Invalid task"
    assert (args.method in ["MaxLogit", "MSP", "MaxEntropy"]), "Invalid method"
    assert (args.model in ["Erfnet", "BisenetV2", "Enet"]), "Invalid model"
    if args.withT != None:
        print("Temperature scaling with T = ", args.withT)
        assert (args.method == "MSP"), "Invalid method (withT is only for MSP)"
        assert (args.withT > 0.0), "Invalid temperature"

    if not os.path.exists('results.txt'):
        open('results.txt', 'w').close()
    file = open('results.txt', 'a')

    modelpath = args.loadDir + args.loadModel
    weightspath = args.loadDir + args.loadWeights

    print("Loading model: " + modelpath)
    print("Loading weights: " + weightspath)

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
        model = ENet(num_classes)
    else:
        raise ValueError("Invalid model")

    if (not args.cpu):
        model = torch.nn.DataParallel(model).cuda()

    def load_my_state_dict(model, state_dict):  # custom function to load model when not all dict elements
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
    # print(torch.load(weightspath).keys())
    # model.load_state_dict(torch.load(weightspath)["state_dict"])
    print("Model and weights LOADED successfully")
    model.eval()

    # image extension management
    if (args.isColab):
        # Colab fs
        if "RoadObsticle21" in args.input[0]:
            extension = "webp"
        elif ("fs_static" in args.input[0]
              or ("RoadAnomaly" in args.input[0] and "RoadAnomaly21" not in args.input[0])):
            extension = "jpg"
        else:
            extension = "png"
        input_path = args.input[0] + '/*.' + extension
    else:
        input_path = args.input[0]

    for path in glob.glob(os.path.expanduser(str(input_path))):
        print(path)
        images = torch.from_numpy(np.array(Image.open(path).convert('RGB'))).unsqueeze(0).float()
        images = images.permute(0, 3, 1, 2)
        with torch.no_grad():
            result = model(images)

        # tasks
        if args.task == 2:
            # methods
            if args.method == "MaxLogit":
                anomaly_result = - torch.max(result.squeeze(0), dim=0)[0].cpu().numpy()
                # anomaly_result = - np.max(result.squeeze(0).data.cpu().numpy(), axis=0)
            elif args.method == "MSP":
                if args.withT:
                    # with Temperature scaling
                    soft_probs_withT = torch.nn.functional.softmax(result.squeeze(0) / args.withT, dim=0)
                    anomaly_result = 1.0 - torch.max(soft_probs_withT, dim=0)[0].cpu().numpy()
                else:
                    soft_probs = torch.nn.functional.softmax(result.squeeze(0), dim=0)
                    anomaly_result = 1.0 - torch.max(soft_probs, dim=0)[0].cpu().numpy()
            elif args.method == "MaxEntropy":
                soft_probs = F.softmax(result.squeeze(0), dim=0)
                log_soft_probs = F.log_softmax(result.squeeze(0), dim=0)
                anomaly_result = - torch.div(torch.sum(soft_probs * log_soft_probs, dim=0),
                                             torch.log(torch.tensor(result.shape[1]))).cpu().numpy()
                # anomaly_result = - torch.div(torch.sum(soft_probs * log_soft_probs, dim=0), np.log(NUM_CLASSES)).cpu().numpy()
        elif args.task == 3:
            anomaly_result = result.squeeze(0).data.cpu().numpy()[19, :, :]
        elif args.task == 4:
            anomaly_result = - torch.max(result.squeeze(0), dim=0)[0].cpu().numpy()

        pathGT = path.replace("images", "labels_masks")
        if "RoadObsticle21" in pathGT:
            pathGT = pathGT.replace("webp", "png")
        if "fs_static" in pathGT:
            pathGT = pathGT.replace("jpg", "png")
        if "RoadAnomaly" in pathGT:
            pathGT = pathGT.replace("jpg", "png")

        mask = Image.open(pathGT)
        ood_gts = np.array(mask)

        if "RoadAnomaly" in pathGT:
            ood_gts = np.where((ood_gts == 2), 1, ood_gts)
        if "LostAndFound" in pathGT:
            ood_gts = np.where((ood_gts == 0), 255, ood_gts)
            ood_gts = np.where((ood_gts == 1), 0, ood_gts)
            ood_gts = np.where((ood_gts > 1) & (ood_gts < 201), 1, ood_gts)

        if "Streethazard" in pathGT:
            ood_gts = np.where((ood_gts == 14), 255, ood_gts)
            ood_gts = np.where((ood_gts < 20), 0, ood_gts)
            ood_gts = np.where((ood_gts == 255), 1, ood_gts)

        if 1 not in np.unique(ood_gts):
            continue
        else:
            ood_gts_list.append(ood_gts)
            anomaly_score_list.append(anomaly_result)
        del result, anomaly_result, ood_gts, mask
        torch.cuda.empty_cache()

    file.write("\n")

    ood_gts = np.array(ood_gts_list)
    anomaly_scores = np.array(anomaly_score_list)

    ood_mask = (ood_gts == 1)
    ind_mask = (ood_gts == 0)

    ood_out = anomaly_scores[ood_mask]
    ind_out = anomaly_scores[ind_mask]

    ood_label = np.ones(len(ood_out))
    ind_label = np.zeros(len(ind_out))

    val_out = np.concatenate((ind_out, ood_out))
    val_label = np.concatenate((ind_label, ood_label))

    prc_auc = average_precision_score(val_label, val_out)
    fpr = fpr_at_95_tpr(val_out, val_label)

    print(f'AUPRC score: {prc_auc * 100.0}')
    print(f'FPR@TPR95: {fpr * 100.0}')

    file.write(('    AUPRC score:' + str(prc_auc * 100.0) + '   FPR@TPR95:' + str(fpr* 100.0)))
    file.close()


if __name__ == '__main__':
    main()
