import argparse
import os

import torch
import torch.optim as optim
import numpy as np

from darknet import DarkNet
from img_loader import Data_loader
from torch.utils.tensorboard import SummaryWriter
from utils import load_classes, predict, evaluate
import os 
os.environ["CUDA_VISIBLE_DEVICES"] = "2"


parser = argparse.ArgumentParser(description='YOLO-v3 Train')
parser.add_argument("--epoch", type=int, default=50)
parser.add_argument("--batch_size", type=int, default=16)
parser.add_argument('--img_size', type=int, default=416)
parser.add_argument('--no_cuda', action='store_true')
parser.add_argument('--max_objects', type=int, default=50)
parser.add_argument("--confidence", type=float, default=0.5)
parser.add_argument("--nms_conf", type=float, default=0.45)
parser.add_argument("--pretrain",type=bool,default=True)


def train():

    args = parser.parse_args()
    use_cuda = torch.cuda.is_available() and not args.no_cuda

    classes = load_classes()
    num_classes = len(classes)

    model = DarkNet(use_cuda, num_classes)
    if use_cuda:
        model = model.cuda()
    optimizer = torch.optim.SGD(model.parameters(),lr = 0.001,momentum=0.9,weight_decay=5e-04)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer,20,0.1)
    if args.pretrain:
        param = (torch.load('coco_weights.pt'))['model']
        for k in list(param.keys()):
            if k.find('pred')>=0 and k.find('6')>=0:
                del param[k]
        model.load_state_dict(param,strict=False)
    
    training_data = Data_loader(
        "/home/leijingshi/data/VOCdevkit/VOC2007/labels/train/",
        "/home/leijingshi/data/VOCdevkit/VOC2007/JPEGImages/",
        img_size=args.img_size,
        max_objects=args.max_objects,
        batch_size=args.batch_size,
        is_cuda=use_cuda)

    test_data = Data_loader(
        "/home/leijingshi/data/VOCdevkit/VOC2007/labels/test/",
        "/home/leijingshi/data/VOCdevkit/VOC2007/JPEGImages/",
        img_size=args.img_size,
        max_objects=args.max_objects,
        batch_size=args.batch_size,
        is_cuda=use_cuda)
        
    writer = SummaryWriter("tensorboard/YOLO")
    iteration = 0
    for epoch in range(args.epoch):

        model.train()
        for batch_i, (imgs, labels) in enumerate(training_data):
            iteration += 1
            optimizer.zero_grad()
            loss, gather_losses = model(imgs, labels)
            loss.backward()
            optimizer.step()
            writer.add_scalar("train_loss", loss, iteration)

            print(f"""[Epoch {epoch+1}/{args.epoch},Batch {batch_i+1}/{training_data.stop_step}] [Losses: x {gather_losses["x"]:.5f}, y {gather_losses["y"]:.5f}, w {gather_losses["w"]:.5f}, h { gather_losses["h"]:.5f}, conf {gather_losses["conf"]:.5f}, cls {gather_losses["cls"]:.5f}, total {loss.item():.5f}, recall: {gather_losses["recall"]:.5f}, precision: {gather_losses["precision"]:.5f}]""")
        
        scheduler.step()
        mAP,l,mIoU = test(test_data,args.nms_conf,args.confidence,model,num_classes,args.img_size,epoch) 
        writer.add_scalar("mAP", mAP, epoch)
        writer.add_scalar("l", l, epoch)
        writer.add_scalar("mIoU", mIoU, epoch)
        
    torch.save(model.cpu().state_dict(),"/home/leijingshi/DL/mid/detect/yolo-v3/weights.pt")
    writer.close()

def test(_data, nms_conf, confidence,model,num_classes,img_size,epoch):
    all_detections = []
    all_annotations = []
    l = []
    
    model.eval()
    for imgs, labels in _data:
        with torch.no_grad():
            prediction = model(imgs)
            outputs = predict(prediction, nms_conf, confidence)
            l.append((model(imgs,labels)[0]).item())
            
        labels = labels.cpu()
        for output, annotations in zip(outputs, labels):
            all_detections.append([np.array([])
                                   for _ in range(num_classes)])
            if output is not None:
                pred_boxes = output[:, :5].cpu().numpy()
                scores = output[:, 4].cpu().numpy()
                pred_labels = output[:, -1].cpu().numpy()

                sort_i = np.argsort(scores)
                pred_labels = pred_labels[sort_i]
                pred_boxes = pred_boxes[sort_i]

                for label in range(num_classes):
                    all_detections[-1][label] = pred_boxes[pred_labels == label]

            all_annotations.append([np.array([])
                                    for _ in range(num_classes)])

            if any(annotations[:, -1] > 0):
                annotation_labels = annotations[annotations[:, -1]
                                                > 0, 0].numpy()
                _annotation_boxes = annotations[annotations[:, -1] > 0, 1:]

                annotation_boxes = np.empty_like(_annotation_boxes)
                annotation_boxes[:, 0] = _annotation_boxes[:,
                                                           0] - _annotation_boxes[:, 2] / 2
                annotation_boxes[:, 1] = _annotation_boxes[:,
                                                           1] - _annotation_boxes[:, 3] / 2
                annotation_boxes[:, 2] = _annotation_boxes[:,
                                                           0] + _annotation_boxes[:, 2] / 2
                annotation_boxes[:, 3] = _annotation_boxes[:,
                                                           1] + _annotation_boxes[:, 3] / 2
                annotation_boxes *= img_size

                for label in range(num_classes):
                    all_annotations[-1][label] = annotation_boxes[annotation_labels == label, :]

    average_precisions, iou = evaluate(
        num_classes, all_detections, all_annotations)

    print(f"""{"-"*40}evaluation.{epoch}{"-"*40}""")
    for c, ap in average_precisions.items():
        print(f"Class '{c}' - AP: {ap}")

    mAP = np.mean(list(average_precisions.values()))
    mIoU = np.mean(iou)
    print(f"mAP: {mAP}")
    print(f"""{"-"*40}end{"-"*40}""")
    model.train()
    
    return mAP,np.mean(l),mIoU

if __name__ == "__main__":
    train()
