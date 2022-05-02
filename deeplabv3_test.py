# If True, returns a model pre-trained on COCO train2017 which contains the same classes as Pascal VOC
# 1. VOC, COCO 데이터 확인
# 2. model에 넣어서 성능 확인
# torchvision.datasets.VOCSegmentation -> pl_bolts.datamodules.VOCDetectionDataModule

import torchvision.models.segmentation
import torchvision.datasets
import pl_bolts.datamodules

classes = [
    "aeroplane",
    "bicycle",
    "bird",
    "boat",
    "bottle",
    "bus",
    "car",
    "cat",
    "chair",
    "cow",
    "diningtable",
    "dog",
    "horse",
    "motorbike",
    "person",
    "pottedplant",
    "sheep",
    "sofa",
    "train",
    "tvmonitor",
]


def main():
    ds = torchvision.datasets.VOCSegmentation(root="./data", download=False)
    model = torchvision.models.segmentation.deeplabv3_resnet50(pretrained=True)

    dm = pl_bolts.datamodules.VOCDetectionDataModule
    pass


if __name__ == "__main__":
    main()
