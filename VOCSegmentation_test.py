# If True, returns a model pre-trained on COCO train2017 which contains the same classes as Pascal VOC
# 1. VOC, COCO 데이터 확인
# 2. model에 넣어서 성능 확인
# torchvision.datasets.VOCSegmentation -> pl_bolts.datamodules.VOCDetectionDataModule

import torchvision.models.segmentation
import torchvision.datasets
from datamodules import VOCSegmentationDataModule
from datamodules.vocdetection_datamodule import CLASSES

# CLASSES[15] == 'person'
def main():
    dm = VOCSegmentationDataModule(data_dir="./data")
    dm.setup()
    dl_train = dm.train_dataloader()

    for batch in dl_train:
        x, y = batch
        break
    pass


if __name__ == "__main__":
    main()
