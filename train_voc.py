import torch
from pytorch_lightning import Trainer, seed_everything

from datamodules import VOCSegmentationDataModule
from datamodules.vocdetection_datamodule import CLASSES
from models import DeepLabV3_resnet50


def cli_main():
    seed_everything(1234)

    dm = VOCSegmentationDataModule(data_dir="./data", num_workers=2, batch_size=4)
    model = DeepLabV3_resnet50(num_classes=dm.num_classes)

    trainer = Trainer(max_epochs=10, gpus=-1)
    trainer.fit(model, datamodule=dm)

    model.to_onnx("./test_256.onnx", torch.randn(1, 3, 256, 256))
    pass


if __name__ == "__main__":
    cli_main()
