from pytorch_lightning import Trainer, seed_everything

from datamodules import OxfordIIITPetDataModule
from models import DeepLabV3_resnet50


def cli_main():
    seed_everything(1234)

    dm = OxfordIIITPetDataModule(data_dir="./data", num_workers=2, batch_size=4)
    model = DeepLabV3_resnet50()

    trainer = Trainer(max_epochs=10, gpus=-1)
    trainer.fit(model, datamodule=dm)
    pass


if __name__ == "__main__":
    cli_main()
