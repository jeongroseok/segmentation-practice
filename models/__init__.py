from argparse import ArgumentParser
from typing import Dict, Tuple

import torch
from pytorch_lightning import LightningModule
from torch import Tensor
from torch.nn import functional as F
from torchvision.models.segmentation import deeplabv3_resnet50
from torchvision.models.segmentation.deeplabv3 import DeepLabHead


class DeepLabV3_resnet50(LightningModule):
    def __init__(
        self,
        lr: float = 1e-4,
        pretrained: bool = True,
        num_classes: int = 3,
    ):
        super().__init__()

        self.num_classes = num_classes
        self.lr = lr

        model = deeplabv3_resnet50(pretrained=pretrained).train()

        self.backbone = model.backbone
        self.classifier = DeepLabHead(2048, num_classes)

        self.criterion = torch.nn.MSELoss(reduction="mean")

    def forward(self, x: Tensor) -> Dict[str, Tensor]:
        input_shape = x.shape[-2:]
        features = self.backbone(x)

        x = features["out"]
        x = self.classifier(x)
        x = F.interpolate(x, size=input_shape, mode="bilinear", align_corners=False)

        return x

    def training_step(self, batch: Tuple[Tensor, Tensor]):
        img, mask = batch
        mask.squeeze_(1)
        mask[mask > 21] = 0
        mask = F.one_hot(mask.long(), 22)[..., 1:].permute(0, 3, 1, 2).float()
        out = self(img)

        # plt.imshow(mask[0].argmax(0).cpu().detach().numpy())
        # plt.imshow(out[0].argmax(0).cpu().detach().numpy())
        
        loss_val = self.criterion(out, mask)
        log_dict = {"train_loss": loss_val}
        return {"loss": loss_val, "log": log_dict, "progress_bar": log_dict}

    def configure_optimizers(self):
        opt = torch.optim.Adam(self.parameters(), lr=self.lr)
        sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=10)
        return [opt], [sch]

    @staticmethod
    def add_model_specific_args(parent_parser):
        parser = ArgumentParser(parents=[parent_parser], add_help=False)
        parser.add_argument(
            "--lr", type=float, default=1e-4, help="adam: learning rate"
        )

        return parser
