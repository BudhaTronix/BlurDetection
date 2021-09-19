import torch
from torch import nn
from torch.nn import functional as F
import pytorch_lightning as pl
import torchvision.models as models
import os
from pytorch_lightning.utilities.cli import LightningCLI
from light_dl import BDDataModule


class LitBlurDetection(pl.LightningModule):
    def __init__(self):
        super().__init__()

        model_ft = models.resnet101(pretrained=True)
        model_ft.conv1 = nn.Conv2d(1, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
        num_ftrs = model_ft.fc.in_features
        num_classes = 6
        model_ft.fc = nn.Linear(num_ftrs, num_classes)

        """backbone = models.resnet50(pretrained=True)
        backbone.conv1 = nn.Conv2d(1, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
        num_filters = backbone.fc.in_features
        layers = list(backbone.children())[:-1]
        self.feature_extractor = nn.Sequential(*layers)

        # use the pretrained model to classify
        num_target_classes = 6
        self.classifier = nn.Linear(num_filters, num_target_classes)"""

    def forward(self, x):
        self.feature_extractor.eval()
        with torch.no_grad():
            representations = self.feature_extractor(x).flatten(1)
        x = self.classifier(representations)

        return x

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-3)
        return optimizer

    def training_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x.double())
        loss = F.mse_loss(logits, y)
        self.log('train_loss', loss)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x.double())
        loss = F.mse_loss(logits, y)
        self.log('val_loss', loss)


# data
os.environ['HTTP_PROXY'] = 'http://proxy:3128/'
os.environ['HTTPS_PROXY'] = 'http://proxy:3128/'


def cli_main():
    cli = LightningCLI(
        LitBlurDetection, BDDataModule, seed_everything_default=1234, save_config_overwrite=True,
    )
    cli.trainer.fit(cli.model, datamodule=cli.datamodule, gpus=1)
    cli.trainer.test(ckpt_path="best")


if __name__ == "__main__":
    cli_main()
