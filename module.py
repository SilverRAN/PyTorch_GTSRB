import pytorch_lightning as pl
import torch
from torchmetrics.functional.classification import multiclass_accuracy

from cifar10_models.densenet import densenet121, densenet161, densenet169
from cifar10_models.googlenet import googlenet
from cifar10_models.inception import inception_v3
from cifar10_models.mobilenetv2 import mobilenet_v2
from cifar10_models.resnet import resnet18, resnet34, resnet50
from cifar10_models.vgg import vgg11_bn, vgg13_bn, vgg16_bn, vgg19_bn
from schduler import WarmupCosineLR

all_classifiers = {
    "vgg11_bn": vgg11_bn(num_classes=43),
    "vgg13_bn": vgg13_bn(num_classes=43),
    "vgg16_bn": vgg16_bn(num_classes=43),
    "vgg19_bn": vgg19_bn(num_classes=43),
    "resnet18": resnet18(num_classes=43),
    "resnet34": resnet34(num_classes=43),
    "resnet50": resnet50(num_classes=43),
    "densenet121": densenet121(num_classes=43),
    "densenet161": densenet161(num_classes=43),
    "densenet169": densenet169(num_classes=43),
    "mobilenet_v2": mobilenet_v2(num_classes=43),
    "googlenet": googlenet(num_classes=43),
    "inception_v3": inception_v3(num_classes=43),
}


class GTSRBModule(pl.LightningModule):
    def __init__(self, hparams):
        super().__init__()
        self.save_hyperparameters(hparams)

        self.criterion = torch.nn.CrossEntropyLoss()

        self.model = all_classifiers[self.hparams.classifier]

    def forward(self, batch):
        images, labels = batch
        predictions = self.model(images)
        loss = self.criterion(predictions, labels)
        accuracy = multiclass_accuracy(
            predictions,
            labels,
            num_classes=predictions.shape[1],
            average="micro",
        )
        return loss, accuracy * 100

    def training_step(self, batch, batch_nb):
        loss, accuracy = self.forward(batch)
        self.log("loss/train", loss)
        self.log("acc/train", accuracy)
        return loss

    def validation_step(self, batch, batch_nb):
        loss, accuracy = self.forward(batch)
        self.log("loss/val", loss)
        self.log("acc/val", accuracy)

    def test_step(self, batch, batch_nb):
        loss, accuracy = self.forward(batch)
        self.log("acc/test", accuracy)

    def configure_optimizers(self):
        optimizer = torch.optim.SGD(
            self.model.parameters(),
            lr=self.hparams.learning_rate,
            weight_decay=self.hparams.weight_decay,
            momentum=0.9,
            nesterov=True,
        )
        total_steps = self.trainer.estimated_stepping_batches
        scheduler = {
            "scheduler": WarmupCosineLR(
                optimizer, warmup_epochs=total_steps * 0.3, max_epochs=total_steps
            ),
            "interval": "step",
            "name": "learning_rate",
        }
        return [optimizer], [scheduler]
