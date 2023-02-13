from torchsummary import summary 

from todels.alexnet import Alexnet, LigthAlexnet
from . import model_tester, MODEL_INPUT_SIZE


@model_tester
def test_alexnet():
    model = Alexnet(in_channels=3, num_classes=10)
    summary(model, MODEL_INPUT_SIZE)


@model_tester
def test_alexnet():
    model = LigthAlexnet(in_channels=3, num_classes=10)
    summary(model, MODEL_INPUT_SIZE)
