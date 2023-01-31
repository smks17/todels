from enum import Enum, auto

from torchsummary import summary

from . import model_tester
from todels.residual.resnext import *

MODULE_INPUT_SIZE = (64,int(128/2),int(128/2))
NUM_CLASSES = 10


class ModuleTypes(Enum):
    Resnext50 = auto(),
    Resnext101 = auto(),
    Resnext152 = auto(),

def create_modules(module_type: ModuleTypes):
    if module_type == ModuleTypes.Resnext50:
        return Resnext50(NUM_CLASSES)
    elif module_type == ModuleTypes.Resnext101:
        return Resnext101(NUM_CLASSES)
    elif module_type == ModuleTypes.Resnext152:
        return Resnext152(NUM_CLASSES)
    else:
        raise NotImplementedError(f"Module type with id {module_type} is not implemented yet!")

@model_tester
def test_resnext50():
    model = create_modules(ModuleTypes.Resnext50)
    summary(model, MODULE_INPUT_SIZE)

# TODO: Figure out is block architecture is True or not and then try to build model faster

# @model_tester
# def test_resnext101():
#     model = create_modules(ModuleTypes.Resnext101)
#     summary(model, MODULE_INPUT_SIZE)

# @model_tester
# def test_resnet152():
#     model = create_modules(ModuleTypes.Resnet152)
#     summary(model, MODULE_INPUT_SIZE)
