from enum import Enum, auto

from torchsummary import summary

from . import model_tester
from todels.residual.resnet import *

MODULE_INPUT_SIZE = (64,int(128/2),int(128/2))
NUM_CLASSES = 10


class ModuleTypes(Enum):
    Resnet18 = auto(),
    Resnet34 = auto(),
    Resnet50 = auto(),
    Resnet101 = auto(),
    Resnet152 = auto(),

def create_modules(module_type: ModuleTypes):
    if module_type == ModuleTypes.Resnet18:
        return Resnet18(NUM_CLASSES)
    elif module_type == ModuleTypes.Resnet34:
        return Resnet34(NUM_CLASSES)
    elif module_type == ModuleTypes.Resnet50:
        return Resnet50(NUM_CLASSES)
    elif module_type == ModuleTypes.Resnet101:
        return Resnet101(NUM_CLASSES)
    elif module_type == ModuleTypes.Resnet152:
        return Resnet152(NUM_CLASSES)
    else:
        raise NotImplementedError(f"Module type with id {module_type} is not implemented yet!")
    
@model_tester
def test_resnet18():
    model = create_modules(ModuleTypes.Resnet18)
    summary(model, MODULE_INPUT_SIZE)

@model_tester
def test_resnet34():
    model = create_modules(ModuleTypes.Resnet34)
    summary(model, MODULE_INPUT_SIZE)

@model_tester
def test_resnet50():
    model = create_modules(ModuleTypes.Resnet50)
    summary(model, MODULE_INPUT_SIZE)

@model_tester
def test_resnet101():
    model = create_modules(ModuleTypes.Resnet101)
    summary(model, MODULE_INPUT_SIZE)

@model_tester
def test_resnet152():
    model = create_modules(ModuleTypes.Resnet152)
    summary(model, MODULE_INPUT_SIZE)
