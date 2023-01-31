from enum import Enum, auto

from torchsummary import summary

from . import model_tester
from todels.residual.resnextblock import ResnextBlockA, ResnextBlockB


BLOCK_INPUT_SIZE = (64,int(128/2),int(128/2))
C = 32

class BlockTypes(Enum):
    Downsample_ResnextBlockA = auto(),
    NoDownsample_ResnextBlockA = auto(),
    Downsample_ResnextBlockB = auto(),
    NoDownsample_ResnextBlockB = auto(),
    

def create_blocks(block_type: BlockTypes):
    if block_type == BlockTypes.Downsample_ResnextBlockA:
        return ResnextBlockA(out_channels_convs=64, C=C, stride=2, downsample=True)
    elif block_type == BlockTypes.NoDownsample_ResnextBlockA:
        return ResnextBlockA(out_channels_convs=64, C=C, stride=1, downsample=False)
    elif block_type == BlockTypes.Downsample_ResnextBlockB:
        return ResnextBlockB(out_channels_convs=64, C=C, stride=2, downsample=True)
    elif block_type == BlockTypes.NoDownsample_ResnextBlockB:
        return ResnextBlockB(out_channels_convs=64, C=C, stride=1, downsample=False)
    else:
        raise NotImplementedError(f"Block type with id {block_type} is not implemented yet!")
    
@model_tester
def test_ResnextBlockA_with_downsample():
    model = create_blocks(BlockTypes.Downsample_ResnextBlockA)
    summary(model, BLOCK_INPUT_SIZE)

@model_tester
def test_ResnextBlockA_without_downsample():
    model = create_blocks(BlockTypes.NoDownsample_ResnextBlockA)
    summary(model, BLOCK_INPUT_SIZE)

@model_tester
def test_ResnextBlockB_with_downsample():
    model = create_blocks(BlockTypes.Downsample_ResnextBlockB)
    summary(model, BLOCK_INPUT_SIZE)

@model_tester
def test_ResnextBlockB_without_downsample():
    model = create_blocks(BlockTypes.NoDownsample_ResnextBlockB)
    summary(model, BLOCK_INPUT_SIZE)
