from enum import Enum, auto

from torchsummary import summary

from . import model_tester
from todels.residual.resnetblock import SimpleResnetBlock, BottleneckBlock


BLOCK_INPUT_SIZE = (64,int(128/2),int(128/2))

class BlockTypes(Enum):
    Downsample_SimpleResnetBlock = auto(),
    NoDownsample_SimpleResnetBlock = auto(),
    Downsample_Bottleneck = auto(),
    NoDownsample_Bottleneck = auto(),
    

def create_blocks(block_type: BlockTypes):
    if block_type == BlockTypes.Downsample_SimpleResnetBlock:
        return SimpleResnetBlock(out_channels_convs=[64,128], stride=(2,2), downsample=True)
    elif block_type == BlockTypes.NoDownsample_SimpleResnetBlock:
        return SimpleResnetBlock(out_channels_convs=[64,64], stride=(1,1), downsample=False)
    elif block_type == BlockTypes.Downsample_Bottleneck:
        return BottleneckBlock(out_channels_convs=64, stride=2, downsample=True)
    elif block_type == BlockTypes.NoDownsample_Bottleneck:
        return BottleneckBlock(out_channels_convs=64, stride=1, downsample=False)
    else:
        raise NotImplementedError(f"Block type with id {block_type} is not implemented yet!")
    
@model_tester
def test_SimpleResnetBlock_with_downsample():
    model = create_blocks(BlockTypes.Downsample_SimpleResnetBlock)
    summary(model, BLOCK_INPUT_SIZE)

@model_tester
def test_SimpleResnetBlock_without_downsample():
    model = create_blocks(BlockTypes.NoDownsample_SimpleResnetBlock)
    summary(model, BLOCK_INPUT_SIZE)

@model_tester
def test_Bottleneck_with_downsample():
    model = create_blocks(BlockTypes.Downsample_Bottleneck)
    summary(model, BLOCK_INPUT_SIZE)

@model_tester
def test_Bottleneck_without_downsample():
    model = create_blocks(BlockTypes.NoDownsample_Bottleneck)
    summary(model, BLOCK_INPUT_SIZE)
