#%%
import autorootcwd

import torch
from torchinfo import summary
from src.utils.registry import DATASET_REGISTRY
from src.data.xca_flow import XCAFlow_DataModule
from src.archs.unet.unet import UNetModel
from src.archs.components.conditional_flow import SchrodingerBridgeConditionalFlowMatcher

use_cuda = torch.cuda.is_available()
device = torch.device("cuda" if use_cuda else "cpu")
batch_size = 16
n_epochs = 10

data_module = DATASET_REGISTRY.get("xca_flow")()
data_module.setup()
train_dataloader = data_module.train_dataloader()
val_dataloader = data_module.val_dataloader()
test_dataloader = data_module.test_dataloader()

model = UNetModel(
    image_size=224,
    in_channels=1,
    model_channels=32,
    out_channels=1,
    num_res_blocks=3,
    attention_resolutions=[16, 8],
    num_classes=None,
).to(device)

# UNetModel은 (t, x) 형태의 입력을 받으므로 input_data로 전달
x = torch.randn(1, 1, 224, 224).to(device)
t = torch.randint(0, 1000, (1,)).to(device)
print(summary(model, input_data=(t, x)))
#%%
