from models import VGG16
import torchsummary
import torch

# model = VGG16(base_dim=64)
# torchsummary.summary(model, (3, 32, 32))


# print (f"PyTorch version:{torch.__version__}") # 1.12.1 이상
# print(f"MPS 장치를 지원하도록 build 되었는지: {torch.backends.mps.is_built()}") # True 여야 합니다.
# print(f"MPS 장치가 사용 가능한지: {torch.backends.mps.is_available()}") # True 여야 합니다.
