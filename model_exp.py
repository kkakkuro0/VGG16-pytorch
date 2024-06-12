from models import VGG16
import torchsummary

model = VGG16(base_dim=64)
torchsummary.summary(model, (3, 32, 32))
