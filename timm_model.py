import timm
import torchsummary
from torch import nn
from torchvision import models
from model.sketch_model import SketchModel

#models=SketchModel("alexnet",99,pretrain=False)
print(timm.list_models("*resne*t*"))
model=timm.create_model('resnetaa50', pretrained=True).cuda()
#model=models.resnet50(pretrained=True).cuda()
# model1=timm.create_model('resnet101', pretrained=True).cuda()
#torchsummary.summary(model,input_size=(3,224,224))
#print(model)
# torchsummary.summary(model1,input_size=(3,224,224))
#print(model)