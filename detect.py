import os
import torch
from PIL import Image
from torch import nn
from torchvision import models
import torchvision.transforms as transforms
i=0 #识别图片计数
test_pa="test"
#加载文件内容
names=os.listdir(test_pa)
name_class=['安妮海瑟薇', '邓超', '贾玲', '柳岩', '刘亦菲', '唐嫣', '泰勒斯威夫特', '王祖贤', '周冬雨', '周杰伦']

for name in names:
    print(name)
    i=i+1
    image_path=os.path.join(test_pa,name)
    image=Image.open(image_path)
    print(image)
    transform = transforms.Compose(
        [transforms.RandomResizedCrop(size=224),
         transforms.RandomHorizontalFlip(),
         transforms.ToTensor(),
         transforms.Normalize((0.5864, 0.5179, 0.5050),
                              (0.3562, 0.3491, 0.3462))])
    image=transform(image)
    print(image.shape)

    model_ft=models.resnet50(pretrained=True)      #需要使用训练时的相同模型
    in_features=model_ft.fc.in_features
    model_ft.fc=nn.Sequential(nn.Linear(in_features,36),
                              nn.Linear(36,10))     #此处也要与训练模型一致
    #加载图像识别算法
    model=torch.load("model/best_model.pth",map_location=torch.device("cpu")) #选择训练后得到的模型文件

    # print(model)
    image=torch.reshape(image,(1,3,224,224))      #修改待预测图片尺寸，需要与训练时一致
    model.eval()
    with torch.no_grad():
        output=model(image)


    print(output) #输出结果
    # print(int(output.argmax(1)))
    print("第{}张图片预测为：{}".format(i,name_class[int(output.argmax(1))]))#结果
