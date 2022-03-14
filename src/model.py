from typing import Dict

import torch
from torch import Tensor
from torchvision import transforms
from torchvision.models import inception_v3
import torch.nn as nn
import torch.nn.functional as F

from src.imagenet_class_labels import id2label


def load_model() -> nn.Module:
    model = inception_v3(pretrained=True)
    model.eval()
    n_params = sum(p.numel() for p in model.parameters())
    print(f'{n_params:,} parameters') # 27,161,264 parameters
    return model

def predict(model: nn.Module, image) -> Dict:

    # resize and normalize input pixel ranges
    img = preprocess(image)

    output = model.forward(img)
    class_idx = torch.max(output.data, 1)[1][0].item()
    label = id2label[class_idx]
    output_probs = F.softmax(output, dim=1)
    confidence =  round(torch.max(output_probs.data, 1)[0][0].item(), 4)

    return {
        'id': class_idx,
        'label': label,
        'confidence': confidence,
    }


def preprocess(img) -> Tensor:
    """
    Inception V3 model from pytorch expects input images with pixel values between -1 and 1
    and dimensions 299 x 299
    """
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]

    preprocess_fn = transforms.Compose([
        transforms.Resize((299,299)),
        transforms.ToTensor(),
        transforms.Normalize(mean, std)
    ])
    image_tensor = preprocess_fn(img)

    # add batch dimension: C x H x W ==> B x C x H x W
    image_tensor = image_tensor.unsqueeze(0)

    return image_tensor

def inverse_preprocess(x: Tensor):
    """"""
    t = x.squeeze(0)

    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]
    t = t.mul(torch.FloatTensor(std).view(3,1,1)).add(torch.FloatTensor(mean).view(3,1,1)) #.numpy()

    im = transforms.ToPILImage()(t) #.convert("RGB") #.Resize((299, 299))

    return im
