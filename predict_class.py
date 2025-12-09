from torchvision import io as tvio
from torchvision import models
from torchvision.models import VGG19_BN_Weights
import torchinfo



input_image=tvio.decode_image('assets/IMG_3436.jpg')
print(type(input_image))
print(input_image.shape,input_image.dtype)
weights=VGG19_BN_Weights.DEFAULT
model=models.vgg19_bn(weights=weights)
print(model)
preprocess=weights.transforms()
batch=preprocess(input_image).unsqueeze(dim=0)
model.eval()
output_logits=model(batch)
print(output_logits.shape,output_logits.dtype)
