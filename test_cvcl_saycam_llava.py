"""
test_cvcl_saycam_llava.py:
- Test local cpkt model trained on SAYCam + LLaVA dataset
"""

import glob
import json
import torch
from torchvision import transforms

from multimodal.multimodal_lit import MultiModalLitModel

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

#load model from local checkpoint
checkpoint = '/projectnb/ivc-ml/ac25/Baby LLaVA/multimodal-baby/experiment_full_1/epoch=13.ckpt'
cvcl = MultiModalLitModel.load_from_checkpoint(checkpoint_path=checkpoint)
cvcl = cvcl.to(device)
cvcl.eval()

#override vocab to include llava pretraining vocab 
vocab_path = '/projectnb/ivc-ml/ac25/Baby LLaVA/multimodal-baby/arjun_misc/vocab.json'
with open(vocab_path) as f:
    cvcl.vocab = json.load(f)
print("CVCL model (SAYCam + LLaVA dataset) loaded!")

#LLavA images need to be resized to 224 x 224
preprocess = transforms.Compose([
            transforms.Resize((224, 224),
                              interpolation=transforms.InterpolationMode.BICUBIC),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])


# create random image tensor
images = torch.rand(4, 3, 224, 224).to(device)
image_features = cvcl.encode_image(images)
print("image features:", image_features.size())

# create random text tensor -> should be lower case 
# texts = ["ball", "puzzle", "car"]
texts = ["ball", "armor"]
#uses spaCy model for tokenization (not sure why there is inconsistency in white space vs spaCy)
texts, texts_len = cvcl.tokenize(texts)

#all texts are padded to 25 tokens + sos + eos
#print(texts)

texts, texts_len = texts.to(device), texts_len.to(device)
texts_features = cvcl.encode_text(texts, texts_len)
print("text features:", texts_features.size())

# test model (default flat embedding -> features are normalized so just use dot product for logits)
logits_per_image, logits_per_text = cvcl(images, texts, texts_len)
print("logits per image:", logits_per_image.size())
print("logits per text:", logits_per_text.size())


