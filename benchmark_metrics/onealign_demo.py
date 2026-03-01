# from transformers import CLIPProcessor
# # from aesthetic_scorer import AestheticScorer
# import torch
# from PIL import Image

# # Load the model
# processor = CLIPProcessor.from_pretrained("/mnt/jfs/model_zoo/aesthetic-scorer")
# model = torch.load("/mnt/jfs/model_zoo/aesthetic-scorer/model.pt")

# # Process an image
# image = Image.open("/data/LoraPipeline/output/gpt4o_judge/jiegeng.png")
# inputs = processor(images=image, return_tensors="pt")["pixel_values"]
# print(type(inputs))
# # Get scores
# with torch.no_grad():
#     scores = model(inputs)

# # Print results
# aesthetic_categories = ["Overall", "Quality", "Composition", "Lighting", "Color", "Depth of Field", "Content"]
# for category, score in zip(aesthetic_categories, scores):
#     print(f"{category}: {score.item():.2f}/5")
# import requests
import torch
from transformers import AutoModelForCausalLM
from transformers.cache_utils import Cache

model = AutoModelForCausalLM.from_pretrained("/mnt/jfs/model_zoo/one-align", trust_remote_code=True, attn_implementation="eager", 
                                             torch_dtype=torch.float16, device_map="auto",
                                             use_cache=False)

from PIL import Image
image = Image.open("/data/LoraPipeline/output/gpt4o_judge/jiegeng.png")
score = model.score([image], task_="aesthetics", input_="image")
print(score)
# task_ : quality | aesthetics; # input_: image | video
