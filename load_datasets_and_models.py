from datasets import load_dataset

from transformers import AutoImageProcessor, DonutSwinForImageClassification
import torch
from datasets import load_dataset


# image_processor = AutoImageProcessor.from_pretrained("naver-clova-ix/donut-base", cache_dir="/data/models/naver-clova-ix/donut-base")
# model = DonutSwinForImageClassification.from_pretrained("naver-clova-ix/donut-base", cache_dir="/data/models/naver-clova-ix/donut-base")


image_processor = AutoImageProcessor.from_pretrained("naver-clova-ix/donut-base")
model = DonutSwinForImageClassification.from_pretrained("naver-clova-ix/donut-base")

model

new_model = DonutSwinForImageClassification.from_pretrained("/home/dasom/.cache/huggingface/hub/models--naver-clova-ix--donut-base/snapshots/a959cf33c20e09215873e338299c900f57047c61")
new_model

# dataset = load_dataset("huggingface/cats-image")
# image = dataset["test"]["image"][0]
# inputs = image_processor(image, return_tensors="pt")

# with torch.no_grad():
#     logits = model(**inputs).logits

# # model predicts one of the 1000 ImageNet classes
# predicted_label = logits.argmax(-1).item()
# print(model.config.id2label[predicted_label])



# dataset download

# dataset = load_dataset(
#     "naver-clova-ix/cord-v2",
#     # split="train",
#     cache_dir="/data/datasets/naver-clova-ix/cord-v2"
# )

# dataset