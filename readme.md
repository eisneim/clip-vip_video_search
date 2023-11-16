
[original CLIP-ViP repo](https://github.com/microsoft/XPretrain/tree/main/CLIP-ViP)

```
$ python search_in_folder.py --phrase "a car is driving on the road" /Volumes/Samsung_T3/_download/test_cut2
```

## sample code:
here is a simple example showing how to use CLIP-Vip's text embeddings and video embeddings to calculate Cosine similarity


pretrained model CLIP-ViP-B/32: [Azure Blob Link](https://hdvila.blob.core.windows.net/dataset/pretrain_clipvip_base_32.pt?sp=r&st=2023-03-16T05:02:41Z&se=2027-05-31T13:02:41Z&spr=https&sv=2021-12-02&sr=b&sig=91OEG2MuszQmr16N%2Bt%2FLnvlwY3sc9CNhbyxYT9rupw0%3D)

CLIP-ViP-B/16: [Azure Blob Link](https://hdvila.blob.core.windows.net/dataset/pretrain_clipvip_base_16.pt?sp=r&st=2023-03-16T05:02:05Z&se=2026-07-31T13:02:05Z&spr=https&sv=2021-12-02&sr=b&sig=XNd7fZSsUhW7eesL3hTfYUMiAvCCN3Bys2TadXlWzFU%3D)

```python
import av
import torch
from torch.nn import functional as F
import numpy as np
from easydict import EasyDict as edict

from transformers.models.clip.configuration_clip import CLIPConfig, CLIPTextConfig, CLIPVisionConfig
from transformers import CLIPProcessor, CLIPTokenizerFast
from transformers import AutoProcessor
from clipvip.CLIP_VIP import CLIPModel, clip_loss


extraCfg = edict({
    "type": "ViP",
    "temporal_size": 12,
    "if_use_temporal_embed": 1,
    "logit_scale_init_value": 4.60,
    "add_cls_num": 3
})

clipconfig = CLIPConfig.from_pretrained("openai/clip-vit-base-patch32")
clipconfig.vision_additional_config = extraCfg

checkpoint = torch.load("YOUR_PATH_TO/CLIP-ViP/pretrain_clipvip_base_32.pt")
cleanDict = { key.replace("clipmodel.", "") : value for key, value in checkpoint.items() }
model =  CLIPModel(config=clipconfig)
model.load_state_dict(cleanDict)

# ------- text embedding -----
tokenizer = CLIPTokenizerFast.from_pretrained("openai/clip-vit-base-patch32")
tokens = tokenizer(["in the forest"], padding=True, return_tensors="pt")
textOutput = model.get_text_features(**tokens)
print(textOutput.shape)

# ------- video embedding -----
def read_video_pyav(container, indices):
    frames = []
    container.seek(0)
    start_index = indices[0]
    end_index = indices[-1]
    for i, frame in enumerate(container.decode(video=0)):
        if i > end_index:
            break
        if i >= start_index and i in indices:
            frames.append(frame)
    return np.stack([x.to_ndarray(format="rgb24") for x in frames])


def sample_frame_indices(clip_len, frame_sample_rate, seg_len):
    converted_len = int(clip_len * frame_sample_rate)
    end_idx = np.random.randint(converted_len, seg_len)
    start_idx = end_idx - converted_len
    indices = np.linspace(start_idx, end_idx, num=clip_len)
    indices = np.clip(indices, start_idx, end_idx - 1).astype(np.int64)
    return indices

processor = AutoProcessor.from_pretrained("microsoft/xclip-base-patch16")
container = av.open("/Volumes/Samsung_T3/_download/test_cut/py_053.mp4")

clip_len = 12
fcount = container.streams.video[0].frames
# sample 12 frames
indices = sample_frame_indices(clip_len=clip_len, frame_sample_rate=fcount//clip_len, seg_len=fcount)
video = read_video_pyav(container, indices)
pixel_values = processor(videos=list(video), return_tensors="pt").pixel_values

inputs = {
        "if_norm": True,
        "pixel_values": pixel_values}
with torch.no_grad():
    video_features = model.get_image_features(**inputs)
print(video_features.shape)

with torch.no_grad():
  sim = F.cosine_similarity(textOutput, video_features, dim=1)
  print(sim) 
  # [ 0.1142 ]

```




