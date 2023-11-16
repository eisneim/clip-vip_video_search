import av
import torch
from torch.nn import functional as F
import numpy as np
from easydict import EasyDict as edict
import argparse, pickle, os

from transformers.models.clip.configuration_clip import CLIPConfig, CLIPTextConfig, CLIPVisionConfig
from transformers import CLIPProcessor, CLIPTokenizerFast
from transformers import AutoProcessor
from clipvip.CLIP_VIP import CLIPModel

#  data preprocessor
tokenizer = CLIPTokenizerFast.from_pretrained("openai/clip-vit-base-patch32")
processor = AutoProcessor.from_pretrained("microsoft/xclip-base-patch16")

def find_mp4_files(directory):
  for root, dirs, files in os.walk(directory):
    for file in files:
      # if file.endswith('.mp4') and file[0] != "." and root.find("segments") > -1:
      if file.endswith('.mp4') and file[0] != ".":
        yield os.path.join(root, file)

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
    frame_sample_rate = max(frame_sample_rate, 1)
    converted_len = int(clip_len * frame_sample_rate)
    # print(clip_len, frame_sample_rate, seg_len, "converted_len", converted_len)
    end_idx = seg_len if converted_len >= seg_len else np.random.randint(converted_len, seg_len)
    start_idx = max(end_idx - converted_len, 0)
    indices = np.linspace(start_idx, end_idx, num=clip_len)
    indices = np.clip(indices, start_idx, end_idx - 1).astype(np.int64)
    # print("indices", indices, "seg_len", seg_len)
    return indices

def prepaer_vect(model, sourceDir):
  print("sourceDir", sourceDir)
  cache = os.path.join(sourceDir, "__embeddings.pickle")
  # only use the first dir
  if os.path.exists(cache):
    with open(cache, "rb") as f:
      pair = pickle.load(f)
    return pair

  pair = {}
  clip_len = 12
  for videoFile in find_mp4_files(sourceDir):
    container = av.open(videoFile)
    fcount = container.streams.video[0].frames
    if fcount < clip_len:
      break
    # sample 12 frames
    indices = sample_frame_indices(clip_len=clip_len, frame_sample_rate=fcount//clip_len, seg_len=fcount)
    video = read_video_pyav(container, indices)
    pixel_values = processor(videos=list(video), return_tensors="pt").pixel_values

    inputs = {
      "if_norm": True,
      "pixel_values": pixel_values}
    with torch.no_grad():
        video_features = model.get_image_features(**inputs)

    pair[videoFile] = video_features[0]
    print(f"parsed video: {videoFile}")

  # save to cache
  with open(cache, 'wb') as handle:
    pickle.dump(pair, handle, protocol=pickle.HIGHEST_PROTOCOL)
  return pair

def main():
  parser = argparse.ArgumentParser(description='chose a folder, parse it, then you can search it')
  parser.add_argument('dir', metavar='N', type=str, nargs='+',
                      help='folder that contains many short video clips')
  parser.add_argument('--phrase', type=str, default="a man standing in woods",
                      help='search phrase ')
  args = parser.parse_args()

  extraCfg = edict({
      "type": "ViP",
      "temporal_size": 12,
      "if_use_temporal_embed": 1,
      "logit_scale_init_value": 4.60,
      "add_cls_num": 3
  })
  #  model set up
  clipconfig = CLIPConfig.from_pretrained("openai/clip-vit-base-patch32")
  clipconfig.vision_additional_config = extraCfg
  checkpoint = torch.load("/Users/teli/www/test/视频向量提取/XPretrain-视频获取/CLIP-ViP/pretrain_clipvip_base_32.pt")
  cleanDict = { key.replace("clipmodel.", "") : value for key, value in checkpoint.items() }
  model =  CLIPModel(config=clipconfig)
  model.load_state_dict(cleanDict)
   # ------- text embedding -----
  tokens = tokenizer([args.phrase], padding=True, return_tensors="pt")
  textOutput = model.get_text_features(**tokens)
  # print(textOutput.shape)

  pair = prepaer_vect(model, args.dir[0])
  files = list(pair.keys())
  if len(files) == 0:
    print("no video find")
    return

  vects = torch.stack([value for _, value in pair.items()])
  
  # search
  with torch.no_grad():
      sim = F.cosine_similarity(textOutput, vects, dim=1)
      print(sim)
  idx = torch.argmax(sim)
  best = files[idx]
  print(best, sim[idx], "样本差异性", torch.std(sim))

  # show alternative results
  some = (sim > 0.2).nonzero()
  others = [files[ii] for ii in some.view(-1)]
  print(others)


if __name__ == "__main__":
  main()
