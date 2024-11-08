from scipy import io
import numpy as np
import os
import json
import cv2
from tqdm import tqdm

mat_file = io.loadmat("./gt.mat")
print("annotation load complete!")

annotation = dict()
annotation['images'] = dict()

for i in tqdm(range(len(mat_file["imnames"][0]))):
  img_list = list()
  rec = mat_file['wordBB'][0][i]
  if isinstance(rec[0][0], np.float32):
    continue
  img_name = mat_file['imnames'][0][i][0].split("/")[1]
  if not os.path.exists(os.path.join("./train", img_name)):
    continue
  
  img = cv2.imread(os.path.join("./train", img_name))
  h, w, _ = img.shape
  
  annotation['images'][img_name] = dict()
  img_anno = annotation['images'][img_name]
  img_anno['words'] = dict()
  img_anno['paragraphs'] = {}
  img_anno['chars'] = {}
  img_anno['img_w'] = w
  img_anno['img_h'] = h
  img_anno['num_patches'] = None
  img_anno['tags'] = []
  img_anno['relations'] = {}
  img_anno['annotation_log'] = {
                "worker": "worker",
                "timestamp": "2024-05-30",
                "tool_version": "",
                "source": None
            }
  img_anno['license_tag'] = {
                "usability": True,
                "public": False,
                "commercial": False,
                "type": None,
                "holder": "SynthText"
            }

  txt_str = ""
  for words in mat_file['txt'][0][i]:
    txt_str += " " + " ".join([w.strip() for w in words.split("\n")])
  txt_str = txt_str.strip().split(" ")

  word_list = list()
  for j in range(len(rec[0][0])):
    x1 = format(float(rec[0][0][j]), "0.6f")
    y1 = format(float(rec[1][0][j]), "0.6f")
    x2 = format(float(rec[0][1][j]), "0.6f")
    y2 = format(float(rec[1][1][j]), "0.6f")
    x3 = format(float(rec[0][2][j]), "0.6f")
    y3 = format(float(rec[1][2][j]), "0.6f")
    x4 = format(float(rec[0][3][j]), "0.6f")
    y4 = format(float(rec[1][3][j]), "0.6f")

    img_anno['words']['{:04d}'.format(j + 1)] = dict()
    img_anno['words']['{:04d}'.format(j + 1)]['transcription'] = txt_str[j]
    img_anno['words']['{:04d}'.format(j + 1)]['points'] = [[x1, y1], [x2, y2], [x3, y3], [x4, y4]]

with open(os.path.join('./train', 'train.json'), 'w', encoding='utf-8') as w:
  json.dump(annotation, w, indent='\t')

print("convert complete!")