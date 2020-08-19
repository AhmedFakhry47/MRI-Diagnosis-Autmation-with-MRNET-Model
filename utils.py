import numpy as np

def augment(image):
  aug=[]
  for s in range(len(image)):
      aug.append(ia.imresize_single_image(image[s], (227, 227)))
  aug=np.array(aug)
  return aug

def clear():
  aug= None
  gc.collect()

def add_rgb(grey_img):
  rgb_img = np.repeat(grey_img[..., np.newaxis], 3, -1)
  return rgb_img

def load_data(path):
  data = []
  for d in sorted(os.listdir(path)):
    if d!='.DS_Store':
      data.append(d)
      print("in iterator ", d)
      clear()
  return data
  
def load_labels(path):
  labels = []
  read = pd.read_csv(path, names=['num', 'hot'])
  labels = list(read['hot'])
  return labels