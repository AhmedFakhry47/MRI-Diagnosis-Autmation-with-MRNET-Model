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

def Average(lst): 
    return sum(lst) / len(lst) 

def data_gen(data,label,path,data_length,batch_size=48):
  all_data = list(zip(data,label))
  random.shuffle(all_data)

  imgs = np.zeros(shape=(batch_size,227,227,3))
  label= np.zeros(shape=(batch_size,1))

  j=0

  while(True):
    chunk = all_data[j*batch_size:(j+1)*batch_size]  
    for i,pair in enumerate(chunk):
      if(pair[0] != '.DS_Store'):
        img = np.load(os.path.join(path,pair[0]),allow_pickle=True)
        img = img.astype(np.uint8)
        img = augment(img)
        img = add_rgb(img)
        imgs[i] = img[0,:,:,:]
        label[i] = pair[1]

        #yield (img_aug[0],pair[1])#np.repeat(pair[1],img_aug.shape[0]))
    yield (imgs,label)  
    j += 1

def combine_accuracies(acc_list):
  weight = float(1/len(acc_list))

  weight_acc = 0
  for acc in acc_list:
    weight_acc += weight * acc
  
  return weight_acc