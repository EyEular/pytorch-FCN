import numpy as np
from PIL import Image

def rle_encode(img):


    print(1)

def rle_decode(img_size, codes):
    if codes == 'nan':
        return np.zeros((img_size[1],img_size[2]))
    #print('eeeee')
    #print(codes)
    s = codes.split()
    starts, lengths = [np.asarray(x, dtype=int) for x in (s[0:][::2], s[1:][::2])]
    # print(img_size)
    starts -= 1
    ends = starts + lengths
    img = np.zeros(img_size[1]*img_size[2])
    for lo, hi in zip(starts, ends):
        img[lo:hi] = 1
    return img.reshape(img_size[1],img_size[2])

def mask_compare(img1, root2):
    img2 = Image.open(root2)
    img2 = np.array(img2)/65535.0

    return (img1 - img2).sum()


