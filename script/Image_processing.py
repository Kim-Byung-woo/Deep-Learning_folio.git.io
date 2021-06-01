#!/usr/bin/env python
# coding: utf-8

# # 디지털 영상의 기초 이해 
# ### ref: https://opencv-python.readthedocs.io/en/latest/index.html
# *   Pixel level
# *   Transformation

# ## Pixel level
# * Add
# * Inverse
# * Threshold

# In[10]:


import os
import cv2
import numpy as np
import matplotlib.pyplot as plt


# In[22]:


def plot_with_histogram(imglist, title=None):
  if title:
    assert len(imglist) == len(title) # assert는 뒤의 조건이 True가 아니면 AssertError를 발생한다.

  for i in range(len(imglist)):
    plt.figure(figsize=(15,7))
    plt.subplot(121)
    plt.imshow(imglist[i], cmap = 'gray') # cmap = color map 
                                           # cmap = 'rainbow'
                                           # cmap = 'Blues'
                                           # cmap = 'autumn'
                                           # cmap = 'RdYlGn'
    if title:
      plt.title(title[i])
      
    plt.subplot(122)
    plt.hist(imglist[i].flatten())
    if title:
      plt.title(title[i])
    plt.show()


# In[23]:


img = cv2.imread('coin.png', cv2.IMREAD_GRAYSCALE) # cv2.IMREAD_GRAYSCALE: 이미지를 Grayscale로 읽어 들입니다.
print(img.dtype)
plot_with_histogram([img])


# ## add

# In[29]:


img_add = img + 50*np.random.random(size=img.shape)
img_add = np.clip(img_add, 0, 255).astype(np.uint8) # numpy.clip(array, min, max): array 내의 element들에 대해서 min 값 보다 작은 값들을 min값으로 바꿔주고 max 값 보다 큰 값들을 max값으로 바꿔주는 함수.
plot_with_histogram([img_add])


# ## Inverse

# In[25]:


img_inv = 255 - img
plot_with_histogram([img_inv])


# ## Threshold
# ### cv2.threshold(src, thresh, maxval, type) 
# ### Parameters:
# * src – input image로 single-channel 이미지.(grayscale 이미지)
# * thresh – 임계값
# * maxval – 임계값을 넘었을 때 적용할 value
# * type – thresholding type

# In[31]:


value = 150
flag = 0

## numpy
if flag:
  img_thres = img.copy()
  img_thres[img < value] = 0
  img_thres[img > value] = 1

## opencv
else:
  # cv2.THRESH_BINARY: threshold보다 크면 value이고 아니면 0으로 바꾸어 줍니다.
  # cv2.THRESH_BINARY_INV: threshold보다 크면 0이고 아니면 value로 바꾸어 줍니다. 
  # cv2.THRESH_TRUNC: threshold보다 크면 value로 지정하고 작으면 기존의 값 그대로 사용한다. 
  # cv2.THRESH_TOZERO: treshold_value보다 크면 픽셀 값 그대로 작으면 0으로 할당한다.
  # cv2.THRESH_TOZERO_INV: threshold_value보다 크면 0으로 작으면 그대로 할당해준다. 
  ret, img_thres = cv2.threshold(img, value, 255, cv2.THRESH_TOZERO)

plot_with_histogram([img_thres])


# ## adaptiveThreshold: 작은 영역별로 thresholding
# ### cv2.adaptiveThreshold(src, maxValue, adaptiveMethod, thresholdType, blockSize, C)
# ### Parameters:
# * src – grayscale image
# * maxValue – 임계값
# * adaptiveMethod – thresholding value를 결정하는 계산 방법
# * thresholdType – threshold type
# * blockSize – thresholding을 적용할 영역 사이즈
# * C – 평균이나 가중평균에서 차감할 값

# In[33]:


th2 = cv2.adaptiveThreshold(img, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 15, 2) # cv2.ADAPTIVE_THRESH_MEAN_C : 주변영역의 평균값으로 결정
th3 = cv2.adaptiveThreshold(img, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 15, 2) # ADAPTIVE_THRESH_GAUSSIAN_C: 주변영역의 Gaussian 윈도우(중앙값만 도드라지게 보고 주변은 잘안보이게 하는 마스킹의 형태) 기반 가중치로 결정
plot_with_histogram([th2, th3], title=['Mean', 'Gaussian'])


# ## Otsu: threshold를 자동으로 계산
# 함수의 flag에 추가로 cv2.THRESH_STSU 를 적용

# In[40]:


ret, th4 = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU) # threshold에서 반환된 ret, th4에는 각각 자동으로 계산된 임계값(threshold)와 이진처리된 이미지입니다.
plot_with_histogram([th4], title=['Otsu'])


# ## Transformation
# * Flip (horizontal, vertical)
# * Resize
# * Translate
# * Rotate
# * Shear

# In[43]:


img = cv2.imread('Lenna_color.png', cv2.IMREAD_GRAYSCALE)
print(img.shape)
plt.imshow(img, 'gray')


# ## flip

# In[15]:


# Extended slice [A:B:C] A: start, B: stop, C:step
val = range(1, 10)
lst = list(a)
# print(lst)

print(lst[::1]) # 처음부터 끝까지 +1 간격으로 출력
print(lst[::3]) # 처음부터 끝까지 +3 간격으로 출력
print(lst[1::2]) # 인덱스1부터 끝까지 +2 간격으로 출력
print(lst[::-1]) # 처음부터 끝까지 -1(역순) 간격으로 출력
print(lst[3::-1]) # 인덱스3부터 끝까지 -1(역순) 간격으로 출력
print(lst[5:1:-1]) # 인덱스5부터 인덱스1까지 -1(역순) 간격으로 출력


# In[49]:


img_hflip = img[:,::-1]
img_vflip = img[::-1]

plt.figure(figsize=(20, 20))
plt.subplot(131)
plt.imshow(img, 'gray')
plt.title('original : {}'.format(img.shape))
plt.subplot(132)
plt.imshow(img_hflip, 'gray')
plt.title('horizontal flip : {}'.format(img_hflip.shape))
plt.subplot(133)
plt.imshow(img_vflip, 'gray')
plt.title('vertical flip : {}'.format(img_vflip.shape))
plt.show()


# ## resize
# ### cv2.resize(img, dsize, fx, fy, interpolation)
# ### Parameters:
# * img – Image
# * dsize – Manual Size. 가로, 세로 형태의 tuple(ex; (100,200))
# * fx – 가로 사이즈의 배수. 2배로 크게하려면 2. 반으로 줄이려면 0.5
# * fy – 세로 사이즈의 배수
# * interpolation – 보간법(default = cv2.INTER_LNIEAR)

# In[52]:


height, width = img.shape
img_shrink = cv2.resize(img, None, fx=0.5, fy=0.5, interpolation=cv2.INTER_AREA) # 이미지를 축소할때에는  cv.INTER_AREA를 권장
img_zoom = cv2.resize(img, (width*2, height*2), interpolation=cv2.INTER_CUBIC) # 이미지 확대할때에는 cv2.INTER_CUBIC 또는 cv2.INTER_LINEAR 권장

plt.figure(figsize=(20, 20))
plt.subplot(131)
plt.imshow(img, 'gray')
plt.title('original : {}'.format(img.shape))
plt.subplot(132)
plt.imshow(img_shrink, 'gray')
plt.title('shrink : {}'.format(img_shrink.shape))
plt.subplot(133)
plt.imshow(img_zoom, 'gray')
plt.title('zoom : {}'.format(img_zoom.shape))
plt.show()


# ### meshgrid(x1, x2..., copy = True, sparse = True, indexing = 'xy'): x축 값(1차원 배열)과 y축(1차원 배열) 값을 받아서 좌표 행렬을 반환
# 
# ### Parameters:
# * x1.x2..: 그리드에 나타낼 1차원 벡터
# * copy: False인 경우 메모리 절약을 위해 오리지널 배열로 반환
# * sparse: True인 경우 메모리 절약을 위해 희소 그리드 반환
# * indexing: 일반좌표-> 'xy', 행렬좌표-> 'ij'

# In[12]:


x = np.arange(10)
y = np.arange(10)
# meshgrid(x1, x2..., copy = True, sparse = True, indexing = 'xy'): 
# x축 값(1차원 배열)과 y축(1차원 배열) 값을 받아서 좌표 행렬을 반환
# copy: False인 경우 메모리 절약을 위해 오리지널 배열로 반환
# sparse: True인 경우 메모리 절약을 위해 희소 그리드 반환
# indexing: 일반좌표-> 'xy', 행렬좌표-> 'ij'
xx, yy = np.meshgrid(x, y, sparse=False) 
xx2, yy2 = np.meshgrid(x, y, sparse=True) 

#plt.figure(figsize=(10, 5))
plt.subplot(121)
plt.scatter(xx,yy)
plt.subplot(122)
plt.scatter(xx2,yy2)
plt.grid()


# ## Translation
# ### cv2.warpAffine(src, M, dsize)
# ### Parameters:
# * src – Image
# * M – 변환 행렬 - 변환행렬은 이동 변환을 위한 2X3의 이차원 행렬입니다. [[1,0,x축이동],[0,1,y축이동]] 형태의 float32 type의 numpy array입니다.
# * dsize (tuple) – output image size(ex; (width=columns, height=rows)

# In[6]:


x = np.arange(512)
y = np.arange(512)
xx, yy = np.meshgrid(x, y, sparse=False) 


# In[56]:


# X축으로 50, Y축으로 100 이동
X = 50
Y = 100

M = np.array([[1, 0, X],
              [0, 1, Y]], dtype=np.float32)
img_translate = cv2.warpAffine(img, M, (width, height))

plt.figure(figsize=(10, 10))
plt.subplot(121)
plt.imshow(img, 'gray')
plt.title('original')
plt.subplot(122)
plt.imshow(img_translate, 'gray')
plt.title('after translation')
plt.show()


# ## Rotate
# ### cv2.getRotationMatrix2D(center, angle, scale) → M
# ### Parameters:
# * center – 이미지의 중심 좌표
# * angle – 회전 각도
# * scale – scale factor

# In[57]:


degree = 45
origin = (width/2, height/2)
scale = 1.
M = cv2.getRotationMatrix2D(origin, degree, scale)
img_rotate = cv2.warpAffine(img, M, (width, height))

plt.figure(figsize=(10, 10))
plt.subplot(121)
plt.imshow(img, 'gray')
plt.title('original')
plt.subplot(122)
plt.imshow(img_rotate, 'gray')
plt.title('after rotation')
plt.show()


# ### shear(영상의 전단 변환): x,y 축 방향으로 영상이 밀리는 것 처럼 보이는 변환(https://gaussian37.github.io/vision-opencv_python_snippets)

# In[67]:


# shear
level = 0.2
M_x = np.array([[1, level, 0], [0, 1, 0]], dtype=np.float32)
M_y = np.array([[1, 0, 0], [level, 1, 0]], dtype=np.float32)
img_shear_x = cv2.warpAffine(img, M_x, (width + int(height * level), height))
img_shear_y = cv2.warpAffine(img, M_y, (width, height + int(width * level)))

plt.figure(figsize=(10, 10))
plt.subplot(131)
plt.imshow(img, 'gray')
plt.title('original')
plt.subplot(132)
plt.imshow(img_shear_x, 'gray')
plt.title('after x shearing')
plt.subplot(133)
plt.imshow(img_shear_y, 'gray')
plt.title('after y shearing')
plt.show()


# ## Filter
# * Blurring
# * Edge detection

# ### Blurring: <br>Image Blurring은 low-pass filter를 이미지에 적용하여 얻을 수 있습니다. 고주파영역을 제거함으로써 노이즈를 제거하거나 경계선을 흐리게 할 수 있습니다.

# ### cv2.blur(src, ksize) - 모든 픽셀에 똑같은 가중치를 부여
# ### Parameters:
# * src – Chennel수는 상관없으나, depth(Data Type)은 CV_8U, CV_16U, CV_16S, CV_32F or CV_64F.
# * ksize – kernel 사이즈(ex; (3,3))

# In[79]:


kernel_size = 9

# mean filter
img_blur_mean = cv2.blur(img, (kernel_size, kernel_size))

plt.figure(figsize=(10, 10))
plt.subplot(121)
plt.imshow(img, 'gray')
plt.title('original')

plt.subplot(122)
plt.imshow(img_blur_mean, 'gray')
plt.title('blurring using mean filter')


# ### cv2.medianBlur(src, ksize) - 중간값을 뽑아서 픽셀값으로 사용
# ### Parameters:	
# * src – Chennel수는 상관없으나, depth(Data Type)은 CV_8U, CV_16U, CV_16S, CV_32F or CV_64F.
# * ksize – 1보다 큰 홀수

# In[80]:


kernel_size = 9

# median filter
img_blur_median = cv2.medianBlur(img, kernel_size)

plt.figure(figsize=(10, 10))
plt.subplot(121)
plt.imshow(img, 'gray')
plt.title('original')

plt.subplot(122)
plt.imshow(img_blur_median, 'gray')
plt.title('blurring using median filter')


# ### cv2.GaussianBlur(img, ksize, sigmaX) - 중심에 있는 픽셀에 높은 가중치를 부여
# ### Parameters:	
# * img – Chennel수는 상관없으나, depth(Data Type)은 CV_8U, CV_16U, CV_16S, CV_32F or CV_64F.
# * ksize – (width, height) 형태의 kernel size. width와 height는 서로 다를 수 있지만 홀수로 지정해야 함.
# * sigmaX – Gaussian kernel standard deviation in X direction.

# In[81]:


kernel_size = 9

# gaussian filter
std = 3
img_blur_gaussian = cv2.GaussianBlur(img, (kernel_size, kernel_size), std)
gaussian_filter = cv2.getGaussianKernel(kernel_size, std)
gaussian_filter = np.outer(gaussian_filter, gaussian_filter.T)

plt.figure(figsize=(15, 15))
plt.subplot(131)
plt.imshow(img, 'gray')
plt.title('original')

plt.subplot(132)
plt.imshow(img_blur_gaussian, 'gray')
plt.title('blurring using gaussian filter')

plt.subplot(133)
plt.imshow(gaussian_filter, 'gray')
plt.title('gaussian_filter')
plt.show()


# ### Edge detection: 이미지 (x,y)에서의 벡터값(크기와 방향, 즉 밝기와 밝기의 변화하는 방향)을 구해서 해당 pixel이 edge에 얼마나 가까운지, 그리고 그 방향이 어디인지 쉽게 알수 있게 합니다.
# * Gradient: 스칼라장(공간)에서 최대의 증가율을 나타내는 벡터장(방향과 힘)

# ### Sobel & Scharr Filter:<br>Gaussian smoothing과 미분을 이용한 방법입니다. 노이즈가 있는 이미지에 적용하면 좋습니다. X축과 Y축을 미분하는 방법으로 경계값을 계산합니다.
# 
# ![sobel_filter](./ref/sobel_filter.JPG "sobel_filter")
# 
# 
# ### cv2.Sobel(src, ddepth, dx, dy, dst, ksize, scale, delta, borderType) → dst
# ### Parameters:
# * src – input image
# * ddepth – output image의 depth, -1이면 input image와 동일.
# * dx – x축 미분 차수.
# * dy – y축 미분 차수.
# * ksize – kernel size(ksize x ksize) - 홀수 값을 사용하며, 최대 31까지 설정할 수 있습니다.

# In[75]:


# sobel
## filter
#M_x = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=np.float32)
#M_y = np.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=np.float32)
#img_sobel_x = cv2.filter2D(img, -1, M_x)
#img_sobel_y = cv2.filter2D(img, -1, M_y)

## opencv
img_sobel_x = cv2.Sobel(img, -1, 1, 0, ksize=1) # depth = -1, dx = 1 
img_sobel_y = cv2.Sobel(img, -1, 0, 1, ksize=1)

plt.figure(figsize=(20, 20))
plt.subplot(141)
plt.imshow(img, 'gray')
plt.title('original')

plt.subplot(142)
plt.imshow(img_sobel_x, 'gray')
plt.title('sobel filter (x)')

plt.subplot(143)
plt.imshow(img_sobel_y, 'gray')
plt.title('sobel filter (y)')

plt.subplot(144)
plt.imshow(img_sobel_x//2+img_sobel_y//2, 'gray')
plt.title('sobel filter (x, y)')


# * Laplacian: 미지의 가로와 세로에 대한 Gradient를 2차 미분한 값입니다. blob(주위의 pixel과 확연한 picel차이를 나타내는 덩어리)검출에 많이 사용됩니다.
# * Canny: 여러 단계의 Algorithm을 통해서 경계를 찾아 냅니다.

# ### Laplacian:<br>이미지의 가로와 세로에 대한 Gradient를 2차 미분한 값입니다. blob(주위의 pixel과 확연한 picel차이를 나타내는 덩어리)검출에 많이 사용됩니다.
# 
# ![Laplacian_filter](./ref/Laplacian_filter.JPG "Laplacian_filter")
# 
# ### cv2.laplacian(src, ddepth, ksize, scale, delta, borderType) → dst
# ### Parameters:
# * src – input image
# * ddepth – output image의 depth, -1이면 input image와 동일.

# In[76]:


# laplacian
img_laplacian = cv2.Laplacian(img, -1)

plt.figure(figsize=(20, 20))
plt.subplot(121)
plt.imshow(img, 'gray')
plt.title('original')

plt.subplot(122)
plt.imshow(img_laplacian, 'gray')
plt.title('laplacial filter')


# ### Canny: 여러 단계의 Algorithm을 통해서 경계를 찾아 냅니다.
# 1. 가우시안 필터를 이용하여 노이즈 제거
# 2. 소벨 필터를 이용화여 Gradient의 방향과 강도 확인
# 3. Edge가 아닌 픽셀 제거
# 4. 추출된 Edge 판별
# 
# ### cv2.Canny(image, threshold1, threshold2[, edges[, apertureSize[, L2gradient]]]) → edge
# ### Parameters:
# * image – 8-bit input image
# * threshold1 – Hysteresis Thredsholding 작업에서의 min 값
# * threshold2 – Hysteresis Thredsholding 작업에서의 max 값

# In[77]:


# canny
thres1, thres2 = 30, 70
img_canny = cv2.Canny(img, thres1, thres2)

plt.figure(figsize=(20, 20))
plt.subplot(121)
plt.imshow(img, 'gray')
plt.title('original')

plt.subplot(122)
plt.imshow(img_canny, 'gray')
plt.title('canny edge')
plt.show()


# In[16]:

import os
import cv2
import numpy as np
import matplotlib.pyplot as plt

path = "ref/Lung_Segmentation/label"
file_list = os.listdir(path)

#%%
def Div_Lung(filename):
    file_path = path + '/' + filename
    img = cv2.imread(file_path, cv2.IMREAD_GRAYSCALE) # cv2.IMREAD_GRAYSCALE: 이미지를 Grayscale로 읽어 들입니다.
    
    img_h = img.shape[0]
    img_w = img.shape[1]

    for i in range(img_h):
        for j in range(int(img_w / 2)):
            val = img[i][j]
            if val > 250: # 원본 색상
                img[i][j] = 128  # 바꿀 색상
                
    cv2.imwrite(f"ref/Lung_Segmentation/result/{filename}",img)
#%%
Div_Lung('resize_CHNCXR_0002_0.png')

#%%
for i in range(len(file_list)):
    file_name = file_list[i]
    Div_Lung(file_name)

#%%
import os
import numpy as np
import matplotlib.pyplot as plt

from skimage.io import imread
from skimage.transform import resize
from sklearn.model_selection import train_test_split

IMG_WIDTH = 256
IMG_HEIGHT = 256
IMG_CHANNELS = 3

data_path = "ref/Lung_Segmentation/"

files = os.listdir(os.path.join(data_path, 'image'))
file_headers = []  #python list
for f in files:
    f1 = os.path.splitext(f)[0]
    file_headers.append(f1)
#print(file_headers)

X_all = np.zeros((len(file_headers), IMG_HEIGHT, IMG_WIDTH, IMG_CHANNELS), dtype=np.uint8)
y_all = np.zeros((len(file_headers), IMG_HEIGHT, IMG_WIDTH, 1), dtype=np.uint8)
#%%
count = 0
for fh in file_headers:
    f1 = os.path.join(data_path, 'image', '{}.png'.format(fh))
    l1 = os.path.join(data_path, 'label', '{}.png'.format(fh))
    
    img = imread(f1)[:,:,:IMG_CHANNELS]
    mask = imread(l1)

    img_h = mask.shape[0]
    img_w = mask.shape[1]

    for i in range(img_h):
      for j in range(int(img_w / 2)):
        val = mask[i][j]
        if val > 250: # 원본 색상
          mask[i][j] = 128  # 바꿀 색상

    mask = np.expand_dims(mask, axis=-1)

    X_all[count] = img
    y_all[count] = mask
    
    count += 1
    
    if count > 10:
        break

#%%
X_all = X_all.astype('float32') / 255.
y_all = y_all.astype('float32') / 255.
#%%
X_all[0]
#%%
y_all[10]
#%%
print(X_all[10].shape)
plt.imshow(np.squeeze(X_all[10]), 'gray') # squeeze: 1차원 축 제거
plt.show()
x_sq = np.squeeze(X_all[10])

print(y_all[10].shape)
plt.imshow(np.squeeze(y_all[10]), 'gray') # squeeze: 1차원 축 제거
plt.show()
y_sq = np.squeeze(y_all[10])


print(mask.shape)
plt.imshow(np.squeeze(mask), 'gray') # squeeze: 1차원 축 제거
plt.show()
msk_sq = np.squeeze(mask)

#%%
import os
import numpy as np
import matplotlib.pyplot as plt

from skimage.io import imread
from skimage.transform import resize
from sklearn.model_selection import train_test_split

IMG_WIDTH = 256
IMG_HEIGHT = 256
IMG_CHANNELS = 3

data_path = "ref/Lung_Segmentation/"

files = os.listdir(os.path.join(data_path, 'image'))
file_headers = []  #python list
for f in files:
    f1 = os.path.splitext(f)[0]
    file_headers.append(f1)
#print(file_headers)

X_all = np.zeros((len(file_headers), IMG_HEIGHT, IMG_WIDTH, IMG_CHANNELS), dtype=np.uint8)
y_all = np.zeros((len(file_headers), IMG_HEIGHT, IMG_WIDTH, 1), dtype=np.bool)


count = 0
for fh in file_headers:
    f1 = os.path.join(data_path, 'image', '{}.png'.format(fh))
    l1 = os.path.join(data_path, 'label', '{}.png'.format(fh))
    
    img = imread(f1)[:,:,:IMG_CHANNELS]
    mask = imread(l1)
    mask = np.expand_dims(mask, axis=-1)
    
    X_all[count] = img
    y_all[count] = mask
    
   
    count += 1
    
    if count > 10:
        break
#%%
print(img[0][0][0])
print(img[0][0][2])

print(np.squeeze(y_all).shape)
print(np.squeeze(y_all[0]).shape)

plt.imshow(np.squeeze(y_all[0]), 'gray') # squeeze: 1차원 축 제거
plt.show()
#%%

arr_3 = np.array([[[ 0,  1,  2,  3],
           [ 4,  5,  6,  7],
           [ 8,  9, 10, 11]],
          
          [[12, 13, 14, 15],
           [16, 17, 18, 19],
           [20, 21, 22, 23]]])



y_test = np.zeros((3, 3, 2), dtype=np.bool) 

a = y_test[1, :, 1]
#%%


y_test = np.zeros((len(file_headers), IMG_HEIGHT, IMG_WIDTH, 2), dtype=np.bool) 

# y_test[0][0][1][1] = 1

img_path = 'ref/Lung_Segmentation/label/resize_CHNCXR_0011_0.png'

mask_r = imread(img_path)
# mask_r = np.expand_dims(mask_r, axis=-1)

mask_l = imread(img_path)
# mask_l = np.expand_dims(mask_l, axis=-1)

img_h = mask_r.shape[0]
img_w = mask_r.shape[1]

for i in range(img_h):
    for j in range(int(img_w / 2)):
        mask_r[i][j] = 0          
        
for i in range(img_h):
    for j in range(int(img_w / 2), img_w):
        mask_l[i][j] = 0


for j in range(img_h):
    for i in range(img_w):
        y_test[0][i][j][0] = mask_r[i][j]
        y_test[0][i][j][1] = mask_l[i][j]



mask_l[145][45]
y_test[0][145][45][1]


print(mask_l.shape)
plt.imshow(mask_l, 'gray') # squeeze: 1차원 축 제거
plt.show()

y_0 = y_test[0, :, :, 0]
print(y_0.shape)
plt.imshow(y_0, 'gray') # squeeze: 1차원 축 제거
plt.show()
#%%
count = 0
for fh in file_headers:
    l1 = os.path.join(data_path, 'label', '{}.png'.format(fh))
    
    mask_r = imread(l1)
    mask_l = imread(l1)
    
    img_h = mask_r.shape[0]
    img_w = mask_r.shape[1]

    for i in range(img_h):
        for j in range(int(img_w / 2)):
            mask_r[i][j] = 0          
            
    for i in range(img_h):
        for j in range(int(img_w / 2), img_w):
            mask_l[i][j] = 0

    for i in range(img_h):
        for j in range(img_w):
            y_test[count][i][j][0] = mask_r[i][j]
            y_test[count][i][j][1] = mask_l[i][j]
    
    count += 1
    
    if count > 10:
        break

#%%
print(y_test.shape)
print(y_test[0][:][:][0].shape)


print(np.squeeze(y_test).shape)
print(np.squeeze(y_test[0]).shape)

plt.imshow(np.squeeze(y_all[0]), 'gray') # squeeze: 1차원 축 제거
plt.show()

#%%

# y_test[0][50][70][0] = mask_l[50][70]
print(np.squeeze(y_all).shape)
print(np.squeeze(y_all[0]).shape)
plt.imshow(np.squeeze(mask), 'gray') # squeeze: 1차원 축 제거
plt.show()

print(mask_r.shape)
plt.imshow(mask_r, 'gray') # squeeze: 1차원 축 제거
plt.show()

print(mask_l.shape)
plt.imshow(mask_l, 'gray') # squeeze: 1차원 축 제거
plt.show()


y_test[0][50][70][0].shape
mask_l[50][70].shape

y_test.shape
print(np.squeeze(y_test).shape)
print(np.squeeze(y_test[0]).shape)
plt.imshow(mask_r, 'gray') # squeeze: 1차원 축 제거
plt.show()



#%%



























