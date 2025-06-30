---
title: "Transfer learning with a pretrained ConvNet"
categories: [AI, Deep Learning]
author: JiHyun Kim
date: 2019-12-22 12:50:00 +0900
tags: [deep learning, image classfication, tensorflow]
hide: false
---

pretrained ConvNet을 사용해서 Transfer learning을 해보자
이 튜토리얼에서 너는 pre-trained network로 부터 transfer learning을 사용해서 강아지와 고양이 사진을 분류할 수 있을것이다.
pre-trained model이란: 큰 데이터 셋에서 이미 이전에 훈련된 저장된 네트워크이다. 특히나 매우 큰 규모의 이미지 분류 작업들에서 사용한다.
이것에 대한 직관은, 모델이 크고 큰 범주의 dataset에서 trained된다면, 이 모델은 visual world에서 generic model로서 효과적으로 작용을 할 것이다.





이 튜토리얼에서 너는 pretrained model을 커스터마이즈하는 두가지 방법에 대해 시도해 볼것이다:
    1. Feature Extraction: 이전 네트워크로부터 배운 representations들을 가지고 새로운 셈플에서 의미있는 특징들을 추출하기 위해
                        전체 모델을 재훈련 시킬 필요가 엇ㅂ다. 베이스 convolutional network가 이미 사진 분류를 위해 유용한 features들을 가지고 있다.
                        하지만 마지막, pretrained model의 분류 파트는 기존의 분류 작업에 해당한다. 또한, 모델이 훈련된 클래스들의 집합에 연속적으로 해당한다.

    2. Fine-Tuning: frozen model의 맨 위 탑레이어들의 몇개를 unfreeze시키고, 새로 추가된 분류 레이어를 합친다. 이것은 우리로하여금 더 높은 차원의 feature에 대해서는 fine tune이 가능하다록 한다. 특정 작업에 feature 대표들이 더 연관되게 만들어준다.


you will follow the general ML workflow를 따를 것이다
1. examine and understand data
2. build an input pipeline(여기서는 Keras ImageDataGenerator)
3. compose our model
4. load in our pretrained base model, pretrained weights
5. stack our classification layers on top
6. train our model
7. evaluate model

```python
from __future__ import absolute_import, division, print_function, unicode_literals

import os

import numpy as np

import matplotlib.pyplot as plt

try:
  # %tensorflow_version only exists in Colab.
  %tensorflow_version 2.x
except Exception:
  pass
import tensorflow as tf

keras = tf.keras
```


###Data Processing
#data download
강아지와 고양이 datasets를 load하기 위해 텐서플로 데이터셋을 사용한다.
tfds 패키지는 pre-defined된 데이터를 가져오기 가장 쉬운 방법이다. 

```python
import tensorflow_datasets as tfds
tfds.disable_progress_bar()
```

tfds.load방식은 데이터를 다운로드 및 캐싱하고, tf.data.Dataset을 리턴해준다.
이러한 결과들은 데이터를 다루고 너의 모델에 piping시키기 위한 강력하고 효과적인 방법들을 제공해준다.

cats_vs_dogs는 표준화된 split을 정의하지 않으므로, 
이 데이터를 (train, validation, test) 80%, 10%, 10%와 같이 각각 분리 시키게끔 subsplit feature를 사용한다.

```python
SPLIT_WEIGHTS = (8, 1, 1)
splits = tfds.Split.TRAIN.subsplit(weighted=SPLIT_WEIGHTS)

(raw_train, raw_validation, raw_test), metadata = tfds.load(
    'cats_vs_dogs', split=list(splits),
    with_info=True, as_supervised=True)
```


tf.data.Dataset 결과값은 (image, label)의 짝으로 나오게 된다. 
이미지는 변수 형태와 3가지의 채널을 가지고 있고, label은 scalar값으로 나오게 된다.
```python
print(raw_train)
print(raw_validation)
print(raw_test)

#결과창:
# <_OptionsDataset shapes: ((None, None, 3), ()), types: (tf.uint8, tf.int64)>
# <_OptionsDataset shapes: ((None, None, 3), ()), types: (tf.uint8, tf.int64)>
# <_OptionsDataset shapes: ((None, None, 3), ()), types: (tf.uint8, tf.int64)>
```

training set에서 처음 두가지의 images와 labels을 보여라
```python
get_label_name = metadata.features['label'].int2str

for image, label in raw_train.take(2):
  plt.figure()
  plt.imshow(image)
  plt.title(get_label_name(label))

#귀여운 강아지와 고양이 사진이 출력되는 것을 볼 수 있다.
```

#Format the Data: 이미지 사이즈를 재 구성 재배열 시키는 단계?
tf.image 모듈을 사용하여 task를 위해 이미지의 규격을 재구성한다.
이미지를 고정시킨 input size로 사이즈를 바꾸고, input 채널의 -1~1의 범위 안으로 재 스케일링 한다.
```python
IMG_SIZE = 160 # All images will be resized to 160x160

def format_example(image, label):
  image = tf.cast(image, tf.float32)
  image = (image/127.5) - 1
  image = tf.image.resize(image, (IMG_SIZE, IMG_SIZE))
  return image, label
```


이 format_example함수를 map메소드를 사용하여 데이터셋의 각각의 아이템에 적용시킨다.
```python
train = raw_train.map(format_example)
validation = raw_validation.map(format_example)
test = raw_test.map(format_example)
```


Now shuffle and batch the data.
이제 데이터를 shuffling하고 batch한다.
```python
BATCH_SIZE = 32
SHUFFLE_BUFFER_SIZE = 1000

train_batches = train.shuffle(SHUFFLE_BUFFER_SIZE).batch(BATCH_SIZE)
validation_batches = validation.batch(BATCH_SIZE)
test_batches = test.batch(BATCH_SIZE)
```


데이터의 batch시킨 이미지의 사이즈를 조사해라
```python
for image_batch, label_batch in train_batches.take(1):
   pass

image_batch.shape
#결과: TensorShape([32, 160, 160, 3])
```

----------------------------------------
pre-trained된 ConvNets에서 base model을 만들어라 

Create the base model from the pre-trained convnets
You will create the base model from the MobileNet V2 model developed at Google. This is pre-trained on the ImageNet dataset, a large dataset of 1.4M images and 1000 classes of web images. ImageNet has a fairly arbitrary research training dataset with categories like jackfruit and syringe, but this base of knowledge will help us tell apart cats and dogs from our specific dataset.

First, you need to pick which layer of MobileNet V2 you will use for feature extraction. Obviously, the very last classification layer (on "top", as most diagrams of machine learning models go from bottom to top) is not very useful. Instead, you will follow the common practice to instead depend on the very last layer before the flatten operation. This layer is called the "bottleneck layer". The bottleneck features retain much generality as compared to the final/top layer.

First, instantiate a MobileNet V2 model pre-loaded with weights trained on ImageNet. By specifying the include_top=False argument, you load a network that doesn't include the classification layers at the top, which is ideal for feature extraction.


