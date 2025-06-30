---
title: "mnist_image_classification"
categories: [AI, Deep Learning]
date: 2019-12-20 12:50:00 +0900
author: JiHyun Kim
tags: [AI, Image Classification, Tensorflow]
published: false
---


```python

import tensorflow as tf
import ssl
ssl._create_default_https_context = ssl._create_unverified_context


mnist = tf.keras.datasets.mnist
(x_train,y_train) ,(x_test,y_test) = mnist.load_data()
x_train,x_test = x_train/255.0-0
.5,x_test/255.0-0.5

print(type(x_train))
print(x_train.shape)
print()
```

```python
# 인공신경망의 구성
print("tf.keras.models.Sequential() 실행함")
model = tf.keras.models.Sequential()
# Sequential: 인공신경망을 구성하는 레이어를 순차적으로 연결하여 구성하는 방법.
# add함수를 통해 순차적으로 레이어와 레이어에서 사용할 기능들을 추가하여 신경망을 구성
print("tf.keras.models.Sequential() 완료")

print("tf.keras.layers.Flatten 실행함")
model.add(tf.keras.layers.Flatten(input_shape=((28,28))))
# Flatten: 이전 레이어에서부터 전달 받은 텐서를 직선화하는 레이어
print("tf.keras.layers.Flatten 완료함")


print("tf.keras.layers.Dense(128) 실행함")
model.add(tf.keras.layers.Dense(128))
print("tf.keras.layers.Dense(128) 완료함")
# Dense: 퍼셉트론으로 이루어진 fully-connected neural network의 한 레이어
#         몇 개의 퍼셉트론을 사용할지를 인자로 전달받으며, 각각의 퍼셉트론에 이전 레이어의 결과를 할당하고, 각각의 연결에 대한 가중치(kernel) 저장
#         output = (input * kernel) + bias
#         kernel은 (좋다고 알려진 방법을 따라) 랜덤하게 초기화

print("tf.keras.layers.Activation('relu') 실행함")
model.add(tf.keras.layers.Activation('relu'))
print("tf.keras.layers.Activation('relu') 완료함")
#Activation: 이전 레이어의 결과에 activation 함수를 적용하는 레이어
    # ReLU함수를 적용해서 뉴럴네트워크가 표현할 수 있는 함수에 nonplinearlity를 추가.
    # 여러개의 Activation함수들: https://www.tensorflow.org/api_docs/python/tf/keras/activations

print("tf.keras.layers.Dropout(0.2) 실행함")
model.add(tf.keras.layers.Dropout(0.2))
# Dropout: 이전 레이어의 activation 함수를 일정 확률로 0으로 만드는 방법
#     이전 레이어의 뉴런을 임의의 비율로 사용하지 않게 함으로써 overfitting을 방지하는 방법 - fully connected network가 안되게 해줌
print("tf.keras.layers.Dropout(0.2) 완료함")

model.add(tf.keras.layers.Dense(10))
model.add(tf.keras.layers.Activation('softmax'))
#softmax: 이전 레이어의 결과값들을 이용해서 최대값을 찾는데 활용하는 함수. 큰 값에 더 큰 가중치를 준다.
            # 이전 레이어의 결과값들이 1<=i<=n이라 할때, softmax 함수에 따라 값을 변화시킨다.
```

```python
model.compile(optimizer = 'adam', loss= 'sparse_categorical_crossentropy',metrics=['accuracy'])
#optimizer: 경사하강법을 수행하는 알고리즘
#loss: 모델을 최적화하는 데 사용하는 목적 함수. 현재 모델에서 사용중인 파라미터와 입력 데이터, 그리고 정답에 대해 정의되는 함수이며,
        # optimizer는 loss함수를 최소화하기 위해 모델에서 사용중인 파라미터를 수정한다.
# metrics: 현재 모델의 성능을 측정하기 위해 사용하는 기준. 트레이닝 과정과는 상관없음

model.fit(x_train,y_train,batch_size = 50, epochs = 10)
#첫번째 인자로 트레이닝 할 데이터를, 두번째 인자로 정답을 전달
# batch_size: gradient descent를 한번 적용하는데 사용하는 데이터 수
# epochs: 전체 트레이닝 데이터를 몇 번 트레이닝할지 설정

model.evaluate(x_test,y_test,verbose=2)
# evaluate: 트레이닝이 끝난 망을 시험하는 함수
#             훈련에 사용하지 않았던 데이터를 이용하여 트레이닝 결과 평가
#             첫번째 인자로 테스트할 데이터를, 두번째 인자로 정답을 전달
#             verbose를 통해서 출력 형태 조절
#                 0: 미표시, 1: 진행막대표시, 2: 결과표시
```


학습결과:
<img src="/images/mnist_train.png">
