"""
정책 신경망 모듈 : 투자 행동을 결정하기 위해 신경망을 관리하는 정책 신경망 클래스 (PolicyNetwork)를 가진다.

* 속성
model : 케라스 라이브러리로 구성한 LSTM 신경망 모델
prob : 가장 최근에 계산한 투자 행동별 확률

* 함수
reset() : prob 변수를 초기화
predict() : 신경망을 통해 투자 행동별 확률 계산
train_on_batch() : 배치 학습을 위한 데이터 생성
save_model() : 학습한 신경망을 파일로 저장
load_model() : 파일로 저장한 신경망을 로드
"""

import numpy as np
from keras.models import Sequential
from keras.layers import Activation, LSTM, Dense, BatchNormalization
from keras.optimizers import sgd

# PolicyNetwork : Keras를 사용하여 LSTM 신경망을 구성함.
# 세 개의 LSTM 층을 256 차원으로 구성하고, 드롭아웃을 50%로 정하여 과적합을 피한다.
class PolicyNetwork:
    def __init__(self, input_dim=0, output_dim=0, lr=0.01):
        self.input_dim = input_dim
        self.lr = lr

        # LSTM 신경망
        # 케라스에서 Sequential 클래스는 전체 신경망을 구성하는 클래스이다.
        # 전체 신경망에서 하나의 노드를 LSTM 클래스로 구성한다.
        self.model = Sequential()

        self.model.add(LSTM(256, input_shape=(1, input_dim),
                            return_sequences=True, stateful=False, dropout=0.5))
        self.model.add(BatchNormalization())
        self.model.add(LSTM(256, return_sequences=True, stateful=False, dropout=0.5))
        self.model.add(BatchNormalization())
        self.model.add(LSTM(256, return_sequences=False, stateful=False, dropout=0.5))
        self.model.add(BatchNormalization())
        self.model.add(Dense(output_dim))
        self.model.add(Activation('sigmoid'))

        # 최적화 알고리즘과 학습 속도를 정하여 신경망 모델을 준비한다.
        # 여기서 기본 학습 알고리즘은 확률적 경사 하강법(SGD)을, 기본 학습 속도(Learning rate, LR)은 0.01로 정함..
        self.model.compile(optimizer=sgd(lr=lr), loss='mse')
        self.prob = None

    def reset(self):    # 초기화
        self.prob = None

    # 신경망을 통해서 학습 데이터와 에이전트 상태를 합한 17 차원의 입력을 받아서
    # 매수와 매도가 수익을 높일 것으로 판단되는 확률을 구한다.
    def predict(self, sample):
        # Sequential 클래스의 predict() 함수는 여러 샘플을 한꺼번에 받아서 신경망의 출력을 반환한다.
        # 하나의 샘플에 대한 결과만을 받고 싶어도 샘플의 배열로 입력값을 구성해야 하기 때문에 2차원 배열로 재구성
        # (NumPy의 array() 함수로 n차원 배열 형식 만들기 가능)
        # NumPy의 reshape() 함수로 다른 차원으로 변환 가능
        self.prob = self.model.predict(np.array(sample).reshape((1, -1, self.input_dim)))[0]
        return self.prob

    # 입력으로 들어온 학습 데이터 집합 x와 레이블 y로 정책 신경망을 학습시킨다.
    # 입력으로 들어올 x와 y는 정책 학습기에서 준비한다.
    # Keras의 Sequential 클래스 함수인 train_on_batch()는 입력으로 들어온 학습 데이터 집합
    # (즉, 1개 배치)로 신경망을 한 번 학습한다.
    def train_on_batch(self, x, y):
        return self.model.train_on_batch(x, y)

    # 학습한 정책 신경망을 파일로 저장
    # 인자로 넘겨지는 model_path는 저장할 파일명을 의미한다.
    def save_model(self, model_path):
        if model_path is not None and self.model is not None:
            self.model.save_weights(model_path, overwrite=True) # save_weights()로 HDF5 파일로 저장

    # 저장한 정책 신경망을 불러오기 위한 함수
    def load_model(self, model_path):
        if model_path is not None:
            self.model.load_weights(model_path)
