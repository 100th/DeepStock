"""
가시화기 모듈 : 정책 신경망을 학습하는 과정에서 에이전트의 투자 상황,
정책 신경망의 투자 결정 상황, 포트폴리오 가치의 상황을 시간에 따라 연속적으로
보여주기 위해 시각화 기능을 담당하는 가시화기 클래스 (Visualizer)를 가진다.

* 속성
fig : 캔버스 같은 역할을 하는 Matplotlib의 Figure 클래스 객체
axes : 차트를 그리기 위한 Matplotlib의 Axes 클래스 객체

* 함수
prepare() : Figure를 초기화하고 일봉 차트를 출력
plot() : 일봉 차트를 제외한 나머지 차트들을 출력
save() : Figure를 그림 파일로 저장
clear() : 일봉 차트를 제외한 나머지 차트들을 초기화

* 가시화기 모듈이 만들어 내는 정보
Figure 제목 : 에포크 및 탐험률
Axes 1 : 종목의 일봉 차트
Axes 2 : 보유 주식 수 및 에이전트 행동 차트
Axes 3 : 정책 신경망 출력 및 탐험 차트
Axes 4 : 포트폴리오 가치 차트

* plot() 함수의 인자
epoch_str : Figure 제목으로 표시한 에포크
num_epoches : 총 수행할 에포크 수
epsilon : 탐험률
action_list : 에이전트가 수행할 수 있는 전체 행동 리스트
actions : 에이전트가 수행한 행동 배열
num_stocks : 주식 보유 수 배열
outvals : 정책 신경망의 출력 배열
exps : 탐험 여부 배열
initial_balance : 초기 자본금
pvs : 포트폴리오 가치 배열
"""

"""
mpl_finance 모듈 다운
https://github.com/matplotlib/mpl_finance
그 후 Anaconda에서 mpl_finance 폴더로 이동하고
$ python setup.py install 로 설치
"""

import numpy as np
import matplotlib.pyplot as plt
from mpl_finance import candlestick_ohlc


class Visualizer:

    def __init__(self):
        self.fig = None  # 캔버스 같은 역할을 하는 Matplotlib의 Figure 클래스 객체
        self.axes = None  # 차트를 그리기 위한 Matplotlib의 Axes 클래스 객체

    def prepare(self, chart_data):
        # 캔버스를 초기화하고 4개의 차트를 그릴 준비
        # 4행 1열 Figure 생성. 각 행에 해당하는 차트는 Axes 객체의 배열로 반환
        # subplots() 함수는 2개의 변수를 Tuple로 반환.
        # 첫 번째는 Figure 객체, 두 번째는 이 Figure 클래스 객체에 포함된 Axes 객체의 배열
        # 검정색 : k, 흰색 : w, 빨간색 : r, 파란색 : b, 초록색 : g, 노란색 : y
        self.fig, self.axes = plt.subplots(nrows=4, ncols=1, facecolor='w', sharex=True)
        for ax in self.axes:
            # 보기 어려운 과학적 표기 비활성화. 모든 Axes가 숫자 표기 단위로.
            ax.get_xaxis().get_major_formatter().set_scientific(False)
            ax.get_yaxis().get_major_formatter().set_scientific(False)
        # 차트 1. 종목 일봉 차트 (에포크에 상관없이 공통된 첫 번째 차트)
        self.axes[0].set_ylabel('Env.')  # y 축 레이블 표시
        # 거래량 가시화
        x = np.arange(len(chart_data)) # np.arrange(3)은 배열 [0, 1, 2] 반환
        volume = np.array(chart_data)[:, -1].tolist()
        self.axes[0].bar(x, volume, color='b', alpha=0.3) # 거래량을 표시하기 위해 막대 차트 그리기
        # ohlc란 open, high, low, close의 약자로 이 순서로된 2차원 배열 (N X 5)
        # 열의 수가 5인 이유는 첫 번째 열은 인덱스
        # N은 일봉의 수
        ax = self.axes[0].twinx()
        ohlc = np.hstack((x.reshape(-1, 1), np.array(chart_data)[:, 1:-1]))
        # self.axes[0]에 봉차트 출력
        # 양봉은 빨간색으로 음봉은 파란색으로 표시
        candlestick_ohlc(ax, ohlc, colorup='r', colordown='b') # colorup(양봉), colordown(음봉)

    def plot(self, epoch_str=None, num_epoches=None, epsilon=None,
            action_list=None, actions=None, num_stocks=None,
            outvals=None, exps=None, learning=None,
            initial_balance=None, pvs=None):
        x = np.arange(len(actions))  # 모든 차트가 공유할 x축 데이터
        # (actions, num_stocks, outvals, exps, pvs의 모든 크기가 같기 때문에 이들 중 하나인 actions의 크기만큼 배열을 생성하여 x축을 사용)
        actions = np.array(actions)  # 에이전트의 행동 배열. Matplotlib이 NumPy 배열을 입력으로 받기 때문에 np로 감싼다.
        outvals = np.array(outvals)  # 정책 신경망의 출력 배열
        pvs_base = np.zeros(len(actions)) + initial_balance  # 초기 자본금 배열.
        # 포트폴리오 가치 차트에서 초기 자본금에 직선을 그어서 포트폴리오 가치와 초기 자본금을 쉽게 비교할 수 있도록 배열(pvs_base)로 준비
        # NumPy zeros() 함수는 인자로 배열의 형태인 shape를 받아서 0으로 구성된 NumPy 배열을 반환
        # zeros(3)은 [0, 0, 0], zeros((2,2))는 [[0, 0], [0, 0]]

        # 차트 2. 에이전트 상태 (에이전트가 수행한 행동을 배경의 색, 보유 주식 수를 라인 차트로)
        colors = ['r', 'b']
        for actiontype, color in zip(action_list, colors):
            for i in x[actions == actiontype]:
                self.axes[1].axvline(i, color=color, alpha=0.1)  # 배경 색으로 행동 표시
                # 매수 행동의 배경 색 = 빨강, 매도 행동의 배경 색 = 파랑
        self.axes[1].plot(x, num_stocks, '-k')  # 보유 주식 수 그리기
        # 내장함수 zip()은 두 개의 배열에서 같은 인덱스의 요소를 순서대로 묶어 준다.
        # zip([1,2,3],[4,5,6])은 [(1,4),(2,5),(3,6)]
        # Matplotlib의 axvline()은 x축 위치에서 세로로 선을 긋는 함수. alpha는 선의 투명도.
        # -k는 검정색 실선

        # 차트 3. 정책 신경망의 출력 및 탐험
        for exp_idx in exps:                        # 탐험을 한 x축 인덱스
            # 탐험을 노란색 배경으로 그리기
            self.axes[2].axvline(exp_idx, color='y')
        for idx, outval in zip(x, outvals):         # 탐험을 하지 않은 지점
            color = 'white'
            if outval.argmax() == 0:
                color = 'r'  # 매수면 빨간색
            elif outval.argmax() == 1:
                color = 'b'  # 매도면 파란색
            # 행동을 빨간색 또는 파란색 배경으로 그리기
            self.axes[2].axvline(idx, color=color, alpha=0.1)
        styles = ['.r', '.b']
        for action, style in zip(action_list, styles):
            # 정책 신경망의 출력을 빨간색 점(매수), 파란색 점(매도)으로 그리기
            self.axes[2].plot(x, outvals[:, action], style)

        # 차트 4. 포트폴리오 가치
        # 초기 자본금을 가로로 곧게 그어서 손익을 쉽게 파악할 수 있도록
        self.axes[3].axhline(initial_balance, linestyle='-', color='gray')
        # fill_between() 함수는 x축 배열과 두 개의 y축 배열을 입력으로 받고 두 y축 배열의
        # 같은 인덱스 위치의 값 사이에 색을 칠한다. where 옵션으로 조건 추가 가능.
        # facecolor 옵션으로 칠할 색 지정. alpha로 투명도 조정 가능.
        # 포트폴리오 가치 > 초기 자본금 : 빨간색
        self.axes[3].fill_between(x, pvs, pvs_base,
                                  where=pvs > pvs_base, facecolor='r', alpha=0.1)
        # 포트폴리오 가치 < 초기 자본금 : 파란색
        self.axes[3].fill_between(x, pvs, pvs_base,
                                  where=pvs < pvs_base, facecolor='b', alpha=0.1)
        # 포트폴리오 가치는 검정색 실선
        self.axes[3].plot(x, pvs, '-k')
        # 학습을 수행한 위치를 표시
        for learning_idx, delayed_reward in learning:
            # 학습 위치를 초록색으로 그리기
            if delayed_reward > 0:
                self.axes[3].axvline(learning_idx, color='r', alpha=0.1)
            else:
                self.axes[3].axvline(learning_idx, color='b', alpha=0.1)

        # 에포크 및 탐험 비율
        self.fig.suptitle('Epoch %s/%s (e=%.2f)' % (epoch_str, num_epoches, epsilon))
        # 캔버스 레이아웃 조정
        plt.tight_layout() # Figure 크기에 알맞게 내부 차트들의 크기 조정
        plt.subplots_adjust(top=.9)

    # Figure를 초기화하고 저장하는 함수
    # 학습 과정에서 변하지 않는 환경에 관한 차트를 제외하고 그 외 차트들을 초기화
    # 입력으로 받는 xlim은 모든 차트의 x축 값 범위를 설정해 줄 튜플
    def clear(self, xlim):
        for ax in self.axes[1:]:
            ax.cla()  # 그린 차트 지우기
            ax.relim()  # x축과 y축 범위(limit)를 초기화
            ax.autoscale()  # 자동 크기 조정 기능 활성화. 스케일 재설정
        # y축 레이블 재설정
        self.axes[1].set_ylabel('Agent')
        self.axes[2].set_ylabel('PG')
        self.axes[3].set_ylabel('PV')
        for ax in self.axes:
            ax.set_xlim(xlim)  # x축 limit 재설정
            ax.get_xaxis().get_major_formatter().set_scientific(False)  # 과학적 표기 비활성화 (있는 그대로 보여주기 위해)
            ax.get_yaxis().get_major_formatter().set_scientific(False)  # 과학적 표기 비활성화
            ax.ticklabel_format(useOffset=False)  # x축 간격을 일정하게 설정

    def save(self, path):
        plt.savefig(path)   # Figure를 그림파일로 저장
