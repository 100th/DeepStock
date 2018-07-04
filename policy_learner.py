"""
정책 학습기 모듈 : 정책 학습기 클래스를 가지고 일련의 학습 데이터를 준비하고 정책 신경망을 학습
"""

import os   # 폴더 생성, 파일 경로 준비
import locale   # 통화(currency) 문자열 포맷
import logging  # 학습 과정 중에 정보를 기록하기 위해
import numpy as np
import settings # 투자 설정, 로깅 설정
from environment import Environment
from agent import Agent
from policy_network import PolicyNetwork
from visualizer import Visualizer


logger = logging.getLogger(__name__)
locale.setlocale(locale.LC_ALL, 'ko_KR.UTF-8')


class PolicyLearner:

    def __init__(self, stock_code, chart_data, training_data=None,
                 min_trading_unit=1, max_trading_unit=2,
                 delayed_reward_threshold=.05, lr=0.01):
        self.stock_code = stock_code  # 종목코드
        self.chart_data = chart_data
        self.environment = Environment(chart_data)  # 환경 객체
        # Environment 클래스는 차트 데이터를 순서대로 읽으면서 주가, 거래량 등의 환경을 제공한다.
        # 에이전트 객체
        self.agent = Agent(self.environment,
                           min_trading_unit=min_trading_unit,
                           max_trading_unit=max_trading_unit,
                           delayed_reward_threshold=delayed_reward_threshold)
        self.training_data = training_data  # 학습 데이터. 학습에 사용할 특징(feature)들을 포함한다.
        self.sample = None
        self.training_data_idx = -1
        # 정책 신경망; 입력 크기(17개) = 학습 데이터의 크기(15개) + 에이전트 상태 크기(2개)
        self.num_features = self.training_data.shape[1] + self.agent.STATE_DIM
        self.policy_network = PolicyNetwork(
            input_dim=self.num_features, output_dim=self.agent.NUM_ACTIONS, lr=lr)
        self.visualizer = Visualizer()  # 가시화 모듈

    def reset(self): # 에포크 초기화 함수 부분
        self.sample = None
        self.training_data_idx = -1 # 학습 데이터를 읽어가면서 이 값은 1씩 증가

    def fit( #학습 함수 선언 부분. 핵심 함수이다.
        # num_epoches : 수행할 반복 학습의 전체 횟수. (너무 크게 잡으면 학습 소요 시간이 너무 길어짐)
        # max_memory : 배치 학습 데이터를 만들기 위해 과거 데이터를 저장할 배열.
        # balance : 에이전트의 초기 투자 자본금을 정해주기 위한 인자
        # discount_factor : 지연 보상이 발생했을 때 그 이전 지연 보상이 발생한 시점과 현재 지연 보상이 발생한
        # 시점 사이에서 수행한 행동들 전체에 현재의 지연 보상을 적용한다.
        # 과거로 갈수록 현재 지연 보상을 적용할 판단 근거가 흐려지기 때문에 먼 과걱의 행동일수록 할인 요인을
        # 적용하여 지연 보상을 약하게 적용한다.
        # start_epsilon : 초기 탐험 비율. 학습이 전혀 되어 있지 않은 초기에는 탐험 비율을 크게 해서
        # 더 많은 탐험, 즉 무작위 투자를 수행하도록 해야 한다. 탐험을 통해 특정 상황에서 좋은 행동과
        # 그렇지 않은 행동을 결정하기 위한 경험을 쌓는다.
        # learning : 학습 유무를 정하는 boolean 값. 학습을 마치면 학습된 정책 신경망 모델이 만들어지는데,
        # 이렇게 학습을 해서 정책 신경망 모델을 만들고자 한다면 learning을 True로,
        # 학습된 모델을 가지고 투자 시뮬레이션만 하려 한다면 learning을 False로 준다.
        self, num_epoches=1000, max_memory=60, balance=10000000,
        discount_factor=0, start_epsilon=.5, learning=True):
        logger.info("LR: {lr}, DF: {discount_factor}, "
                    "TU: [{min_trading_unit}, {max_trading_unit}], "
                    "DRT: {delayed_reward_threshold}".format(
            lr=self.policy_network.lr,
            discount_factor=discount_factor,
            min_trading_unit=self.agent.min_trading_unit,
            max_trading_unit=self.agent.max_trading_unit,
            delayed_reward_threshold=self.agent.delayed_reward_threshold
        ))

        # 가시화 준비
        # 차트 데이터는 변하지 않으므로 미리 가시화
        self.visualizer.prepare(self.environment.chart_data)

        # 가시화 결과 저장할 폴더 준비
        epoch_summary_dir = os.path.join(   # 폴더의 경로를 변수로 저장
            settings.BASE_DIR, 'epoch_summary/%s/epoch_summary_%s' % (
                self.stock_code, settings.timestr))
        if not os.path.isdir(epoch_summary_dir):    # 해당 경로가 없으면 이 경로를 구성하는 폴더들을 생성
            os.makedirs(epoch_summary_dir)

        # 에이전트 초기 자본금 설정
        self.agent.set_balance(balance)

        # 학습에 대한 정보 초기화
        max_portfolio_value = 0
        epoch_win_cnt = 0

        # 학습 반복
        for epoch in range(num_epoches):
            # 에포크 관련 정보 초기화
            loss = 0. # 정책 신경망의 결과가 학습 데이터와 얼마나 차이가 있는지를 저장하는 변수. 학습 중 줄어드는게 좋음.
            itr_cnt = 0 # 수행한 에포크 수를 저장
            win_cnt = 0 # 수행한 에포크 중에서 수익이 발생한 에포크 수를 저장. 포트폴리오 가치가 초기 자본금보다 높아진 에포크 수.
            exploration_cnt = 0 # 무작위 투자를 수행한 횟수. epsilon이 0.1이고 100번의 투자 결정이 있으면 약 10번의 무작위 투자를 함
            batch_size = 0
            pos_learning_cnt = 0    # 수익이 발생하여 긍정적 지연 보상을 준 수
            neg_learning_cnt = 0    # 손실이 발생하여 부정적 지연 보상을 준 수

            # 메모리 초기화
            # 메모리 리스트에 저장하는 데이터는 샘플, 행동, 즉시보상, 정책 신경망의 출력, 포트폴리오 가치,
            # 보유 주식 수, 탐험 위치, 학습 위치이다.
            memory_sample = []
            memory_action = []
            memory_reward = []
            memory_prob = []
            memory_pv = []
            memory_num_stocks = []
            memory_exp_idx = []
            memory_learning_idx = []

            # 환경, 에이전트, 정책 신경망 초기화
            self.environment.reset()
            self.agent.reset()
            self.policy_network.reset()
            self.reset()

            # 가시화 초기화
            self.visualizer.clear([0, len(self.chart_data)])    # 2, 3, 4번째 차트를 초기화함. x축 데이터 범위를 파라미터로

            # 학습을 진행할 수록 탐험 비율 감소
            # 무작위 투자 비율인 epsilon 값을 정함
            # fit() 함수의 인자로 넘어오는 최초 무작위 투자 비율인 start_epsilon 값에 현재 epoch 수에 학습 진행률을 곱해서 정함
            # ex) start_epsilon이 0.3이면 첫 번째 에포크에서는 30%의 확률로 무작위 투자를 진행함.
            # 수행할 에포크 수가 100이라고 했을 때, 50번째 에포크에서는 0.3 * (1 - 49/99) = 0.51
            if learning:
                epsilon = start_epsilon * (1. - float(epoch) / (num_epoches - 1))
            else:
                epsilon = 0

            while True:
                # 샘플 생성
                next_sample = self._build_sample()
                if next_sample is None: # 마지막까지 데이터를 다 읽은 것이므로 반복문 종료
                    break

                # 정책 신경망 또는 탐험에 의한 행동 결정
                # 매수와 매도 중 하나를 결정. 이 행동 결정은 무작위 투자 비율인 epsilon 값의 확률로 무작위로 하거나
                # 그렇지 않은 경우 정책 신경망의 출력을 통해 결정된다. 정책 신경망의 출력은 매수를 했을 때와 매도를 했을 때의
                # 포트폴리오 가치를 높일 확률을 의미한다. 즉 매수에 대한 정책 신경망 출력이 매도에 대한 출력보다 높으면 매수, 반대는 매도
                # decide_action() 함수가 반환하는 값은 세 가지.
                # 결정한 행동인 action, 결정에 대한 확신도인 confidence, 무작위 투자 유무인 exploration.
                action, confidence, exploration = self.agent.decide_action(
                    self.policy_network, self.sample, epsilon)

                # 결정한 행동을 수행하고 (act 함수) 즉시 보상과 지연 보상 획득 (act가 반환함)
                immediate_reward, delayed_reward = self.agent.act(action, confidence)

                # 행동 및 행동에 대한 결과를 기억 (메모리에 저장)
                memory_sample.append(next_sample) # 각 데이터를 메모리에 추가
                memory_action.append(action)
                memory_reward.append(immediate_reward)
                memory_pv.append(self.agent.portfolio_value)
                memory_num_stocks.append(self.agent.num_stocks)
                memory = [( # 학습데이터의 샘플, 에이전트 행동, 즉시보상, 포트폴리오 가치, 보유 주식 수를 저장하는 배열
                    memory_sample[i],
                    memory_action[i],
                    memory_reward[i])
                    for i in list(range(len(memory_action)))[-max_memory:]
                ]
                if exploration: # 무작위 투자로 행동을 결정한 경우에 현재의 인덱스를 memory_exp_idx에 저장
                    memory_exp_idx.append(itr_cnt)
                    memory_prob.append([np.nan] * Agent.NUM_ACTIONS) # memory_prob은 정책 신경망의 출력을 그대로 저장하는 배열
                                            # 무작위 투자에서는 정책 신경망의 출력이 없기 때문에 NumPy의 Not A Number(nan) 값을 넣어줌
                else: # 무작위 투자가 아닌 경우 정책 신경망의 출력을 그대로 저장
                    memory_prob.append(self.policy_network.prob)
                # 메모리 변수들의 목적은 (1) 학습에서 배치 학습 데이터로 사용 (2) 가시화기에서 차트를 그릴 때 사용

                # 반복에 대한 정보 갱신
                batch_size += 1 # 배치 크기
                itr_cnt += 1    # 반복 카운팅 횟수
                exploration_cnt += 1 if exploration else 0  # 무작위 투자 횟수 (탐험을 한 경우에만)
                win_cnt += 1 if delayed_reward > 0 else 0   # 수익이 발생한 횟수를 증가시킴 (지연 보상이 0보다 큰 경우에만)

                # 지연 보상이 발생한 경우 학습을 수행하는 부분
                # 학습 모드이고 지연 보상이 존재할 경우 정책 신경망 갱신
                # 학습 없이 메모리가 최대 크기만큼 다 찼을 경우 즉시 보상으로 지연 보상을 대체하여 학습을 진행
                if delayed_reward == 0 and batch_size >= max_memory:
                    delayed_reward = immediate_reward
                    self.agent.base_portfolio_value = self.agent.portfolio_value
                if learning and delayed_reward != 0:
                    # 배치 학습 데이터 크기
                    batch_size = min(batch_size, max_memory) # 배치 데이터 크기는 memory 변수의 크기인 max_memory보다는 작아야 함
                    # 배치 학습 데이터 생성
                    x, y = self._get_batch(
                        memory, batch_size, discount_factor, delayed_reward)
                    if len(x) > 0:  # 긍정 학습과 부정 학습 횟수 세기
                        if delayed_reward > 0:
                            pos_learning_cnt += 1
                        else:
                            neg_learning_cnt += 1
                        # 정책 신경망 갱신
                        loss += self.policy_network.train_on_batch(x, y)    # 준비한 배치 데이터로 학습을 진행함
                        memory_learning_idx.append([itr_cnt, delayed_reward])   # 학습이 진행된 인덱스를 저장함
                    batch_size = 0  # 학습이 진행되었으니 배치 데이터 크기를 초기화함

            # 에포크 관련 정보 가시화
            num_epoches_digit = len(str(num_epoches))   # 총 에포크 수의 문자열 길이를 확인함. 총 에포크 수가 1000이면 길이는 4
            epoch_str = str(epoch + 1).rjust(num_epoches_digit, '0')    # 현재 에포크 수를 4자리 문자열로 만들어 줌. 첫 에포크는 1 -> 0001로 문자열을 자리수에 맞게 정렬

            self.visualizer.plot(
                epoch_str=epoch_str, num_epoches=num_epoches, epsilon=epsilon,
                action_list=Agent.ACTIONS, actions=memory_action,
                num_stocks=memory_num_stocks, outvals=memory_prob,
                exps=memory_exp_idx, learning=memory_learning_idx,
                initial_balance=self.agent.initial_balance, pvs=memory_pv
            )
            self.visualizer.save(os.path.join(  #가시화한 에포크 수행 결과를 파일로 저장함
                epoch_summary_dir, 'epoch_summary_%s_%s.png' % (
                    settings.timestr, epoch_str)))

            # 에포크 관련 정보 로그 기록
            # 총 에포크 중에서 몇 번째 에포크를 수행했는지, 탐험률, 탐험 횟수, 매수 횟수, 매도 횟수, 관망 횟수,
            # 보유 주식 수, 포트폴리오 가치, 긍정적 학습 횟수, 부정적 학습 횟수, 학습 손실을 로그로 남긴다.
            if pos_learning_cnt + neg_learning_cnt > 0:
                loss /= pos_learning_cnt + neg_learning_cnt
            logger.info("[Epoch %s/%s]\tEpsilon:%.4f\t#Expl.:%d/%d\t"
                        "#Buy:%d\t#Sell:%d\t#Hold:%d\t"
                        "#Stocks:%d\tPV:%s\t"
                        "POS:%s\tNEG:%s\tLoss:%10.6f" % (
                            epoch_str, num_epoches, epsilon, exploration_cnt, itr_cnt,
                            self.agent.num_buy, self.agent.num_sell, self.agent.num_hold,
                            self.agent.num_stocks,
                            locale.currency(self.agent.portfolio_value, grouping=True),
                            pos_learning_cnt, neg_learning_cnt, loss))

            # 학습 관련 정보 갱신
            # 하나의 에포크 수행이 완료되었기 때문에 전체 학습에 대한 통계 정보를 갱신한다.
            # 관리하고 있는 학습 통계 정보는 달성한 최대 포트 폴리오 가치 max_portfolio_value와 쉭이 발생한 에포크의 수 epoch_win_cnt이다.
            max_portfolio_value = max(
                max_portfolio_value, self.agent.portfolio_value)
            if self.agent.portfolio_value > self.agent.initial_balance:
                epoch_win_cnt += 1
        # 여기 까지가 학습 반복 for문의 코드 블록이다.

        # 학습 관련 정보 로그 기록
        logger.info("Max PV: %s, \t # Win: %d" % (
            locale.currency(max_portfolio_value, grouping=True), epoch_win_cnt))

    def _get_batch(self, memory, batch_size, discount_factor, delayed_reward):  # 미니 배치 데이터 생성
        x = np.zeros((batch_size, 1, self.num_features))    # x는 일련의 학습 데이터 및 에이전트 상태
        # x 배열의 형태는 배치 데이터 크기, 학습 데이터 특징 크기로 2차원으로 구성됨.
        y = np.full((batch_size, self.agent.NUM_ACTIONS), 0.5)  # y는 일련의 지연 보상
        # y 배열의 형태는 배치 데이터 크기, 정책 신경망이 결정하는 에이전트 행동의 수. 2차원으로. 0.5 일괄적으로 채워둠.

        for i, (sample, action, reward) in enumerate(
                reversed(memory[-batch_size:])):
            x[i] = np.array(sample).reshape((-1, 1, self.num_features)) # 특징벡터 지정
            y[i, action] = (delayed_reward + 1) / 2 # 지연 보상으로 정답(레이블)을 설정하여 학습 데이터 구성
            # 지연 보상이 1인 경우 1로, -1인 경우 0으로 레이블을 지정
            if discount_factor > 0:
                y[i, action] *= discount_factor ** i    # 할인 요인이 있을 경우
        return x, y

    # 학습 데이터를 구성하는 샘플 하나를 생성하는 함수
    def _build_sample(self):
        self.environment.observe()  # 차트 데이터의 현재 인덱스에서 다음 인덱스 데이터를 읽음
        if len(self.training_data) > self.training_data_idx + 1:    # 학습 데이터의 다음 인덱스가 존재하는지 확인
            self.training_data_idx += 1
            self.sample = self.training_data.iloc[self.training_data_idx].tolist()  # 인덱스의 데이터를 받아와서 sample로 저장
            self.sample.extend(self.agent.get_states())     # sample에 에이전트 상태를 15개에서 +2개하여 17개로.
            return self.sample
        return None

    # 학습된 정책 신경망 모델로 주식투자 시뮬레이션을 진행
    def trade(self, model_path=None, balance=2000000):
        if model_path is None:
            return
        self.policy_network.load_model(model_path=model_path)   # 학습된 신경망 모델을 정책 신경망 객체의 load_model로 적용
        self.fit(balance=balance, num_epoches=1, learning=False)
        # 이 함수는 학습된 정책 신경망으로 투자 시뮬레이션을 하는 것이므로 반복 투자를 할 필요가 없기 때문에
        # 총 에포크 수 num_epoches를 1로 주고, learning 인자에 False를 넘겨준다.
        # 이렇게 하면 학습을 진행하지 않고 정책 신경망에만 의존하여 투자 시뮬레이션을 진행한다.
        # 물론 무작위 투자는 수행하지 않음.
