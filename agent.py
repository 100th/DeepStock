"""
Agent : 투자 행동을 수행하고 투자금과 보유 주식을 관리

* 속성
initial_balance : 초기 투자금
balance : 현금 잔고
num_stocks : 보유 주식 수
portfolio_value : 포트폴리오 가치(투자금 잔고 + 주식 현재가 * 보유 주식 수)

* 함수
reset() : 에이전트의 상태를 초기화
set_balance() : 초기 자본금을 설정
get_states() : 에이전트 상태를 획득
decide_action() : 탐험 또는 정책 신경망에 의한 행동 결정
validate_action() : 행동의 유효성 판단
decide_trading_unit() : 매수 또는 매도할 주식 수 결정
act() : 행동 수행
"""
import numpy as np

class Agent:
    # 에이전트 상태가 구성하는 값 개수
    STATE_DIM = 2  # 주식 보유 비율, 포트폴리오 가치 비율

    # 매매 수수료 및 세금
    TRADING_CHARGE = 0  # 거래 수수료 미고려 (일반적으로 0.015%)
    TRADING_TAX = 0  # 거래세 미고려 (실제 0.3%)

    # 행동에 특정 값을 부여하여 상수로 다룸
    ACTION_BUY = 0  # 매수
    ACTION_SELL = 1  # 매도
    ACTION_HOLD = 2  # 홀딩 (관망)
    ACTIONS = [ACTION_BUY, ACTION_SELL]  # 인공 신경망에서 확률을 구할 행동들
    NUM_ACTIONS = len(ACTIONS)  # 인공 신경망에서 고려할 출력값의 개수
    #매수와 매도에 대한 확률만 계산하고, 결정한 행동을 할 수 없을 때만 관망 행동을 한다.

    def __init__(
        self, environment, min_trading_unit=1, max_trading_unit=2,
        delayed_reward_threshold=.05):
        # Environment 객체
        self.environment = environment  # 현재 주식 가격을 가져오기 위해 환경 참조

        # 최소 매매 단위, 최대 매매 단위, 지연보상 임계치
        self.min_trading_unit = min_trading_unit  # 최소 단일 거래 단위
        self.max_trading_unit = max_trading_unit  # 최대 단일 거래 단위. 이 값을 크게 잡으면 결정한 행동에 대한 확신이 높을 떄 더 많이 매수 또는 매도 하도록 설계.
        self.delayed_reward_threshold = delayed_reward_threshold  # 지연보상 임계치. 손익률이 이 값을 넘으면 지연 보상이 발생

        # Agent 클래스의 속성
        self.initial_balance = 0  # 초기 자본금. 투자 시작 시점의 보유 현금.
        self.balance = 0  # 현재 현금 잔고
        self.num_stocks = 0  # 보유 주식 수
        self.portfolio_value = 0  # balance + num_stocks * {현재 주식 가격}
        self.base_portfolio_value = 0  # 직전 학습 시점의 PV. 현재 포트폴리오 가치가 증가했는지 감소했는지를 비교할 기준이 됨.
        self.num_buy = 0  # 매수 횟수
        self.num_sell = 0  # 매도 횟수
        self.num_hold = 0  # 홀딩 횟수
        self.immediate_reward = 0  # 즉시 보상. 행동을 수행한 시점에서 수익이 발생하면 1, 아니면 -1

        # Agent 클래스의 상태
        self.ratio_hold = 0  # 주식 보유 비율 (현재 보유 주식 수 / 최대 보유 주식 수)
        self.ratio_portfolio_value = 0  # 포트폴리오 가치 비율. 직전 지연 보상이 발생했을 때의 포트폴리오 가치 대비 현재 포트폴리오 가치 비율

    # Agent 클래스의 함수 : Get/Set 함수
    # 학습 단계에서 한 에포크마다 에이전트의 상태를 초기화해야 함.
    def reset(self):
        self.balance = self.initial_balance
        self.num_stocks = 0
        self.portfolio_value = self.initial_balance
        self.base_portfolio_value = self.initial_balance
        self.num_buy = 0
        self.num_sell = 0
        self.num_hold = 0
        self.immediate_reward = 0
        self.ratio_hold = 0
        self.ratio_portfolio_value = 0

    # 에이전트의 초기 자본금 설정
    def set_balance(self, balance):
        self.initial_balance = balance

    # 에이전트의 상태를 반환
    def get_states(self):
        # 주식 보유 비율 = 보유 주식 수 / (포트폴리오 가치 / 현재 주가)
        # 현재 상태에서 가장 많이 가질 수 있는 주식 수 대비 현재 보유한 주식의 비율
        # 주식 수가 너무 적으면 매수의 관점에서 투자하며, 너무 많으면 매도의 관점에서 투자한다.
        # 즉, 보유 주식 수를 투자 행동 결정에 영향을 주기 위해서 정책 신경망의 입력에 포함해야 함.
        self.ratio_hold = self.num_stocks / int(
            self.portfolio_value / self.environment.get_price())

        # 포트폴리오 가치 비율 = 현재 포트폴리오 가치 / 기준 포트폴리오 가치
        # 기준 포트폴리오 가치 : 직전에 목표 수익 또는 손익률을 달성했을 때의 포트폴리오 가치
        # 이 값은 현재 수익이 발생했는지 손실이 발생했는지 판단할 수 있는 값
        # 0에 가까우면 손실이 큰 것이고, 1보다 크면 수익이 발생했다는 뜻
        # 수익률이 목표 수익률에 가까우면 매도의 관점에서 투자
        # 수익률이 투자 행동 결정에 영향을 줄 수 있기 때문에 이 값을 에이전트의 상태로 정하고 정책 신경망의 입력값에 포함
        self.ratio_portfolio_value = self.portfolio_value / self.base_portfolio_value
        return (
            self.ratio_hold,
            self.ratio_portfolio_value
        )

    # Agent 클래스의 함수 : 행동 결정 및 유효성 검사 함수
    # 에이전트가 행동을 결정하고 결정한 행동의 유효성을 검사하는 함수를 보여줌
    # 입력으로 들어온 epsilon의 확률로 무작위로 행동을 결정하고, 그렇지 않은 경우에 정책 신경망을 통해 행동을 결정
    def decide_action(self, policy_network, sample, epsilon):
        confidence = 0.
        # 탐험 결정
        if np.random.rand() < epsilon:  # 0에서 1 사이의 랜덤 값을 생성하고 (NumPy에서 나옴 random 모듈) 이 값이 엡실론보다 작으면 무작위로 행동을 결정
            exploration = True
            # randint(low, hight=None) 함수는 high를 넣지 않은 경우 0에서 low 사이의 정수를 랜덤으로 생성하고
            # high를 넣은 경우 low에서 high 사이의 정수를 생성.
            action = np.random.randint(self.NUM_ACTIONS)  # 무작위로 행동 결정. 여기서 NUM_ACTIONS는 2이다. 그러므로 랜덤으로 0(매수) 또는 1(매도) 값이 결정됨
        else:
            exploration = False
            # 탐험을 하지 않는 경우 정책 신경망을 통해 행동을 결정.
            # 정책 신경망 클래스의 predict() 함수를 사용하여 현재의 상태에서 매수와 매도의 확률을 받아온다.
            # 이렇게 받아온 확률 중에서 큰 값을 선택하여 행동으로 결정한다.
            probs = policy_network.predict(sample)  # 각 행동에 대한 확률
            action = np.argmax(probs)
            confidence = probs[action]
        return action, confidence, exploration

    # '신용 매수', '공매도'는 고려 안한다.
    # 신용 매수 : 잔금이 부족하더라도 돈을 빌려서 매수를 하는 것
    # 공매도 : 주식을 보유하지 않고 있더라도 미리 매도하고 나중에 주식을 사서 갚는 방식의 거래

    # 이렇게 결정한 행동은 특정 상황에서는 수행할 수 없을 수도 있다.
    # 예를 들어, 매수를 결정했는데 잔금이 1주를 매수하기에도 부족한 경우
    # 매도를 결정했는데 보유하고 있는 주식이 하나도 없는 경우에 결정한 행동을 수행할 수 없다.
    # 그래서 결정한 행동을 수행할 수 있는지를 확인하기 위해 validate_action() 함수를 사용.
    def validate_action(self, action):
        validity = True
        if action == Agent.ACTION_BUY:
            # 매수 결정에 대해서 적어도 1주를 살 수 있는지 잔금 확인. 설정된 거래 수수료까지 고려.
            if self.balance < self.environment.get_price() * (
                1 + self.TRADING_CHARGE) * self.min_trading_unit:
                validity = False
        elif action == Agent.ACTION_SELL:
            # 주식 잔고가 있는지 확인
            if self.num_stocks <= 0:
                validity = False
        return validity

    # Agent 클래스의 함수 : 매수/매도 단위 결정 함수
    # 정책 신경망에서 결정한 행동의 확률에 따라서 매수 또는 매도의 단위를 조정하는 함수
    # 확률이 높을수록 매수 또는 매도하는 단위를 크게 정해준다.
    def decide_trading_unit(self, confidence):
        if np.isnan(confidence):
            return self.min_trading_unit
        added_trading = max(min(
            int(confidence * (self.max_trading_unit - self.min_trading_unit)),
            self.max_trading_unit-self.min_trading_unit
        ), 0)
        return self.min_trading_unit + added_trading

    # Agent 클래스의 함수 : 투자 행동 수행 함수 (행동 준비)
    # 인자로 받은 action : 탐험 또는 정책 신경망을 통해 결정한 행동으로 매수와 매도를 의미하는 0 또는 1의 값
    # confidence : 정책 신경망을 통해 결정한 경우 결정한 행동에 대한 소프트맥스 확률 값.
    def act(self, action, confidence):
        if not self.validate_action(action):    # 이 행동을 할 수 있는지 확인
            action = Agent.ACTION_HOLD      # 할 수 없으면 관망

        # 환경 객체에서 현재의 주가 받아오기
        # 이 가격은 매수 금액, 매도 금액, 포트폴리오 가치를 계산할 때 사용됨.
        curr_price = self.environment.get_price()

        # 즉시 보상 초기화. 에이전트가 행동을 할 때마다 결정되기 때문에.
        self.immediate_reward = 0

        # Agent 클래스의 함수 : 투자 행동 수행 함수 (매수)
        # 매수
        if action == Agent.ACTION_BUY:
            # 매수할 단위(살 주식 수)를 판단
            trading_unit = self.decide_trading_unit(confidence)
            balance = self.balance - curr_price * (1 + self.TRADING_CHARGE) * trading_unit  # 매수 후 잔금 확인
            # 보유 현금이 모자랄 경우 보유 현금으로 가능한 만큼 최대한 매수
            if balance < 0:
                # 결정한 매수 단위가 최대 단일 거래 단위를 넘어가면 최대 단일 거래 단위로 제한하고
                # 최소 거래 단위보다 최소한 1주를 매수하도록 한다.
                trading_unit = max(min(
                    int(self.balance / (
                        curr_price * (1 + self.TRADING_CHARGE))), self.max_trading_unit),
                    self.min_trading_unit
                )
            # 매수할 단위에 수수료를 적용하여 총 매수 금액 산정
            invest_amount = curr_price * (1 + self.TRADING_CHARGE) * trading_unit
            self.balance -= invest_amount  # 보유 현금을 갱신. 이 금액을 현재 잔금에서 뺌.
            self.num_stocks += trading_unit  # 보유 주식 수를 갱신. 주식 보유 수를 투자 단위만큼 늘려줌.
            self.num_buy += 1  # 매수 횟수 증가 (통계 정보)

        # Agent 클래스의 함수 : 투자 행동 수행 함수 (매도 및 관망)
        # 매도
        elif action == Agent.ACTION_SELL:
            # 매도할 단위를 판단
            trading_unit = self.decide_trading_unit(confidence)
            # 보유 주식이 모자랄 경우 가능한 만큼 최대한 매도
            # 현재 가지고 있는 주식 수보다 결정한 매도 단위가 많으면 안 되기 때문에
            # 현재 보유 주식 수를 최대 매도 단위로 제한함.
            trading_unit = min(trading_unit, self.num_stocks)
            # 매도 금액 계산. 정해준 매도 수수료와 거래세를 모두 고려.
            invest_amount = curr_price * (
                1 - (self.TRADING_TAX + self.TRADING_CHARGE)) * trading_unit
            self.num_stocks -= trading_unit  # 보유 주식 수를 갱신. 보유 주식 수를 매도한 만큼 뺌.
            self.balance += invest_amount  # 보유 현금을 갱신. 매도하여 현금화한 금액을 잔고에 더해 줌.
            self.num_sell += 1  # 매도 횟수 증가 (통계를 위한 변수)

        # 홀딩
        # 결정한 매수나 매도 행동을 수행할 수 없는 경우에 적용.
        # 홀딩은 아무 것도 하지 않는 것이기 때문에 보유 주식 수나 잔고에 영향을 미치지 않는다.
        # 대신 가격 변동은 있을 수 있으므로 포트폴리오 가치는 변동된다.
        elif action == Agent.ACTION_HOLD:
            self.num_hold += 1  # 홀딩 횟수 증가

        # Agent 클래스의 함수 : 투자 행동 수행 함수 (포트폴리오 가치 갱신 및 지연 보상 판단)
        # 포트폴리오 가치 갱신
        # 포트폴리오 가치는 잔고, 주식 보요 수, 현재 주식 가격에 의해 결정된다.
        self.portfolio_value = self.balance + curr_price * self.num_stocks
        # 기준 포트폴리오 가치에서 현재 포트폴리오 가치의 등락률을 계산한다.
        # (기준 포트폴리오 가치는 과거에 학습을 수행한 시점의 포트폴리오 가치를 의미)
        profitloss = (
            (self.portfolio_value - self.base_portfolio_value) / self.base_portfolio_value)

        # 즉시 보상 판단 : 수익이 발생한 상태면 1, 그렇지 않으면 -1.
        self.immediate_reward = 1 if profitloss >= 0 else -1

        # 지연 보상 판단
        # 지연 보상은 포트폴리오 가치의 등락률인 profitloss가 지연 보상 임계치인
        # delayed_reward_threshold를 수익으로 초과하는 경우 1, 손실로 초과하는 경우 -1, 그 외엔 0
        # 여기서는 지연 보상이 0이 아닌 경우 학습을 수행한다.
        # 즉, 지연 보상 임계치를 초과하는 수익이 났으면 이전에 했던 행동들이 잘했다고 보고 긍정적으로 학습하고,
        # 지연 보상 임계치를 초과하는 손실이 났으면 이전 행동들에 문제가 있다고 보고 부정적으로 학습한다.
        if profitloss > self.delayed_reward_threshold:
            delayed_reward = 1
            # 목표 수익률 달성하여 기준 포트폴리오 가치 갱신
            self.base_portfolio_value = self.portfolio_value
        elif profitloss < -self.delayed_reward_threshold:
            delayed_reward = -1
            # 손실 기준치를 초과하여 기준 포트폴리오 가치 갱신
            self.base_portfolio_value = self.portfolio_value
        else:
            delayed_reward = 0
        return self.immediate_reward, delayed_reward
