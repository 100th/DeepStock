# DeepStock
> Not yet

## About
> Not yet

## Development Environment
- Windows 10 64bit
- Python 3.6
- PyCharm
- Atom

## Library
- NumPy
- Pandas
- Matplotlib
- Tensorflow
- Keras

## Since
> 2018.06.22 ~

## File Order
#### 1. environment : 투자할 종목의 차트 데이터를 관리하는 모듈
- chart_data : 주식 종목의 차트 데이터
- observation : 현재 관측치
- idx : 차트 데이터에서의 현재 위치
  
- reset() : idx와 observation을 초기화
- observe() : idx를 다음 위치로 이동하고 observation을 업데이트
- get_price() : 현재 observation에서 종가를 획득
  
#### 2. agent : 투자 행동을 수행하고 투자금과 보유 주식을 관리
- initial_balance : 초기 투자금
- balance : 현금 잔고
- num_stocks : 보유 주식 수
- portfolio_value : 포트폴리오 가치(투자금 잔고 + 주식 현재가 * 보유 주식 수)
  
- reset() : 에이전트의 상태를 초기화
- set_balance() : 초기 자본금을 설정
- get_states() : 에이전트 상태를 획득
- decide_action() : 탐험 또는 정책 신경망에 의한 행동 결정
- validate_action() : 행동의 유효성 판단
- decide_trading_unit() : 매수 또는 매도할 주식 수 결정
- act() : 행동 수행

#### 3. policy_network : 투자 행동을 결정하기 위해 신경망을 관리하는 정책 신경망 클래스 (PolicyNetwork)를 가짐
- model : 케라스 라이브러리로 구성한 LSTM 신경망 모델
- prob : 가장 최근에 계산한 투자 행동별 확률

- reset() : prob 변수를 초기화
- predict() : 신경망을 통해 투자 행동별 확률 계산
- train_on_batch() : 배치 학습을 위한 데이터 생성
- save_model() : 학습한 신경망을 파일로 저장
- load_model() : 파일로 저장한 신경망을 로드

#### 4. visualizer : Agent의 투자 상황, 정책 신경망의 투자 결정 상황, 포트폴리오 가치의 상황의 시각화 기능 담당
- fig : 캔버스 같은 역할을 하는 Matplotlib의 Figure 클래스 객체
- axes : 차트를 그리기 위한 Matplotlib의 Axes 클래스 객체

- prepare() : Figure를 초기화하고 일봉 차트를 출력
- plot() : 일봉 차트를 제외한 나머지 차트들을 출력
- save() : Figure를 그림 파일로 저장
- clear() : 일봉 차트를 제외한 나머지 차트들을 초기화

##### * 가시화기 모듈이 만들어 내는 정보
- Figure 제목 : 에포크 및 탐험률
- Axes 1 : 종목의 일봉 차트
- Axes 2 : 보유 주식 수 및 에이전트 행동 차트
- Axes 3 : 정책 신경망 출력 및 탐험 차트
- Axes 4 : 포트폴리오 가치 차트

##### * plot() 함수의 인자
- epoch_str : Figure 제목으로 표시한 에포크
- num_epoches : 총 수행할 에포크 수
- epsilon : 탐험률
- action_list : 에이전트가 수행할 수 있는 전체 행동 리스트
- actions : 에이전트가 수행한 행동 배열
- num_stocks : 주식 보유 수 배열
- outvals : 정책 신경망의 출력 배열
- exps : 탐험 여부 배열
- initial_balance : 초기 자본금
- pvs : 포트폴리오 가치 배열

#### 5. policy_learner : 정책 학습기 클래스를 가지고 일련의 학습 데이터를 준비하고 정책 신경망을 학습

#### 6. creon

#### 7. data_manager

#### 8. main

#### 9. main_notraining : 비학습 투자 시뮬레이션

#### 10. agent_custom

#### 11. policy_network_custom

#### 12. policy_network_dnn

#### 13. data_manager_custom

#### 14. main_custom
