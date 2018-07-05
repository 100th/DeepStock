"""
준비한 차트 데이터와 학습 데이터로 강화학습을 진행.
주식 데이터를 읽고, 차트 데이터와 학습 데이터를 준비하고, 주식투자 강화학습을 실행하는 모듈
"""

import logging
import os
import settings
import data_manager
from policy_learner import PolicyLearner


if __name__ == '__main__':
    stock_code = '005930'  # 삼성전자

    # 로그 기록
    log_dir = os.path.join(settings.BASE_DIR, 'logs/%s' % stock_code)
    timestr = settings.get_time_str()
    if not os.path.exists('logs/%s' % stock_code):
        os.makedirs('logs/%s' % stock_code)
    file_handler = logging.FileHandler(filename=os.path.join(
        log_dir, "%s_%s.log" % (stock_code, timestr)), encoding='utf-8')
    stream_handler = logging.StreamHandler()
    file_handler.setLevel(logging.DEBUG)
    stream_handler.setLevel(logging.INFO)
    logging.basicConfig(format="%(message)s",
                        handlers=[file_handler, stream_handler], level=logging.DEBUG)

    # 주식 데이터 준비
    chart_data = data_manager.load_chart_data(  # 차트 데이터를 Pandas DataFrame 객체로 불러온다
        os.path.join(settings.BASE_DIR,
                     'data/chart_data/{}.csv'.format(stock_code)))
    prep_data = data_manager.preprocess(chart_data)     # 불러온 차트 데이터를 전처리해서 학습 데이터를 만들 준비함
    # training_data는 차트 데이터의 열들, 전처리에서 추가된 열들, 학습 데이터의 열들이 모두 포함된 데이터.
    training_data = data_manager.build_training_data(prep_data)

    # 기간 필터링
    training_data = training_data[(training_data['date'] >= '2017-01-01') &
                                  (training_data['date'] <= '2017-12-31')]  # 2017년 전체 데이터 사용
    training_data = training_data.dropna()

    # 차트 데이터 분리
    features_chart_data = ['date', 'open', 'high', 'low', 'close', 'volume']    # 차트 데이터 열 지정
    chart_data = training_data[features_chart_data] # 차트 데이터 분리

    # 학습 데이터 분리
    features_training_data = [  # 학습 데이터의 열 지정
        'open_lastclose_ratio', 'high_close_ratio', 'low_close_ratio',
        'close_lastclose_ratio', 'volume_lastvolume_ratio',
        'close_ma5_ratio', 'volume_ma5_ratio',
        'close_ma10_ratio', 'volume_ma10_ratio',
        'close_ma20_ratio', 'volume_ma20_ratio',
        'close_ma60_ratio', 'volume_ma60_ratio',
        'close_ma120_ratio', 'volume_ma120_ratio'
    ]
    training_data = training_data[features_training_data]   # 학습 데이터 분리

    # 강화학습 시작
    # 정책 학습기 객체 생성 (종목 코드, 차트 데이터, 학습 데이터, 최소 투자 단위, 최대 투자 단위,
    # 지연 보상 임계치, 학습 속도)
    policy_learner = PolicyLearner(
        stock_code=stock_code, chart_data=chart_data, training_data=training_data,
        min_trading_unit=1, max_trading_unit=2, delayed_reward_threshold=.2, lr=.001)
    # 생성한 정책 학습기 객체의 fit() 함수 호출 (이때 초기 자본금, 수행할 에포크 수, 할인 요인, 초기 탐험률)
    policy_learner.fit(balance=10000000, num_epoches=1000,
                       discount_factor=0, start_epsilon=.5)

    # 정책 신경망을 파일로 저장
    model_dir = os.path.join(settings.BASE_DIR, 'models/%s' % stock_code)   # 저장 폴더 경로 설정
    if not os.path.isdir(model_dir):
        os.makedirs(model_dir)
    model_path = os.path.join(model_dir, 'model_%s.h5' % timestr)   # 파일명 설정
    policy_learner.policy_network.save_model(model_path)    # 이 경로에 저장
