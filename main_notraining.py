"""
비학습 투자 시뮬레이션
"""
import logging
import os
import settings
import data_manager
from policy_learner import PolicyLearner


if __name__ == '__main__':
    stock_code = '005930'  # 삼성전자
    model_ver = '20180202000545'    # 정책 신경망 모델의 버전명을 지정해 줌

    # 로그 기록
    log_dir = os.path.join(settings.BASE_DIR, 'logs/%s' % stock_code)
    timestr = settings.get_time_str()
    file_handler = logging.FileHandler(filename=os.path.join(
        log_dir, "%s_%s.log" % (stock_code, timestr)), encoding='utf-8')
    stream_handler = logging.StreamHandler()
    file_handler.setLevel(logging.DEBUG)
    stream_handler.setLevel(logging.INFO)
    logging.basicConfig(format="%(message)s",
        handlers=[file_handler, stream_handler], level=logging.DEBUG)

    # 주식 데이터 준비
    chart_data = data_manager.load_chart_data(
        os.path.join(settings.BASE_DIR,
                     'data/chart_data/{}.csv'.format(stock_code)))
    prep_data = data_manager.preprocess(chart_data)
    training_data = data_manager.build_training_data(prep_data)

    # 기간 필터링
    training_data = training_data[(training_data['date'] >= '2018-01-01') &
                                  (training_data['date'] <= '2018-01-31')]
    training_data = training_data.dropna()

    # 차트 데이터 분리
    features_chart_data = ['date', 'open', 'high', 'low', 'close', 'volume']
    chart_data = training_data[features_chart_data]

    # 학습 데이터 분리
    features_training_data = [
        'open_lastclose_ratio', 'high_close_ratio', 'low_close_ratio',
        'close_lastclose_ratio', 'volume_lastvolume_ratio',
        'close_ma5_ratio', 'volume_ma5_ratio',
        'close_ma10_ratio', 'volume_ma10_ratio',
        'close_ma20_ratio', 'volume_ma20_ratio',
        'close_ma60_ratio', 'volume_ma60_ratio',
        'close_ma120_ratio', 'volume_ma120_ratio'
    ]
    training_data = training_data[features_training_data]

    # 비 학습 투자 시뮬레이션 시작
    # 종목 코드, 차트 데이터, 학습 데이터, 최소 및 최대 투자 단위를 지정해 줌.
    # 기존 main 모듈에서 지정해 줬던 지연 보상 기준(delayed_reward_threshold)과 학습 속도(lr)는 입력하지 않아도 됨.
    # 비학습 투자 시뮬레이션에서는 사용하지 않는 인자들이기 때문에.
    policy_learner = PolicyLearner(
        stock_code=stock_code, chart_data=chart_data, training_data=training_data,
        min_trading_unit=1, max_trading_unit=3)
    # 준비한 정책 학습기 객체의 trade() 함수를 호출.
    # 이는 비학습 투자 시뮬레이션을 수행하기 위해 인자들을 적절히 설정하여 fit() 함수를 호출하는 함수.
    # trade() 함수에서는 fit() 함수를 호출할 때 수행할 에포크 수인 num_epoches를 1로, learning 인자에 False를 줌.
    policy_learner.trade(balance=10000000,
                         model_path=os.path.join(
                             settings.BASE_DIR,
                             'models/{}/model_{}.h5'.format(stock_code, model_ver)))

    # 기존 main 모듈 마지막 부분의 '정책 신경망을 파일로 저장' 코드 블록은 제거
    # 이미 저장된 정책 신경망 모듈을 사용했고 추가적으로 학습을 하지 않았기 때문에.
    # 만약 추가적인 학습을 수행하여 모델을 새로 저장하고 싶다면 코드 블록을 제거하지 않고 그대로 두면 됨.
