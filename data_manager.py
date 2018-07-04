import pandas as pd
import numpy as np

# CSV 파일 읽어오기
def load_chart_data(fpath):
    chart_data = pd.read_csv(fpath, thousands=',', header=None)
    chart_data.columns = ['date', 'open', 'high', 'low', 'close', 'volume']
    return chart_data

# 종가와 거래량의 이동 평균값 구하기
def preprocess(chart_data):
    prep_data = chart_data
    windows = [5, 10, 20, 60, 120]
    for window in windows:
        prep_data['close_ma{}'.format(window)] = prep_data['close'].rolling(window).mean()
        prep_data['volume_ma{}'.format(window)] = (
            prep_data['volume'].rolling(window).mean()) # 롤링 평균 = 이동 평균
    return prep_data

# 특징(feature)을 계산한다.
def build_training_data(prep_data):
    training_data = prep_data

    # 시가/전일종가 비율(open_lastclose_ratio)
    # 첫 번째 행은 전일 값이 없거나 그 값이 있더라도 알 수 없기 때문에 전일 종가 비율을 구하지 못함
    # 그래서 두 번째 행부터 마지막 행까지 'open_lastclose_ratio' 열에 시가/전일 종가 비율을 저장함
    # 시가/전일종가 비율 = (현재 종가 - 전일 종가) / 전일 종가
    training_data['open_lastclose_ratio'] = np.zeros(len(training_data))
    training_data.loc[1:, 'open_lastclose_ratio'] = \
        (training_data['open'][1:].values - training_data['close'][:-1].values) / \
        training_data['close'][:-1].values
    # 고가/종가 비율(high_close_ratio)
    training_data['high_close_ratio'] = \
        (training_data['high'].values - training_data['close'].values) / \
        training_data['close'].values
    # 저가/종가 비율(low_close_ratio)
    training_data['low_close_ratio'] = \
        (training_data['low'].values - training_data['close'].values) / \
        training_data['close'].values
    # 종가/전일종가 비율(close_lastclose_ratio)
    training_data['close_lastclose_ratio'] = np.zeros(len(training_data))
    training_data.loc[1:, 'close_lastclose_ratio'] = \
        (training_data['close'][1:].values - training_data['close'][:-1].values) / \
        training_data['close'][:-1].values
    # 거래량/전일거래량 비율(volume_lastvolume_ratio)
    training_data['volume_lastvolume_ratio'] = np.zeros(len(training_data))
    training_data.loc[1:, 'volume_lastvolume_ratio'] = \
        (training_data['volume'][1:].values - training_data['volume'][:-1].values) / \
        training_data['volume'][:-1]\
            .replace(to_replace=0, method='ffill') \
            .replace(to_replace=0, method='bfill').values   # 거래량 값이 0이면 이전의 0이 아닌 값으로 바꿈

    # 이동평균 종가 비율과 이동평균 거래량 비율 구하기
    # 이동평균 종가 비율 = (현재 종가 - 이동 평균) / 이동 평균
    # 이동평균 거래량 비율 = 같은 방식으로
    windows = [5, 10, 20, 60, 120] # 각 윈도우에 대해서 이동평균 종가 비율, 이동ㅊ평균 거래량 비율을 구한다.
    for window in windows:
        training_data['close_ma%d_ratio' % window] = \
            (training_data['close'] - training_data['close_ma%d' % window]) / \
            training_data['close_ma%d' % window]
        training_data['volume_ma%d_ratio' % window] = \
            (training_data['volume'] - training_data['volume_ma%d' % window]) / \
            training_data['volume_ma%d' % window]

    return training_data


# chart_data = pd.read_csv(fpath, encoding='CP949', thousands=',', engine='python')
