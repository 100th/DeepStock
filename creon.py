import time
import win32com.client
import pandas as pd


class Creon:
    def __init__(self):
        self.obj_CpCodeMgr = win32com.client.Dispatch('CpUtil.CpCodeMgr')
        self.obj_CpCybos = win32com.client.Dispatch('CpUtil.CpCybos')
        self.obj_StockChart = win32com.client.Dispatch('CpSysDib.StockChart')

    def creon_7400_주식차트조회(self, code, date_from, date_to):  # 코드, 시작일, 종료일
        b_connected = self.obj_CpCybos.IsConnect    # 연결 확인
        if b_connected == 0:
            print("연결 실패")
            return None

        list_field_key = [0, 1, 2, 3, 4, 5, 8]
        list_field_name = ['date', 'time', 'open', 'high', 'low', 'close', 'volume']
        dict_chart = {name: [] for name in list_field_name}

        # 차트 데이터를 받아오기 위해 API 호출
        self.obj_StockChart.SetInputValue(0, 'A'+code)
        self.obj_StockChart.SetInputValue(1, ord('1'))  # 0: 개수, 1: 기간
        self.obj_StockChart.SetInputValue(2, date_to)  # 종료일
        self.obj_StockChart.SetInputValue(3, date_from)  # 시작일
        self.obj_StockChart.SetInputValue(5, list_field_key)  # 필드 (획들할 필드를 리스트로 입력)
        self.obj_StockChart.SetInputValue(6, ord('D'))  # 'D', 'W', 'M', 'm', 'T'   # 차트 데이터 단위를 Day로 설정
        self.obj_StockChart.BlockRequest()  # 입력한 설정에 따라 데이터를 요청

        # 요청 결과 및 출력물
        status = self.obj_StockChart.GetDibStatus() # 요청 결과 상태를 받아옴. 0이면 정상
        msg = self.obj_StockChart.GetDibMsg1()
        print("통신상태: {} {}".format(status, msg))
        if status != 0:
            return None

        cnt = self.obj_StockChart.GetHeaderValue(3)  # 수신개수. 결과 출력물의 개수 확인
        for i in range(cnt):
            dict_item = (
                {name: self.obj_StockChart.GetDataValue(pos, i) # 값을 받아옴
                for pos, name in zip(range(len(list_field_name)), list_field_name)}
            )
            for k, v in dict_item.items():
                dict_chart[k].append(v)     # 여기서 받은 값을 dict_chart 딕셔너리에 추가

        print("차트: {} {}".format(cnt, dict_chart))
        return pd.DataFrame(dict_chart, columns=list_field_name)    # 이렇게 구성된 dict_chart를 pandas DataFrame 객체로 만들어서 반환

# Creon 클래스 사용
if __name__ == '__main__':
    creon = Creon()
    print(creon.creon_7400_주식차트조회('035420', 20150101, 20171231))
# Anaconda Prompt를 관리자 권한으로 실행하는 것을 추천
