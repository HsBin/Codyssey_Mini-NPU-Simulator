import json
import time

DATA_FILE = "data.json"
EPSILON = 1e-9
REPEAT_COUNT = 10

#정규화 라벨 함수.
def normalize_label(label):
    """외부 라벨을 프로그램 내부 표준 라벨로 변환한다."""
    if not isinstance(label, str):
        return None

    label = label.strip().lower()

    if label == "+" or label == "cross":
        return "Cross"

    if label == "x":
        return "X"

    return None


#행렬 클래스
class Matrix:

    def __init__(self, data):
        self.data = data
        self.size = len(data) if isinstance(data, list) else 0

    #2차원 배열인지 검증하는 메소드
    def validate(self, expected_size):
        if not isinstance(self.data, list):
            return False, "2차원 배열(list) 형식이 아닙니다."

        if len(self.data) != expected_size:
            return False, f"행 개수가 {expected_size}개가 아닙니다."

        for row_index, row in enumerate(self.data, start=1):
            if not isinstance(row, list):
                return False, f"{row_index}번째 행이 list 형식이 아닙니다."

            if len(row) != expected_size:
                return False, f"{row_index}번째 행의 열 개수가 {expected_size}개가 아닙니다."

            for col_index, value in enumerate(row, start=1):
                if isinstance(value, bool) or not isinstance(value, (int, float)):
                    return False, f"{row_index}행 {col_index}열에 숫자가 아닌 값이 있습니다."

        return True, ""

    #행렬 반환
    def get(self, row, col):
        return self.data[row][col]

    #보너스 기능 대비 메소드
    @classmethod
    def generate_cross(cls, size):
        data = [[0.0 for _ in range(size)] for _ in range(size)]
        center = size // 2

        for i in range(size):
            data[center][i] = 1.0
            data[i][center] = 1.0

        return cls(data)