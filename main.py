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

class MiniNPU:
    """Mini NPU Simulator 전체 동작 담당"""

    def __init__(self):
        self.epsilon = EPSILON
        self.repeat_count = REPEAT_COUNT
        self.filters = {}

    #mac 연산.
    def mac(self, pattern, filter_matrix):
        total = 0.0

        for row in range(pattern.size):
            for col in range(pattern.size):
                total += pattern.get(row, col) * filter_matrix.get(row, col)

        return total

    #비교 결과 처리.
    def decide(self, score_a, score_b, label_a, label_b):
        #둘다 완전 동점이어도 앱실론보다 낮음. = 0
        if abs(score_a - score_b) < self.epsilon:
            return "UNDECIDED"

        if score_a > score_b:
            return label_a

        return label_b

    #행렬 입력하는 메소드.
    def input_matrix(self, name, size):
        while True:
            print(f"\n{name} ({size}줄 입력, 공백 구분)")
            rows = []
            input_error = False

            for row_number in range(size):
                try:
                    #공백 구분해서 입력받고, 입력받은 데이터 각각에 실수화 처리하고 리스트형태로 초기화.
                    row = list(map(float, input(f"{row_number + 1}행: ").split()))

                #숫자로 변환할 수 없는 값이 입력된 경우 처리
                except ValueError:
                    print(f"입력 형식 오류: 각 줄에 {size}개의 숫자를 공백으로 구분해 입력하세요.")
                    input_error = True
                    break

                #입력받는 개수 다르면 다시 입력받기.
                if len(row) != size:
                    print(f"입력 형식 오류: 각 줄에 {size}개의 숫자를 공백으로 구분해 입력하세요.")
                    input_error = True
                    break

                rows.append(row)

            #다시 입력하럽 반복하기 전 안내문 출력
            if input_error:
                print(f"{name} 입력을 처음부터 다시 시도합니다.")
                continue

            #입력받은 숫자들로 행렬 객체 생성 및 검증.
            matrix = Matrix(rows)
            valid, reason = matrix.validate(size)

            if valid:
                return matrix

            print(f"입력 오류: {reason}")
            print(f"{name} 입력을 처음부터 다시 시도합니다.")

        