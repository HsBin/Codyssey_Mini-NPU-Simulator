from utils import MATRIX_SCHEMA

#행렬 클래스
class Matrix:

    def __init__(self, data):
        self.data = data
        self.size = len(data) if isinstance(data, list) else 0 #list[list] 2차원 배열을 처음부터 if문에 적으면 더 깔끔함.

    #2차원 배열인지 검증하는 메소드
    def validate(self, expected_size):
        #행렬 자체 검사
        if (
            not isinstance(self.data, MATRIX_SCHEMA["type"])
            or len(self.data) != expected_size
        ):
            return False, f"행 개수가 {expected_size}개가 아닙니다."

        # 각 행을 하나의 반복문으로 검사
        for row in self.data:
            if (
                not isinstance(row, MATRIX_SCHEMA["row_type"])
                or len(row) != expected_size
                or not all(
                    isinstance(value, MATRIX_SCHEMA["value_types"])
                    and not isinstance(value, bool)
                    for value in row
                )
            ):
                return False, f"{expected_size}x{expected_size} 숫자 행렬 형식이 아닙니다."

        return True, ""

    #행렬 특정 자리 반환
    def get(self, row, col):
        return self.data[row][col]

    # 2차원 행렬을 1차원 리스트로 변환
    def flatten(self):
        flat_data = []

        for row in self.data:
            for value in row:
                flat_data.append(value)

        return flat_data


    #보너스 기능 대비 메소드
    # Cross 행렬 생성
    @classmethod
    def create_cross(cls, size):
        data = []

        #짝수의 경우 두줄씩 표현.
        if size % 2 == 0:
            centers = (size // 2 - 1, size // 2)
        else:
            centers = (size // 2,)

        for row in range(size):
            new_row = []

            for col in range(size):
                if row in centers or col in centers:
                    new_row.append(1.0)
                else:
                    new_row.append(0.0)

            data.append(new_row)

        return cls(data)


    # X 행렬 생성
    @classmethod
    def create_x(cls, size):
        data = []

        for row in range(size):
            new_row = []

            for col in range(size):
                if row == col or row + col == size - 1:
                    new_row.append(1.0)
                else:
                    new_row.append(0.0)

            data.append(new_row)

        return cls(data)

