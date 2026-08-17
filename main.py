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

    #json파일 로드.
    def load_json(self):
        try:
            with open(DATA_FILE, "r", encoding="utf-8") as file:
                return json.load(file)
        except FileNotFoundError:
            print(f"오류: {DATA_FILE} 파일을 찾을 수 없습니다.")
        except json.JSONDecodeError as error:
            print(f"오류: JSON 형식이 올바르지 않습니다. ({error})")
        except OSError as error:
            print(f"오류: 파일을 읽을 수 없습니다. ({error})")

        return None

    #data에서 filters 가져오는 메소드.
    def load_filters(self, data):
        #원래 있던 필터 섞이지 않게 초기화, json데이터에서 filters 키의 값 가져오기.
        self.filters = {}
        raw_filters = data.get("filters")

        #딕셔너리인지 확인.
        if not isinstance(raw_filters, dict):
            print("오류: filters 항목이 없거나 형식이 잘못되었습니다.")
            return

        #사이즈별로 키 이름에 맞춰서 초기화.
        for size in (5, 13, 25):
            size_key = f"size_{size}"
            filter_group = raw_filters.get(size_key)

            if not isinstance(filter_group, dict):
                print(f"✗ {size_key}: 필터 그룹이 없거나 형식이 잘못되었습니다.")
                continue

            #정상 처리된 필터 저장
            normalized_filters = {}
            #해당 필터에서 발생한 오류메시지들 저장.
            errors = []

            for filter_key, filter_data in filter_group.items():
                #필터 키 라벨 정규화
                label = normalize_label(filter_key)

                if label is None:
                    errors.append(f"알 수 없는 필터 라벨: {filter_key}")
                    continue

                #필터 데이터 가진 행렬 객체 생성.
                matrix = Matrix(filter_data)
                #행렬 객체 검증.
                valid, reason = matrix.validate(size)

                if not valid:
                    errors.append(f"{label} 필터 오류: {reason}")
                    continue

                #정상처리된 필터 딕셔너리에 검증된 행렬 객체 저장.
                normalized_filters[label] = matrix

            #필터들 없으면 오류메시지에 저장.
            if "Cross" not in normalized_filters:
                errors.append("Cross 필터가 없습니다.")

            if "X" not in normalized_filters:
                errors.append("X 필터가 없습니다.")

            #에러 하나라도 있으면, 필터 키 저장 안하고 continue
            if errors:
                print(f"✗ {size_key}: " + " / ".join(errors))
                continue

            self.filters[size] = normalized_filters
            print(f"✓ {size_key:<7} 필터 로드 완료 (Cross, X)")

    #패턴분석 메소드
    def analyze_patterns(self, data):
        results = []
        patterns = data.get("patterns")

        #딕셔너리 형태인지 확인.
        if not isinstance(patterns, dict):
            results.append({
                "case": "patterns",
                "status": "FAIL",
                "reason": "patterns 항목이 없거나 형식이 잘못되었습니다."
            })
            return results

        #패턴 분석 결과 미리 미리 초기화.
        for key, value in patterns.items():
            result = {
                "case": key,
                "status": "FAIL",
                "reason": "",
                "cross_score": None,
                "x_score": None,
                "decision": None,
                "expected": None
            }

            #패턴 값이 딕셔너리인지 확인.
            if not isinstance(value, dict):
                result["reason"] = "패턴 데이터 형식이 잘못되었습니다."
                results.append(result)
                continue

            parts = key.split("_")

            #key값 형식 확인.
            if len(parts) != 3 or parts[0] != "size":
                result["reason"] = "패턴 키가 size_{N}_{idx} 형식이 아닙니다."
                results.append(result)
                continue

            #패턴 키 사이즈 숫자인지 확인.
            try:
                size = int(parts[1])
            except ValueError:
                result["reason"] = "패턴 키에서 크기를 읽을 수 없습니다."
                results.append(result)
                continue

            #필터키 로드 확인
            if size not in self.filters:
                result["reason"] = f"size_{size} 필터가 정상적으로 로드되지 않았습니다."
                results.append(result)
                continue

            #기대결과 정규화해서 변수에 저장.
            expected = normalize_label(value.get("expected"))
            result["expected"] = expected

            #정규화 잘 됐는지 확인.
            if expected is None:
                result["reason"] = "expected 라벨을 Cross/X로 변환할 수 없습니다."
                results.append(result)
                continue

            #행렬 입력받고, 검증까지.
            pattern = Matrix(value.get("input"))
            valid, reason = pattern.validate(size)

            #검증 성공여부 판별.
            if not valid:
                result["reason"] = f"패턴 크기/형식 오류: {reason}"
                results.append(result)
                continue

            #각 필터 값 지역 변수에 따로 초기화.
            cross_filter = self.filters[size]["Cross"]
            x_filter = self.filters[size]["X"]

            #각 필터 별 스코어 계산 후, 비교결과 저장.
            cross_score = self.mac(pattern, cross_filter)
            x_score = self.mac(pattern, x_filter)
            decision = self.decide(cross_score, x_score, "Cross", "X")

            result["cross_score"] = cross_score
            result["x_score"] = x_score
            result["decision"] = decision

            if decision == expected:
                result["status"] = "PASS"
            elif decision == "UNDECIDED":
                result["reason"] = f"두 점수 차이가 epsilon({self.epsilon})보다 작아 UNDECIDED로 판정됨"
            else:
                result["reason"] = f"판정 {decision}이 expected {expected}와 다름"

            results.append(result)

        return results

    #평균 계산 시간 계산 메소드
    def benchmark(self, pattern, filter_matrix):
        total_time = 0.0

        for _ in range(self.repeat_count):
            start = time.perf_counter()
            self.mac(pattern, filter_matrix)
            end = time.perf_counter()
            total_time += (end - start) * 1000

        return total_time / self.repeat_count

    #십자가 행렬 생성 메소드.
    def create_cross_matrix(self, size):
        data = []
        center = size // 2

        for row in range(size):
            new_row = []

            for col in range(size):
                if row == center or col == center:
                    new_row.append(1.0)
                else:
                    new_row.append(0.0)

            data.append(new_row)

        return Matrix(data)

    #평균 계산 시간 분석 결과 출력 메소드
    def print_performance(self, cases):
        print("\n#---------------------------------------")
        print(f"# 성능 분석 (평균/{self.repeat_count}회)")
        print("#---------------------------------------")
        print(f"{'크기':<10}{'평균 시간(ms)':>16}{'연산 횟수(N²)':>18}")
        print("-" * 44)

        for size, pattern, filter_matrix in cases:
            avg_time = self.benchmark(pattern, filter_matrix)
            print(f"{size}x{size:<7}{avg_time:>16.6f}{size * size:>18}")

    #모드1 진행 메소드.
    def run_user_mode(self):
        print("\n#---------------------------------------")
        print("# [1] 필터 입력")
        print("#---------------------------------------")

        filter_a = self.input_matrix("필터 A", 3)
        filter_b = self.input_matrix("필터 B", 3)

        print("\n✓ 필터 A 저장 완료")
        print("✓ 필터 B 저장 완료")

        print("\n#---------------------------------------")
        print("# [2] 패턴 입력")
        print("#---------------------------------------")

        pattern = self.input_matrix("패턴", 3)

        score_a = self.mac(pattern, filter_a)
        score_b = self.mac(pattern, filter_b)
        decision = self.decide(score_a, score_b, "A", "B")
        average_time = self.benchmark(pattern, filter_a)

        print("\n#---------------------------------------")
        print("# [3] MAC 결과")
        print("#---------------------------------------")
        print(f"A 점수: {score_a}")
        print(f"B 점수: {score_b}")
        print(f"연산 시간(평균/{self.repeat_count}회): {average_time:.6f} ms")

        if decision == "UNDECIDED":
            print(f"판정: 판정 불가 (|A-B| < {self.epsilon})")
        else:
            print(f"판정: {decision}")

        self.print_performance([(3, pattern, filter_a)])

    #모드2 진행 메소드.
    def run_json_mode(self):
        data = self.load_json()

        if data is None:
            return

        print("\n#---------------------------------------")
        print("# [1] 필터 로드")
        print("#---------------------------------------")

        self.load_filters(data)

        print("\n#---------------------------------------")
        print("# [2] 패턴 분석 (라벨 정규화 적용)")
        print("#---------------------------------------")

        results = self.analyze_patterns(data)

        for result in results:
            print(f"\n--- {result['case']} ---")

            #대표적으로 십자가 점수부터 none이면 데이터가 오염되었다는 뜻
            if result.get("cross_score") is not None:
                print(f"Cross 점수: {result['cross_score']}")
                print(f"X 점수: {result['x_score']}")
                print(
                    f"판정: {result['decision']} | "
                    f"expected: {result['expected']} | "
                    f"{result['status']}"
                )
            else:
                print(f"판정: 처리 불가 | {result['status']}")

            if result["reason"]:
                print(f"사유: {result['reason']}")

        #성능분석 케이스들 저장할곳.
        performance_cases = []

        #대표 예시로 십자가 3x3 행렬 만들고 대충 비교해볼 패턴도 추가.
        cross_3 = self.create_cross_matrix(3)
        performance_cases.append((3, cross_3, cross_3))

        for size in (5, 13, 25):
            if size in self.filters:
                pattern = self.create_cross_matrix(size)
                performance_cases.append((size, pattern, self.filters[size]["Cross"]))

        self.print_performance(performance_cases)

        #-------------------- 결과 요약 출력. -------------------------------------------------------
        total = len(results)
        passed = 0

        for result in results:
            if result["status"] == "PASS":
                passed += 1

        failed = total - passed

        print("\n#---------------------------------------")
        print("# [4] 결과 요약")
        print("#---------------------------------------")
        print(f"총 테스트: {total}개")
        print(f"통과: {passed}개")
        print(f"실패: {failed}개")

        if failed > 0:
            print("\n실패 케이스:")

            for result in results:
                if result["status"] == "FAIL":
                    print(f"- {result['case']}: {result['reason']}")


    #시뮬레이터 메인 시작 메소드
    def run(self):
        print("=== Mini NPU Simulator ===")

        while True:
            print("\n[모드 선택]")
            print("1. 사용자 입력 (3x3)")
            print("2. data.json 분석")

            choice = input("선택: ").strip()

            if choice == "1":
                self.run_user_mode()
                break

            if choice == "2":
                self.run_json_mode()
                break

            print("입력 오류: 1 또는 2를 입력하세요.")


if __name__ == "__main__":
    simulator = MiniNPU()
    simulator.run()