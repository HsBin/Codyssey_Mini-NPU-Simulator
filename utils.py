DATA_FILE = "data.json"
EPSILON = 1e-9
REPEAT_COUNT = 10

FILTER_SIZES = (5, 13, 25)

MATRIX_SCHEMA = {
    "type": list,
    "row_type": list,
    "value_types": (int, float)
}

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

