# 먼저 얼굴을 찾는 모델과 마스크 검출을 위한 모델을 서칭

# 실시간 웹캠을 읽기
#       웹캠이 계속 켜져있는 경우 이미지를 리더
#       캡쳐가 꺼지는 경우 브레이크


# 웹캠에서 넘어온 이미지 처리
#   이미지의 높이와 너비를 추출한다.
#   이미지를 전처리 한다.
#   카페모델(얼굴을 찾는 모델) 인풋에 전처리한 사진을 넣는다.
#   포워딩 시켜서 얼굴을 추츨 후 변수 지
#   웹캠 이미들을 변수에 저장한다.

#   마스크를 착용했는지
#   confidence 값 -> detective[0,0,i,2]
#   바운딩 박스 구함
#   원본 이미지에서 얼굴을 추출


from person_db import Person
from person_db import Face
from person_db import PersonDB
