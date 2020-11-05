from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.models import load_model
import numpy as np
import cv2
import face_recognition

facenet = cv2.dnn.readNet('models/deploy.prototxt', 'models/res10_300x300_ssd_iter_140000.caffemodel')
# model : 마스크 검출 모델
model = load_model('models/mask_detector.model')

cap = cv2.VideoCapture(0)
i = 0

while cap.isOpened():
    ret, img = cap.read()
    print('img.shape', img.shape)

    if not ret:
        break

    # 이미지의 높이와 너비 추출
    h, w = img.shape[:2]
    print('hight : {} , width : {}'.format(h, w))

    # 이미지 전처리
    # ref. https://www.pyimagesearch.com/2017/11/06/deep-learning-opencvs-blobfromimage-works/
    blob = cv2.dnn.blobFromImage(img, scalefactor=1., size=(300, 300), mean=(104., 177., 123.))
    # print('blob.shape', blob.shape)
    # print(blob)

    # facenet의 input으로 blob을 설정
    facenet.setInput(blob)
    # facenet 결과 추론, 얼굴 추출 결과가 dets의 저장
    dets = facenet.forward()

    # print(dets)
    # print('dets.shape', dets.shape)
    # print('dets.shape[2] :: ', dets.shape[2])

    result_img = img.copy()
    # print('result_img.shape', result_img.shape)

    for i in range(dets.shape[2]):

        # 검출한 결과가 신뢰도
        confidence = dets[0, 0, i, 2]

        # 신뢰도를 0.5로 임계치 지정
        if confidence < 0.5:
            continue
        # print('confidence :: ', confidence * 100)
        # 바운딩 박스를 구함
        x1 = int(dets[0, 0, i, 3] * w) # 박스 시작점 x 좌표
        y1 = int(dets[0, 0, i, 4] * h) # 박스 시작점 y 좌표
        x2 = int(dets[0, 0, i, 5] * w) # 박스 끝점 x 좌표
        y2 = int(dets[0, 0, i, 6] * h) # 박스 끝점 y 좌표

        #load DB
        # dir_name = "result"
        # pdb = PersonDB()
        # pdb.load_db(dir_name)
        # pdb.print_persons()

        # 원본 이미지에서 얼굴영역 추출
        face = img[y1:y2, x1:x2]
        # print(type(face))
        # print('face.shape :: ', face.shape)

        # 추출한 얼굴영역을 전처리
        face_input = cv2.resize(face, dsize=(224, 224))
        face_input = cv2.cvtColor(face_input, cv2.COLOR_BGR2RGB)
        face_input = preprocess_input(face_input)
        face_input = np.expand_dims(face_input, axis=0)

        # print(face_input.shape)

        # 마스크 검출 모델로 결과값 return
        mask, nomask = model.predict(face_input).squeeze()

        def test(color, label):
            cv2.rectangle(result_img, pt1=(x1, y1), pt2=(x2, y2), thickness=2, color=color, lineType=cv2.LINE_AA)
            cv2.putText(result_img, text=label, org=(x1, y1 - 10), fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=0.8,
                        color=color, thickness=2, lineType=cv2.LINE_AA)

        # 마스크를 꼈는지 안겼는지에 따라 라벨링해줌
        if mask > nomask:
            color = (0, 255, 0)
            label = 'Mask %d%%' % (mask * 100)
            print(label)
            test(color, label)



        else:
            color = (0, 0, 255)
            label = 'No Mask %d%%' % (nomask * 100)
            print(label)
            test(color, label)
    # cv2.imshow('img', face)
