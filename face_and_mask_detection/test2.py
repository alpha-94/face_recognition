from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.models import load_model
import numpy as np
import cv2
import face_recognition


class Models:
    # default
    # proto_file = 'models/deploy.prototxt'
    # face_model_file = 'models/res10_300x300_ssd_iter_140000.caffemodel'
    # mask_model_file = 'models/mask_detector.model'
    def __init__(self, proto_file, face_model_file, mask_model_file):
        self.face_model = None
        self.mask_model = None
        self.proto_file = proto_file
        self.face_model_file = face_model_file
        self.mask_model_file = mask_model_file

    def face(self):
        self.face_model = cv2.dnn.readNet(self.proto_file, self.face_model_file)
        return self.face_model

    def mask(self):
        self.mask_model = load_model(self.mask_model_file)
        return self.mask_model


class Image_Preprocessing:
    def __init__(self, img, face_model):
        self.blob = None
        self.dets = None
        self.face_input = None
        self.img = img
        self.face_model = face_model
        self.w, self.h = self.img.shape[:2]
        self.x1, self.x2, self.y1, self.y2 = (0, 0, 0, 0)

    def get_blob(self, scalefactor=1., size=(300, 300), mean=(104., 177., 123.)):
        self.blob = cv2.dnn.blobFromImage(self.img, scalefactor=scalefactor, size=size, mean=mean)
        self.face_model.setInput(self.blob)
        self.dets = self.face_model.forward()

        return self.dets

    def tf_preprocess(self, dets_2):
        self.x1 = int(self.dets[0, 0, dets_2, 3] * self.w)  # 박스 시작점 x 좌표
        self.y1 = int(self.dets[0, 0, dets_2, 4] * self.h)  # 박스 시작점 y 좌표
        self.x2 = int(self.dets[0, 0, dets_2, 5] * self.w)  # 박스 끝점 x 좌표
        self.y2 = int(self.dets[0, 0, dets_2, 6] * self.h)  # 박스 끝점 y 좌표

        face = self.img[self.y1:self.y2, self.x1:self.x2]

        face_input = cv2.resize(face, dsize=(224, 224))
        face_input = cv2.cvtColor(face_input, cv2.COLOR_BGR2RGB)
        face_input = preprocess_input(face_input)
        self.face_input = np.expand_dims(face_input, axis=0)

        return self.face_input

    def label_mask(self, result_img, color, label):
        cv2.rectangle(result_img, pt1=(self.x1, self.y1), pt2=(self.x2, self.y2), thickness=2, color=color,
                      lineType=cv2.LINE_AA)
        cv2.putText(result_img, text=label, org=(self.x1, self.y1 - 10), fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                    fontScale=0.8,
                    color=color, thickness=2, lineType=cv2.LINE_AA)


if __name__ == '__main__':
    # path
    proto_file = 'models/deploy.prototxt'
    face_model_file = 'models/res10_300x300_ssd_iter_140000.caffemodel'
    mask_model_file = 'models/mask_detector.model'

    # static Variable
    i = 0

    models = Models(proto_file, face_model_file, mask_model_file)

    cap = cv2.VideoCapture(0)

    while cap.isOpened():
        ret, img = cap.read()
        if not ret:
            break

        ip = Image_Preprocessing(img, models.face)

        dets = ip.get_blob()

        result_img = img.copy()

        for i in range(dets.shapep[2]):
            confidence = dets[0, 0, i, 2]

            if confidence < 0.5:
                continue

            face_input = ip.tf_preprocess(i)

            mask, nomask = models.mask().pridict(face_input).squeeze()

            if mask > nomask:
                color = (0, 255, 0)
                label = 'Mask %d%%' % (mask * 100)
            else:
                color = (0, 0, 255)
                label = 'No Mask %d%%' % (nomask * 100)

            ip.label_mask(result_img, color, label)

        cv2.imshow('img', result_img)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

