# face_recog.py

import face_recognition
import cv2
import camera
import os
import numpy as np
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input


class FaceRecog():
    def __init__(self):
        # Using OpenCV to capture from device 0. If you have trouble capturing
        # from a webcam, comment the line below out and use a video file
        # instead.
        self.camera = camera.VideoCamera()
        self.facenet = cv2.dnn.readNet('models/deploy.prototxt', 'models/res10_300x300_ssd_iter_140000.caffemodel')
        self.known_face_encodings = []
        self.known_face_names = []
        self.face = []

        # Load sample pictures and learn how to recognize it.
        dirname = 'knowns'
        files = os.listdir(dirname)
        for filename in files:
            name, ext = os.path.splitext(filename)
            if ext == '.jpeg':
                self.known_face_names.append(name)
                pathname = os.path.join(dirname, filename)
                img = face_recognition.load_image_file(pathname)
                test1 = cv2.imread(pathname)
                # print(test1.shape[:2])
                h, w = test1.shape[:2]
                # print(test1.shape) # (960, 721, 3)
                blob = cv2.dnn.blobFromImage(img, scalefactor=1., size=(300, 300), mean=(104., 177., 123.))
                self.facenet.setInput(blob)
                dets = self.facenet.forward()

                for i in range(dets.shape[2]):
                    # print(name)
                    # 검출한 결과가 신뢰도
                    confidence = dets[0, 0, i, 2]

                    # 신뢰도를 0.5로 임계치 지정
                    if confidence < 0.5:
                        continue
                    # print('confidence :: ', confidence * 100)
                    # 바운딩 박스를 구함
                    x1 = int(dets[0, 0, i, 3] * w)  # 박스 시작점 x 좌표
                    y1 = int(dets[0, 0, i, 4] * h)  # 박스 시작점 y 좌표
                    x2 = int(dets[0, 0, i, 5] * w)  # 박스 끝점 x 좌표
                    y2 = int(dets[0, 0, i, 6] * h)  # 박스 끝점 y 좌표

                    # load DB
                    # dir_name = "result"
                    # pdb = PersonDB()
                    # pdb.load_db(dir_name)
                    # pdb.print_persons()

                    # 원본 이미지에서 얼굴영역 추출
                    face = img[y1:y2, x1:x2]
                    # print(type(face))
                    # print('face.shape :: ', face.shape)

                    self.face.append(face)

                for i in self.face:
                    print(i.shape)
                    face_encoding = face_recognition.face_encodings(i)
                    if len(face_encoding) > 0:
                        print('ok')
                        self.known_face_encodings.append(face_encoding[0])
                    else:
                        print("이미지에 얼굴이 없습니다!")


                # face_encoding = face_recognition.face_encodings(img)[0]
                # self.known_face_encodings.append(face_encoding)

        # Initialize some variables
        self.face_locations = []
        self.face_encodings = []
        self.face_names = []
        self.process_this_frame = True

    def __del__(self):
        del self.camera

    def get_frame(self):
        # Grab a single frame of video
        frame = self.camera.get_frame()

        # Resize frame of video to 1/4 size for faster face recognition processing
        small_frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)

        # Convert the image from BGR color (which OpenCV uses) to RGB color (which face_recognition uses)
        rgb_small_frame = small_frame[:, :, ::-1]

        # Only process every other frame of video to save time
        if self.process_this_frame:
            # Find all the faces and face encodings in the current frame of video

            self.face_locations = face_recognition.face_locations(rgb_small_frame)
            self.face_encodings = face_recognition.face_encodings(rgb_small_frame, self.face_locations)

            self.face_names = []
            for face_encoding in self.face_encodings:
                # See if the face is a match for the known face(s)
                distances = face_recognition.face_distance(self.known_face_encodings, face_encoding)
                min_value = min(distances)

                # tolerance: How much distance between faces to consider it a match. Lower is more strict.
                # 0.6 is typical best performance.
                name = "Unknown"
                if min_value < 0.6:
                    index = np.argmin(distances)
                    name = self.known_face_names[index]

                self.face_names.append(name)

        self.process_this_frame = not self.process_this_frame

        # Display the results
        for (top, right, bottom, left), name in zip(self.face_locations, self.face_names):
            # Scale back up face locations since the frame we detected in was scaled to 1/4 size
            top *= 4
            right *= 4
            bottom *= 4
            left *= 4

            # Draw a box around the face
            cv2.rectangle(frame, (left, top), (right, bottom), (0, 0, 255), 2)

            # Draw a label with a name below the face
            cv2.rectangle(frame, (left, bottom - 35), (right, bottom), (0, 0, 255), cv2.FILLED)
            font = cv2.FONT_HERSHEY_DUPLEX
            cv2.putText(frame, name, (left + 6, bottom - 6), font, 1.0, (255, 255, 255), 1)

        return frame

    def get_jpg_bytes(self):
        frame = self.get_frame()
        # We are using Motion JPEG, but OpenCV defaults to capture raw images,
        # so we must encode it into JPEG in order to correctly display the
        # video stream.
        ret, jpg = cv2.imencode('.jpg', frame)
        return jpg.tobytes()


if __name__ == '__main__':
    face_recog = FaceRecog()
    print(face_recog.known_face_names)
    while True:
        frame = face_recog.get_frame()

        # show the frame
        cv2.imshow("Frame", frame)
        key = cv2.waitKey(1) & 0xFF

        # if the `q` key was pressed, break from the loop
        if key == ord("q"):
            break

    # do a bit of cleanup
    cv2.destroyAllWindows()
    print('finish')
