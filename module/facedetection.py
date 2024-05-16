import numpy as np
import cv2 as cv
import joblib
import os

def detect_faces(frame, detector, recognizer, svc, mydict):
    height, width = frame.shape[:2]
    input_size = (320, 320)
    scale_x, scale_y = width / input_size[0], height / input_size[1]
    
    frame_resized = cv.resize(frame, input_size)
    faces = detector.detect(frame_resized)
    names = []
    if faces[1] is not None:
        for face in faces[1]:
            face_align = recognizer.alignCrop(frame_resized, face)
            face_feature = recognizer.feature(face_align)
            face_feature_flattened = face_feature.flatten()
            test_predict = svc.predict([face_feature_flattened])
            if test_predict[0] < len(mydict):
                result = mydict[test_predict[0]]
            else:
                result = "Unknown" 
            names.append(result)
            
            face[0] *= scale_x
            face[1] *= scale_y
            face[2] *= scale_x
            face[3] *= scale_y
            for i in range(4, 14, 2):
                face[i] *= scale_x
                face[i+1] *= scale_y
    return faces, names

def visualize(input, faces, names, fps, thickness=2):
    if faces[1] is not None:
        for idx, (face, name) in enumerate(zip(faces[1], names)):
            print('Face {}, top-left coordinates: ({:.0f}, {:.0f}), box width: {:.0f}, box height {:.0f}, score: {:.2f}'.format(idx, face[0], face[1], face[2], face[3], face[-1]))

            coords = face[:-1].astype(np.int32)
            cv.rectangle(input, (coords[0], coords[1]), (coords[0]+coords[2], coords[1]+coords[3]), (0, 255, 0), thickness)
            cv.putText(input, name, (coords[0], coords[1] - 10), cv.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), thickness)

            cv.circle(input, (coords[4], coords[5]), 2, (255, 0, 0), thickness)
            cv.circle(input, (coords[6], coords[7]), 2, (0, 0, 255), thickness)
            cv.circle(input, (coords[8], coords[9]), 2, (0, 255, 0), thickness)
            cv.circle(input, (coords[10], coords[11]), 2, (255, 0, 255), thickness)
            cv.circle(input, (coords[12], coords[13]), 2, (0, 255, 255), thickness)
    cv.putText(input, 'FPS: {:.2f}'.format(fps), (1, 16), cv.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

def load_models(face_detection_model_path, face_recognition_model_path, svc_path):
    if not os.path.exists(svc_path):
        raise FileNotFoundError(f"SVC model path does not exist: {svc_path}")

    svc = joblib.load(svc_path)
    mydict = ['MinhThuan', 'MinhTri', 'NguyenMinh', 'NhatThong','QuangKhai']

    if not os.path.exists(face_detection_model_path):
        raise FileNotFoundError(f"Face detection model path does not exist: {face_detection_model_path}")

    if not os.path.exists(face_recognition_model_path):
        raise FileNotFoundError(f"Face recognition model path does not exist: {face_recognition_model_path}")

    try:
        detector = cv.FaceDetectorYN.create(face_detection_model_path, "", (320, 320), 0.9, 0.3, 5000)
        recognizer = cv.FaceRecognizerSF.create(face_recognition_model_path, "")
        return detector, recognizer, svc, mydict
    except cv.error as e:
        raise RuntimeError(f"Error loading models: {e}")