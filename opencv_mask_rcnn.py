
import sys
from matplotlib.pyplot import close
import numpy as np
import cv2

# 220325
print(cv2.__version__)

def drawBox(img, classId, conf, left, top, right, bottom):
    # Draw a bounding box.
    cv2.rectangle(img, (left, top), (right, bottom), colors[classId], 2)

    label = f'{classes[classId]}: {conf:.2f}'

    labelSize, baseLine = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 1)
    top = max(top, labelSize[1])
    cv2.rectangle(img, (left - 1, top - labelSize[1] - baseLine),
                  (left + labelSize[0], top), colors[classId], -1)
    cv2.putText(img, label, (left, top - baseLine), cv2.FONT_HERSHEY_SIMPLEX,
                0.6, (0, 0, 0), 1, cv2.LINE_AA)

# 모델 & 설정 파일
model = 'mask_rcnn/frozen_inference_graph.pb'
config = 'mask_rcnn/mask_rcnn_inception_v2_coco_2018_01_28_.pbtxt'
class_labels = 'mask_rcnn/coco_90.names'
confThreshold = 0.6
maskThreshold = 0.3

# 테스트 이미지 파일
img_files = ['dog.jpg', 'traffic.jpg', 'sheep.jpg', 'bus.jpg', 'people.jpg', 'stop sign.jpg']
# 네트워크 생성
net = cv2.dnn.readNet(model, config)

if net.empty():
    print('Net open failed!')
    sys.exit()

# class name file open
classes = []
with open(class_labels, 'rt') as f:
    classes = f.read().rstrip('\n').split('\n')

# random color 지정
colors = np.random.uniform(0, 255, size=(len(classes), 3))

# 실행
for f in img_files:
    img = cv2.imread(f)

    if img is None:
        continue

    # blob 내부에서 Auto resize 하므로, 별도로 input image resize 할 필요 없음
    blob = cv2.dnn.blobFromImage(img, swapRB=True) # 블롭 생성 & 추론
    net.setInput(blob)
    boxes, masks = net.forward(['detection_out_final', 'detection_masks'])
    # detection_out_final / bounding box 정보를 가지고 있음 / boxes.shape=(1, 1, 100, 7)
    # detection_masks / 각 bounding box의 Mask 정보 / masks.shape=(100, 90, 15, 15)

    h, w = img.shape[:2]
    numClasses = masks.shape[1]  # 90
    numDetections = boxes.shape[2]  # 100, Detection 최대 개수

    boxesToDraw = []
    for i in range(numDetections):
        box = boxes[0, 0, i]  
        print(box[0].shape) # shape 확인, box.shape=(7,)
        mask = masks[i]  # mask.shape=(90, 15, 15)
        score = box[2]
        if score > confThreshold:
            classId = int(box[1]) # box[a,b,c] → a는 0, b은 class id, c는 confidence(확률) 정보

            x1 = int(w * box[3])
            y1 = int(h * box[4])
            x2 = int(w * box[5])
            y2 = int(h * box[6])

            x1 = max(0, min(x1, w - 1))
            y1 = max(0, min(y1, h - 1))
            x2 = max(0, min(x2, w - 1))
            y2 = max(0, min(y2, h - 1))

            boxesToDraw.append([img, classId, score, x1, y1, x2, y2])
            classMask = mask[classId]

            # 객체별 15x15 마스크를 바운딩 박스 크기로 resize한 후, 불투명 컬러로 표시
            classMask = cv2.resize(classMask, (x2 - x1 + 1, y2 - y1 + 1))
            mask = (classMask > maskThreshold)

            roi = img[y1:y2+1, x1:x2+1][mask]
            img[y1:y2+1, x1:x2+1][mask] = (0.7 * colors[classId] + 0.3 * roi).astype(np.uint8)

    # 객체별 바운딩 박스 그리기 & 클래스 이름 표시
    for box in boxesToDraw: 
        drawBox(*box) # * Tuple 형태러 전달

    t, _ = net.getPerfProfile()
    label = 'Inference time: %.2f ms' % (t * 1000.0 / cv2.getTickFrequency())
    cv2.putText(img, label, (10, 30), cv2.FONT_HERSHEY_SIMPLEX,
                0.7, (0, 0, 255), 1, cv2.LINE_AA)

    cv2.imshow('img', img)
    if cv2.waitKey(5000) == 27:
        break


cv2.destroyAllWindows()


