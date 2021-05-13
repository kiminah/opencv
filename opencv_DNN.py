import cv2, time, sys

# load model
model_path = 'model/opencv_face_detector_uint8.pb'
config_path = 'model/opencv_face_detector.pbtxt'

net = cv2.dnn.readNetFromTensorflow(model_path, config_path)

conf_threshold = 0.7

# initialize video source, default 0 (webcam)
# video_path = '1_num-Copy1.mp4'
# cap = cv2.VideoCapture(video_path)
if len(sys.argv) < 3:
    sys.exit()

src = sys.argv[1]
dst = sys.argv[2]
cap = cv2.VideoCapture(src)
idx = 0

fourcc = cv2.VideoWriter_fourcc('m', 'p', '4', 'v')
out = cv2.VideoWriter('%s_output_opencv_dnn.mp4' % (src.split('.')[0]), fourcc, cap.get(cv2.CAP_PROP_FPS),
                      (int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))))

frame_count, tt = 0, 0

while cap.isOpened():
    ret, img = cap.read()
    if not ret:
        break

    frame_count += 1

    start_time = time.time()

    # prepare input
    result_img = img.copy()
    h, w, _ = result_img.shape
    blob = cv2.dnn.blobFromImage(result_img, 1.0, (300, 300), [104, 117, 123], False, False)
    net.setInput(blob)

    # inference, find faces
    detections = net.forward()

    # postprocessing
    for i in range(detections.shape[2]):
        confidence = detections[0, 0, i, 2]
        if confidence > conf_threshold:
            x1 = int(detections[0, 0, i, 3] * w)
            y1 = int(detections[0, 0, i, 4] * h)
            x2 = int(detections[0, 0, i, 5] * w)
            y2 = int(detections[0, 0, i, 6] * h)

            # draw rects
            cv2.rectangle(result_img, (x1, y1), (x2, y2), (255, 255, 255), int(round(h / 150)), cv2.LINE_AA)
            cv2.putText(result_img, '%.2f%%' % (confidence * 100.), (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 1,
                        (255, 255, 255), 2, cv2.LINE_AA)

            # save
            cropped = img[y1: y2, x1: x2]
            fname = "{0}{1:05d}.jpg".format(dst, idx)
            cv2.imwrite(fname, cropped)
            idx += 1

    # inference time
    tt += time.time() - start_time
    fps = frame_count / tt
    cv2.putText(result_img, 'FPS(dnn): %.2f' % (fps), (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2,
                cv2.LINE_AA)

    #     # visualize
    #     cv2.imshow('result', result_img)
    #     if cv2.waitKey(1) == ord('q'):
    #         break

    out.write(result_img)

cap.release()
out.release()
cv2.destroyAllWindows()