from __future__ import print_function
import argparse
import numpy as np
import cv2
import os
import glob


def detectBarcode(img):
    # HYPERPARAMETER
    filterX = np.array([[-1, 0, 1],
                        [-2, 0, 2],
                        [-1, 0, 1]])
    filterY = filterX.transpose()

    BLUR_KSIZE = (15, 15)
    SIGMA_X = 6

    CLOSE_KSIZE = (int(20 + 0.4 * img.shape[0] // 100), int(3 + 0.1 * img.shape[1] // 100))
    CLOSE_ITER = 3
    MORPH_KSIZE = (3, 3)
    ERODE_ITER = 30
    DILATE_ITER = 15

    # CODE
    barcodeContour = []
    for i in range(2):
        edgeX = cv2.filter2D(img, -1, filterX)
        edgeY = cv2.filter2D(img, -1, filterY)
        procImg = cv2.subtract(edgeX, edgeY) if i == 0 else cv2.subtract(edgeY, edgeX)

        procImg = cv2.GaussianBlur(procImg, BLUR_KSIZE, SIGMA_X)

        _, procImg = cv2.threshold(procImg, 30, 255, cv2.THRESH_BINARY)

        struct = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (CLOSE_KSIZE[i], CLOSE_KSIZE[1 - i]))
        procImg = cv2.morphologyEx(procImg, cv2.MORPH_CLOSE, struct, iterations=CLOSE_ITER)

        struct = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, MORPH_KSIZE)
        procImg = cv2.erode(procImg, struct, iterations=ERODE_ITER)
        procImg = cv2.dilate(procImg, struct, iterations=DILATE_ITER)

        contours, _ = cv2.findContours(procImg, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        if contours:
            contour = sorted(contours, key=cv2.contourArea, reverse=True)[0]
            barcodeContour.append(contour)
    bigger = max(barcodeContour, key=cv2.contourArea)
    (x, y, w, h) = cv2.boundingRect(bigger)

    return x, y, x + w, y + h


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument("-d", "--dataset", required=True, help="path to the dataset folder")
    ap.add_argument("-r", "--detectset", required=True, help="path to the detectset folder")
    ap.add_argument("-f", "--detect", required=True, help="path to the detect file")
    args = vars(ap.parse_args())

    dataset = args["dataset"]
    detectset = args["detectset"]
    detectfile = args["detect"]

    # 결과 영상 저장 폴더 존재 여부 확인
    if (not os.path.isdir(detectset)):
        os.mkdir(detectset)

    # 결과 영상 표시 여부
    verbose = False

    # 검출 결과 위치 저장을 위한 파일 생성
    f = open(detectfile, "wt", encoding="UTF-8")  # UT-8로 인코딩
    import time

    start = time.time()
    # 바코드 영상에 대한 바코드 영역 검출
    for imagePath in glob.glob(dataset + "/*.jpg"):
        print(imagePath, '처리중...')

        # 영상을 불러오고 그레이 스케일 영상으로 변환
        image = cv2.imread(imagePath)
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        # 바코드 검출
        points = detectBarcode(gray)

        # 바코드 영역 표시
        detectimg = cv2.rectangle(image, (points[0], points[1]), (points[2], points[3]), (0, 255, 0), 2)  # 이미지에 사각형 그리기

        # 결과 영상 저장
        loc1 = imagePath.rfind("\\")
        loc2 = imagePath.rfind(".")
        fname = 'result/' + imagePath[loc1 + 1: loc2] + '_res.jpg'
        cv2.imwrite(fname, detectimg)

        # 검출한 결과 위치 저장
        f.write(imagePath[loc1 + 1: loc2])
        f.write("\t")
        f.write(str(points[0]))
        f.write("\t")
        f.write(str(points[1]))
        f.write("\t")
        f.write(str(points[2]))
        f.write("\t")
        f.write(str(points[3]))
        f.write("\n")
        if verbose:
            cv2.imshow("image", image)
            cv2.waitKey(0)
    print(time.time() - start)