# Barcode-Detector
OpenCV 라이브러리를 이용한 Python 기반 바코드 검출기
\
93% 정확도 도달

#사용법
### 1. 바코드 검출
```
$ python detect.py -d dataset -r result -f detect.dat
```
### 2. 정확도 검사
```
$ python accuracy.py --reference accuracy/ref.dat --detect detect.dat
```

#파일 설명

1. detect.py
   - dataset 폴더에 존재하는 이미지에 대해 영상처리 모듈을 통한 바코드 검출
   - result 폴더에 Bounding box 결과 가시화
2. accuracy.py
   - accuracy/ref.dat(정답레이블)와 detect.dat(검출 결과 레이블)에 대해 정확도 비교