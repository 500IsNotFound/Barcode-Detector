import numpy as np
import argparse
import pdb

# read filename and position data from a given file
# and create a dictionary
def readRectData(fname):
	file = open(fname, 'r', encoding='UTF8')

	pos_dict = {}
	num = 0
	for line in file:
		data = line.split()
		pos_dict[data[0].replace(u"\ufeff", '')] = [int(x) for x in data[1:]]

		num = num + 1

	print('total number of list: ', num)

	file.close()

	return num, pos_dict

# calculate a ratio of intersection of two rectangle against
# reference rectangle : True Positive Rate (TPR)
# calculate a ratio of intersection of two rectangle against
# detected rectangle : Precision
# arguments: lists of format [sx, sy, ex, ey]
def calcRatioOfIntersectArea(rect1, rect2):
	# r2.sx > r1.ex or r2.ex < r1.sx : no intersection
	if (rect2[0] > rect1[2]) or (rect2[2] < rect1[0]):
		return [0.0, 0.0, 0.0, 0.0]
	# r2.sy > r1.ey or r2.ey < r1.sy : no intersection
	if (rect2[1] > rect1[3]) or (rect2[3] < rect1[1]):
		return [0.0, 0.0, 0.0, 0.0]

	sx = max([rect1[0], rect2[0]])
	ex = min([rect1[2], rect2[2]])
	sy = max([rect1[1], rect2[1]])
	ey = min([rect1[3], rect2[3]])

	area_actual = (rect1[2] - rect1[0]) * (rect1[3] - rect1[1])
	area_detected = (rect2[2] - rect2[0]) * (rect2[3] - rect2[1])
	true_positive = (ex - sx) * (ey - sy)
	false_positive = area_detected - true_positive

	precision = true_positive / area_actual
	recall = true_positive / area_detected

	area_of_union = area_actual + area_detected - true_positive

	if true_positive <= 0 :
		iou = 0
	else:
		iou = true_positive / area_of_union

	if (precision + recall) == 0 :
		f1_score = 0
	else :
		f1_score = 2*(precision*recall) / (precision + recall)

	return [precision, recall, f1_score, iou]

if __name__ == '__main__' :
	# 명령행 인자 처리
	ap = argparse.ArgumentParser()
	ap.add_argument("-r", "--reference", required=True,
		help="File on reference position of plate")
	ap.add_argument("-d", "--detect", required=True,
		help="File on detected position of plate")

	args = vars(ap.parse_args())

	# read reference data
	ref_num, ref_data = readRectData(args["reference"])
	# read detection data
	res_num, res_data = readRectData(args["detect"])

	# save accuracies to file
	file = open('accuracy.dat', 'w', encoding='UTF8')

	# find a correspoding item in reference with given detection item
	# create a list with accuracies
	accuracy = []
	idx = 0
	for key in ref_data:
		ref = ref_data.get(key)
		res = res_data.get(key)
		if(res != None):
			[pre, rec, f1, iou] = calcRatioOfIntersectArea(ref, res)
			accuracy.append([pre, rec, f1, iou])
			txt = key + ' ' + str(pre) + ' ' + str(rec) + ' ' + str(f1) +' '+ str(iou)+ '\n'
			file.write(txt)
			idx = idx + 1
		#else:
		#	accuracy.append(0.0)
		#idx = idx + 1
	#print(accuracy)

	# calculate average of accuracies
	sum_precision = 0.0
	sum_recall = 0.0
	sum_f1_score = 0.0
	sum_iou = 0.0

	for precision, recall, f1_score, iou in accuracy:
		sum_precision = sum_precision + precision
		sum_recall = sum_recall + recall
		sum_f1_score = sum_f1_score + f1_score
		sum_iou = sum_iou + iou

	avg_precision = sum_precision / idx
	avg_recall = sum_recall / idx
	avg_f1_score = sum_f1_score / idx
	avg_iou = sum_iou / idx

	file.close()
	print("Accuracy data in accuracy.dat")
	print("==============================")
	print("average Precision: ", round(avg_precision, 2))
	print("average Recall: ", round(avg_recall, 2))
	print("average F1_Score: ", round(avg_f1_score, 2))
	print("average IOU_Score: ", round(avg_iou, 2))