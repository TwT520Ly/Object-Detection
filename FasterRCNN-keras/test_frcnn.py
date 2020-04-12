import os
import cv2
import numpy as np
import sys
import pickle
from optparse import OptionParser
import time
from keras_frcnn import config
import keras_frcnn.resnet as nn
from keras import backend as K
from keras.layers import Input
from keras.models import Model
from keras_frcnn import roi_helpers

# 增加递归程度
sys.setrecursionlimit(40000)

# 构造参数解析器
parser = OptionParser()

# ‘dest’是存储的变量，可以访问dest的值得到参数值
# 测试集路径
parser.add_option("-p", "--path", dest="test_path", help="Path to test data.", default='data/')
# roi数量
parser.add_option("-n", "--num_rois", dest="num_rois",
				help="Number of ROIs per iteration. Higher means more memory use.", default=32)
# 超参数配置文件
parser.add_option("--config_filename", dest="config_filename", help=
				"Location to read the metadata related to the training (generated when training).",
				default="config.pickle")

(options, args) = parser.parse_args()

# 寻找是否有测试集路径
if not options.test_path:   # if filename is not given
	parser.error('Error: path to test data must be specified. Pass --path to command line')

# 输出配置文件
config_output_filename = options.config_filename

# 加载配置文件，C是一个config对象
with open(config_output_filename, 'rb') as f_in:
	C = pickle.load(f_in)

# turn off any data augmentation at test time
C.use_horizontal_flips = False
C.use_vertical_flips = False
C.rot_90 = False

img_path = options.test_path

def format_img(img, C):
	img_min_side = float(C.im_size)
	(height,width,_) = img.shape

	# 对原始图片进行放缩，固定最短边，按照缩放比例进行放缩
	if width <= height:
		f = img_min_side/width
		new_height = int(f * height)
		new_width = int(img_min_side)
	else:
		f = img_min_side/height
		new_width = int(f * width)
		new_height = int(img_min_side)
	img = cv2.resize(img, (new_width, new_height), interpolation=cv2.INTER_CUBIC)
	# BGR->RGB
	img = img[:, :, (2, 1, 0)]
	img = img.astype(np.float32)
	img[:, :, 0] -= C.img_channel_mean[0]
	img[:, :, 1] -= C.img_channel_mean[1]
	img[:, :, 2] -= C.img_channel_mean[2]
	# 像素缩放
	img /= C.img_scaling_factor
	# HWC->CHW 维度转换（高维转置）
	img = np.transpose(img, (2, 0, 1))
	# NCHW
	img = np.expand_dims(img, axis=0)
	return img

# Method to transform the coordinates of the bounding box to its original size
# 将缩放后的图像框放大到原始图像中，平移+放大
def get_real_coordinates(ratio, x1, y1, x2, y2):

	real_x1 = int(round(x1 // ratio))
	real_y1 = int(round(y1 // ratio))
	real_x2 = int(round(x2 // ratio))
	real_y2 = int(round(y2 // ratio))

class_mapping = C.class_mapping

if 'bg' not in class_mapping:
	class_mapping['bg'] = len(class_mapping)

# 反向映射字典，key: 序号值，value: 类别名
class_mapping = {v: k for k, v in class_mapping.items()}
print(class_mapping)
# 将类别名对应为一个颜色，key: 类别名，value: RGB颜色
class_to_color = {class_mapping[v]: np.random.randint(0, 255, 3) for v in class_mapping}
# 用户的命令行参数对config中的num_rois进行修改
C.num_rois = int(options.num_rois)

if K.image_dim_ordering() == 'th':
	# 元组，生成三维的，但是后两个为空维度
	input_shape_img = (3, None, None)
	input_shape_features = (1024, None, None)
else:
	input_shape_img = (None, None, 3)
	input_shape_features = (None, None, 1024)

# 实例化一个keras-tensor
img_input = Input(shape=input_shape_img)
roi_input = Input(shape=(C.num_rois, 4))
feature_map_input = Input(shape=input_shape_features)

# define the base network (resnet here, can be VGG, Inception, etc)
shared_layers = nn.nn_base(img_input, trainable=True)

# define the RPN, built on the base layers
# 每一个anchor生成的框的个数，此处为9个
num_anchors = len(C.anchor_box_scales) * len(C.anchor_box_ratios)
rpn_layers = nn.rpn(shared_layers, num_anchors)

classifier = nn.classifier(feature_map_input, roi_input, C.num_rois, nb_classes=len(class_mapping), trainable=True)

# 输入为img_input，网络层结构为rpn_layers
model_rpn = Model(img_input, rpn_layers)
model_classifier_only = Model([feature_map_input, roi_input], classifier)

model_classifier = Model([feature_map_input, roi_input], classifier)

model_rpn.load_weights(C.model_path, by_name=True)
model_classifier.load_weights(C.model_path, by_name=True)

model_rpn.compile(optimizer='sgd', loss='mse')
model_classifier.compile(optimizer='sgd', loss='mse')

all_imgs = []

classes = {}

bbox_threshold = 0.8

visualise = True

# 遍历所有测试集的图片
for idx, img_name in enumerate(sorted(os.listdir(img_path))):
	if not img_name.lower().endswith(('.bmp', '.jpeg', '.jpg', '.png', '.tif', '.tiff')):
		continue
	# 输出图片名字
	print(img_name)
	# 记录开始时间
	st = time.time()
	# 合并图片名和根目录路径
	filepath = os.path.join(img_path,img_name)
	# 读取图片
	img = cv2.imread(filepath)
	# 图像预处理：大小缩放，去均值，像素缩放
	X = format_img(img, C) # NCHW

	# 从NCHW访问索引N=0的数组（三维），并将RGB->BGR，再进行转置得到HWC，全部过程都是深拷贝
	img_scaled = np.transpose(X.copy()[0, (2, 1, 0), :, :], (1, 2, 0)).copy()
	# BGR补上RGB顺序的均值
	img_scaled[:, :, 0] += 123.68
	img_scaled[:, :, 1] += 116.779
	img_scaled[:, :, 2] += 103.939

	# int8转换
	img_scaled = img_scaled.astype(np.uint8)

	if K.image_dim_ordering() == 'tf':
		# NCHW->NHWC 提高tensorflow计算效率
		X = np.transpose(X, (0, 2, 3, 1))

	# get the feature maps and output from the RPN
	# 调用网络计算结果，X->NHWC，输出的结果就是rpn_layer的返回值
	# Y1: classification
	# Y2: regression
	# F: features
	[Y1, Y2, F] = model_rpn.predict(X)
	# roi: region of interest
	R = roi_helpers.rpn_to_roi(Y1, Y2, C, K.image_dim_ordering(), overlap_thresh=0.7)

	# convert from (x1,y1,x2,y2) to (x,y,w,h)
	R[:, 2] -= R[:, 0]
	R[:, 3] -= R[:, 1]

	# apply the spatial pyramid pooling to the proposed regions
	bboxes = {}
	probs = {}

	for jk in range(R.shape[0]//C.num_rois + 1):
		ROIs = np.expand_dims(R[C.num_rois*jk:C.num_rois*(jk+1), :], axis=0)
		if ROIs.shape[1] == 0:
			break

		if jk == R.shape[0]//C.num_rois:
			#pad R
			curr_shape = ROIs.shape
			target_shape = (curr_shape[0],C.num_rois,curr_shape[2])
			ROIs_padded = np.zeros(target_shape).astype(ROIs.dtype)
			ROIs_padded[:, :curr_shape[1], :] = ROIs
			ROIs_padded[0, curr_shape[1]:, :] = ROIs[0, 0, :]
			ROIs = ROIs_padded

		[P_cls, P_regr] = model_classifier_only.predict([F, ROIs])

		for ii in range(P_cls.shape[1]):

			if np.max(P_cls[0, ii, :]) < bbox_threshold or np.argmax(P_cls[0, ii, :]) == (P_cls.shape[2] - 1):
				continue

			cls_name = class_mapping[np.argmax(P_cls[0, ii, :])]

			if cls_name not in bboxes:
				bboxes[cls_name] = []
				probs[cls_name] = []

			(x, y, w, h) = ROIs[0, ii, :]

			cls_num = np.argmax(P_cls[0, ii, :])
			try:
				(tx, ty, tw, th) = P_regr[0, ii, 4*cls_num:4*(cls_num+1)]
				tx /= C.classifier_regr_std[0]
				ty /= C.classifier_regr_std[1]
				tw /= C.classifier_regr_std[2]
				th /= C.classifier_regr_std[3]
				x, y, w, h = roi_helpers.apply_regr(x, y, w, h, tx, ty, tw, th)
			except:
				pass
			bboxes[cls_name].append([16*x, 16*y, 16*(x+w), 16*(y+h)])
			probs[cls_name].append(np.max(P_cls[0, ii, :]))

	all_dets = []

	for key in bboxes:
		bbox = np.array(bboxes[key])

		new_boxes, new_probs = roi_helpers.non_max_suppression_fast(bbox, np.array(probs[key]), overlap_thresh=0.5)
		for jk in range(new_boxes.shape[0]):
			(x1, y1, x2, y2) = new_boxes[jk,:]

			cv2.rectangle(img_scaled,(x1, y1), (x2, y2), class_to_color[key],2)

			textLabel = '{}: {}'.format(key,int(100*new_probs[jk]))
			all_dets.append((key,100*new_probs[jk]))

			(retval,baseLine) = cv2.getTextSize(textLabel,cv2.FONT_HERSHEY_COMPLEX,1,1)
			textOrg = (x1, y1-0)

			cv2.rectangle(img_scaled, (textOrg[0] - 5, textOrg[1]+baseLine - 5), (textOrg[0]+retval[0] + 5, textOrg[1]-retval[1] - 5), (0, 0, 0), 2)
			cv2.rectangle(img_scaled, (textOrg[0] - 5,textOrg[1]+baseLine - 5), (textOrg[0]+retval[0] + 5, textOrg[1]-retval[1] - 5), (255, 255, 255), -1)
			cv2.putText(img_scaled, textLabel, textOrg, cv2.FONT_HERSHEY_DUPLEX, 1, (0, 0, 0), 1)
	# 计算耗费的时间
	print('Elapsed time = {}'.format(time.time() - st))
	# 显示原图
	cv2.imshow('img', img_scaled)
	cv2.waitKey(0)
	#cv2.imwrite('./imgs/{}.png'.format(idx),img_scaled)
	print(all_dets)
