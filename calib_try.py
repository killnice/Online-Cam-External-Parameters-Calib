import json
import numpy as np
import cv2
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import np_lie as lie
import time
from scipy.optimize import minimize


class map_point:
	def __init__(self, id, pos, pt, des):
		self.id = id
		self.pos = pos
		self.pt = pt
		self.des = des
class OnlineExternalParametersCalib:
	def __init__(self, file_name):
		with open(file_name, 'r') as fp:
			self.data = json.load(fp)
			fx = self.data['fl_x']
			fy = self.data['fl_y']
			cx = self.data['cx']
			cy = self.data['cy']
			self.K = np.array([[fx, 0, cx],
							   [0, fy, cy],
							   [0, 0,  1 ]],dtype=np.float64)
		self.global_map = list()
		self.local_map = list()
		self.opti_info = list()
	def keypointToPoint(self, keypoint):
		point = np.zeros(len(keypoint) * 2, np.float32)
		for i in range(len(keypoint)):
			point[i * 2] = keypoint[i].pt[0]
			point[i * 2 + 1] = keypoint[i].pt[1]
		point = point.reshape(-1, 2)
		return point

	def read_data(self, index):
		img_name = self.data['frames'][index]['file_path']
		# img_name = img_name[:2] + 'fox/' + img_name[2:]
		# print("image", img_name)
		img = cv2.imread('fox/' + img_name)
		# print(img.shape)
		pre_idx = 0
		while img is None:
			pre_idx += 1
			img_name = self.data['frames'][index + pre_idx]['file_path']
			img = cv2.imread('fox/' + img_name)
		pose = np.array(self.data['frames'][index]['transform_matrix'])

		return img, pose

	def pixel2cam(self, uv):
		piexl = np.array([uv[0], uv[1], 1]).T
		return np.matmul(np.linalg.inv(self.K), piexl)

	def rawpixel2cam(self, pts):
		out = list()
		for uv in pts:
			piexl = np.array([uv[0], uv[1], 1]).T
			out.append(np.matmul(np.linalg.inv(self.K), piexl))
		return np.array(out)[:, :2]

	def reprojecterr_arr(self, uv, P, T):
		P_in_cam = T @ P.T
		P_in_cam = P_in_cam.T
		P_in_cam = P_in_cam[:, :3]
		s = P_in_cam[:, 2]
		self.K @ P_in_cam.T / s
		err_arr = (np.array([uv[:, 0], uv[:, 1]]) - (self.K @ P_in_cam.T / s)[:2]).T
		return err_arr

	def reprojecterr(self, uv, P, T):
		P_in_cam = T @ P.T
		P_in_cam = P_in_cam.T
		P_in_cam = P_in_cam[:, :3]
		s = P_in_cam[:, 2]
		err = (np.array([uv[:, 0], uv[:, 1]]) - (self.K @ P_in_cam.T / s)[:2]).T
		return np.mean(err, axis=0)  # numpy.linalg.norm

	def J_pixelerr_to_se3(self, PinWorld, T):  # 重投影误差到李代数的一阶微分关系（雅可比）
		fx = self.K[0, 0]
		fy = self.K[1, 1]
		P_in_cam = T @ PinWorld.T
		P_in_cam = P_in_cam.T
		X = P_in_cam[0]
		Y = P_in_cam[1]
		Z = P_in_cam[2]
		J = - np.array([[-fx*X*Y/(Z*Z),   fx+fx*X*X/(Z*Z), -fx*Y/Z, fx/Z, 0, -fx*X/(Z*Z)],
					  [ -fy-fy*Y*Y/(Z*Z), fy*X*Y/(Z*Z),    fy*X/Z,  0, fy/Z, -fy*Y/(Z*Z)]])
		return J

	def update_ExternalParameter(self, uv, P, T):
		# uv: 特征点集合
		# P: 地图点集合
		# T: n+1帧图像位姿 4x4
		last_time = time.time()
		pose = T
		opti_time = 0
		err_init = self.reprojecterr(uv, P, pose)
		err = err_init
		lr = 0.0000001
		if np.mean(np.abs(err_init)) > 20:
			# print('初始位姿不准确, 不进行优化, 初始重投影误差为', err_init)
			return False, T
		reprojecterr_arr = list()

		while opti_time < 50:  # 设定迭代次数限制
			opti_time += 1
			J_arr = list()
			err = self.reprojecterr(uv, P, pose)   # 计算重投影误差
			for i in range(P.shape[0]):
				J = self.J_pixelerr_to_se3(P[i], pose)
				J_arr.append(J)
				
			J_mean = np.mean(np.array(J_arr), axis=0)
			se3 = lie.SE3_log(pose)  # 4x4 李群 --> 1x6 李代数 [fai, rou]
			dx = err @ J_mean  # 计算梯度
			se3 -= lr*dx  #梯度更新se3
			pose = lie.SE3_exp(se3)
			reprojecterr_arr.append(np.mean(np.abs(err)))
			# if np.mean(np.abs(err)) < 0.05:
			# 	break
			# print('J_mean', J_mean, 'err', err, dx)

		if np.mean(np.abs(err)) <= 1.5:
			self.opti_info.append(reprojecterr_arr)
			print('优化前', err_init, '优化后',
				  err, '位置变化', (pose[:3, 3] - T[:3, 3]) * 1000, "mm", "用时", time.time() - last_time, "s")
			return True, pose
		else:
			# print('没有收敛, 优化前的重投影误差', err_init, '优化后的重投影误差', err, (pose[:3, 3] - T[:3, 3]) * 1000, "mm")
			return False, pose

	def multi_match_and_map(self, index0, num):
		hessian = 400
		surf = cv2.xfeatures2d.SURF_create(hessianThreshold=hessian)
		small_map = list()
		for i in range(index0, index0+num+1):
			img, pose = self.read_data(i)
			kp, des = surf.detectAndCompute(img, None)
			info = dict()
			info['pose'] = pose
			info['org_kp'] = np.array(kp)
			info['des'] = des
			small_map.append(info)


	def match_and_map(self, index1, index2):
		img1, pose1 = self.read_data(index1)
		img2, pose2 = self.read_data(index2)
		hessian = 800
		surf = cv2.xfeatures2d.SURF_create(hessianThreshold=hessian)
		# descriptor = cv2.SURF(hessian) #将Hessian Threshold设置为400,阈值越大能检测的特征就越少
		kp1, des1 = surf.detectAndCompute(img1, None)
		kp2, des2 = surf.detectAndCompute(img2, None)
		kp1 = np.array(kp1)
		kp2 = np.array(kp2)
		bf = cv2.BFMatcher()
		matches = bf.knnMatch(des1, des2, 2)
		good = []
		goodmatch = []
		for m, n in matches:
			if m.distance < 0.5 * n.distance:
				good.append([m])
				goodmatch.append(m)
		img3 = cv2.drawMatchesKnn(img1, kp1, img2, kp2, good, None, flags=2)
		src_pts = np.array([kp1[m.queryIdx].pt for m in goodmatch])
		dst_pts = np.array([kp2[m.trainIdx].pt for m in goodmatch])
		src_pts_norm = self.rawpixel2cam(src_pts)
		dst_pts_norm = self.rawpixel2cam(dst_pts)
		point3D = cv2.triangulatePoints(pose1[:3], pose2[:3], src_pts_norm.T, dst_pts_norm.T)

		point3D /= point3D[3]
		map3D = point3D.T
		isvalid, pose2 = self.update_ExternalParameter(dst_pts, map3D, pose2)
		if isvalid:
			point3D = cv2.triangulatePoints(pose1[:3], pose2[:3], src_pts_norm.T, dst_pts_norm.T)
			point3D /= point3D[3]
			self.local_map.append(point3D)

		cv2.imshow('frame', cv2.resize(img3, None, fx=0.4, fy=0.4))


if __name__ == '__main__':
	file_name = './fox/transforms.json'
	Calib = OnlineExternalParametersCalib(file_name)
	# 特征点匹配 -->三角化得到初始地图点+初始位姿 --> pnp+BA 使用初始位姿为初值pose，使用最小二乘法得到最小重投影误差 --> 精确的相机位姿
	# Calib.read_data(0)
	for i in range(1, 20):
		Calib.match_and_map(i, i + 1)
		key = cv2.waitKey(1)
		if key == 27:
			# np.save("opti_info", Calib.opti_info)
			break









