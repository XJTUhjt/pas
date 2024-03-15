import numpy as np
import math
from scipy import *
from crowd_sim.envs.grid_utils import *
import warnings
warnings.filterwarnings('ignore')


############## Sensor_grid ##################
# 0:  empty
# 1 : occupied
# 0.5 :  unknown/occluded
#############################################


# Find the unknown cells using polygons for faster computation. (Result similar to ray tracing)
def generateSensorGrid(label_grid, ego_dict, ref_dict, map_xy, FOV_radius, res=0.1):
	x_local, y_local = map_xy
	
	center_ego = ego_dict['pos']
	occluded_id = []
	visible_id = []

	# get the maximum and minimum x and y values in the local grids
	x_shape = x_local.shape[0]
	y_shape = x_local.shape[1]	

	#记录多边形点列表
	polygon_points_humanid_list = []

	id_grid = label_grid[1].copy()


	unique_id = np.unique(id_grid) # does not include ego (robot) id
   
	# cells not occupied by ego itself
	mask = np.where(label_grid[0]!=2, True, False)

	# no need to do ray tracing if no object on the grid
	if np.all(label_grid[0,mask]==0.):
		sensor_grid = np.zeros((x_shape, y_shape))
	
	else:
		sensor_grid = np.zeros((x_shape, y_shape)) 

		ref_pos = np.array(ref_dict['pos'])
		ref_r = np.array(ref_dict['r'])

		# Find the cells that are occluded by the obstructing human agents 寻找被agent遮挡的grid id
		# reorder humans according to their distance from the robot. 根据距离机器人的远近给人群排序
		distance = [np.linalg.norm(center-center_ego) for center in ref_pos] #返回每个人的距离
		sort_indx = np.argsort(distance) #返回升序排列的索引
	
		unchecked_id = np.array(ref_dict['id'])[sort_indx] #没有见检查过的机器人，根据升序排列
		# Create occlusion polygons starting from closest humans. Reject humans that are already inside the polygons.
		#用平行四边形判断某人是否被遮挡

		for center, human_radius, h_id in zip(ref_pos[sort_indx], ref_r[sort_indx], unchecked_id):	
			# if human is already occluded, then just pass
			if h_id in occluded_id:
				continue

			hmask = (label_grid[1,:,:]==h_id)
			sensor_grid[hmask] = 1.

			alpha = math.atan2(center[1]-center_ego[1], center[0]-center_ego[0])

			theta = math.asin(np.clip(human_radius/np.sqrt((center[1]-center_ego[1])**2 + (center[0]-center_ego[0])**2), -1., 1.))

			
			# 4 or 5 polygon points
			# 2 points from human
			x1 = center_ego[0] + human_radius/np.tan(theta)*np.cos(alpha-theta)
			y1 = center_ego[1] + human_radius/np.tan(theta)*np.sin(alpha-theta)

			x2 = center_ego[0] + human_radius/np.tan(theta)*np.cos(alpha+theta)
			y2 = center_ego[1] + human_radius/np.tan(theta)*np.sin(alpha+theta)


			# Choose points big/small enough to cover the region of interest in the grid
			if x1 <= center_ego[0]:
				x3 = -12. 
			else:
				x3 = 12. 

			if center_ego[0] == x1:
				# print(center_ego[0], center_ego[1], x1, y1, x3, alpha, theta)
				if center_ego[1] < center[1]: 
					y3 = 12
				else:
					y3 = -12
			else:
				y3 = np.clip(linefunction(center_ego[0], center_ego[1], x1, y1, x3), -12., 12.)

			# assert not np.isnan(y3), 'y3 is nan'

			if x2 <= center_ego[0]:
				x4 = -12. 
			else:
				x4 = 12. 

			if center_ego[0] == x2:
				# print(center_ego[0], center_ego[1], x1, y1, x3, alpha, theta)
				if center_ego[1] < center[1]: 
					y4 = 12
				else:
					y4 = -12
			else:
				y4 = np.clip(linefunction(center_ego[0],center_ego[1],x2,y2,x4), -12., 12.)

			# assert not np.isnan(y3), 'y4 is nan'

			polygon_points = np.array([[x1, y1], [x2, y2], [x4, y4],[x3, y3]], dtype=np.float32)

			#判断是否是五边形并给出中间点
			def if_5_poly(points):
				#如果他们在一根轴上。则是四边形
				if points[2][0] == points[3][0] or points[2][1] == points[3][1]:
					x = y = 0
				else:
					#斜率为正
					if points[2][1] - points[3][1] / points[2][0] - points[3][0] > 0:
						if (points[2][0] + points[3][0]) / 2 > 0:
							x = 12
							y = -12
						else:
							x = -12
							y = 12
					else:
						if (points[2][0] + points[3][0]) / 2 > 0:
							x = y =12
						else:
							x = y = -12
							
				return x, y
			
			x5, y5 = if_5_poly(polygon_points)
			polygon_points_humanid_list.append([h_id, x1, y1, x3, y3, x5, y5, x4, y4, x2, y2])

			grid_points = np.array([x_local.flatten(), y_local.flatten()])	
			#occupied mask，通过平行四边形算出遮挡的区域
			occ_mask = parallelpointinpolygon(grid_points.T, polygon_points)
			occ_mask = occ_mask.reshape(x_local.shape)
			sensor_grid[occ_mask] = 0.5

			# check if any agent is fully inside the polygon 检查是否被完全覆盖
			for oid in unchecked_id:
				oid_mask = (label_grid[1,:,:]==oid)
				# if any agent is fully inside the polygon store in the occluded_id and opt from unchecked_id			
				if np.all(sensor_grid[oid_mask] == 0.5):
					occluded_id.append(oid)
					unchecked_id = np.delete(unchecked_id, np.where(unchecked_id==h_id))

	# Set cells out side of field of view as unknown
	FOVmask = point_in_circle(x_local, y_local, ego_dict['pos'], FOV_radius, res) 
	sensor_grid[np.invert(FOVmask)] = 0.5
 
	#找到任何可以看到的id，在FOV范围内
	for id in unique_id:
		mask1 = (label_grid[1,:,:]==id)
		if np.any(sensor_grid[mask1] == 1.):
			sensor_grid[mask1] = 1. 
			visible_id.append(int(id))
	
		else:
			pass

	#删除不在visible_id内的数据
	if len(polygon_points_humanid_list) != 0:
		filtered_polygon_points = [data for data in polygon_points_humanid_list if data[0] in visible_id]
		#可见有数据
		if len(filtered_polygon_points) != 0:
			polygon_points_humans_np = np.vstack(filtered_polygon_points)
			#补全矩阵维度：
			for lack_num in range(len(ref_dict['id']) - polygon_points_humans_np.shape[0]):
				polygon_points_humans_np = np.vstack([polygon_points_humans_np, [999, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]])
			result_polygon_points_humans_np = polygon_points_humans_np
		else:
		#有数据但是不在FOV内，创建0矩阵
			polygon_points_humans_np = np.array([[999, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]])
			#补全矩阵维度
			for lack_num in range(len(ref_dict['id']) - 1):
				polygon_points_humans_np = np.vstack([polygon_points_humans_np, [999, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]])

			result_polygon_points_humans_np = polygon_points_humans_np
	#等于零的话也要输出结果np矩阵，形状为human_num * 11
	else:
		#初始化矩阵
		polygon_points_humans_np = np.array([[999, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]])
		#补全矩阵维度
		for lack_num in range(len(ref_dict['id']) - 1):
			polygon_points_humans_np = np.vstack([polygon_points_humans_np, [999, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]])

		result_polygon_points_humans_np = polygon_points_humans_np

	# print(result_polygon_points_humans_np)
	# assert not np.isnan(result_polygon_points_humans_np).any(), "NaN values found in result_polygon_points_humans_np!"
		
	return visible_id, sensor_grid, result_polygon_points_humans_np

