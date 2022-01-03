# -*- coding: utf-8 -*-
import os
import argparse
import cv2
import json
import scipy.io as sio
import hnswlib
import numpy as np
from multiprocessing.pool import ThreadPool


class HNSW(object):
	def __init__ (self, ef = 1000, metric='l2', dim = 512, num_elements= 1000000, M = 80):
		self.ef = ef
		self.metric = metric
		self.dim = dim
		self.num_elements = num_elements
		self.M = M
		self.cur_num_of_elements = 0
		self.model = None
		self.is_add = False
		self.meta = {'user':{}, 'face':{}}
		if self.dim and self.metric and self.num_elements and self.ef and self.M:
			self.create_model(metric=self.metric, dim = self.dim, num_elements = self.num_elements, ef = self.ef, M = self.M)
			
	def create_model(self, metric='l2', dim = 512, num_elements = 1000000, ef = 1000, M = 80):
		if metric:
			self.metric = metric
		if dim:
			self.dim = dim
		if num_elements:
			self.num_elements = num_elements
		if ef:
			self.ef = ef
		if M:
			self.M = M
		if self.dim and self.metric and self.num_elements and self.ef and self.M:
				self.model =  hnswlib.Index(space=self.metric, dim=self.dim)
				self.model.init_index(max_elements=self.num_elements, ef_construction=self.ef, M=self.M)
				self.model.set_ef(self.ef)

	def add(self, signatures, labels,  usr_ids, usr_names, num_threads = -1):
		if self.is_add:
			print("do not training with index is loaded or trained!")
			return False
		if self.cur_num_of_elements > self.num_elements:
			print("num of element nhiều quá")
			self.is_add = True
			return False
		else:
			self.cur_num_of_elements += len(labels)
			self.model.add_items(signatures, labels,num_threads = num_threads)
			for i in range(len(labels)):
				if str(usr_ids[i]) not in self.meta["user"]:
					self.meta["user"][str(usr_ids[i])] = {}
					self.meta["user"][str(usr_ids[i])]["face_id"] = []
					self.meta["user"][str(usr_ids[i])]["name"] = usr_names[i]
				self.meta["user"][str(usr_ids[i])]["face_id"].append(float(labels[i]))
				self.meta["face"][str(labels[i])] = usr_ids[i]
			self.is_add = True
			return True

	def save_model(self, file_pathmodel, name):
		path_bin = os.path.join(file_pathmodel,name + ".bin")
		path_json = os.path.join(file_pathmodel,name + "_meta.json")
		self.model.save_index(path_bin)
		data_meta_save = {"ef":self.ef, "metric":self.metric, "dim":self.dim, "num_elements":self.num_elements, "M":self.M, 
		"cur_num_of_elements": self.cur_num_of_elements, "meta":self.meta}
		with open(path_json,"w+") as f:
			json.dump(data_meta_save,f)

	def search(self, v, k, num_threads = -1):
		k = min(self.num_elements, k)
		labels, distances = self.model.knn_query(v, k=k, num_threads = num_threads)
		return labels, distances

	def load_model(self, file_pathmodel, name):
		path_bin = os.path.join(file_pathmodel,name + ".bin")
		path_json = os.path.join(file_pathmodel,name + "_meta.json")
		if not os.path.exists(path_bin) or not os.path.exists(path_json) :
			print("not found weight")
		else:
			data = {}
			with open(path_json) as f:
				data = json.loads(f.read())
			self.ef = data["ef"]
			self.metric = data["metric"]
			self.dim = data["dim"]
			self.num_elements = data["num_elements"]
			self.M = data["M"]
			self.cur_num_of_elements = data["cur_num_of_elements"]
			self.is_add = True
			self.meta = data["meta"]
			self.model = hnswlib.Index(space=self.metric, dim=self.dim)
			self.model.load_index(path_bin)
	def del_model(self):
		del self.model

if __name__ =="__main__":
	hnsw = HNSW(ef = 5, metric='l2', dim = 512, num_elements= 10, M = 5)
	signatures = np.float32(np.random.random((10, 512)))
	labels = np.array([1,1,2,3,4,5,6,7,8,9])
	print(labels)
	hnsw.add(signatures, labels, num_threads = -1)
	print(hnsw.search( signatures[0], 3, num_threads = -1))
