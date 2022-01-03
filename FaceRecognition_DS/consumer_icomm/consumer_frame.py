import time
import argparse
from ICCObject import ImageObj
# from insightface import face_model
import cv2
import utils
from hnsw import HNSW
import os
import numpy as np
import datetime
from icommlib.rabbitmq import ICRabbitMQ
import json
import pickle
import math
# import darknet.darknet
import glob
import pika
# from face_detectors import RetinafaceBatchTRT
from face_model import FaceModelBatchTRT
from landmark_model import LandmarkModelBatchTRT
from face_tracker import FaceTracker

def check_update_data(path_index):
    global hnsw_model
    global time_check_update
    lst_md = []
    for root, _, fnames in os.walk(path_index):
        for fn in fnames:
            lst_md.append(os.stat(os.path.join(root, fn)).st_mtime)
    if len(lst_md) == 0:
        tm = ''
    else:
        tm = time.ctime(max(lst_md))
    if tm != time_check_update:
        time_check_update = tm
        for droot, _, fnames in os.walk(path_index):
            for fn in fnames:
                if fn.lower().endswith(".bin"):
                    try:
                        print('<!> Reload HNSW model')
                        hnsw_model = HNSW()
                        hnsw_model.load_model(droot, fn[:-4])
                    except Exception as e:
                        print(e)
    return hnsw_model



def search_top_k(model_hnsw, face_embedding, top_k_face = 21*6, top_k_user = 3):
    '''
        HNSW search for single embedding
    '''
    face_embedding = np.array([np.squeeze(face_embedding)])
    label, distance = model_hnsw.search(face_embedding, k=min(top_k_face, model_hnsw.cur_num_of_elements))
    distances = [float(distance[0][i]) for i in range(top_k_face)]
    id_faces = [str(label[0][i]) for i in range(top_k_face)]
    usr_ids = [model_hnsw.meta["face"][str(label[0][i])] for i in range(top_k_face)]
    person_names = [model_hnsw.meta["user"][str(usr_id)]["name"] for ix, usr_id in enumerate(usr_ids)]
    return filter_by_user_id(usr_ids, person_names, distances, id_faces, top_k = top_k_user)

def filter_by_user_id(usr_ids, person_names, distances, id_faces, top_k = 3):
    '''
        Filter result from face id to user-id (remove duplicate from augmentation images to user-id)
        Default: top 3 user-id
    '''
    # Sort by distance
    updi = sorted(zip(usr_ids, person_names, distances, id_faces), key = lambda x: x[2])
    usr_ids = [x for x, _, _, _ in updi]
    person_names = [x for _, x, _, _ in updi]
    distances = [x for _, _, x, _ in updi]
    id_faces = [x for _, _, _, x in updi]


    lst_usr_ids      = []
    lst_person_names = []
    lst_distances    = []
    lst_id_faces     = []
    for iu in range(len(usr_ids)):
        if usr_ids[iu] not in lst_usr_ids:
            lst_usr_ids.append(usr_ids[iu])
            lst_person_names.append(person_names[iu])
            lst_distances.append(distances[iu])
            lst_id_faces.append(id_faces[iu])
        if len(lst_usr_ids) == top_k:
            break
    return lst_usr_ids, lst_person_names, lst_distances, lst_id_faces

def filter_by_landmark(img_faces, bboxes, points):
    '''
        Filter noisy face by landmark
    '''
    _, out_landmark_area = model_landmark.predict_batch(np.array(img_faces))
    choosed = np.where(np.array(out_landmark_area) > noisy_faces_thresh)
    if len(choosed[0]) > 0:
        choosed = [ix[0] for ix in choosed]
        return np.array(img_faces)[choosed], np.array(bboxes)[choosed], np.array(points)[choosed]
    else:
        return [], [], []

def merge_flip_result(normal_result, flip_result, top_k_user = 3):
    usr_ids, person_names, distances, id_faces = normal_result
    flip_usr_ids, flip_person_names, flip_distances, flip_id_faces  = flip_result
    usr_ids.extend(flip_usr_ids)
    person_names.extend(flip_person_names)
    distances.extend(flip_distances)
    id_faces.extend(flip_id_faces)
    return filter_by_user_id(usr_ids, person_names, distances, id_faces, top_k = top_k_user) 


def callback_func(body):
    global hnsw_model
    global rab_push
    global channel_push
    global svm_model
    print('='*50)
    camera_url = body.get("camera_url", "")
    path_full_image = body.get("path_full_image", "")
    start_time = body.get("start_time", "")
    time_stamp = body.get("time_stamp", -1)
    path_stream_hash = body.get("path_stream_hash", "")
    bboxes = body.get("bboxes", [])
    points = body.get("points", [])
    # print(bboxes, points)
    h = body.get("H", -1)
    w = body.get("W", -1)
    hnsw_model = check_update_data(path_indexs)
    # path_save_background = '/home/tungnguyen2/checkin_icomm/Service_v3/src/background_{}.jpg'.format(camera_url.split('@')[-1])
    # svm_model = check_update_data_svm(path_indexs)
    st_time = time.time()
    img = cv2.imread(path_full_image) # normal opencv bgr image

    if img is not None:
        img_copy = img.copy()
        img_obj = ImageObj(img)
        # points, bboxes = model.get_input(img_obj.img, scales=scale)
        # bboxes, points = model.get_inputs(img_obj.img)
        bboxes = np.array(bboxes)
        points = np.array(points)
        if len(bboxes) > 0:
            img_faces = np.array(model.get_face_align(img_obj.img, bboxes, points)) # RGB faces
            assert len(img_faces) == len(bboxes), "Length not equal: {} vs {}".format(len(img_faces), len(bboxes))
            if len(img_faces) > 0:
                data_push = {
                             "camera_url": camera_url,
                             "start_time": start_time,
                             "path_full_image": path_full_image,
                             "time_stamp": time_stamp,
                             "face_info": [],
                             "H": h,
                             "W": w
                            }
                face_ebbs = model.get_features(img_faces)
                if check_flip:
                    face_ebbs_flip = model.get_features(img_faces, flip = 1)
                if filter_noisy_faces:
                    filted = 0
                _, landmark_areas = model_landmark.predict_batch(img_faces)
                for i in range(len(img_faces)):
                    face_ebb = face_ebbs[i]
                    face_ebb_flip = face_ebbs_flip[i]
                    x1, y1, x2, y2 = bboxes[i][:4]
                    usr_ids, person_names, distances, id_faces = search_top_k(hnsw_model, face_ebb.copy())
                    if check_flip:
                        flip_usr_ids, flip_person_names, flip_distances, flip_id_faces = search_top_k(hnsw_model, face_ebb_flip.copy())
                        normal_result = (usr_ids, person_names, distances, id_faces)
                        flip_result   = (flip_usr_ids, flip_person_names, flip_distances, flip_id_faces)
                        print('Using flip result')
                        print(usr_ids, person_names, distances)
                        print(flip_usr_ids, flip_person_names, flip_distances)
                        usr_ids, person_names, distances, id_faces = merge_flip_result(normal_result, flip_result)
                        print(usr_ids, person_names, distances)
                    id_tracking = tracker_model.fit(face_ebb.copy(), time_stamp)
                    if filter_noisy_faces and min(distances) > noisy_faces_thresh_conf: # Remove noisy face
                        if landmark_areas[i] < noisy_faces_thresh_area:
                            filted += 1
                            continue

                    path_folder_crop = utils.gen_folder_name(path_data_image_crop, path_stream_hash,
                                                             datetime.datetime.strptime(start_time, "%Y-%m-%d %H:%M:%S"))
                    path_img_crop = os.path.join(path_folder_crop, str(time_stamp) + "_{0}.jpg".format(str(i)))
                    # cv2.imwrite(path_img_crop, np.transpose(img_faces[i], (1, 2, 0)))
                    cv2.imwrite(path_img_crop, cv2.cvtColor(img_faces[i], cv2.COLOR_RGB2BGR))
                    data_push["face_info"].append({"usr_ids": [int(usr_id) for usr_id in usr_ids],
                                                   "person_names": person_names,
                                                   "distances": distances,
                                                   "id_faces": id_faces,
                                                   "id_tracking": id_tracking,
                                                   "box": bboxes[i][:4].tolist(),
                                                   "path_face": path_img_crop,
                                                   "is_noisy": 1 if ((landmark_areas[i] < noisy_faces_thresh_area) and (min(distances) > noisy_faces_thresh_conf)) else 0
                                                   })
                if filter_noisy_faces:
                    print('Filted out {} from {} face images'.format(filted, len(img_faces)))
                if len(data_push["face_info"]) > 0:
                    try:
                        ICRabbitMQ.publish_message(channel_push, queue_name_push, data_push)
                    except pika.exceptions.StreamLostError:
                        rab = ICRabbitMQ(host, virtual_host, usr_name, password)
                        channel_push = rab.init_queue(queue_name_push)
                        ICRabbitMQ.publish_message(channel_push, queue_name_push, data_push)
                    # Visualize

                    cv2.imwrite(path_full_image, cv2.resize(img_obj.img, (480, 320))) #cv2.resize(img_obj.img, (640, 480))
                    if min(distances) <= 1.1:
                        print('-->', id_faces, [int(usr_id) for usr_id in usr_ids], distances, path_img_crop)
                else:
                    print('<!> Could not detect any face, so we remove it')
                    os.remove(path_full_image) #cv2.resize(img_obj.img, (640, 480))


            # # Save storage from 00:00 AM to 05:00 AM
            # h_now = datetime.datetime.strptime(start_time, "%Y-%m-%d %H:%M:%S").hour
            # if 0 < h_now < 5:
            #     os.remove(path_full_image)
            # else:
            #     cv2.imwrite(path_full_image, cv2.resize(img_obj.img, (140, 108)))
    else:
        print('[X] Image error:', path_full_image)
    print("inference time/ FPS: ", time.time() - st_time, 1.0/(time.time() - st_time + 0.001))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='consumer frame')
    parser.add_argument('--scale', default="405,720", help='')
    parser.add_argument('--gpu', default=0, help='')
    args = parser.parse_args()

    gpu_device = int(args.gpu)
    scale = list(map(int, args.scale.split(",")))

    params = utils.load_parameter()
    path_indexs = params["path_indexs"]
    check_flip = params["check_flip"]
    path_data_image_crop = params["path_data_image_crop"]

    usr_name = params["usr_name"]
    password = params["password"]
    host = params["host"]
    virtual_host = params["virtual_host"]
    queue_name = "camera-checkin-frame"
    queue_name_push = "camera-checkin-recog"

    
    filter_noisy_faces = True if params['filter_noisy_faces'] else False
    noisy_faces_thresh_area = params['noisy_faces_thresh_area']
    noisy_faces_thresh_conf = params['noisy_faces_thresh_conf']
    # embedding_model = face_model.FaceModel(image_size, path_model, path_ga_model, gpu_device, det, flip, threshold, model_face_detection = None)
    model = FaceModelBatchTRT(embedding_weight = "weights/resnet160-mask-251121-batchsize_1-fp32.trt",
                                detector_weight = None,
                                embedding_batch_size = 1)
    # if filter_noisy_faces:
    model_landmark = LandmarkModelBatchTRT()
    tracker_model = FaceTracker(
                        max_time_exists = 10,    # in second
                        euclid_threshold = 1.1
        )
    
    while True:
        # try:
        svm_model = None
        time_check_update = ""
        time_check_update_svm = ""
        hnsw_model = check_update_data(path_indexs)
        # svm_model = check_update_data_svm(path_indexs)
        rab = ICRabbitMQ(host, virtual_host, usr_name, password)
        channel = rab.init_queue(queue_name)
        # channel_push = rab.init_queue(queue_name_push)

        print(" *wait message")

        def callback(ch, method, _, body):
            body = json.loads(body.decode("utf-8"))
            callback_func(body)
            ch.basic_ack(delivery_tag=method.delivery_tag)
            # print("receive done: ")

        channel.basic_qos(prefetch_count=10)
        channel.basic_consume(queue=queue_name, on_message_callback=callback)
        channel.start_consuming()
        # except Exception as ve:
        #     print('<!> ERROR ->', ve)
        
        #     time_check_update = ""
        #     hnsw_model = check_update_data(path_indexs)
        #     # svm_model = check_update_data_svm(path_indexs)
        #     rab = ICRabbitMQ(host, virtual_host, usr_name, password)
        #     channel      = rab.init_queue(queue_name)
        #     channel_push = rab.init_queue(queue_name_push)