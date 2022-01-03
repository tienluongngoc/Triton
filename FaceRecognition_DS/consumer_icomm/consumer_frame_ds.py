from hnsw import HNSW
import utils
import time
import os
import numpy as np
from ic_rabbitmq import ICRabbitMQ
import sys
import cv2
from landmark_model import LandmarkModelBatchTRT
import json
from datetime import datetime
import pika

time_format = "%Y-%m-%d %H:%M:%S"
model_landmark = LandmarkModelBatchTRT("weights/landmark_model.engine")


def check_update_data(path_index):
    global hnsw_model
    global time_check_update
    global time_format
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


def search_top_k(model_hnsw, face_embedding, top_k_face=21*6, top_k_user=3):
    '''
        HNSW search for single embedding
    '''
    face_embedding = np.array([np.squeeze(face_embedding)])
    label, distance = model_hnsw.search(face_embedding, k=min(
        top_k_face, model_hnsw.cur_num_of_elements))
    distances = [float(distance[0][i]) for i in range(top_k_face)]
    id_faces = [str(label[0][i]) for i in range(top_k_face)]
    usr_ids = [model_hnsw.meta["face"]
               [str(label[0][i])] for i in range(top_k_face)]
    person_names = [model_hnsw.meta["user"]
                    [str(usr_id)]["name"] for ix, usr_id in enumerate(usr_ids)]
    return filter_by_user_id(usr_ids, person_names, distances, id_faces, top_k=top_k_user)


def filter_by_user_id(usr_ids, person_names, distances, id_faces, top_k=3):
    '''
        Filter result from face id to user-id (remove duplicate from augmentation images to user-id)
        Default: top 3 user-id
    '''
    # Sort by distance
    updi = sorted(zip(usr_ids, person_names, distances,
                  id_faces), key=lambda x: x[2])
    usr_ids = [x for x, _, _, _ in updi]
    person_names = [x for _, x, _, _ in updi]
    distances = [x for _, _, x, _ in updi]
    id_faces = [x for _, _, _, x in updi]

    lst_usr_ids = []
    lst_person_names = []
    lst_distances = []
    lst_id_faces = []
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


def merge_flip_result(normal_result, flip_result, top_k_user=3):
    usr_ids, person_names, distances, id_faces = normal_result
    flip_usr_ids, flip_person_names, flip_distances, flip_id_faces = flip_result
    usr_ids.extend(flip_usr_ids)
    person_names.extend(flip_person_names)
    distances.extend(flip_distances)
    id_faces.extend(flip_id_faces)
    return filter_by_user_id(usr_ids, person_names, distances, id_faces, top_k=top_k_user)


def callback_func(body):
    global hnsw_model
    global rab_push
    global channel_push
    global svm_model
    global filter_noisy_faces
    print('='*50)
    hnsw_model = check_update_data(path_indexs)

    st_time = time.time()

    feature = body['object']['person']['hair'].split(" ")[1:-1]
    
    id_tracking = int(feature[0])
    path_full_image =feature[1]
    path_img_crop = feature[2]
    path_aligned_image =feature[3]
    feature = np.array([[float(x) for x in feature[4:]]])
    print(path_aligned_image," : ",feature[0][:10])
    norm = np.linalg.norm(feature, axis=1, keepdims=True)
    face_ebb = np.divide(feature, norm)

    camera_url = body['place']['name']
    bbox = body['object']['bbox']
    start_time = body['@timestamp']
    time_stamp = datetime.strptime(start_time, time_format)
    time_stamp = datetime.timestamp(time_stamp)
    x1, y1, x2, y2 = int(bbox['topleftx']), int(bbox['toplefty']), int(
        bbox['bottomrightx']), int(bbox['bottomrighty'])
    
    # image_crop = cv2.imread(path_aligned_image) 
    # # image_crop = cv2.imread("image_aligned/crop.jpg")
    # if filter_noisy_faces:
    #     filted = 0

    # _, landmark_areas = model_landmark.predict_batch(
    #     np.array([image_crop]))

    # usr_ids, person_names, distances, id_faces = search_top_k(
    #     hnsw_model, face_ebb.copy())

    # print(person_names)
    # print(distances)
    # # Remove noisy face
    # if filter_noisy_faces and min(distances) > noisy_faces_thresh_conf:
    #     if landmark_areas[0] < noisy_faces_thresh_area:
    #         filted += 1
    # else:

    #     # pussmessage
    #     data_push = {
    #         "camera_url": camera_url,
    #         "start_time": start_time,
    #         "path_full_image": path_full_image,
    #         "time_stamp": time_stamp,
    #         "face_info": [],
    #         "H": 1080,
    #         "W": 1920
    #     }
    #     data_push["face_info"].append({"usr_ids": [int(usr_id) for usr_id in usr_ids],
    #                                    "person_names": person_names,
    #                                    "distances": distances,
    #                                    "id_faces": id_faces,
    #                                    "id_tracking": id_tracking,
    #                                    "box": [x1, x2, y1, y2],
    #                                    "path_face": path_img_crop,
    #                                    "is_noisy": 1 if ((landmark_areas[0] < noisy_faces_thresh_area) and (min(distances) > noisy_faces_thresh_conf)) else 0
    #                                    })

    #     # pushrabbit
    #     if filter_noisy_faces:
    #         print('Filted out {} from '.format(
    #             filted))
    #     # if len(data_push["face_info"]) > 0:
    #     #     try:
    #     #         ICRabbitMQ.publish_message(
    #     #             channel_push, queue_name_push, data_push)
    #     #     except pika.exceptions.StreamLostError:
    #     #         rab_push = ICRabbitMQ(host, virtual_host_push, usr_name, password)
    #     #         channel_push = rab_push.init_queue(queue_name_push)
    #     #         ICRabbitMQ.publish_message(
    #     #             channel_push, queue_name_push, data_push)
    #     #     # Visualize

    #     #     # cv2.resize(img_obj.img, (640, 480))
    #     #     if min(distances) <= 1.1:
    #     #         print(
    #     #             '-->', id_faces, [int(usr_id) for usr_id in usr_ids], distances, path_img_crop)
    #     # else:
    #     #     print('<!> Could not detect any face, so we remove it')
    #     #     os.remove(path_full_image)  # cv2.resize(img_obj.img, (640, 480))

    # print("inference time/ FPS: ", time.time() -
    #       st_time, 1.0/(time.time() - st_time + 0.001))

    # sys.exit()


if __name__ == "__main__":
    params = utils.load_parameter()
    hnsw_model = None
    path_indexs = params['path_indexs']
    check_flip = params["check_flip"]
    path_data_image_crop = params["path_data_image_crop"]
    filter_noisy_faces = True if params['filter_noisy_faces'] else False
    noisy_faces_thresh_area = params['noisy_faces_thresh_area']
    noisy_faces_thresh_conf = params['noisy_faces_thresh_conf']


# init rabbit
    usr_name = params["usr_name"]
    password = params["password"]
    host = params["host"]
    virtual_host_receive = params["virtual_host_receive"]
    virtual_host_push = params["virtual_host_push"]
    queue_name_receive = params["queue_name_receive"]
    queue_name_push = params["queue_name_push"]

    time_check_update = "-1"

    # print(path_indexs)

    # signatures = np.float32(np.random.random((10, 512)))
    # print(hnsw_model.search(signatures[0], 3, num_threads=-1))

    while True:
        # try:
        svm_model = None
        time_check_update = ""
        time_check_update_svm = ""
        hnsw_model = check_update_data(path_indexs)
        # svm_model = check_update_data_svm(path_indexs)
        rab_receive = ICRabbitMQ(
            host, virtual_host_receive, usr_name, password)
        rab_push = ICRabbitMQ(host, virtual_host_push, usr_name, password)
        channel_receive = rab_receive.init_queue(queue_name_receive)
        # channel_push = rab_push.init_queue(queue_name_push)
        print(" *wait message")

        def callback(ch, method, _, body):
            body = json.loads(body.decode("utf-8"))

            callback_func(body)
            ch.basic_ack(delivery_tag=method.delivery_tag)
            print("receive done: ")

        channel_receive.basic_qos(prefetch_count=10)
        channel_receive.basic_consume(
            queue=queue_name_receive, on_message_callback=callback)
        channel_receive.start_consuming()
        # except Exception as ve:
        #     print('<!> ERROR ->', ve)

        #     time_check_update = ""
        #     hnsw_model = check_update_data(path_indexs)
        #     # svm_model = check_update_data_svm(path_indexs)
        #     rab = ICRabbitMQ(host, virtual_host, usr_name, password)
        #     channel      = rab.init_queue(queue_name)
        #     channel_push = rab.init_queue(queue_name_push)
