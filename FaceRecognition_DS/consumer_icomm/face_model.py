import cv2
import numpy as np
from utils.image import ImageData
# from utils.getter import get_model
from common import face_preprocess
from face_processors import FaceEmbeddingBatchTRT
from face_detectors import RetinafaceBatchTRT
# from detectors.retinaface_batch import RetinafaceBatchTRT
import time

class FaceModelBase:
    def __init__(self):
        self.embedd_model = None
        self.detector_input_size = None
        self.detector = None
        self.embedd_model = None
        self.embedding_batch_size = None

    def visualize_detect(self, base_image, bboxes, points):
        '''
            Visualize detection
        '''
        vis_image = base_image.copy()
        for i in range(len(bboxes)):
            pt1 = tuple(map(int, bboxes[i][0:2]))
            pt2 = tuple(map(int, bboxes[i][2:4]))
            cv2.rectangle(vis_image, pt1, pt2, (0, 255, 0), 1)
            for lm_pt in points[i]:
                cv2.circle(vis_image, tuple(map(int, lm_pt)), 3, (0, 0, 255), 3)
        return vis_image

    def get_inputs(self, bgr_image, threshold = 0.8, polygons = None):
        '''
            Get boxes & 5 landmark points
            Input:
                - bgr_image: BGR image
            Output:
                - bboxes: face bounding boxes
                - points: 5 landmark points for each cor-response face
        '''
        # vis_image = bgr_image.copy()
        # image = ImageData(bgr_image, self.detector_input_size)
        # image.resize_image(mode='pad')     
        bboxes, points = self.detector.detect([bgr_image], threshold=threshold, polygons = polygons)
        bboxes = bboxes[0]
        points = points[0]
        # if len(bboxes) > 0:
        #     # Post processing
        #     bboxes = bboxes[:, :4]
        #     bboxes /= image.scale_factor
        #     points /= image.scale_factor
        # del image
        return bboxes, points

    @staticmethod
    def get_face_align(bgr_img, bboxes, points, image_size='112,112'):
        '''
            Align face from given bounding boxes and landmark points
        '''
        rgb_img = cv2.cvtColor(bgr_img, cv2.COLOR_BGR2RGB)
        aligned = []
        for i in range(len(bboxes)):
            bbox_i = bboxes[i]
            points_i = points[i]
            nimg = face_preprocess.preprocess(rgb_img, bbox_i, points_i, image_size=image_size)
            aligned.append(nimg)
        return aligned

    def get_features(self, aligned, flip = 0):
        '''
            Extract embedding for face
            Input:
                - aligned: batch of RGB aligned images
        '''
        return self.embedd_model.get_features(aligned, batch_size = self.embedding_batch_size, flip = flip)


class FaceModelBatchTRT(FaceModelBase):
    '''
        Face model with batch inference (both Retinaface + Embedding)
    '''
    def __init__(self,
        detector_batch_size = 1,
        detector_weight = 'weights/retinaface_r50_v1-1x3x640x640-fp16.trt',
        detector_input_size = (640, 640), 
        detector_post_process_type = 'BATCH',
        detector_engine = 'TRT',
        embedding_batch_size = 4,
        embedding_weight = "weights/resnet124-batchsize_8-fp32.trt",
        embedding_input_size = (112, 112),
        embedding_engine = 'TRT'):
        '''
            Init Detector & Embedding Extractor
        '''
        super(FaceModelBatchTRT, self).__init__()
        self.detector_input_size = detector_input_size
        self.detector = None
        if detector_weight is not None:
            self.detector = RetinafaceBatchTRT(model_path = detector_weight, \
                                            batch_size = detector_batch_size,\
                                            input_shape = detector_input_size, \
                                            post_process_type = detector_post_process_type,
                                            engine = detector_engine)
            self.detector.prepare(nms=0.4)
        self.embedd_model = None
        if embedding_weight is not None:
            self.embedd_model = FaceEmbeddingBatchTRT(trt_model_path = embedding_weight, \
                                            batch_size = embedding_batch_size, \
                                            input_shape=embedding_input_size,
                                            engine = embedding_engine)


    def get_inputs_batch(self, list_bgr_image, threshold = 0.6, polygons = None):
        '''
            Batch-processing: get boxes & 5 landmark points
            Input:
                - list_bgr_image: list BGR image [B x W x H x 3]
            Output:
                - bboxes: face bounding boxes [B x N x 4]
                - points: 5 landmark points for each cor-response face [B x N x 5 x 2]
        ''' 
        list_bboxes, list_points = self.detector.detect(list_bgr_image, threshold=threshold, polygons = polygons)
        return list_bboxes, list_points

    @staticmethod
    def get_face_align(bgr_img, bboxes, points, image_size='112,112'):
        '''
            Align face from given bounding boxes and landmark points
        '''
        rgb_img = cv2.cvtColor(bgr_img, cv2.COLOR_BGR2RGB)
        aligned = []
        for i in range(len(bboxes)):
            bbox_i = bboxes[i]
            points_i = points[i]
            nimg = face_preprocess.preprocess(rgb_img, bbox_i, points_i, image_size=image_size)
            aligned.append(nimg)
        return aligned

    def get_features(self, aligned, flip = 0):
        '''
            Extract embedding for face
            Input:
                - aligned: batch of RGB aligned images
        '''
        return self.embedd_model.get_features(aligned, flip = flip)

if __name__ == '__main__':
    # face_model = FaceModel()
    # image = cv2.imread('test_images/lumia.jpg')
    # bboxes, points = face_model.get_inputs(image)
    # vis = face_model.visualize_detect(image, bboxes, points)
    # cv2.imwrite('visualize_1.jpg', vis)
    # image = cv2.imread('test_images/Stallone.jpg')
    # bboxes, points = face_model.get_inputs(image)
    # vis = face_model.visualize_detect(image, bboxes, points)
    # cv2.imwrite('visualize_2.jpg', vis)
    face_model = FaceModelBatchTRT(detector_post_process_type = 'SINGLE')
    for i in range(100):
        image1 = cv2.imread('test_images/lumia.jpg')
        image2 = cv2.imread('test_images/TH.png')
        image3 = cv2.imread('test_images/TH1.jpg')
        batch_image = [image1, image2, image3, image1, image2, image3]
        
        list_bboxes, list_points = face_model.get_inputs_batch(batch_image)
        t0 = time.time()
        for i in range(len(list_bboxes)):
            bboxes = list_bboxes[i]
            points = list_points[i]
            print('image {} detected {} faces'.format(i, len(bboxes)))
            img_faces = face_model.get_face_align(batch_image[i], bboxes, points)
            # for ix, face in enumerate(img_faces):
            #     bgr = cv2.cvtColor(face, cv2.COLOR_RGB2BGR)
            #     cv2.imwrite('sample/{}_face_{}.jpg'.format(i, ix), bgr)
            embedd = face_model.get_features(img_faces)
        t1 = time.time()
        print('All cost: {}'.format(t1 - t0))
