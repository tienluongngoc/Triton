import os
import cv2
import argparse
import time
import numpy as np
from utils.image import ImageData
from detectors.retinaface import *
import onnxruntime
from exec_backends.trt_loader import TrtModel
from shapely.geometry import Point
from shapely.geometry.polygon import Polygon

def softmax(z):
    assert len(z.shape) == 2
    s = np.max(z, axis=1)
    s = s[:, np.newaxis] # necessary step to do broadcasting
    e_x = np.exp(z - s)
    div = np.sum(e_x, axis=1)
    div = div[:, np.newaxis] # dito
    return e_x / div

def distance2bbox(points, distance, max_shape=None):
    """Decode distance prediction to bounding box.

    Args:
        points (Tensor): Shape (n, 2), [x, y].
        distance (Tensor): Distance from the given point to 4
            boundaries (left, top, right, bottom).
        max_shape (tuple): Shape of the image.

    Returns:
        Tensor: Decoded bboxes.
    """
    x1 = points[:, 0] - distance[:, 0]
    y1 = points[:, 1] - distance[:, 1]
    x2 = points[:, 0] + distance[:, 2]
    y2 = points[:, 1] + distance[:, 3]
    if max_shape is not None:
        x1 = x1.clamp(min=0, max=max_shape[1])
        y1 = y1.clamp(min=0, max=max_shape[0])
        x2 = x2.clamp(min=0, max=max_shape[1])
        y2 = y2.clamp(min=0, max=max_shape[0])
    return np.stack([x1, y1, x2, y2], axis=-1)

def distance2kps(points, distance, max_shape=None):
    """Decode distance prediction to bounding box.

    Args:
        points (Tensor): Shape (n, 2), [x, y].
        distance (Tensor): Distance from the given point to 4
            boundaries (left, top, right, bottom).
        max_shape (tuple): Shape of the image.

    Returns:
        Tensor: Decoded bboxes.
    """
    preds = []
    for i in range(0, distance.shape[1], 2):
        px = points[:, i%2] + distance[:, i]
        py = points[:, i%2+1] + distance[:, i+1]
        if max_shape is not None:
            px = px.clamp(min=0, max=max_shape[1])
            py = py.clamp(min=0, max=max_shape[0])
        preds.append(px)
        preds.append(py)
    return np.stack(preds, axis=-1)

def filter_by_polygons(list_det, list_landmarks, polygons):
    '''
        Filter face by polygons
    '''
    list_new_det = []
    list_new_landmarks = []
    for ib in range(len(list_det)): # Foreach image
        if len(list_det[ib]) > 0:
            new_det       = []
            new_landmarks = []
            checked = [True]*len(list_det[ib])
            for polygon in polygons: # Foreach polygon
                polygon_shapely = Polygon(polygon)
                for i in range(len(list_det[ib])): # Foreach face
                    if checked[i]:
                        print(list_det[ib][i])
                        x1, y1, x2, y2 = list_det[ib][i][:4]
                        point = Point((x1+x2)/2, (y1+y2)/2)
                        if polygon_shapely.contains(point):
                            new_det.append(list_det[ib][i])
                            new_landmarks.append(list_landmarks[ib][i])
                            print(polygon, 'contain', point)
                            checked[i] = False
                        else:
                            print(polygon, 'not contain', point)
            list_new_det.append(np.array(new_det))
            list_new_landmarks.append(np.array(new_landmarks))
        else:
            list_new_det.append([])
    return list_new_det, list_new_landmarks

class RetinafaceBatchTRT(object):
    def __init__(self, model_path='weights/retinaface_r50_v1-1x3x640x640.trt', input_shape = (640, 640), batch_size = 1, post_process_type = 'SINGLE', engine = 'TRT'):
        '''
            TensorRT-Retinaface with batch-inference
        '''
        print('[INFO] Create Retinaface with {} engine'.format(engine))
        self.engine = engine
        if self.engine == 'TRT':
            self.model = TrtModel(model_path)
        elif self.engine == 'ONNX':
            self.model = onnxruntime.InferenceSession(model_path)
        else:
            raise NotImplementedError("Current support only TRT and ONNX engine")
        self.input_shape = input_shape
        self.rac = 'net3'
        self.masks = False
        self.batch_size = batch_size
        self.post_process_type = post_process_type
        self.indices = None #[4, 0, 1, 5, 2, 3, 6, 7, 8]
        for size in self.input_shape:
            if size % 32 != 0:
                raise ValueError("Current support only size which is multiple of 32 for compabilities")
        
        self.prepare()

    def prepare(self, nms: float = 0.4, **kwargs):
        self.nms_threshold = nms
        self.landmark_std = 1.0
        _ratio = (1.,)
        fmc = 3
        if self.rac == 'net3':
            _ratio = (1.,)
        elif self.rac == 'net3l':
            _ratio = (1.,)
            self.landmark_std = 0.2
        else:
            assert False, 'rac setting error %s' % self.rac

        if fmc == 3:
            self._feat_stride_fpn = [32, 16, 8]
            self.anchor_cfg = {
                '32': {'SCALES': (32, 16), 'BASE_SIZE': 16, 'RATIOS': _ratio, 'ALLOWED_BORDER': 9999},
                '16': {'SCALES': (8, 4), 'BASE_SIZE': 16, 'RATIOS': _ratio, 'ALLOWED_BORDER': 9999},
                '8': {'SCALES': (2, 1), 'BASE_SIZE': 16, 'RATIOS': _ratio, 'ALLOWED_BORDER': 9999},
            }
        self.use_landmarks = True
        self.fpn_keys = []

        for s in self._feat_stride_fpn:
            self.fpn_keys.append('stride%s' % s)

        self._anchors_fpn = dict(zip(self.fpn_keys, generate_anchors_fpn(cfg=self.anchor_cfg)))
        for k in self._anchors_fpn:
            v = self._anchors_fpn[k].astype(np.float32)
            self._anchors_fpn[k] = v
        
        self._num_anchors = dict(zip(self.fpn_keys, [anchors.shape[0] for anchors in self._anchors_fpn.values()]))
        # Create anchor
        # print('Rebuild anchor')
        self.anchor_plane_cache = {}
        for _idx, s in enumerate(self._feat_stride_fpn):
            stride = int(s)
            width = int(self.input_shape[0]/stride)
            height = int(self.input_shape[1]/stride)
            K = width * height
            A = self._num_anchors['stride%s' % s]
            key = (height, width, stride)
            anchors_fpn = self._anchors_fpn['stride%s' % s]
            anchors = anchors_plane(height, width, stride, anchors_fpn)
            self.anchor_plane_cache[key] = np.tile(anchors.reshape((K * A, 4)), (self.batch_size, 1, 1))
        if self.engine == 'TRT':
            self.warm_up()

    def warm_up(self):
        '''
            Warm up NMS jit
        '''
        print('Warming up NMS jit...')
        tik = time.time()
        image = cv2.imread('test_images/lumia.jpg', cv2.IMREAD_COLOR)
        _ = self.detect([image], threshold=0.1)
        tok = time.time()
        print('Warming up complete, time cost = {}'.format(tok - tik))

    def detect(self, list_bgr_img, threshold: float = 0.6, log = False, polygons = None):
        # Preprocess
        list_image_data = self.preprocess_batch(list_bgr_img)
        batch_data = np.array([cv2.cvtColor(img.transformed_image, cv2.COLOR_BGR2RGB) for img in list_image_data], dtype = np.float32)
        # Allocate
        n_img       = len(batch_data)
        if n_img % self.batch_size == 0:
            to_pad = 0
        else:
            to_pad = ((n_img//self.batch_size)+1)*self.batch_size - n_img
        # print('To pad: {}'.format(to_pad))
        n_batch = (n_img + to_pad)//self.batch_size
        paded_batch = np.zeros((self.batch_size * n_batch, 3, self.input_shape[1], self.input_shape[0]), dtype = np.float32)
        # Preprocess
        aligned             = np.transpose(batch_data, (0, 3, 1, 2))
        paded_batch[:n_img] = aligned
        # print('will infer for {} batches'.format(n_batch))
        list_det = []
        list_landmarks = []
        for i in range(n_batch):
            lower = i*self.batch_size
            higher = (i+1)*self.batch_size
            rs = self.detect_single_batch(paded_batch[lower:higher], threshold=threshold)
            list_det.extend(rs[0])
            list_landmarks.extend(rs[1])
        # Divive by scale factor
        list_det = list_det[:n_img]
        # assert len(list_det) == len(list_image_data), \
        #     "Number of output not equal number of input: {} vs {}".format(len(list_det), len(list_image_data))
        for i in range(len(list_image_data)):
            list_det[i]       =  list_det[i][:, :4]/list_image_data[i].scale_factor
            list_landmarks[i] /= list_image_data[i].scale_factor
        if polygons is not None:
            return filter_by_polygons(list_det, list_landmarks, polygons)
        return list_det, list_landmarks


    def detect_single_batch(self, batch_img, threshold: float = 0.6):
        batch_size = len(batch_img)
        assert batch_size == self.batch_size, "Model define with batch_size = {}, your input: {}".format(self.batch_size, batch_size) 
        tik = time.time()
        if self.engine == 'TRT':
            net_out = self.model.run(batch_img)
        else:
            ort_inputs = {self.model.get_inputs()[0].name: batch_img}
            net_out = self.model.run(None, ort_inputs)
        tok = time.time()
        # Sort cause output model while convert to TensorRT is shuffled
        if self.indices is None:
            if self.input_shape[0] == 640:
                target_shapes = [(1, 4, 20, 20), (1, 8, 20, 20), (1, 20, 20, 20), (1, 4, 40, 40), (1, 8, 40, 40), (1, 20, 40, 40), (1, 4, 80, 80), (1, 8, 80, 80), (1, 20, 80, 80)]
            # elif self.input_shape[0] == 896:
            #     target_shapes = [(1, 4, 20, 20), (1, 8, 20, 20), (1, 20, 20, 20), (1, 4, 40, 40), (1, 8, 40, 40), (1, 20, 40, 40), (1, 4, 80, 80), (1, 8, 80, 80), (1, 20, 80, 80)]
            else:
                raise NotImplementedError("Do it yourself")
            shapes = [net_out[i].shape for i in range(len(net_out))]
            self.indices = []
            for ix, ts in enumerate(target_shapes):
                for iy, shape in enumerate(shapes):
                    if ts == shape:
                        self.indices.append(iy)
            print('Recalculate indices: ', self.indices)
        sorted_net_out = [net_out[i] for i in self.indices]
        if self.post_process_type == 'BATCH':
            list_det, list_landmarks = self.postprocess_batch(sorted_net_out, threshold = threshold, batch_size = batch_size)
            return list_det, list_landmarks
        else:
            list_det = []
            list_landmarks = []
            for i in range(batch_size):
                single_net_out = [np.expand_dims(layer_out[i], 0) for layer_out in sorted_net_out]
                det, landmarks = self.postprocess(single_net_out, threshold = threshold)
                list_det.append(det)
                list_landmarks.append(landmarks)
            return list_det, list_landmarks

    def preprocess_batch(self, list_bgr_img):
        '''
            Preprocess image for batch-inference
        '''
        list_image_data = []
        for ix, bgr_image in enumerate(list_bgr_img):
            # print(bgr_image.shape)
            image = ImageData(bgr_image, self.input_shape)
            image.resize_image(mode='pad')     
            list_image_data.append(image)
        return list_image_data

    def postprocess_batch(self, net_out, threshold, batch_size):
        '''
            Post process for batch-inference
        '''
        proposals_list_batch = {i : [] for i in range(batch_size)}
        scores_list_batch = {i : [] for i in range(batch_size)}
        landmarks_list_batch = {i : [] for i in range(batch_size)}
        # t0 = time.time()
        # Foreach FPN layer
        for _idx, s in enumerate(self._feat_stride_fpn):
            _key = 'stride%s' % s
            stride = int(s)
            if self.use_landmarks:
                idx = _idx * 3
            else:
                idx = _idx * 2
            if self.masks:
                idx = _idx * 4

            A = self._num_anchors['stride%s' % s]
            
            scores_batch = net_out[idx]
            scores_batch = scores_batch[:, A:, :, :]
            idx += 1
            bbox_deltas_batch = net_out[idx]
            height, width = bbox_deltas_batch.shape[2], bbox_deltas_batch.shape[3]

            # K = height * width
            key = (height, width, stride)
            anchors_batch = self.anchor_plane_cache[key]

            scores_batch = clip_pad(scores_batch, (height, width))
            scores_batch = scores_batch.transpose((0, 2, 3, 1)).reshape((batch_size, -1, 1))

            bbox_deltas_batch = clip_pad(bbox_deltas_batch, (height, width))
            bbox_deltas_batch = bbox_deltas_batch.transpose((0, 2, 3, 1))
            bbox_pred_len = bbox_deltas_batch.shape[3] // A
            bbox_deltas_batch = bbox_deltas_batch.reshape((batch_size, -1, bbox_pred_len))
            proposals_batch = bbox_pred_batch(anchors_batch, bbox_deltas_batch)
            
            
            # Get proposal
            scores_batch = scores_batch.reshape((batch_size, -1))
            order_batch = np.argwhere(scores_batch >= threshold)

            # Get landmark
            if self.use_landmarks:
                idx += 1
                landmark_deltas_batch = net_out[idx]
                landmark_deltas_batch = clip_pad(landmark_deltas_batch, (height, width))
                landmark_pred_len = landmark_deltas_batch.shape[1] // A
                landmark_deltas_batch = landmark_deltas_batch.transpose((0, 2, 3, 1)).reshape((batch_size, -1, 5, landmark_pred_len // 5))
                landmark_deltas_batch *= self.landmark_std
                landmarks = landmark_pred_batch(anchors_batch, landmark_deltas_batch)

            # Foreach image
            for ib in range(batch_size): 
                order = [id[1] for id in order_batch if id[0] == ib]
                proposals_list_batch[ib].append(proposals_batch[ib, order])
                scores_list_batch[ib].append(scores_batch[ib, order].reshape((-1, 1)))
                if self.use_landmarks:
                    landmarks_list_batch[ib].append(landmarks[ib, order])
                
        # Foreach image
        list_det = []
        list_landmarks = []
        for ib in range(batch_size):
            proposals_list = proposals_list_batch[ib]
            scores_list = scores_list_batch[ib]
            landmarks_list = landmarks_list_batch[ib]

            proposals = np.vstack(proposals_list)
            landmarks = None
            if proposals.shape[0] == 0:
                if self.use_landmarks:
                    landmarks = np.zeros((0, 5, 2))
                list_det.append(np.zeros((0, 5)))
                list_landmarks.append(landmarks)
                continue

            scores = np.vstack(scores_list)
            scores_ravel = scores.ravel()
            order = scores_ravel.argsort()[::-1]
            proposals = proposals[order, :]
            scores = scores[order]

            if self.use_landmarks:
                landmarks = np.vstack(landmarks_list)
                landmarks = landmarks[order].astype(np.float32, copy=False)

            pre_det = np.hstack((proposals[:, 0:4], scores)).astype(np.float32, copy=False)
            keep = nms(pre_det, thresh=self.nms_threshold)
            det = np.hstack((pre_det, proposals[:, 4:]))
            det = det[keep, :]
            if self.use_landmarks:
                landmarks = landmarks[keep]
            # t1 = time.time()
            list_det.append(det)
            list_landmarks.append(landmarks)
        return list_det, list_landmarks
    
    def postprocess(self, net_out, threshold):
        proposals_list = []
        scores_list = []
        mask_scores_list = []
        landmarks_list = []
        t0 = time.time()
        for _idx, s in enumerate(self._feat_stride_fpn):
            _key = 'stride%s' % s
            stride = int(s)
            if self.use_landmarks:
                idx = _idx * 3
            else:
                idx = _idx * 2
            if self.masks:
                idx = _idx * 4

            A = self._num_anchors['stride%s' % s]

            scores = net_out[idx]
            # print(scores.shape, idx)
            scores = scores[:, A:, :, :]
            idx += 1
            bbox_deltas = net_out[idx]
            height, width = bbox_deltas.shape[2], bbox_deltas.shape[3]

            # K = height * width
            key = (height, width, stride)
            anchors = self.anchor_plane_cache[key][0]
            scores = clip_pad(scores, (height, width))
            scores = scores.transpose((0, 2, 3, 1)).reshape((-1, 1))

            bbox_deltas = clip_pad(bbox_deltas, (height, width))
            bbox_deltas = bbox_deltas.transpose((0, 2, 3, 1))
            bbox_pred_len = bbox_deltas.shape[3] // A
            bbox_deltas = bbox_deltas.reshape((-1, bbox_pred_len))

            proposals = bbox_pred(anchors, bbox_deltas)

            scores_ravel = scores.ravel()
            order = np.where(scores_ravel >= threshold)[0]
            proposals = proposals[order, :]
            scores = scores[order]

            proposals_list.append(proposals)
            scores_list.append(scores)

            if self.use_landmarks:
                idx += 1
                landmark_deltas = net_out[idx]
                landmark_deltas = clip_pad(landmark_deltas, (height, width))
                landmark_pred_len = landmark_deltas.shape[1] // A
                landmark_deltas = landmark_deltas.transpose((0, 2, 3, 1)).reshape((-1, 5, landmark_pred_len // 5))
                landmark_deltas *= self.landmark_std
                landmarks = landmark_pred(anchors, landmark_deltas)
                landmarks = landmarks[order, :]
                landmarks_list.append(landmarks)

        proposals = np.vstack(proposals_list)
        landmarks = None
        if proposals.shape[0] == 0:
            if self.use_landmarks:
                landmarks = np.zeros((0, 5, 2))
            return np.zeros((0, 5)), landmarks

        scores = np.vstack(scores_list)

        scores_ravel = scores.ravel()
        order = scores_ravel.argsort()[::-1]
        proposals = proposals[order, :]
        scores = scores[order]

        if self.use_landmarks:
            landmarks = np.vstack(landmarks_list)
            landmarks = landmarks[order].astype(np.float32, copy=False)
        if self.masks:
            mask_scores = np.vstack(mask_scores_list)
            mask_scores = mask_scores[order]
            pre_det = np.hstack((proposals[:, 0:4], scores, mask_scores)).astype(np.float32, copy=False)
        else:
            pre_det = np.hstack((proposals[:, 0:4], scores)).astype(np.float32, copy=False)
        keep = nms(pre_det, thresh=self.nms_threshold)
        det = np.hstack((pre_det, proposals[:, 4:]))
        det = det[keep, :]
        if self.use_landmarks:
            landmarks = landmarks[keep]
        t1 = time.time()
        return det, landmarks

class SCRFDBatchTRT:
    def __init__(self,
                model_path='weights/scrfd-896x896-batchsize-1-fp32.trt',
                input_shape = (896, 896),
                batch_size = 1, # Current support only batchsize = 1
                engine = 'TRT'):
        self.taskname = 'detection'
        self.batched = False
        self.engine = engine
        self.input_size = input_shape
        if self.engine == 'TRT':
            self.model = TrtModel(model_path)
        elif self.engine == 'ONNX':
            self.model = onnxruntime.InferenceSession(model_path)
        else:
            raise NotImplementedError("Current support only TRT and ONNX engine")
        self.center_cache = {}
        self.nms_thresh = 0.4
        self.fmc = 3
        self._feat_stride_fpn = [8, 16, 32]
        self._num_anchors = 2
        self.use_kps = True
        self.indices = None
        
    def prepare(self, ctx_id, **kwargs):
        if ctx_id<0:
            self.model.set_providers(['CPUExecutionProvider'])
        nms_thresh = kwargs.get('nms_thresh', None)
        if nms_thresh is not None:
            self.nms_thresh = nms_thresh
        input_size = kwargs.get('input_size', None)
        if input_size is not None:
            if self.input_size is not None:
                print('warning: det_size is already set in scrfd model, ignore')
            else:
                self.input_size = input_size

    def forward(self, batch_img, thresh):
        scores_list = []
        bboxes_list = []
        kpss_list = []

        if self.engine == 'TRT':
            net_outs = self.model.run(batch_img)
        else:
            ort_inputs = {self.model.get_inputs()[0].name: batch_img}
            net_outs = self.model.run(None, ort_inputs)
            print([out.shape for out in net_outs])
        if self.indices is None:
            if self.input_size[0] == 640:
                target_shapes = [(1, 12800, 1), (1, 3200, 1), (1, 800, 1), (1, 12800, 4), (1, 3200, 4), (1, 800, 4), (1, 12800, 10), (1, 3200, 10), (1, 800, 10)]
            elif self.input_size[0] == 896:
                target_shapes = [(1, 25088, 1), (1, 6272, 1), (1, 1568, 1), (1, 25088, 4), (1, 6272, 4), (1, 1568, 4), (1, 25088, 10), (1, 6272, 10), (1, 1568, 10)]
            else:
                raise NotImplementedError("Not support image size")
            shapes = [net_outs[i].shape for i in range(len(net_outs))]
            print('Shapes:', shapes)
            print('Target shapes: ', target_shapes)
            self.indices = []
            for ix, ts in enumerate(target_shapes):
                for iy, shape in enumerate(shapes):
                    if ts == shape:
                        self.indices.append(iy)
            print('Recalculate indices: ', self.indices)
        sorted_net_out = [net_outs[i] for i in self.indices]
        input_height = batch_img.shape[2]
        input_width = batch_img.shape[3]
        fmc = self.fmc
        lst_scores_list = []
        lst_bboxes_list = []
        lst_kpss_list = []
        for ix in range(len(batch_img)):
            scores_list = []
            bboxes_list = []
            kpss_list   = []
            for idx, stride in enumerate(self._feat_stride_fpn):
                # If model support batch dim, take first output

                scores = sorted_net_out[idx][0]
                bbox_preds = sorted_net_out[idx + fmc][0]
                bbox_preds = bbox_preds * stride
                if self.use_kps:
                    kps_preds = sorted_net_out[idx + fmc * 2][0] * stride

                height = input_height // stride
                width = input_width // stride
                K = height * width
                key = (height, width, stride)
                if key in self.center_cache:
                    anchor_centers = self.center_cache[key]
                else:
                    anchor_centers = np.stack(np.mgrid[:height, :width][::-1], axis=-1).astype(np.float32)
                    anchor_centers = (anchor_centers * stride).reshape( (-1, 2) )
                    if self._num_anchors>1:
                        anchor_centers = np.stack([anchor_centers]*self._num_anchors, axis=1).reshape( (-1,2) )
                    if len(self.center_cache)<100:
                        self.center_cache[key] = anchor_centers

                pos_inds = np.where(scores>=thresh)[0]
                bboxes = distance2bbox(anchor_centers, bbox_preds)
                pos_scores = scores[pos_inds]
                pos_bboxes = bboxes[pos_inds]
                scores_list.append(pos_scores)
                bboxes_list.append(pos_bboxes)
                if self.use_kps:
                    kpss = distance2kps(anchor_centers, kps_preds)
                    #kpss = kps_preds
                    kpss = kpss.reshape( (kpss.shape[0], -1, 2) )
                    pos_kpss = kpss[pos_inds]
                    kpss_list.append(pos_kpss)
            lst_scores_list.append(scores_list)
            lst_bboxes_list.append(bboxes_list)
            lst_kpss_list.append(kpss_list)
        return lst_scores_list, lst_bboxes_list, lst_kpss_list

    def preprocess_batch(self, list_bgr_img, input_size):
        '''
            Preprocess batch of image
        '''
        batch_size = len(list_bgr_img)
        batch_image = np.zeros((batch_size, input_size[1], input_size[0], 3), dtype=np.uint8)
        batch_scale = []
        for ix, img in enumerate(list_bgr_img):
            im_ratio = float(img.shape[0]) / img.shape[1]
            model_ratio = float(input_size[1]) / input_size[0]
            if im_ratio>model_ratio:
                new_height = input_size[1]
                new_width = int(new_height / im_ratio)
            else:
                new_width = input_size[0]
                new_height = int(new_width * im_ratio)
            scale = float(new_height) / img.shape[0]
            resized_img = cv2.resize(img, (new_width, new_height))
            batch_image[ix, :new_height, :new_width, :] = cv2.cvtColor(resized_img, cv2.COLOR_BGR2RGB)
            batch_scale.append(scale)
        batch_image = (batch_image.astype('float32') - 127.5)/128.0 # Normalization
        batch_image = np.transpose(batch_image, (0, 3, 1, 2))
        return batch_image, batch_scale

    def detect(self, list_bgr_img, threshold=0.5, input_size = None, polygons = None):
        assert input_size is not None or self.input_size is not None
        input_size = self.input_size if input_size is None else input_size
            
        # Preprocess batch of image
        batch_image, batch_scale = self.preprocess_batch(list_bgr_img, input_size)
        # Infer
        lst_scores_list, lst_bboxes_list, lst_kpss_list = self.forward(batch_image, threshold)
        list_bboxes = []
        list_points = []
        for ix in range(len(batch_image)):
            scores_list = lst_scores_list[ix]
            bboxes_list = lst_bboxes_list[ix]
            kpss_list   = lst_kpss_list[ix]
            det_scale   = batch_scale[ix]
            scores = np.vstack(scores_list)
            scores_ravel = scores.ravel()
            order = scores_ravel.argsort()[::-1]
            bboxes = np.vstack(bboxes_list) / det_scale
            if self.use_kps:
                kpss = np.vstack(kpss_list) / det_scale
            pre_det = np.hstack((bboxes, scores)).astype(np.float32, copy=False)
            pre_det = pre_det[order, :]
            keep = self.nms(pre_det)
            det = pre_det[keep, :]
            if self.use_kps:
                kpss = kpss[order,:,:]
                kpss = kpss[keep,:,:]
            else:
                kpss = None
            list_bboxes.append(det)
            list_points.append(kpss)
        if polygons is not None:
            return filter_by_polygons(list_bboxes, list_points, polygons)
        return list_bboxes, list_points

    def nms(self, dets):
        thresh = self.nms_thresh
        x1 = dets[:, 0]
        y1 = dets[:, 1]
        x2 = dets[:, 2]
        y2 = dets[:, 3]
        scores = dets[:, 4]

        areas = (x2 - x1 + 1) * (y2 - y1 + 1)
        order = scores.argsort()[::-1]

        keep = []
        while order.size > 0:
            i = order[0]
            keep.append(i)
            xx1 = np.maximum(x1[i], x1[order[1:]])
            yy1 = np.maximum(y1[i], y1[order[1:]])
            xx2 = np.minimum(x2[i], x2[order[1:]])
            yy2 = np.minimum(y2[i], y2[order[1:]])

            w = np.maximum(0.0, xx2 - xx1 + 1)
            h = np.maximum(0.0, yy2 - yy1 + 1)
            inter = w * h
            ovr = inter / (areas[i] + areas[order[1:]] - inter)

            inds = np.where(ovr <= thresh)[0]
            order = order[inds + 1]

        return keep

if __name__ == '__main__':
    import time
    import glob
    model = RetinafaceBatchTRT(model_path='weights/retinaface_r50_v1.onnx',
                            input_shape = (640, 640),
                            batch_size = 1,
                            post_process_type = 'SINGLE',
                            engine = 'ONNX')
    # model = SCRFDBatchTRT()
    for path in glob.glob('test_images/retina_mask/*')[:10]:
        
        tik = time.time()
        img = cv2.imread(path)
        # list_bboxes, list_points = model.detect(img, input_size = (896, 896))
        list_bboxes, list_points = model.detect([img])
        list_bboxes = list_bboxes[0]
        list_points = list_points[0]
        for ix, det in enumerate(list_bboxes): 
           
            color = (0, 255, 0)
  
            pt1 = tuple(map(int, det[0:2]))
            pt2 = tuple(map(int, det[2:4]))
            cv2.rectangle(img, pt1, pt2, color, 2)
            for point in list_points[ix]:
                cv2.circle(img, (int(point[0]), int(point[1])), 2, (255, 0, 0), -1)
        cv2.imwrite(path.replace('retina_mask', 'retina_mask_result'), img)
        tok = time.time()
        print(path, tok-tik)