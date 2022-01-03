import argparse
import cv2
import sys
import numpy as np
import os
import datetime
from skimage import transform as trans
import glob
import time
import onnxruntime
from exec_backends.trt_loader import TrtModel

def transform(data, center, output_size, scale, rotation):
    scale_ratio = scale
    rot = float(rotation) * np.pi / 180.0
    t1 = trans.SimilarityTransform(scale=scale_ratio)
    cx = center[0] * scale_ratio
    cy = center[1] * scale_ratio
    t2 = trans.SimilarityTransform(translation=(-1 * cx, -1 * cy))
    t3 = trans.SimilarityTransform(rotation=rot)
    t4 = trans.SimilarityTransform(translation=(output_size / 2,
                                                output_size / 2))
    t = t1 + t2 + t3 + t4
    M = t.params[0:2]
    cropped = cv2.warpAffine(data,
                             M, (output_size, output_size),
                             borderValue=0.0)
    return cropped, M


def trans_points2d(pts, M):
    new_pts = np.zeros(shape=pts.shape, dtype=np.float32)
    for i in range(pts.shape[0]):
        pt = pts[i]
        new_pt = np.array([pt[0], pt[1], 1.], dtype=np.float32)
        new_pt = np.dot(M, new_pt)
        #print('new_pt', new_pt.shape, new_pt)
        new_pts[i] = new_pt[0:2]

    return new_pts


def trans_points3d(pts, M):
    scale = np.sqrt(M[0][0] * M[0][0] + M[0][1] * M[0][1])
    #print(scale)
    new_pts = np.zeros(shape=pts.shape, dtype=np.float32)
    for i in range(pts.shape[0]):
        pt = pts[i]
        new_pt = np.array([pt[0], pt[1], 1.], dtype=np.float32)
        new_pt = np.dot(M, new_pt)
        #print('new_pt', new_pt.shape, new_pt)
        new_pts[i][0:2] = new_pt[0:2]
        new_pts[i][2] = pts[i][2] * scale

    return new_pts


def trans_points(pts, M):
    if pts.shape[1] == 2:
        return trans_points2d(pts, M)
    else:
        return trans_points3d(pts, M)


class LandmarkModelBatchTRT:
    def __init__(self,
                model_path = 'weights/landmark_model.trt',
                im_size=192,
                max_batch_size = 4,
                engine = 'TRT'):
        print('[INFO] Load facial landmark model with {} engine'.format(engine))
        image_size = (im_size, im_size)
        self.image_size = image_size
        self.engine = engine
        if self.engine == 'TRT':
            self.model = TrtModel(model_path)
        elif self.engine == 'ONNX':
            self.model = onnxruntime.InferenceSession(model_path)
        else:
            raise NotImplementedError("Current support only TRT & ONNX engine")
        self.max_batch_size = max_batch_size
        # Variable for pre-processing
        bbox = np.array([0, 0, 112, 112])
        w, h = (bbox[2] - bbox[0]), (bbox[3] - bbox[1])
        self.center = (bbox[2] + bbox[0]) / 2, (bbox[3] + bbox[1]) / 2
        self._scale = self.image_size[0] * 2 / 3.0 / max(w, h)

    def predict(self, img):
        height, width, _ = img.shape
        assert width == 112 and height == 112, "Current support only width = height = 112"
        out = []
        infer_time = 0
        input_blob = np.zeros((1, 3) + self.image_size, dtype=np.float32)
        rimg, M = transform(img, self.center, self.image_size[0], self._scale, 0)
        # rimg = cv2.cvtColor(rimg, cv2.COLOR_BGR2RGB)
        rimg = np.transpose(rimg, (2, 0, 1))  #3*112*112, RGB
        input_blob[0] = rimg
        t1 = time.time()
        # print(np.amin(input_blob), np.amax(input_blob), np.mean(input_blob))
        pred = self.model.run(input_blob)[0][0]
        t2 = time.time()
        # print(pred, pred.shape)
        if pred.shape[0] >= 3000:
            pred = pred.reshape((-1, 3))
        else:
            pred = pred.reshape((-1, 2))
        pred[:, 0:2] += 1
        pred[:, 0:2] *= (self.image_size[0] // 2)
        if pred.shape[1] == 3:
            pred[:, 2] *= (self.image_size[0] // 2)
        IM = cv2.invertAffineTransform(M)
        pred = trans_points(pred, IM)
        out.append(pred)
        # Get area to remove noise
        out = np.squeeze(np.array(out))
        tlx = np.amin(out[:, 0])
        tly = np.amin(out[:, 1])
        brx = np.amax(out[:, 0])
        bry = np.amax(out[:, 1])
        infer_time += t2 - t1
        return out, (brx-tlx)*(bry-tly)/112/112, infer_time
    
    def predict_batch(self, batch_img):
        '''
            Input: batch of RGB image
        '''
        height, width, _ = batch_img[0].shape
        assert width == 112 and height == 112, "Current support only width = height = 112"
        pred_landmark = []
        pred_area = []     
        n_img, height, width, _ = batch_img.shape
        # Pre-processing
        input_blob = np.zeros((n_img, 3) + self.image_size, dtype=np.float32)
        for i in range(n_img):
            rimg, M = transform(batch_img[i], self.center, self.image_size[0], self._scale, 0)
            # rimg = cv2.cvtColor(rimg, cv2.COLOR_BGR2RGB)
            input_blob[i] = np.transpose(rimg, (2, 0, 1))  # 3*112*112, RGB
        # Inference
        # t1 = time.time()
        preds = np.zeros((n_img, 212))
        total_batch = n_img//self.max_batch_size if n_img % self.max_batch_size == 0 else n_img//self.max_batch_size + 1
        for i in range(total_batch):
            lower = i*self.max_batch_size
            higher = min((i+1)*self.max_batch_size, n_img)
            preds[lower:higher] = self.model.run(input_blob[lower: higher])[0]
        # t2 = time.time()
        # print(t2 - t1)
        # Post-processing
        preds = preds.reshape((n_img, -1, 2))
        preds[:, :, 0:2] += 1
        preds[:, :, 0:2] *= (self.image_size[0] // 2)
        if preds.shape[2] == 3:
            preds[:, :, 2] *= (self.image_size[0] // 2)
        IM = cv2.invertAffineTransform(M)
        for i in range(n_img):
            pred = trans_points(preds[i], IM)
            pred_landmark.append(pred)
            # Get area to remove noise
            out = np.squeeze(np.array(pred))
            tlx = np.amin(out[:, 0])
            tly = np.amin(out[:, 1])
            brx = np.amax(out[:, 0])
            bry = np.amax(out[:, 1])
            pred_area.append((brx-tlx)*(bry-tly)/112/112)
        return pred_landmark, pred_area

    def get_cluster_quality(self, batch_img):
        '''
            Calculate mean landmark area (also called cluster quality)
        '''
        pred_landmark, pred_area = self.predict_batch(batch_img)
        return np.mean(np.array(pred_area)), pred_area

if __name__ == '__main__':
    import time
    handler = LandmarkModelBatchTRT('weights/landmark_model.engine')
    all_ids = glob.glob('image_aligned/*')
    path_out = './'
    if not os.path.exists(path_out):
        os.mkdir(path_out)
    total_ids = len(all_ids)
    # Single inference
    # tik = time.time()
    # infer_times = 0
    # for i in range(1):
    #     im_path = all_ids[i]
    #     # print(im_path)
    #     img     = cv2.imread(im_path)
    #     out_array, _, infer_time = handler.predict(img)
    #     infer_times += infer_time
    #     # print(out_array)
    #     for p in out_array:
    #         cv2.circle(img, p.astype('int32'), 1, (0, 0, 255), 1)
    #     filename = im_path.split('/')[-1]
    #     cv2.imwrite(os.path.join(path_out, filename), img)
    # tok = time.time() 
    # fps = total_ids/(tok-tik)
    # print(fps, tok - tik, infer_times, infer_times/(tok-tik))

    # Batch inference
    infer_times = 0
    lst_img = []
    for i in range(total_ids):
        im_path = all_ids[i]
        lst_img.append(cv2.imread(im_path))
    tik = time.time()

    out_landmark, out_landmark_area = handler.predict_batch(np.array(lst_img))
    for i in range(len(out_landmark)):
        img = lst_img[i]
        im_path = all_ids[i]
        landmark_area = out_landmark_area[i]
        for p in out_landmark[i]:
            cv2.circle(img, p.astype('int32'), 1, (0, 0, 255), 1)
        filename = im_path.split('/')[-1].split('.')[0]
        cv2.imwrite(os.path.join(path_out, filename + '_' + str(landmark_area) + '.jpg'), img)
    tok = time.time() 
    fps = total_ids/(tok-tik)
    print(fps, tok - tik)