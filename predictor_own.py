import tensorflow as tf
from utils import bbox_utils, data_utils, drawing_utils, io_utils, train_utils, landmark_utils
import blazeface
import time
import numpy as np
import cv2
import matplotlib.pyplot as plt
import time
import glob
import os
from data_preparation_aic_orig_editted_face_test import ds

orig_vid=False

def coco_pck_amp(est_keys,true_keypoints):
    dist=1000
    torso_diam=np.linalg.norm(true_keypoints[-1,0:2] - true_keypoints[-2,0:2])
    est_key=est_keys[-2:,0:2]
    true_keypoint=true_keypoints[-2:,0:2]
    
    dist_all= np.array([ np.linalg.norm(true_keypoint[x,:] - est_key[x,:]) for x in range(est_key.shape[0])])
    
    
    return np.sum(dist_all<torso_diam/5)

args = io_utils.handle_args()
if args.handle_gpu:
    io_utils.handle_gpu_compatibility()
    

batch_size = 1
use_custom_images = False
custom_image_path = "data/images/"
hyper_params = train_utils.get_hyper_params()
img_size = hyper_params["img_size"]

data_types = data_utils.get_data_types()
data_shapes = data_utils.get_data_shapes()
padding_values = data_utils.get_padding_values()

# if use_custom_images:
#     img_paths = data_utils.get_custom_imgs(custom_image_path)
#     total_items = len(img_paths)
#     test_data = tf.data.Dataset.from_generator(lambda: data_utils.custom_data_generator(
#                                                img_paths, img_size, img_size), data_types, data_shapes)
# else:
#     test_split = "train[80%:]"
#     test_data, info = data_utils.get_dataset("the300w_lp", test_split)
#     total_items = data_utils.get_total_item_size(info, test_split)
#     test_data = test_data.map(lambda x: data_utils.preprocessing(x, img_size, img_size))
# #

test_data=ds
test_data = test_data.padded_batch(batch_size, padded_shapes=data_shapes, padding_values=padding_values)

files = glob.glob('test/pos/*.jpeg')
for f in files:
    os.remove(f)
    
files = glob.glob('test/neg/*.jpeg')
for f in files:
    os.remove(f)

model = blazeface.get_model(hyper_params)
model_path = io_utils.get_model_path()

# checkpoint_dir = 'training_checkpoints_aligned_batches_cont300/'
# checkpoint_prefix = os.path.join(checkpoint_dir, "ckpt_{epoch}")
    
# model.load_weights('D:/Downloads/tf-blazeface-master/training_checkpoints_aligned_batches_faces_only_mids/ckpt_2')


# model.load_weights('D:/Downloads/tf-blazeface-master/training_checkpoints_aligned_batches_cont300/ckpt_160')

# model.load_weights('D:/Downloads/tf-blazeface-master/training_checkpoints_aligned_test5/ckpt_115')
# model.load_weights('D:/Downloads/tf-blazeface-master/training_checkpoints_aligned_test4/ckpt_282')

# model.load_weights('D:/Downloads/tf-blazeface-master/training_checkpoints_aligned_batches_face_mids_test5_cont1001e4_rev/ckpt_131')


# model.load_weights('D:/Downloads/tf-blazeface-master/training_checkpoints_aligned_batches_test1_rev/ckpt_82')

# model.load_weights('D:/Downloads/tf-blazeface-master/training_checkpoints_aligned_batches_face_mids_test5_cont210_rev/ckpt_400')

model.load_weights('D:/Downloads/tf-blazeface-master/training_checkpoints_aligned_batches_face_mids_test5_rev1_rot/ckpt_400')



# model.load_weights('D:/Downloads/tf-blazeface-master/training_checkpoints_aligned_batches/100/ckpt_1')

# model.load_weights('D:/Downloads/tf-blazeface-master/training_checkpoints_aligned_batches/ckpt_70')


# model.load_weights('D:/Downloads/tf-blazeface-master/training_checkpoints_aligned_batches_face_midshipapart/ckpt_243')


# model.load_weights('D:/Downloads/tf-blazeface-master/training_checkpoints_aligned_batches_face_mids_ownpriors/ckpt_183')


# model.load_weights('D:/Downloads/tf-blazeface-master/training_checkpoints_aligned_batches_face_mids/ckpt_238')


# model.load_weights('D:/Downloads/tf-blazeface-master/training_checkpoints_aligned_batches_all3_face_ratios_123/ckpt_182')


# model.load_weights('D:/Downloads/tf-blazeface-master/training_checkpoints_aligned_batches_all3_face_prior_aic/ckpt_26')


# model.load_weights('D:/Downloads/tf-blazeface-master/training_checkpoints_aligned_batches_all3_face_prior_aic_nonorm/ckpt_36')


# model.load_weights('D:/Downloads/tf-blazeface-master/training_checkpoints_aligned_batches_faces_ratios_reverse/ckpt_99')



# model.load_weights('D:/Downloads/tf-blazeface-master/training_checkpoints_aligned_batches_face_only_cont105/ckpt_24')

# model.load_weights('/mnt/sda1/downloads/tf-blazeface-master/trained/blazeface80_epochs15_any139.h5')
# 
prior_boxes = bbox_utils.generate_prior_boxes(hyper_params["feature_map_shapes"], hyper_params["aspect_ratios"])

# prior_numpy1=np.load('prior_boxes.npy')
# prior_numpy=np.zeros((prior_numpy1.shape))
# prior_numpy[:,0]=(prior_numpy1[:,0]+prior_numpy1[:,2])/2
# prior_numpy[:,1]=(prior_numpy1[:,1]+prior_numpy1[:,3])/2
# prior_numpy[:,2]=prior_numpy1[:,2]-prior_numpy1[:,0]
# prior_numpy[:,3]=prior_numpy1[:,3]-prior_numpy1[:,1]
# prior_boxes=tf.convert_to_tensor(prior_numpy, dtype=tf.float32)
        


variances = hyper_params["variances"]
total_landmarks = hyper_params["total_landmarks"]
landmark_variances = total_landmarks * variances[0:2]
variances += landmark_variances


    

cc=0
co=0
pck_test=list([])
a=time.time()
for image_data in test_data:
    if cc<100:
        cc=cc+1;
        img, coord1, label1 = image_data
        pred_deltas, pred_scores = model.predict_on_batch(img)
        max(pred_scores[0])
        plt.plot(sorted(pred_scores[0]))
    
        pred_scores[0]=pred_scores[0]
        pred_deltas *= variances
        #
        pred_bboxes_and_landmarks = bbox_utils.get_bboxes_and_landmarks_from_deltas(prior_boxes, pred_deltas)
        pred_bboxes_and_landmarks = tf.clip_by_value(pred_bboxes_and_landmarks, 0, 1)
        #
        pred_scores = tf.cast(pred_scores, tf.float32)
        #
        weighted_suppressed_data = bbox_utils.weighted_suppression(pred_scores[0], pred_bboxes_and_landmarks[0])
        #
        weighted_bboxes = weighted_suppressed_data[..., 0:4]
        weighted_landmarks = weighted_suppressed_data[..., 4:]
        #
        if orig_vid:
            denormalized_bboxes = bbox_utils.denormalize_bboxes(weighted_bboxes, sizes_list[cc-1][0], sizes_list[cc-1][1])
            weighted_landmarks = tf.reshape(weighted_landmarks, (-1, total_landmarks, 2))
            denormalized_landmarks = landmark_utils.denormalize_landmarks(weighted_landmarks, sizes_list[cc-1][0], sizes_list[cc-1][1])
            
        else:
            denormalized_bboxes = bbox_utils.denormalize_bboxes(weighted_bboxes, img_size, img_size)
            weighted_landmarks = tf.reshape(weighted_landmarks, (-1, total_landmarks, 2))
            denormalized_landmarks = landmark_utils.denormalize_landmarks(weighted_landmarks, img_size, img_size)
        
        
    
    # b=time.time()
    # print(b-a)
        
        if sum(sum(sum(denormalized_landmarks.numpy())))>0:
            # cc += 1
    
            
            if orig_vid:
                # from utils import bbox_utils, data_utils, drawing_utils, io_utils, train_utils, landmark_utils
    
                # imgt=drawing_utils.draw_bboxes_with_landmarks_orig(img[0], denormalized_bboxes, denormalized_landmarks,sizes_list[cc-1],tf.image.convert_image_dtype(temps_orig[cc-1], tf.float32))
                # img= np.array(imgt)
                img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    
                cv2.imwrite('test/pos/'+'img'+str(cc).zfill(4)+'.jpeg',cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
            else:
                
                imgt=drawing_utils.draw_bboxes_with_landmarks(img[0], denormalized_bboxes, denormalized_landmarks)
                img = np.array(imgt)
                img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        
                skeleton = (128*label1).numpy().astype(int)
                rec= (128*coord1).numpy().astype(int)[0][0]
                # plt.imshow(img)
                for ii in range(6):
                    cv2.circle(img, center=tuple(skeleton[0][0][ii][0:2]), radius=1, color=(0, 255, 0), thickness=2)
                img = cv2.rectangle(img, (rec[1],rec[0]), (rec[3],rec[2]), (0, 255, 0) , 1)
                img = cv2.putText(np.array(img), str(cc),(10,10), cv2.FONT_HERSHEY_SIMPLEX , 0.4, (0,5,255), 1, cv2.LINE_AA) 
                # plt.imshow(img)
                    
                cv2.imwrite('test/pos/'+'img'+str(cc).zfill(4)+'.jpeg',(img))
            
            
            # time.sleep(5)
            
            # lands_np=denormalized_landmarks.numpy()
            # candidates=lands_np[:,0,0]
            # arg_candidates=np.argsort(candidates)
            # candidates[arg_candidates[0]]
            # stoppage=0
            # max_pck=list([])
            # for ind in range(candidates.shape[0]-1,0,-1):
            #     if sum(sum(lands_np[arg_candidates[ind],:,:]))>0:
            #         if stoppage==0:
            #             max_pck.append(coco_pck_amp(lands_np[arg_candidates[ind],:,:],all_labels[co,0,:,:]*128))
                        
            #     else:
            #         stoppage=1
            
                
            #     # plt.imshow(imgt)
            # pck_test.append(max(max_pck))
            # print(max(max_pck))
                
            # time.sleep(3.5)
        else:
            imgt=drawing_utils.draw_bboxes_with_landmarks(img[0], denormalized_bboxes, denormalized_landmarks)
            img = np.array(imgt)
            img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
            skeleton = (128*label1).numpy().astype(int)
            rec= (128*coord1).numpy().astype(int)[0][0]
            # plt.imshow(img)
            for ii in range(6):
                cv2.circle(img, center=tuple(skeleton[0][0][ii][0:2]), radius=1, color=(0, 0, 255), thickness=2)
            img = cv2.rectangle(np.array(img), (rec[1],rec[0]), (rec[3],rec[2]), (0, 0, 255) , 1)
            
            # plt.imshow(img)
            
            cv2.imwrite('test/neg/'+'img'+str(cc).zfill(4)+'.jpeg',(img))

    # co += 1
    
print(np.mean(pck_test)/2)
    