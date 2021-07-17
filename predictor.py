import tensorflow as tf
from utils import bbox_utils, data_utils, drawing_utils, io_utils, train_utils, landmark_utils
import blazeface

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

if use_custom_images:
    img_paths = data_utils.get_custom_imgs(custom_image_path)
    total_items = len(img_paths)
    test_data = tf.data.Dataset.from_generator(lambda: data_utils.custom_data_generator(
                                               img_paths, img_size, img_size), data_types, data_shapes)
else:
    test_split = "train[80%:]"
    test_data, info = data_utils.get_dataset("the300w_lp", test_split)
    total_items = data_utils.get_total_item_size(info, test_split)
    test_data = test_data.map(lambda x: data_utils.preprocessing(x, img_size, img_size))
    
# train_split = "train[:80%]"
# val_split = "train[80%:]"
# train_data, info = data_utils.get_dataset("the300w_lp", train_split)
# val_data, _ = data_utils.get_dataset("the300w_lp", val_split)
# train_total_items = data_utils.get_total_item_size(info, train_split)
# val_total_items = data_utils.get_total_item_size(info, val_split)
# #
# img_size = hyper_params["img_size"]

# train_data = train_data.map(lambda x : data_utils.preprocessing(x, img_size, img_size, augmentation.apply))
# val_data = val_data.map(lambda x : data_utils.preprocessing(x, img_size, img_size))

#

test_data=ds_val
test_data = test_data.padded_batch(batch_size, padded_shapes=data_shapes, padding_values=padding_values)



model = blazeface.get_model(hyper_params)
model_path = io_utils.get_model_path()
model.load_weights('D:/Downloads/tf-blazeface-master/trained/blazeface_model_weights_85.h5')

# model.load_weights('C:/Users/altius/Downloads/blazeface80_epochs15_any139.h5')

prior_boxes = bbox_utils.generate_prior_boxes(hyper_params["feature_map_shapes"], hyper_params["aspect_ratios"])

variances = hyper_params["variances"]
total_landmarks = hyper_params["total_landmarks"]
landmark_variances = total_landmarks * variances[0:2]
variances += landmark_variances


for image_data in test_data:
    img, lands, coords = image_data
    print(img.shape)
    pass
    
    # ind=0
    # pred_deltas, pred_scores = model.predict_on_batch(img)
    # pred_deltas *= variances
    # #
    # pred_bboxes_and_landmarks = bbox_utils.get_bboxes_and_landmarks_from_deltas(prior_boxes, pred_deltas)
    # pred_bboxes_and_landmarks = tf.clip_by_value(pred_bboxes_and_landmarks, 0, 1)
    # #
    # pred_scores = tf.cast(pred_scores, tf.float32)
    # #
    # weighted_suppressed_data = bbox_utils.weighted_suppression(pred_scores[ind], pred_bboxes_and_landmarks[ind])
    # #
    # weighted_bboxes = weighted_suppressed_data[..., 0:4]
    # weighted_landmarks = weighted_suppressed_data[..., 4:]
    # #
    # denormalized_bboxes = bbox_utils.denormalize_bboxes(weighted_bboxes, img_size, img_size)
    # weighted_landmarks = tf.reshape(weighted_landmarks, (-1, total_landmarks, 2))
    # denormalized_landmarks = landmark_utils.denormalize_landmarks(weighted_landmarks, img_size, img_size)
    # drawing_utils.draw_bboxes_with_landmarks(img[ind], denormalized_bboxes, denormalized_landmarks)

    ind=0
    pred_deltas, pred_scores = model.predict_on_batch(img)
    pred_deltas *= variances
    #
    pred_bboxes_and_landmarks = bbox_utils.get_bboxes_and_landmarks_from_deltas(prior_boxes, pred_deltas)
    pred_bboxes_and_landmarks = tf.clip_by_value(pred_bboxes_and_landmarks, 0, 1)
    #
    pred_scores = tf.cast(pred_scores, tf.float32)
    #
    weighted_suppressed_data = bbox_utils.weighted_suppression(pred_scores[ind]*10, pred_bboxes_and_landmarks[ind])
    #
    weighted_bboxes = weighted_suppressed_data[..., 0:4]
    weighted_landmarks = weighted_suppressed_data[..., 4:]
    #
    denormalized_bboxes = bbox_utils.denormalize_bboxes(weighted_bboxes, img_size, img_size)
    weighted_landmarks = tf.reshape(weighted_landmarks, (-1, total_landmarks, 2))
    denormalized_landmarks = landmark_utils.denormalize_landmarks(weighted_landmarks, img_size, img_size)
    drawing_utils.draw_bboxes_with_landmarks(img[ind], denormalized_bboxes, denormalized_landmarks)
    
# for item in weighted_landmarks:
#     print(item)