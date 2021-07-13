import os
import tensorflow as tf
from pynvml import *
import tensorflow.keras as keras
import numpy as np
import cv2
from utils import bbox_utils, data_utils, io_utils, train_utils, drawing_utils, landmark_utils
hyper_params = train_utils.get_hyper_params()

# Load coordinates of keypoints
all_labels=np.load('all_labels_faces.npy')

# load coordinates of the box
all_coord=np.load('all_coord_faces.npy')

# load file names
X_train_filenames = np.load('new_names_faces.npy')

# specify the path to save the model
checkpoint_dir = 'training_checkpoints_aligned_batches_faces_ratios_reverse/'


number_of_epochs = 100
train_lim=50000

# load prequisits of the generator class
def get_data_shapes():
    """Generating dataset parameter shapes for tensorflow datasets.
    outputs:
        shapes = output shapes for (images, ground truth boxes, ground truth landmarks)
    """
    return ([None, None, None], [None, None], [None, None, None])

def calculate_actual_outputs(prior_boxes, gt_boxes, gt_landmarks, hyper_params):
    """Calculate ssd actual output values.
    Batch operations supported.
    inputs:
        prior_boxes = (total_bboxes, [center_x, center_y, width, height])
            these values in normalized format between [0, 1]
        gt_boxes = (batch_size, gt_box_size, [y1, x1, y2, x2])
            these values in normalized format between [0, 1]
        gt_landmarks = (batch_size, gt_box_size, total_landmarks, [x, y])
            these values in normalized format between [0, 1]
        hyper_params = dictionary

    outputs:
        actual_deltas = (batch_size, total_bboxes, [delta_bbox_y, delta_bbox_x, delta_bbox_h, delta_bbox_w, delta_landmark_x0, delta_landmark_y0, ..., delta_landmark_xN, delta_landmark_yN])
        actual_labels = (batch_size, total_bboxes, [1 or 0])
    """
    batch_size = tf.shape(gt_boxes)[0]
    iou_threshold = hyper_params["iou_threshold"]
    variances = hyper_params["variances"]
    total_landmarks = hyper_params["total_landmarks"]
    landmark_variances = total_landmarks * variances[0:2]
    # Calculate iou values between each bboxes and ground truth boxes
    iou_map = bbox_utils.generate_iou_map(bbox_utils.convert_xywh_to_bboxes(prior_boxes), gt_boxes)
    # Get max index value for each row
    max_indices_each_gt_box = tf.argmax(iou_map, axis=2, output_type=tf.int32)
    # IoU map has iou values for every gt boxes and we merge these values column wise
    merged_iou_map = tf.reduce_max(iou_map, axis=2)
    #
    pos_cond = tf.greater(merged_iou_map, iou_threshold)
    #
    gt_landmarks = tf.reshape(gt_landmarks, (batch_size, -1, total_landmarks * 2))
    gt_boxes_and_landmarks = tf.concat([gt_boxes, gt_landmarks], -1)
    gt_boxes_and_landmarks_map = tf.gather(gt_boxes_and_landmarks, max_indices_each_gt_box, batch_dims=1)
    expanded_gt_boxes_and_landmarks = tf.where(tf.expand_dims(pos_cond, -1), gt_boxes_and_landmarks_map, tf.zeros_like(gt_boxes_and_landmarks_map))
    actual_deltas = bbox_utils.get_deltas_from_bboxes_and_landmarks(prior_boxes, expanded_gt_boxes_and_landmarks) / (variances + landmark_variances)
    #
    actual_labels = tf.expand_dims(tf.cast(pos_cond, dtype=tf.float32), -1)
    #
    return actual_deltas, actual_labels

def get_padding_values():
    """Generating padding values for missing values in batch for tensorflow datasets.
    outputs:
        paddings = padding values with dtypes for (images, ground truth boxes, ground truth landmarks)
    """
    return (tf.constant(0, tf.float32), tf.constant(0, tf.float32), tf.constant(0, tf.float32))

# define the data generator for batch generation
class My_Custom_Generator(keras.utils.Sequence) :
  
    def __init__(self, image_filenames, labels_coord, batch_size) :
      self.image_filenames = image_filenames
      self.labels = labels_coord[0]
      self.coord = labels_coord[1]
      self.batch_size = batch_size

    def __len__(self) :
      return (np.ceil(len(self.image_filenames) / float(self.batch_size))).astype(np.int)

    
    def __getitem__(self, idx) :
        train_mode=0

          
        batch_x = self.image_filenames[idx * self.batch_size : (idx+1) * self.batch_size]
        batch_y = self.labels[idx * self.batch_size : (idx+1) * self.batch_size,:,:,:]
        batch_z = self.coord[idx * self.batch_size : (idx+1) * self.batch_size,:,:]
        
        label=batch_y
        coord=batch_z
        
        tot_data=batch_y.shape[0]

        prior_boxes = bbox_utils.generate_prior_boxes(hyper_params["feature_map_shapes"], hyper_params["aspect_ratios"])

        data = np.zeros([tot_data, 128, 128, 3])
        omit_list=list([])
        c=0
        
        for i in range(tot_data):
            try:

                FileName = batch_x[i]

                I = cv2.imread(FileName)

                if I.shape[0]>5:
                    
                    img_dec=tf.image.convert_image_dtype(I, tf.float32)
                    img1 = tf.image.resize(img_dec, [128,128])
                    img = tf.image.convert_image_dtype(img1, tf.float32)
                    img=(img - 0.5) / 0.5
                    data[i]=img
                    c += 1
                else:
                    omit_list.append(i)
            except:
               pass
            
 
        actual_deltas, actual_labels = calculate_actual_outputs(prior_boxes, coord,label, hyper_params)       
        return data, (actual_deltas, actual_labels)

# you may set any condition based on the success of the above commands
if True:
    
    from tensorflow.keras.callbacks import ModelCheckpoint, TensorBoard, LearningRateScheduler
    from tensorflow.keras.optimizers import SGD, Adam
    import augmentation
    from ssd_loss import CustomLoss
    from utils import bbox_utils, data_utils, io_utils, train_utils, drawing_utils, landmark_utils
    import blazeface
    
    args = io_utils.handle_args()
    if args.handle_gpu:
        io_utils.handle_gpu_compatibility()
    
    # set the batch size, epochs
    batch_size = 512
    load_weights = False
    hyper_params = train_utils.get_hyper_params()
        
       
    # initialize none values
    data_shapes = data_utils.get_data_shapes()
    padding_values = data_utils.get_padding_values()

    # instantiate the batch generator
    my_training_batch_generator = My_Custom_Generator(X_train_filenames[0:train_lim], (all_labels[0:train_lim],all_coord[0:train_lim]) ,batch_size)

    # use tf keras strategy maker to run on multiple gpus
    # strategy = tf.distribute.MirroredStrategy()

    # report available gpus
    # print('Number of devices: {}'.format(strategy.num_replicas_in_sync))
    
    # using the strategy train the following model
    # with strategy.scope():

    model = blazeface.get_model(hyper_params)

    custom_losses = CustomLoss(hyper_params["neg_pos_ratio"], hyper_params["loc_loss_alpha"])

    model.compile(optimizer=Adam(learning_rate=1e-3),
              loss=[custom_losses.loc_loss_fn, custom_losses.conf_loss_fn])
    
    
    
    blazeface.init_model(model)
        
    #define the call back
    class PrintLR(tf.keras.callbacks.Callback):
        def on_epoch_end(self, epoch, logs=None):
            print('\nLearning rate for epoch {} is {}'.format(epoch + 1,
                                                          model.optimizer.lr.numpy()))
            free_mem()
            print(time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()))

    # specify the folder for saving 
    checkpoint_prefix = os.path.join(checkpoint_dir, "ckpt_{number_of_epochs}")

    def decay(epoch):
      # if epoch < 3:
        return 1e-3
          
    # add a call back to save model weights
    callbacks = [
                
        tf.keras.callbacks.ModelCheckpoint(filepath=checkpoint_prefix,
                                           save_weights_only=True)
    ]   

    history=model.fit_generator(generator=my_training_batch_generator, epochs=number_of_epochs, steps_per_epoch = int(train_lim // batch_size),callbacks=callbacks)   
    model.summary()
      
    # save model history and model parameters
    sha=model.history.history
    np.save(checkpoint_dir+'/history.npy',sha)
    