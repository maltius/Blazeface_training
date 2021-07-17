import platform
import numpy as np
import tensorflow as tf
import cv2
import time
import augmentation
import matplotlib.pyplot as plt
import os




# label3= np.load('/mnt/sda1/downloads/BlazePose-tensorflow-master/data_configs/cocos_17.npy')
# files= np.load('/mnt/sda1/downloads/BlazePose-tensorflow-master/data_configs/files_17.npy')


path='D:/Downloads/vid_test/'

files = sorted([f for f in os.listdir(path) if f.endswith('.jpg')])

portion=len(files)


# portion=files.shape[0]

temps=list([])
temps_orig=list()
sizes_list=list()
all_labels=np.zeros((portion,1,6,2)).astype(np.float32)
all_coord=np.zeros((portion,1,4)).astype(np.float32)

new_files=list([])

# config = tf.compat.v1.ConfigProto()
# config.gpu_options.allow_growth = True
# sess = tf.Session(config=config)

# gpu_devices = tf.config.experimental.list_physical_devices('GPU')
# for device in gpu_devices:
#     tf.config.experimental.set_memory_growth(device, False)

# img = tf.io.read_file(FileName)
# img = tf.image.decode_image(img)
# img_shape = img.shape
            
# in the end put everything in 

# coord1[0,:,:]=all_coord[0,:,:]

# actual_deltas, actual_labels = calculate_actual_outputs(prior_boxes, all_coord,all_labels, hyper_params)

c=0
for t in range(0,portion):
    if t%500==0:
        print('t is: ',t)
    # hold it 'image' key
    FileName = path+files[t]
    I = cv2.imread(FileName)
    
    # param=4
    # if param>0 and param<10:
    #     if True:
    #         II=cv2.resize(I,(int(I.shape[1]*param),I.shape[0]))
    #         I2=II[:,int(II.shape[1]/2-I.shape[1]/2):int(II.shape[1]/2+I.shape[1]/2),:]

    #         # I=I[:,int(I.shape[1]/4):3*int(I.shape[1]/4),:]
    # sizes_list.append((I.shape[0],I.shape[1]))
    # temps_orig.append(I)
    I2 = cv2.rotate(I,cv2.cv2.ROTATE_90_CLOCKWISE)
    
    if True:
        # img1 = tf.io.read_file(FileName)
        img_dec=tf.image.convert_image_dtype(I2, tf.float32)
        # img_dec = tf.image.decode_image(img1)
        img = tf.image.resize(img_dec, [128,128])
        # img = tf.image.convert_image_dtype(img1, tf.float32)
        img_shape = img_dec.shape

        
        
        # tf.expand_dims(image_data["landmarks_2d"], 0)
        # img = img/255.0
        
    
        item=(img - 0.5) / 0.5
        
        c=c+1
        # item['landmarks_2d']=label
        
        temps.append(item)
        
all_labels=all_labels[0:c,:,:]
all_coord=all_coord[0:c,:,:]        

if len(temps)>5:
    train_dataset=tf.data.Dataset.from_tensor_slices((temps))
    
    train_dataset1=tf.data.Dataset.from_tensor_slices((all_labels))
    # train_dataset1 = tf.convert_to_tensor(train_dataset1,dtype=tf.float32)
    train_dataset2=tf.data.Dataset.from_tensor_slices((all_coord))
    # train_dataset2 = tf.convert_to_tensor(train_dataset2,dtype=tf.float32)
    
    
    ds = tf.data.Dataset.zip((train_dataset,train_dataset2,train_dataset1)) 
    train_total_items=portion


cc=0
for elements in ds.__iter__():
    if cc<5:
        img=np.asarray(elements[0])
        skeleton = (128*elements[2]).numpy().astype(int)
        rec= (128*elements[1]).numpy().astype(int)[0]
        # plt.imshow(img)
        for ii in range(6):
            cv2.circle(img, center=tuple(skeleton[0][ii][0:2]), radius=1, color=(0, 255, 0), thickness=2)
        img = cv2.rectangle(img, (rec[1],rec[0]), (rec[3],rec[2]), (255, 0, 0) , 1)
        print(type(elements))
        print(elements[1]*128)
        print(elements[2]*128)
        plt.imshow(img)
        # cv2.imwrite('sams/'+str(cc).zfill(4)+'.jpg',img*256)
        cc=cc+1
        
        # skeleton=label3[t,:,:]
        # for i in range(0,17):
        #     # cv2.circle(I, center=tuple(skeleton[i][0:2].astype(int)), radius=1, color=(255, 0, 0), thickness=2)
        #     cv2.putText(I,str(i), tuple(skeleton[i][0:2].astype(int)), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0,0,255),thickness=4)
        # plt.imshow(I)
        plt.show()

        
# coco_blaze_face['val']=temp

# train_set=coco_blaze_face['train']




# np.save('aic_labels_faces_mids.npy',all_labels)
# np.save('aic_coord_faces_mids.npy',all_coord)
# np.save('aic_new_names_faces_mids.npy',np.array(new_files))




# # features = tf.data.Dataset.features.FeaturesDict(temp)
# cc=0
# for elements in ds.__iter__():
#     if cc<1:
#         plt.imshow(elements[0])
#         print(type(elements))
#         print(elements[1])
#         print(elements[2])
#         cc=cc+1
#         plt.show()


# temp=list([])
# portion=100
# all_labels=np.zeros((portion,1,6,2)).astype(np.float32)
# all_coord=np.zeros((portion,1,4)).astype(np.float32)

# img = tf.io.read_file(FileName)
# img = tf.image.decode_image(img)
# img_shape = img.shape
            
# in the end put everything in 


# c=0
# for t in range(label3.shape[0]-portion*4,label3.shape[0]):
#     # if t%500==0:
#     #     print(t)
#     # hold it 'image' key
#     FileName = "D:/Downloads/aic-single/aic_persons_17_single/"+files[t]
#     I = cv2.imread(FileName)
#     hasel=label3[t,6,0]*label3[t,9,1]*label3[t,6,1]*label3[t,9,0]*label3[t,13,0]*label3[t,13,1]
#     max_h=max(abs(label3[t,6,0]-label3[t,9,0]),abs(label3[t,6,0]-label3[t,9,0]))
    
    
#     if I.shape[0]>5 and hasel>0 and max_h>85 and c<portion:
#         if c%500==0:
#             print(c)
#         # img1 = tf.io.read_file(FileName)
#         img_dec=tf.image.convert_image_dtype(I, tf.float32)
#         # img_dec = tf.image.decode_image(img1)
#         img = tf.image.resize(img_dec, [128,128])
#         # img = tf.image.convert_image_dtype(img1, tf.float32)
#         img_shape = img_dec.shape
        
#         # choose indicies to select among everything
#         ind=[6,9,6,9,13,9]
        
#         # store it in "landmarks_2d" key
#         label=label3[t,ind,0:2]
        

        
#         # label[4,:]=(label3[t,5,0:2]+label3[t,6,0:2])*0.5
#         label[5,:]=(label3[t,6,0:2]+label3[t,9,0:2])*0.5
        
#         label[ :, 0] *= (128 / img_shape[1])
#         label[ :, 1] *= (128 / img_shape[0])

#         all_labels[c,:,:,:]=np.reshape(label/128,(1,label.shape[0],label.shape[1])).astype(np.float32)
        
#         minx=min(label[0:5,1]/128)
#         maxx=max(label[0:5,1]/128)
#         miny=min(label[0:5,0]/128)
#         maxy=max(label[0:5,0]/128)
        
#         coord=np.array([minx,miny,maxx,maxy]).astype(np.float32)
#         all_coord[c,:,:]=np.reshape(coord,(1,4)).astype(np.float32)
        
#         # tf.expand_dims(image_data["landmarks_2d"], 0)
#         # img = img/255.0
        
    
#         item=(img - 0.5) / 0.5
#         c=c+1
#         # item['landmarks_2d']=label
        
#         temp.append(item)

# # coco_blaze_face['val']=temp

# # train_set=coco_blaze_face['train']
# # all_labels=all_labels[0:c,:,:]

# train_dataset=tf.data.Dataset.from_tensor_slices((temp))

# train_dataset1=tf.data.Dataset.from_tensor_slices((all_labels))
# # train_dataset1 = tf.convert_to_tensor(train_dataset1,dtype=tf.float32)
# train_dataset2=tf.data.Dataset.from_tensor_slices((all_coord))
# # train_dataset2 = tf.convert_to_tensor(train_dataset2,dtype=tf.float32)


# ds_val = tf.data.Dataset.zip((train_dataset,train_dataset2,train_dataset1)) 
# val_total_items=portion

# # # features = tf.data.Dataset.features.FeaturesDict(temp)
# # cc=0
# # for elements in ds.__iter__():
# #     if cc<1:
# #         plt.imshow(elements[0])
# #         print(type(elements))
# #         print(elements[1])
# #         print(elements[2])
# #         cc=cc+1
# # # skeleton = label.astype(int)
# # # img = I
# # # # plt.imshow(img)
# # # for i in range(skeleton.shape[0]):
# # #     cv2.circle(img, center=tuple(skeleton[i][0:2]), radius=2, color=(0, 255, 0), thickness=2)
        
# # # plt.imshow(I)

# # # np.save('all_labels.npy',all_labels)
# # # np.save('all_coord.npy',all_coord)

# cc=0
# for elements in ds_val.__iter__():
#     if cc<1:
        # img=np.asarray(elements[0])
        # skeleton = (128*elements[2]).numpy().astype(int)
        # rec= (128*elements[1]).numpy().astype(int)[0]
        # # plt.imshow(img)
        # for ii in range(6):
        #     cv2.circle(img, center=tuple(skeleton[0][ii][0:2]), radius=1, color=(0, 255, 0), thickness=2)
        # img = cv2.rectangle(img, (rec[1],rec[0]), (rec[3],rec[2]), (255, 0, 0) , 1) 
        # print(type(elements))
        # print(elements[1]*128)
        # print(elements[2]*128)
        # plt.imshow(img)
        # # cv2.imwrite('sams/'+str(cc).zfill(4)+'.jpg',img*256)
        # cc=cc+1

# for tr in range(5000,5500,2):
#     # tr=3007     
#     img=np.asarray(temp[tr])
#     skeleton = (128*all_labels[tr]).astype(int)
#     rec= (128*all_coord[tr]).astype(int)[0]
#     # plt.imshow(img)
#     for ii in range(6):
#         cv2.circle(img, center=tuple(skeleton[0][ii][0:2]), radius=1, color=(0, 255, 0), thickness=2)
#     img = cv2.rectangle(img, (rec[1],rec[0]), (rec[3],rec[2]), (255, 0, 0) , 1) 
#     # print(type(elements))
#     # print(elements[1]*128)
#     # print(elements[2]*128)
#     plt.imshow(img)
#     plt.show()
#     time.sleep(1)
#     # cv2.imwrite('sams/'+str(cc).zfill(4)+'.jpg',img*256)
#     # cc=cc+1


if False:
    img=np.copy(I)
    skeleton = temp_label.astype(int)
    plt.imshow(img)
    # skeleton = (ex_labels[cc,:,:]).astype(int)

    for ii in range(skeleton.shape[0]):
        # cv2.circle(img, center=tuple(skeleton[ii][0:2]), radius=1, color=(0, 255, 0), thickness=2)

        cv2.putText(img,str(ii), tuple(skeleton[ii][0:2].astype(int)), cv2.FONT_HERSHEY_SIMPLEX, 0.3, (255,0,0),thickness=1)

    cv2.imwrite('img.jpeg',img)