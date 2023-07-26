import time
import tensorflow as tf

from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from tensorflow.python.keras.utils import np_utils
import h5py
from operator import truediv
import scipy.io as sio
import spectral
import numpy as np
import pandas as pd
# Commented out IPython magic to ensure Python compatibility.
import matplotlib.pyplot as plt
import tensorflow_addons as tfa
import seaborn as sns
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report, cohen_kappa_score

import keras
from keras import layers
from keras.models import Model
from keras.layers import Dropout, Input, Conv2D, Conv3D, MaxPool3D, Flatten, Dense, Reshape, BatchNormalization

from swinmodel import PatchMerging, PatchExtract, PatchEmbedding, SwinTransformer

""" os.environ["CUDA_VISIBLE_DEVICES"] = '0'
config = tf.ConfigProto()
config.gpu_options.per_procsss_gpu_memory_fraction = 0.5
config.gpu_options.allow_growth = True
sess = tf.Session(config=config)
 """
# Data Preparation

#loading dataset
# X = sio.loadmat('/root/autodl-tmp/HSI/IP1/Indian_pines_corrected.mat')['indian_pines_corrected']
# y = sio.loadmat('/root/autodl-tmp/HSI/IP1/Indian_pines_gt.mat')['indian_pines_gt']
X = sio.loadmat('D:\myproject\SWIN HSI/IP1/Indian_pines_corrected.mat')['indian_pines_corrected']
y = sio.loadmat('D:\myproject\SWIN HSI/IP1/Indian_pines_gt.mat')['indian_pines_gt']

# classes
num_classes = 16
# Proportion of the sample used for testing
test_ratio = 0.7
# Size of the extracted patch around each pixel
# 不行：26，27，28，29，30
patch1_size = 25

# Number of principal components obtained by dimensionality reduction using PCA (original data: 30)
pca_components =30
#dataset ='PU'
dataset = 'IP'
#dataset = 'SA'
# dataset = 'PU'
# dataset = 'XZ'
# dataset = 'HC'
# dataset = 'HU'
# dataset = 'LK'
#训练次数
num_epochs = 100

#其他需要修改的参数：保存位置，加载位置，导出 图的颜色

# Apply PCA transform to hyperspectral data X
def applyPCA(X, numComponents):
    newX = np.reshape(X, (-1, X.shape[2]))
    pca = PCA(n_components=numComponents, whiten=True)
    newX = pca.fit_transform(newX)
    newX = np.reshape(newX, (X.shape[0], X.shape[1], numComponents))
    return newX

# Extract the patch around each pixel and create it in a format that matches the keras processing windowsize还有一个8(originally windowSize=5)
def createImageCubes(X, y, windowSize=5, removeZeroLabels=True):
    # for X to padding
    margin = int(windowSize / 2)  # hybirdcnn是（（windowsize-1）/2）
    zeroPaddedX = padWithZeros(X, margin=margin)
    # split patches
    patchesData = np.zeros((X.shape[0] * X.shape[1], windowSize, windowSize, X.shape[2]))
    patchesLabels = np.zeros((X.shape[0] * X.shape[1]))
    patchIndex = 0
    for r in range(margin, zeroPaddedX.shape[0] - margin):
        for c in range(margin, zeroPaddedX.shape[1] - margin):
            patch = zeroPaddedX[r - margin:r + margin + 1, c - margin:c + margin + 1]
            patchesData[patchIndex, :, :, :] = patch
            patchesLabels[patchIndex] = y[r - margin, c - margin]
            patchIndex = patchIndex + 1
    if removeZeroLabels:
        patchesData = patchesData[patchesLabels > 0, :, :, :]
        patchesLabels = patchesLabels[patchesLabels > 0]
        patchesLabels -= 1
    return patchesData, patchesLabels

# def createImageCubes(X, y, windowSize=8, removeZeroLabels=True):
#     # for X to padding
#     margin = int(windowSize / 2)  # hybirdcnn是（（windowsize-1）/2）
#     zeroPaddedX = padWithZeros(X, margin=margin)
#     # split patches
#     patchesData = np.zeros((X.shape[0] * X.shape[1], windowSize, windowSize, X.shape[2]))
#     patchesLabels = np.zeros((X.shape[0] * X.shape[1]))
#     patchIndex = 0
#     for r in range(margin, zeroPaddedX.shape[0] - margin):
#         for c in range(margin, zeroPaddedX.shape[1] - margin):
#             patch = zeroPaddedX[r - margin:r + margin + 1, c - margin:c + margin + 1]
#             patchesData[patchIndex, :, :, :] = patch
#             patchesLabels[patchIndex] = y[r - margin, c - margin]
#             patchIndex = patchIndex + 1
#     if removeZeroLabels:
#         patchesData = patchesData[patchesLabels > 0, :, :, :]
#         patchesLabels = patchesLabels[patchesLabels > 0]
#         patchesLabels -= 1
#     return patchesData, patchesLabels

def splitTrainTestSet(X, y, testRatio, randomState=345):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=testRatio, random_state=randomState, stratify=y)
    return X_train, X_test, y_train, y_test

def padWithZeros(X, margin=2):
    newX = np.zeros((X.shape[0] + 2 * margin, X.shape[1] + 2 * margin, X.shape[2]))
    x_offset = margin
    y_offset = margin
    newX[x_offset:X.shape[0] + x_offset, y_offset:X.shape[1] + y_offset, :] = X
    return newX


print('Hyperspectral data shape: ', X.shape)
print('Label shape: ', y.shape)

print('\n... ... PCA tranformation ... ...')
X_pca = applyPCA(X, numComponents=pca_components)
print('Data shape after PCA: ', X_pca.shape)

print('\n... ... create data cubes ... ...')
X_pca, y = createImageCubes(X_pca, y, windowSize=patch1_size)
print('Data cube X shape: ', X_pca.shape)
print('Data cube y shape: ', y.shape)

print('\n... ... create train & test data ... ...')
Xtrain, Xtest, ytrain, ytest = splitTrainTestSet(X_pca, y, test_ratio)
print('Xtrain shape: ', Xtrain.shape)
print('Xtest  shape: ', Xtest.shape)

print('\n... ... to keras ... ...')
# Change the shape of Xtrain, Ytrain to match keras
Xtrain = Xtrain.reshape(-1, patch1_size, patch1_size, pca_components, 1)
Xtest  = Xtest.reshape(-1, patch1_size, patch1_size, pca_components, 1)
ytrain = np_utils.to_categorical(ytrain)
# #ytrain = keras.utils.to_categorical(ytrain, num_classes)
ytest = np_utils.to_categorical(ytest, num_classes)
print('Xtrain shape: ', Xtrain.shape)
print('Xtest  shape: ', Xtest.shape)
print('ytrain shape: ',{ytrain.shape})
print('ytest shape:' , {ytest.shape})


# Parameters and Functions
input_shape = (patch1_size, patch1_size,pca_components) #need to follow loading.shape and afterPCA

output_units = 11 #(hybrid is 9 )11

# patch_size = (2, 2)  # 2-by-2 sized patches
# dropout_rate = 0.03  # Dropout rate (origin0.03)
# num_heads = 8  # Attention heads
# embed_dim = 64  # Embedding dimension (origin64)
# num_mlp = 256  # MLP layer size
# qkv_bias = True  # Convert embedded patches to query, key, and values with a learnable additive value
# window_size = 2  # Size of attention window
# shift_size = 1  # Size of shifting window
# image_dimension = 25  # Initial image size (origin32) （afterPCA25）（afterCNN module 17）

patch_size = (2, 2)  # 2-by-2 sized patches
dropout_rate = 0.03  # Dropout rate (origin0.03)
num_heads = 8  # 8 Attention heads 1：4 ， 2：2 ，3：1
embed_dim = 64  # Embedding dimension (origin64)
num_mlp = 256  # MLP layer size 256
qkv_bias = True  # Convert embedded patches to query, key, and values with a learnable additive value
window_size = 2  # Size of attention window 2
shift_size = 1  # Size of shifting window 1
image_dimension = 25  # Initial image size (origin32) （afterPCA25）（afterCNN module 17）



num_patch_x = input_shape[0] // patch_size[0]
num_patch_y = input_shape[1] // patch_size[1]

learning_rate = 1e-3
batch_size = 128
validation_split = 0.1
weight_decay = 0.0001
label_smoothing = 0.1

# Modeling
#input = layers.Input((patch1_size, patch1_size, pca_components, 1))
input = layers.Input(input_shape)

#x = layers.RandomCrop(image_dimension, image_dimension)(conv_layer4)
x = tf.keras.layers.RandomCrop(image_dimension, image_dimension)(input)
x = tf.keras.layers.RandomFlip("horizontal")(x)
#print('random flip:',x.shape)
x = PatchExtract(patch_size)(x)
# x = PatchExtract(patch_size)(input)
print('patch extract:',x.shape)
x = PatchEmbedding(num_patch_x * num_patch_y, embed_dim)(x)
print('patch embedding:',x.shape)
# x = SwinTransformer(
#     dim=embed_dim,
#     num_patch=(num_patch_x, num_patch_y),
#     num_heads=num_heads,
#     window_size=window_size,
#     shift_size=0,
#     num_mlp=num_mlp,
#     qkv_bias=qkv_bias,
#     dropout_rate=dropout_rate,
# )(x)
# print('swin1:',x.shape)
x = SwinTransformer(
    dim=embed_dim,
    num_patch=(num_patch_x, num_patch_y),
    num_heads=num_heads,
    window_size=window_size,
    shift_size=shift_size,
    num_mlp=num_mlp,
    qkv_bias=qkv_bias,
    dropout_rate=dropout_rate,
)(x)
print('swin2:',x.shape)
x = PatchMerging((num_patch_x, num_patch_y), embed_dim=embed_dim)(x)
print('patchmerging:',x.shape)
x = layers.GlobalAveragePooling1D()(x)
print('global average pooling:',x.shape)
output = layers.Dense(num_classes, activation="softmax")(x)
print('out:',x.shape)


#train

model = keras.Model(input, output)
model.summary()
s = time.time()
model.compile(
    loss=keras.losses.CategoricalCrossentropy(label_smoothing=label_smoothing),
optimizer=tfa.optimizers.AdamW(
       learning_rate=learning_rate, weight_decay=weight_decay
#     optimizer=tf.keras.optimizers.SGD(
#         learning_rate=learning_rate
    ),
    metrics=[
        keras.metrics.CategoricalAccuracy(name="accuracy"),
        #keras.metrics.TopKCategoricalAccuracy(5, name="top-5-accuracy"),
    ],
)
history = model.fit(x=Xtrain, y=ytrain, batch_size=batch_size, epochs=num_epochs,validation_split=validation_split, )
e = time.time()
print(e-s)
#AC
loss, accuracy = model.evaluate(Xtest, ytest)
print(f"Test loss: {round(loss, 2)}")
print(f"Test accuracy: {round(accuracy * 100, 2)}%")

# log for save to csv:
hist_df = pd.DataFrame(history.history)
# hist_csv_file = '/root/autodl-tmp/result/IP/history.csv'
hist_csv_file = 'result/IP/history.csv'
with open(hist_csv_file, mode='w') as f:
    hist_df.to_csv(f)
#loss show
# plt.plot(history.history["loss"], label="train_loss")
# plt.plot(history.history["val_loss"], label="val_loss")
# plt.xlabel("Epochs")
# plt.ylabel("Loss")
# plt.title("Train and Validation Losses Over Epochs", fontsize=14)
# plt.legend()
# plt.grid()
# plt.show()

# # rate show
# loss, accuracy, top_5_accuracy = model.evaluate(Xtest, ytest)
# print(f"Test loss: {round(loss, 2)}")
# print(f"Test accuracy: {round(accuracy * 100, 2)}%")
# print(f"Test top 5 accuracy: {round(top_5_accuracy * 100, 2)}%")

# Classification Map

# rate
def AA_andEachClassAccuracy(confusion_matrix):
    counter = confusion_matrix.shape[0]
    list_diag = np.diag(confusion_matrix)
    list_raw_sum = np.sum(confusion_matrix, axis=1)
    each_acc = np.nan_to_num(truediv(list_diag, list_raw_sum))
    average_acc = np.mean(each_acc)
    return each_acc, average_acc

def reports(X_test, y_test, name):
    # start = time.time()
    Y_pred = model.predict(X_test)
    y_pred = np.argmax(Y_pred, axis=1)
    # end = time.time()
    # print(end - start)
    if name == 'IP':
        target_names = ['Alfalfa', 'Corn-notill', 'Corn-mintill', 'Corn'
            , 'Grass-pasture', 'Grass-trees', 'Grass-pasture-mowed',
                        'Hay-windrowed', 'Oats', 'Soybean-notill', 'Soybean-mintill',
                        'Soybean-clean', 'Wheat', 'Woods', 'Buildings-Grass-Trees-Drives',
                        'Stone-Steel-Towers']
    elif name == 'SA':
        target_names = ['Brocoli_green_weeds_1', 'Brocoli_green_weeds_2', 'Fallow', 'Fallow_rough_plow',
                        'Fallow_smooth',
                        'Stubble', 'Celery', 'Grapes_untrained', 'Soil_vinyard_develop', 'Corn_senesced_green_weeds',
                        'Lettuce_romaine_4wk', 'Lettuce_romaine_5wk', 'Lettuce_romaine_6wk', 'Lettuce_romaine_7wk',
                        'Vinyard_untrained', 'Vinyard_vertical_trellis']
    elif name == 'PU':
        target_names = ['Asphalt', 'Meadows', 'Gravel', 'Trees', 'Painted metal sheets', 'Bare Soil', 'Bitumen',
                        'Self-Blocking Bricks', 'Shadows']
    elif dataset == 'XZ':
        target_names = ['Bareland1', 'Lakes', 'Coals', 'Cement', 'Crops1', 'Tress', 'Bareland2',
                        'Crops', 'Red-Titles']
    elif dataset == 'HC':
        target_names = ['Strawberry', 'Cowpea', 'Soybean', 'Sorghum', 'Water spinach', 'Watermelon', 'Greens',
                        'Trees', 'Grass', 'Red roof', 'Gray roof', 'Plastic', 'Bare soil', 'Road', 'Bright object',
                        'Water']
    elif dataset == 'HU':
        target_names = ['Red roof', 'Road', 'Bare soil', 'Cotton', 'Cotton firewood', 'Rape', 'Chinese cabbage',
                        'Pakchoi', 'Cabbage', 'Tuber mustard', 'Brassica parachinensis', 'Brassica chinensis',
                        'Small Brassica chinensis', 'Lactuca sativa', 'Celtuce',
                        'Film covered lettuce', 'Romaine lettuce', 'Carrot', 'White radish', 'Garlic sprout',
                        'Broad bean', 'Tree']
    elif dataset == 'LK':
        target_names = ['Corn', 'Cotton', 'SeSame', 'Broad-leaf soybean', 'Narrow-leaf soybean', 'Rice', 'Water',
                        'Roads and houses', 'Mixed weed']
    elif dataset == 'XA':
        target_names = ['Acer negundo Linn', 'Willow', 'Elm', 'Paddy', 'Chinese Pagoda Tree',
                        'Fraxinus chinensis', 'Koelreuteria paniculata', 'Water', 'Bare land',
                        'Paddy stubble', 'Robinia pseudoacacia', 'Corn', 'Pear', 'Soya', 'Alamo', 'Vegetable field',
                        'Sparsewood', 'Meadow', 'Peach', 'Building']
    classification = classification_report(np.argmax(y_test, axis=1), y_pred, target_names=target_names)
    oa = accuracy_score(np.argmax(y_test, axis=1), y_pred)
    confusion = confusion_matrix(np.argmax(y_test, axis=1), y_pred)
    each_acc, aa = AA_andEachClassAccuracy(confusion)
    kappa = cohen_kappa_score(np.argmax(y_test, axis=1), y_pred)
    score = model.evaluate(X_test, y_test, batch_size=32)
    Test_Loss = score[0] * 100
    Test_accuracy = score[1] * 100


    return classification, confusion, Test_Loss, Test_accuracy, oa * 100, each_acc * 100, aa * 100, kappa * 100



classification, confusion, Test_loss, Test_accuracy, oa, each_acc, aa, kappa = reports(Xtest,ytest,dataset)
classification = str(classification)
confusion = str(confusion)
print(classification)
# file_name = "/root/autodl-tmp/result/IP/classification_report.txt"
file_name = "result/IP/classification_report.txt"

with open(file_name, 'w') as x_file:
    x_file.write('{} Test loss (%)'.format(Test_loss))
    x_file.write('\n')
    x_file.write('{} Test accuracy (%)'.format(Test_accuracy))
    x_file.write('\n')
    x_file.write('\n')
    x_file.write('{} Kappa accuracy (%)'.format(kappa))
    x_file.write('\n')
    x_file.write('{} Overall accuracy (%)'.format(oa))
    x_file.write('\n')
    x_file.write('{} Average accuracy (%)'.format(aa))
    x_file.write('\n')
    x_file.write('\n')
    x_file.write('{}'.format(classification))
    x_file.write('\n')
    x_file.write('{}'.format(confusion))

#
# # confusion matrix Evaluation
# # cm
# Y_pred_test = model.predict(Xtest)
# y_pred_test = np.argmax(Y_pred_test, axis=1)
# confusion = confusion_matrix(np.argmax(ytest, axis=1), y_pred_test, labels=np.unique(np.argmax(ytest, axis=1)))
# cm_sum = np.sum(confusion, axis=1, keepdims=True)
# cm_perc = confusion / cm_sum.astype(float) * 100
# annot = np.empty_like(confusion).astype(str)
# nrows, ncols = confusion.shape
# for i in range(nrows):
#     for j in range(ncols):
#         c = confusion[i, j]
#         p = cm_perc[i, j]
#         if i == j:
#             s = cm_sum[i]
#             annot[i, j] = '%.1f%%\n%d/%d' % (p, c, s)
#         elif c == 0:
#             annot[i, j] = ''
#         else:
#             annot[i, j] = '%.1f%%\n%d' % (p, c)
#
# if dataset == 'IP':
#     target_names = ['Alfalfa', 'Corn-notill', 'Corn-mintill', 'Corn'
#         , 'Grass-pasture', 'Grass-trees', 'Grass-pasture-mowed',
#                     'Hay-windrowed', 'Oats', 'Soybean-notill', 'Soybean-mintill',
#                     'Soybean-clean', 'Wheat', 'Woods', 'Buildings-Grass-Trees-Drives',
#                     'Stone-Steel-Towers']
# elif dataset == 'SA':
#     target_names = ['Brocoli_green_weeds_1', 'Brocoli_green_weeds_2', 'Fallow', 'Fallow_rough_plow',
#                     'Fallow_smooth',
#                     'Stubble', 'Celery', 'Grapes_untrained', 'Soil_vinyard_develop', 'Corn_senesced_green_weeds',
#                     'Lettuce_romaine_4wk', 'Lettuce_romaine_5wk', 'Lettuce_romaine_6wk', 'Lettuce_romaine_7wk',
#                     'Vinyard_untrained', 'Vinyard_vertical_trellis']
# elif dataset == 'PU':
#     target_names = ['Asphalt', 'Meadows', 'Gravel', 'Trees', 'Painted metal sheets', 'Bare Soil', 'Bitumen',
#                     'Self-Blocking Bricks', 'Shadows']
# elif dataset == 'XZ':
#     target_names = ['Bareland1', 'Lakes', 'Coals', 'Cement', 'Crops1', 'Tress', 'Bareland2',
#                     'Crops', 'Red-Titles']
# elif dataset == 'HC':
#     target_names = ['Strawberry', 'Cowpea', 'Soybean', 'Sorghum', 'Water spinach', 'Watermelon', 'Greens',
#                     'Trees', 'Grass', 'Red roof', 'Gray roof', 'Plastic', 'Bare soil', 'Road','Bright object', 'Water']
# elif dataset == 'HU':
#     target_names = ['Red roof', 'Road', 'Bare soil', 'Cotton', 'Cotton firewood', 'Rape', 'Chinese cabbage',
#                     'Pakchoi', 'Cabbage', 'Tuber mustard', 'Brassica parachinensis', 'Brassica chinensis', 'Small Brassica chinensis', 'Lactuca sativa', 'Celtuce',
#                     'Film covered lettuce','Romaine lettuce', 'Carrot', 'White radish', 'Garlic sprout', 'Broad bean', 'Tree']
# elif dataset == 'LK':
#     target_names = ['Corn', 'Cotton', 'SeSame', 'Broad-leaf soybean', 'Narrow-leaf soybean', 'Rice', 'Water',
#                     'Roads and houses', 'Mixed weed']
# elif dataset == 'XA' :
#     target_names  = ['Acer negundo Linn', 'Willow', 'Elm', 'Paddy', 'Chinese Pagoda Tree',
#            'Fraxinus chinensis', 'Koelreuteria paniculata', 'Water', 'Bare land',
#            'Paddy stubble', 'Robinia pseudoacacia', 'Corn', 'Pear', 'Soya', 'Alamo', 'Vegetable field',
#            'Sparsewood', 'Meadow', 'Peach', 'Building']
# cm = pd.DataFrame(confusion, index=np.unique(target_names), columns=np.unique(target_names))
# cm.index.name = 'Actual'
# cm.columns.name = 'Predicted'
# fig, ax = plt.subplots(figsize=(15, 10))
# plt.rcParams.update({'font.size': 12})
# #plt.rc('font', family='Lato Light', size=12)
# sns.heatmap(cm, cmap="PuBu", annot=annot, fmt='', vmax=1000, vmin=0, ax=ax)
# # plt.savefig("/root/autodl-tmp/result/IP/cm.png")
# plt.savefig("result/IP/cm.png")
# plt.show()
#
#
# #show
# deepx16_colors = np.array([
#     [0, 0, 0], [190, 36, 73], [218, 70, 76],
#     [236, 97, 69], [247, 131, 77], [252, 170, 95],
#     [253, 200, 119], [254, 227, 145], [254, 245, 175],
#     [247, 252, 179], [232, 246, 156], [202, 233, 157],
#     [166, 219, 164], [126, 203, 164], [89, 180, 170],
#     [59, 146, 184], [68, 112, 177]], np.int16)
# deepx9_colors = np.array([
#     [0, 0, 0], [211, 60, 78], [244, 109, 67], [252, 172, 96], [254, 224, 139], [254, 254, 190], [230, 245, 152], [169, 220, 164], [102, 194, 165], [50, 134, 188],
# ],np.int16)
# # K after PCA
# #show
# def Patch(data, height_index, width_index):
#     height_slice = slice(height_index, height_index + patch1_size)
#     width_slice = slice(width_index, width_index + patch1_size)
#     patch = data[height_slice, width_slice, :]
#     return patch
#
#
# # X = sio.loadmat('/root/autodl-tmp/HSI/IP1/Indian_pines_corrected.mat')['indian_pines_corrected']
# # y = sio.loadmat('/root/autodl-tmp/HSI/IP1/Indian_pines_gt.mat')['indian_pines_gt']
# X = sio.loadmat('D:\myproject\SWIN HSI/IP1/Indian_pines_corrected.mat')['indian_pines_corrected']
# y = sio.loadmat('D:\myproject\SWIN HSI/IP1/Indian_pines_gt.mat')['indian_pines_gt']
#
#
# height = y.shape[0]
# width = y.shape[1]
#
# X = applyPCA(X, numComponents= pca_components)
# X = padWithZeros(X, patch1_size//2)
# # 逐像素预测类别
# outputs = np.zeros((height,width))
# for i in range(height):
#     for j in range(width):
#         if int(y[i,j]) == 0:
#             continue
#         else :
#             image_patch = Patch(X, i, j)
#             X_test_image = image_patch.reshape(1, image_patch.shape[0], image_patch.shape[1], image_patch.shape[2],
#                                                1).astype('float32')
#             prediction = (model.predict(X_test_image))
#             prediction = np.argmax(prediction, axis=1)
#            # prediction = net(X_test_image)
#             #prediction = np.argmax(prediction.detach().cpu().numpy(), axis=1)
#             outputs[i][j] = prediction+1
#
# predict_image = spectral.imshow(classes = outputs.astype(int),figsize =(7,7))
# # spectral.save_rgb("/root/autodl-tmp/result/IP/predictions.jpg", outputs.astype(int), colors=deepx16_colors)
# spectral.save_rgb("result/IP/predictions.jpg", outputs.astype(int), colors=deepx16_colors)
