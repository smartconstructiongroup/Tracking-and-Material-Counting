# -*- coding: utf-8 -*-

from keras.optimizers import SGD
from keras.layers import Input, merge, ZeroPadding2D
from keras.layers.core import Dense, Dropout, Activation
from keras.layers.convolutional import Convolution2D
from keras.layers.pooling import AveragePooling2D, GlobalAveragePooling2D, MaxPooling2D
from keras.layers.normalization import BatchNormalization
from keras.models import Model
import keras.backend as K
from custom_layers.scale_layer import Scale
from sklearn.metrics import log_loss
from sklearn.metrics import confusion_matrix
from load_Palette import load_palette
import numpy as np
import cv2
import os
import pickle



def densenet161_model(img_rows, img_cols, color_type=1, nb_dense_block=4, growth_rate=48, nb_filter=96, reduction=0.5, dropout_rate=0.0, weight_decay=1e-4, num_classes=None):
    '''
    DenseNet 161 Model for Keras

    Model Schema is based on 
    https://github.com/flyyufelix/DenseNet-Keras

    ImageNet Pretrained Weights 
    Theano: https://drive.google.com/open?id=0Byy2AcGyEVxfVnlCMlBGTDR3RGs
    TensorFlow: https://drive.google.com/open?id=0Byy2AcGyEVxfUDZwVjU2cFNidTA

    # Arguments
        nb_dense_block: number of dense blocks to add to end
        growth_rate: number of filters to add per dense block
        nb_filter: initial number of filters
        reduction: reduction factor of transition blocks.
        dropout_rate: dropout rate
        weight_decay: weight decay factor
        classes: optional number of classes to classify images
        weights_path: path to pre-trained weights
    # Returns
        A Keras model instance.
    '''
    eps = 1.1e-5

    # compute compression factor
    compression = 1.0 - reduction

    # Handle Dimension Ordering for different backends
    global concat_axis
    if K.image_dim_ordering() == 'tf':
      concat_axis = 3
      img_input = Input(shape=(224, 224, 3), name='data')
    else:
      concat_axis = 1
      img_input = Input(shape=(3, 224, 224), name='data')

    # From architecture for ImageNet (Table 1 in the paper)
    nb_filter = 96
    nb_layers = [6,12,36,24] # For DenseNet-161

    # Initial convolution
    x = ZeroPadding2D((3, 3), name='conv1_zeropadding')(img_input)
    x = Convolution2D(nb_filter, 7, 7, subsample=(2, 2), name='conv1', bias=False)(x)
    x = BatchNormalization(epsilon=eps, axis=concat_axis, name='conv1_bn')(x)
    x = Scale(axis=concat_axis, name='conv1_scale')(x)
    x = Activation('relu', name='relu1')(x)
    x = ZeroPadding2D((1, 1), name='pool1_zeropadding')(x)
    x = MaxPooling2D((3, 3), strides=(2, 2), name='pool1')(x)

    # Add dense blocks
    for block_idx in range(nb_dense_block - 1):
        stage = block_idx+2
        x, nb_filter = dense_block(x, stage, nb_layers[block_idx], nb_filter, growth_rate, dropout_rate=dropout_rate, weight_decay=weight_decay)

        # Add transition_block
        x = transition_block(x, stage, nb_filter, compression=compression, dropout_rate=dropout_rate, weight_decay=weight_decay)
        nb_filter = int(nb_filter * compression)

    final_stage = stage + 1
    x, nb_filter = dense_block(x, final_stage, nb_layers[-1], nb_filter, growth_rate, dropout_rate=dropout_rate, weight_decay=weight_decay)

    x = BatchNormalization(epsilon=eps, axis=concat_axis, name='conv'+str(final_stage)+'_blk_bn')(x)
    x = Scale(axis=concat_axis, name='conv'+str(final_stage)+'_blk_scale')(x)
    x = Activation('relu', name='relu'+str(final_stage)+'_blk')(x)

    x_fc = GlobalAveragePooling2D(name='pool'+str(final_stage))(x)
    x_fc = Dense(1000, name='fc6')(x_fc)
    x_fc = Activation('softmax', name='prob')(x_fc)

    model = Model(img_input, x_fc, name='densenet')

    if K.image_dim_ordering() == 'th':
      # Use pre-trained weights for Theano backend
      weights_path = 'imagenet_models/densenet161_weights_th.h5'
    else:
      # Use pre-trained weights for Tensorflow backend
      weights_path = 'imagenet_models/densenet161_weights_tf.h5'

    model.load_weights(weights_path, by_name=True)

    # Truncate and replace softmax layer for transfer learning
    # Cannot use model.layers.pop() since model is not of Sequential() type
    # The method below works since pre-trained weights are stored in layers but not in the model
    x_newfc = GlobalAveragePooling2D(name='pool'+str(final_stage))(x)
    x_newfc = Dense(num_classes, name='fc6')(x_newfc)
    x_newfc = Activation('softmax', name='prob')(x_newfc)

    model = Model(img_input, x_newfc)

    # Learning rate is changed to 0.001
    sgd = SGD(lr=1e-3, decay=1e-6, momentum=0.9, nesterov=True)
    model.compile(optimizer=sgd, loss='categorical_crossentropy', metrics=['accuracy'])

    return model



def conv_block(x, stage, branch, nb_filter, dropout_rate=None, weight_decay=1e-4):
    '''Apply BatchNorm, Relu, bottleneck 1x1 Conv2D, 3x3 Conv2D, and option dropout
        # Arguments
            x: input tensor 
            stage: index for dense block
            branch: layer index within each dense block
            nb_filter: number of filters
            dropout_rate: dropout rate
            weight_decay: weight decay factor
    '''
    eps = 1.1e-5
    conv_name_base = 'conv' + str(stage) + '_' + str(branch)
    relu_name_base = 'relu' + str(stage) + '_' + str(branch)

    # 1x1 Convolution (Bottleneck layer)
    inter_channel = nb_filter * 4  
    x = BatchNormalization(epsilon=eps, axis=concat_axis, name=conv_name_base+'_x1_bn')(x)
    x = Scale(axis=concat_axis, name=conv_name_base+'_x1_scale')(x)
    x = Activation('relu', name=relu_name_base+'_x1')(x)
    x = Convolution2D(inter_channel, 1, 1, name=conv_name_base+'_x1', bias=False)(x)

    if dropout_rate:
        x = Dropout(dropout_rate)(x)

    # 3x3 Convolution
    x = BatchNormalization(epsilon=eps, axis=concat_axis, name=conv_name_base+'_x2_bn')(x)
    x = Scale(axis=concat_axis, name=conv_name_base+'_x2_scale')(x)
    x = Activation('relu', name=relu_name_base+'_x2')(x)
    x = ZeroPadding2D((1, 1), name=conv_name_base+'_x2_zeropadding')(x)
    x = Convolution2D(nb_filter, 3, 3, name=conv_name_base+'_x2', bias=False)(x)

    if dropout_rate:
        x = Dropout(dropout_rate)(x)

    return x


def transition_block(x, stage, nb_filter, compression=1.0, dropout_rate=None, weight_decay=1E-4):
    ''' Apply BatchNorm, 1x1 Convolution, averagePooling, optional compression, dropout 
        # Arguments
            x: input tensor
            stage: index for dense block
            nb_filter: number of filters
            compression: calculated as 1 - reduction. Reduces the number of feature maps in the transition block.
            dropout_rate: dropout rate
            weight_decay: weight decay factor
    '''

    eps = 1.1e-5
    conv_name_base = 'conv' + str(stage) + '_blk'
    relu_name_base = 'relu' + str(stage) + '_blk'
    pool_name_base = 'pool' + str(stage) 

    x = BatchNormalization(epsilon=eps, axis=concat_axis, name=conv_name_base+'_bn')(x)
    x = Scale(axis=concat_axis, name=conv_name_base+'_scale')(x)
    x = Activation('relu', name=relu_name_base)(x)
    x = Convolution2D(int(nb_filter * compression), 1, 1, name=conv_name_base, bias=False)(x)

    if dropout_rate:
        x = Dropout(dropout_rate)(x)

    x = AveragePooling2D((2, 2), strides=(2, 2), name=pool_name_base)(x)

    return x


def dense_block(x, stage, nb_layers, nb_filter, growth_rate, dropout_rate=None, weight_decay=1e-4, grow_nb_filters=True):
    ''' Build a dense_block where the output of each conv_block is fed to subsequent ones
        # Arguments
            x: input tensor
            stage: index for dense block
            nb_layers: the number of layers of conv_block to append to the model.
            nb_filter: number of filters
            growth_rate: growth rate
            dropout_rate: dropout rate
            weight_decay: weight decay factor
            grow_nb_filters: flag to decide to allow number of filters to grow
    '''

    eps = 1.1e-5
    concat_feat = x

    for i in range(nb_layers):
        branch = i+1
        x = conv_block(concat_feat, stage, branch, growth_rate, dropout_rate, weight_decay)
        concat_feat = merge([concat_feat, x], mode='concat', concat_axis=concat_axis, name='concat_'+str(stage)+'_'+str(branch))

        if grow_nb_filters:
            nb_filter += growth_rate

    return concat_feat, nb_filter


def clear_all():
    """Clears all the variables from the workspace of the spyder application."""
    gl = globals().copy()
    for var in gl:
        if var[0] == '_': continue
        if 'func' in str(globals()[var]): continue
        if 'module' in str(globals()[var]): continue

        del globals()[var]

if __name__ == '__main__':

   # Example to fine-tune on 3000 samples from Cifar10

    img_rows, img_cols = 224, 224 # Resolution of inputs
    channel = 3
    batch_size = 8
    nb_epoch = 10
    num_classes = 3
    # Load cub200 data. Please implement your own load_data() module for your own dataset
    X_train1, Y_train1, X_valid1, Y_valid1 =  load_palette(img_rows, img_cols)
   
    
    
                
                
    import keras
    Y_train1 = keras.utils.to_categorical(Y_train1, num_classes)
    Y_valid1 = keras.utils.to_categorical(Y_valid1, num_classes)
    
    model = densenet161_model(img_rows=img_rows, img_cols=img_cols, color_type=channel, num_classes=num_classes)

    # Start Fine-tuning
    model.fit(X_train1, Y_train1,
              batch_size=batch_size,
              nb_epoch=nb_epoch,
              shuffle=True,
              verbose=1,
              validation_data=(X_valid1, Y_valid1),
              )
    
#
#    model.save("densenet161-Net30.model")
    # model.save_weights('densenet161-bird-weights.h5')                                                                                                                             
#    clear_all() 
    # Make predictions
    predictions_valid = model.predict(X_valid1, batch_size=batch_size, verbose=1)

    # Cross-entropy loss score
#    score = log_loss(Y_valid, predictions_valid)


    predictions_valid = model.predict(X_valid1, batch_size=batch_size, verbose=1)
#    scores=model.evaluate(X_valid,Y_valid ,verbose=0)
#    score = log_loss(Y_valid, predictions_valid)
    pred = np.round(predictions_valid)
    pred_acc = pred == Y_valid1
    pred_acc1 = (np.sum(pred) / np.size(Y_valid1,0))
#    print('Test score:', scores[0])
#    print('Test accuracy:', scores[1])
    print('Prediction accuracy:', pred_acc1)
    Y1=np.array(Y_valid1, dtype='float32')
    yy=np.array(Y_train1, dtype='float32')
    Y2=np.array(predictions_valid, dtype='float32')
    y1 = np.argmax(Y1 ,axis=1)
    y2 = np.argmax(Y2 ,axis=1)
  
    
    Confusion_Matrix=confusion_matrix(y1, y2)
    conf_our_head = open('conf_palet.pickle', 'wb')
    pickle.dump(Confusion_Matrix, conf_our_head)
    conf_our_head.close()
    
    our_head = open('our_palette.pickle', 'wb')
    pickle.dump(predictions_valid, our_head)
    our_head.close()
    train_path='dataset/train'
    test_path='dataset/test'
    
    
    for i in range(len(X_train1)):
        name='img'+str(i)+'.jpg'
        cv2.imwrite(os.path.join(train_path , name), X_train1[i,:,:,:])
        
    for i in range(len(X_valid1)):
        name='img'+str(i)+'.jpg'
        cv2.imwrite(os.path.join(test_path , name), X_valid1[i,:,:,:]) 
        
    np.savetxt('Valid.txt', y1, delimiter=',')   
    np.savetxt('Train.txt', yy, delimiter=',')  
    np.savetxt('Predict.txt', y2, delimiter=',')       
        
    
    # Y_train = open('Y_train.pickle', 'wb')
    # pickle.dump(Y_train1, Y_train)
    # Y_train.close()
    
    # Y_valid = open('Y_valid.pickle', 'wb')
    # pickle.dump(Y_valid1,Y_valid)
    # Y_valid.close()
    
    
    # Y2 = open('Y_predict.pickle', 'wb')
    # pickle.dump(y2, Y2)
    # Y2.close()
    
    # X_train = open('X_train.pickle', 'wb')
    # pickle.dump(X_train1, X_train)
    # X_train.close()
    
    # X_valid = open('X_valid.pickle', 'wb')
    # pickle.dump(X_valid1, X_valid)
    # X_valid.close()
    
    # with open('train.pickle','wb') as f:
    #     pickle.dump([X_train1, Y_train1],f)
    
    # with open('Y_predict.pickle','wb') as f:
    #     pickle.dump(y2,f)
    
