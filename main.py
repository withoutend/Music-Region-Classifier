#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
from preprocess import *
import os
import keras
from keras.utils import to_categorical
from keras.layers import *
from keras.layers.convolutional import Conv2D
from keras.models import *
from keras.layers import Activation
from keras.layers.core import Dense, Flatten, Dropout
import matplotlib.image as mpimg
import pydot
from keras.utils import multi_gpu_model
import csv
import random


# In[2]:


import itertools
def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)

    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.tight_layout()


# In[5]:


def unison_shuffled_copies(a, b, c):
    assert len(a) == len(b)
    p = np.random.permutation(len(a))
    return a[p], b[p], c[p]


# In[183]:


def model_construct(shape,num_class):
    
    ini=keras.initializers.glorot_uniform(seed=None)

    model=Sequential()

    #model.add(keras.layers.BatchNormalization(input_shape=(shape[1],shape[2],shape[3])))
    model.add(Conv2D(64,[2,2],activation="elu",strides=(2,2),data_format="channels_last",input_shape=(shape[1],shape[2],shape[3])))#
    model.add(keras.layers.BatchNormalization())
    model.add(MaxPooling2D(pool_size=(2,2),padding='same',data_format=None))
    #model.add(Dropout(0.5))

    #model.add(keras.layers.BatchNormalization())
    model.add(Conv2D(128,[2,2], strides=(2,2),activation="elu"))
    model.add(keras.layers.BatchNormalization())
    model.add(MaxPooling2D(pool_size=(2,2),padding='same',data_format=None))
    model.add(Dropout(0.5))
    
    #model.add(keras.layers.BatchNormalization())
    model.add(Conv2D(256,[2,2],strides=2,activation="elu"))
    model.add(keras.layers.BatchNormalization())
    model.add(MaxPooling2D(pool_size=(2,2), padding='same',data_format=None))
    model.add(Dropout(0.5))
    
    #model.add(keras.layers.BatchNormalization())
    model.add(Conv2D(512,[2,2],strides=(2,2),activation="elu"))
    model.add(keras.layers.BatchNormalization())
    model.add(MaxPooling2D(pool_size=(2,2), padding='same',data_format=None))
    model.add(Dropout(0.5))

    #model.add(keras.layers.BatchNormalization())
    model.add(Flatten())
    model.add(Dense(1024,activation="relu"))
    #model.add(keras.layers.BatchNormalization())
    
    model.add(Dense(num_class,activation="sigmoid"))#sigmoid
    
    model.summary()
    
    rmsp= keras.optimizers.RMSprop(lr=0.001, rho=0.9, epsilon=None, decay=0.1)
    model.compile(loss='binary_crossentropy',optimizer='adam',metrics=['accuracy'])#binary_crossentropy
    
    return model


# In[7]:


def load_data(mag_data,label_data,name_data):
    
    #audio file path
    path = "/home/learning/Final Project/Data"
    dirs = os.listdir( path )
    label=0
    country_dict={}
    num_per_class=[0]
    count=0
    
    for dirc in dirs:
        if(os.path.isdir(path+'/'+dirc) and dirc!='.ipynb_checkpoints'):
            sub_folder=os.listdir(path+'/'+dirc)
            
            #construct dictionary of label(int) and country(string)
            country_dict[label]=dirc   
            
            print('move to '+path+'/'+dirc)
            #print(label)
            for file in sub_folder:
                
                #if it is wave file
                if(file[-4:]=='.wav'):#if(file=='Alan_Walker_-_Different_World_feat._Sofia_Carson_K-391_CORSAK_Lyric_Video-m-PJmmvyP10.wav'):#if(file[-4:]=='.wav'):
                    #this is mew song
                    #print(name_data)
                    if(file[-15:-4] not in name_data):
                        print('    '+file+'...fft...')
                        count+=1
                        
                        for i in range(5):
                            #record url
                            name_data.append(file[-15:-4])
                            #print(file[-15:-4])
                            #apply fft
                            mag,phase=fft_array(filename=path+'/'+dirc+'/'+file[:-4],frame_length=20,overlap=0,white_noise_std_in_dB=1,interval=[i,0])
                        
                            #record mag
                            mag_data.append(mag)
                        
                            #record country label
                            label_data.append([label])
                        
                    #already seen this song
                    else:
                        
                        #find its index
                        index=name_data.index(file[-15:-4])
                        
                        #update its country label
                        for i in range(6):
                            label_data[index+i].append(label)
                            print("++++++++++++++++")
                            #print(index+i)
                            #print(label_data[index+i])
            label+=1
            num_per_class.append(count)

    return country_dict,label,num_per_class


# In[153]:


def my_predict(model,path):
    mag_data=[]
    for file in os.listdir(path):
        if(file[-4:]=='.wav'):
            print('    '+file+'...fft...')      
            for i in range(5):
                    #apply fft
                    mag,phase=fft_array(filename=path+'/'+file[:-4],frame_length=20,overlap=0,white_noise_std_in_dB=1,interval=[i,0])        
                    #record mag
                    mag_data.append(mag)
                

    mag_data=np.array(mag_data)
    print(np.array(mag_data).shape)
    #print(mag_data[0][10])
    x=np.zeros([mag_data.shape[0],128,128])
    

    #crop it into 128*128 per song(2.56sec)
    for i in range(mag_data.shape[0]):
        for j in range(128):
            for k in range(128):
                x[i][j][k]=mag_data[i][j][k]
    x= x.reshape( x.shape[0],  x.shape[1],  x.shape[2], 1)
    prediction=model.predict(x)
        
    return np.mean(prediction,axis=0)




if __name__ == '__main__':
# In[9]:

    convert_all()
    #list can pass by reference, so I use list
    mag_data=[]
    label_data=[]
    name_data=[]

    country_dict,num_class,num_per_class=load_data(mag_data,label_data,name_data)

    print(country_dict)

    #-------- mag post-preprocess

    mag_data=np.array(mag_data)
    x_train=np.zeros([mag_data.shape[0],128,128])

    #crop it into 128*128 per song(2.56sec)
    for i in range(x_train.shape[0]):
        for j in range(128):
            for k in range(128):
                #print(i)
                x_train[i][j][k]=mag_data[i][j][k]
    x_train= x_train.reshape( x_train.shape[0],  x_train.shape[1],  x_train.shape[2], 1)

    #-------label post-preprocess

    y_train=np.zeros([len(label_data),num_class])

    #encode as one hot vector, more accurately, encode as nmulti-class one-hot vector
    for i in range(len(label_data)):
        for j in label_data[i]:
            y_train[i][j]=1


    x_train,y_train, name=unison_shuffled_copies(x_train,y_train,np.array(name_data))

    x_test=x_train[:400]
    y_test=y_train[:400]
    x_train=x_train[400:]
    y_train=y_train[400:]


    #defind the model
    model=model_construct( x_train.shape,num_class)

    #train the model
    history=model.fit( x_train,y_train,batch_size=1024,epochs=50,validation_split=0.1,shuffle=True,verbose=2)



    # Plot training & validation accuracy values
    plt.plot(history.history['acc'])
    plt.plot(history.history['val_acc'])
    plt.title('Model accuracy')
    plt.ylabel('Accuracy')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Validation'], loc='upper left')
    plt.legend(['Train', 'Validation'], loc='upper left')
    #plt.imshow(x_train[3].reshape(128,128).T,origin='lower')
    plt.show()

    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('Model loss')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Validation'], loc='upper left')
    plt.legend(['Train', 'Validation'], loc='upper left')
    #plt.imshow(x_train[3].reshape(128,128).T,origin='lower')
    plt.show()

    from sklearn.metrics import confusion_matrix
    prediction=model.predict(x_test)
    cm=confusion_matrix(y_test.argmax(axis=1), prediction.argmax(axis=1))


    plot_confusion_matrix(cm, classes=range(num_class),
                          title='Confusion matrix')


    metrics = model.evaluate(x_test, y_test, verbose=0)
    print("Metrics(Test loss & Test Accuracy): ")
    print(metrics) 


    model.save('CNN_bc_adam_bs1024_epoch50.h5')



    ones=0
    index_onne=np.zeros(num_class)
    one=np.zeros(num_class)
    two=0
    cou=np.zeros(num_class)
    for i in range(y_train.shape[0]):
        for itemY in y_train[i]:
            if(itemY==1):
                ones+=1
                
        if(ones==1):
            cou[int(np.unravel_index(np.argmax(item, axis=None), item.shape)[0])]+=1
            
        one[ones-1]+=1
        if(ones>=2):
            two+=1
        else:
            print(y_train[i])
        ones=0


    ones=0
    index_onne=np.zeros(num_class)
    one=np.zeros(num_class)
    two=0
    cou=np.zeros(num_class)
    for i in range(y_train.shape[0]):
        for itemY in y_train[i]:
            if(itemY==1):
                ones+=1
                
        if(ones==1):
            cou[int(np.unravel_index(np.argmax(y_train[i], axis=None), y_train[i].shape)[0])]+=1
            
        one[ones-1]+=1
        if(ones>=2):
            two+=1
        else:
            print(y_train[i])
        ones=0


    x_labels = np.array(['Turkey', 'Russia', 'United Kingdom','Indonesia',  'United States', 'Thailand', 'Mexico',
                      'Germany', 'Philippines','South Korea', 'India','France','Taiwan', 'Japan','Poland','Brazil'])
    plt.bar(range(num_class),cou)
    plt.title('unique song class distribution')
    plt.xticks(range(num_class), x_labels, rotation='vertical')
    plt.ylabel('songs')
    plt.show()

    plt.title('popular song distribution')
    plt.ylabel('shared songs')
    plt.xlabel('countries')#
    plt.bar(range(1,num_class+1),one,1/1.5)

    p=my_predict(model,path='/home/learning/Final Project/Hachi2')

    p=model.predict(x_train[0:2])
    print(y_train[0])

    plt.bar(range(num_class),p[0])
    plt.title('Most posiibly region')
    plt.xticks(range(num_class), x_labels, rotation='vertical')
    #plt.legend()
    plt.show()
    print(country_dict)


    taiwan_share=0
    song_share=np.zeros(num_class)
    taiwan_one=np.zeros(num_class)
    song_in_taiwan=0
    array_t=[]
    for i in range(y_train.shape[0]):
        if(y_train[i][12]==1):
            song_in_taiwan+=1
            for itemY in y_train[i]:
                if(itemY==1):
                    taiwan_share+=1
            song_share[taiwan_share]+=1
            array_t.append(taiwan_share)
            taiwan_share=0
    plt.title('shared song in Taiwan')
    plt.ylabel('shared countries')
    plt.xlabel('song number')#
    plt.bar(range(num_class),song_share)


    # In[ ]:


