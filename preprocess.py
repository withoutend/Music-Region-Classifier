import matplotlib.pyplot as plt
from scipy.fftpack import fft,ifft
from scipy.io import wavfile # get the api
import math
import cmath
import numpy as np
import pydub
from pydub import AudioSegment
from os import path
import wave
import librosa
#import pyfftw
#import seaborn
def mp3_to_wav(file, path):
    path=path+'/'
    sound = AudioSegment.from_mp3(path+file)
    sound.export(path+file[:-4]+'.wav', format="wav")
def convert_all():
    path = "/home/learning/Final Project/Data"
    dirs = os.listdir( path )

    # 输出所有文件和文件夹

    for dirc in dirs:
        if(os.path.isdir(path+'/'+dirc)):
            sub_folder=os.listdir(path+'/'+dirc)
            print('move to '+path+'/'+dirc)
            for file in sub_folder:
                if(file[-4:]=='.mp3'):#found mp3 file
                    if((file[:-4]+'.wav') not in sub_folder):#if no corresponing wav file found
                        print(file+'...convert...')
                        mp3_to_wav(file,path+'/'+dirc)#convert to wav
                    #os.remove(path+'/'+dirc+'/'+file)
                    
def fft_array(filename,channels=2,frame_length=40,overlap=10,bit_depth=16,white_noise_std_in_dB=-10,interval=[0,-1]):
    print("")
    mag=[]
    phase=[]
    data, fs = librosa.load(filename+'.wav', sr=44100,mono=True)
    #fs, data = wavfile.read(filename+'.wav')
    
    #get data in interval
    if(interval[1]==-1):
        square_sum=[item for item in data[int(fs*interval[0]):]]
    elif(interval[1]==0):
        square_sum=[item for item in data[int(len(data)*(interval[0]+1)/6):int(len(data)*(interval[0]+1)/6)+int(2.56*fs)]]
    else:
        square_sum=[item for item in data[int(fs*interval[0]):int(fs*interval[1])]]
    rms=np.array(square_sum)
    #print(rms)
    
    #add white noise
    if(white_noise_std_in_dB<=0):
        rms+=np.random.normal(0, (2**(bit_depth-1))*(10**(white_noise_std_in_dB/10)), size=rms.shape[0])
    
    
    num_sample=int(fs*frame_length/1e3)
    #window = np.hamming(num_sample)
    begin_frame=0
    
    #fft frame by frame
    while((begin_frame+fs*frame_length/1e3)<=rms.size):
        #print(begin_frame)
        y=rms[begin_frame:int(begin_frame+fs*frame_length/1e3)]
        
        #apply hamming window
        #print('fftw')
        yy=fft(y)#(np.multiply(window, y), axis=0)

        yf2 = abs(yy)[0:int(num_sample/2+1)] 

        yp=np.array([cmath.phase(item) for item in yy])
        yp2=yp[0:int(num_sample/2+1)]
        
        mag.append(yf2)
        phase.append(yp2)
        begin_frame+=int(fs*(frame_length-overlap)/1e3)

    return np.array(mag),np.array(phase)

def ifft_array(mag,phase,channels=2,frame_length=40,overlap=10,bit_depth=16,fs=44100):
    bit_type={8:np.uint8,16:np.int16,32:np.int32}
    data=[]
    num_sample=(mag[0].size-1)*2
    inverse_window=1./np.hamming(num_sample)
    frame_mag=[]
    frame_phase=[]
    frame_complex=[]
    for frame in range(mag.shape[0]):
        
        #reconstruct complex array
        for item in mag[frame]:
            frame_mag.append(item)
        for item in mag[frame][::-1][1:-1]:
            frame_mag.append(item)
    
        for item in phase[frame]:
            frame_phase.append(item)
        for item in phase[frame][::-1][1:-1]:
            frame_phase.append(-item)  
        
        for i in range(len(frame_mag)):
            frame_complex.append(cmath.rect(frame_mag[i],frame_phase[i]))
        
        #apply ifft 
        if(frame!=0):
            data=data[0:(len(data)-int(num_sample*overlap/frame_length))]
        
        #apply inverse hamming window
        data.extend(ifft(frame_complex, axis=0).real)#np.multiply(inverse_window,ifft(frame_complex, axis=0).real)
        
        #clear memory
        frame_mag=[]
        frame_phase=[]
        frame_complex=[]
    
    #get real part, convert to proper bit-depth  
    maxv = np.iinfo(bit_type[bit_depth]).max
    #print(bit_type[bit_depth])
    return np.array([maxv*item for item in data],dtype=bit_type[bit_depth])#/channels ,dtype=np.int16

def low_pass_filter(mag,phase,cut_off=500,decay_in_dB=-10,gain_in_dB=-3,frame_length=40,overlap=10,bit_depth=16,fs=44100):
    num_sample=(mag[0].size-1)*2
    
    #compute frequency resolution
    f_resolution=fs/num_sample
    
    #scalar array for lpf
    lpf_mag=[]
    lpf_phase=[]
    
    #transfer dB into ratio
    decay=10**(decay_in_dB/10)
    gain=10**(gain_in_dB/10)
    for i in range(mag[0].size):
        f=i*f_resolution
        if(f>=cut_off):
            lpf_mag.append(decay**math.log(f/cut_off,2))  
            #lpf_phase.append(-math.atan(f/cut_off))
        else:
            lpf_mag.append(1)
            #lpf_phase.append(0)
        #do noting on phase
        lpf_phase.append(0)
        
    return np.multiply(mag*gain,np.array(lpf_mag)),np.array(phase+np.array(lpf_phase))

'''
##### how to use this #####
-------use on track------
mag,phase=fft_array('p232_014',white_noise_std_in_dB=1,interval=[1,3])
mag,phase=low_pass_filter(mag,phase,cut_off=20000,decay_in_dB=-10)
array=ifft_array(mag,phase)
wavfile.write('tem.wav',rate=44100,data=array)

-----only use for generate label-------
mag,phase=fft_array('untitled',white_noise_std_in_dB=1,interval=[a,b])
give white noise std a positive int will erase the noise it add
a, b is the interval(in seconds) we want to apply
-----apply a high boost filter -------
mag,phase=low_pass_filter(mag,phase,gain_in_dB=3,cut_off=2000,decay_in_dB=10)
'''