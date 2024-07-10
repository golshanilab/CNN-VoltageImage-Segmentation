import numpy as np
import zarr
import matplotlib.pyplot as plt
import tifffile as tif
from scipy import signal
import os
import sys

def pb_correct(im_data,fs):
    
    medn = np.median(im_data,axis=0)
    im_data = (im_data - medn[np.newaxis,:])
    bfilt,afilt = signal.butter(2, 10,'highpass', fs=fs)
    clean_data = signal.filtfilt(bfilt,afilt,im_data, axis=0)
    
    return clean_data

def training_data_generation(stacks, frames):
    data_name = os.path.basename(os.path.dirname(stacks))
    ROIs_Filepath = os.path.normpath(stacks + '/ROIs.tif')
    RegisteredData_Filepath = os.path.normpath(stacks + '/Registered.tif')

    rois = tif.imread(ROIs_Filepath)
    rois = np.sum(rois, axis=0)
    imageData = tif.imread(RegisteredData_Filepath, aszarr=True)
    imageData = zarr.open(imageData)

    data_full_length, m, n = imageData.shape
    data_chunk_size = frames
    num_chunks = data_full_length // data_chunk_size

    for i in range(num_chunks):
        start = i * data_chunk_size
        end = start + data_chunk_size
        data = imageData[start:end]

        avg_image = np.average(data, axis=0)
        avg_image = avg_image/np.max(avg_image)
        filtered = pb_correct(data, 500)
        max_med = np.max(filtered, axis=0) - np.median(filtered, axis=0)
        max_med = max_med/np.max(max_med)

        cols = n//32 - 1
        rows = m//32 - 1

        for j in range(cols):
            for k in range(rows):
                x = j*32
                y = k*32
                roi = rois[y:y+64, x:x+64]
                mxm = max_med[y:y+64, x:x+64]
                avg = avg_image[y:y+64, x:x+64]

                tif.imwrite(os.path.normpath(stacks+"/training_data/images/"+data_name+"_set_"+str(i)+"_"+str(j)+"_"+str(k)+".tif"), np.array([avg,mxm]))
                tif.imwrite(os.path.normpath(stacks+"/training_data/masks/"+data_name+"_set_"+str(i)+"_"+str(j)+"_"+str(k)+".tif"), roi)



if __name__ == "__main__":

    ## inputs: data path and number of frames
    stacks_path = os.path.normpath(sys.argv[1])
    frames = int(sys.argv[2])
    training_data_generation(stacks_path, frames)



       




