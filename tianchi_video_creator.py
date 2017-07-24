import os
from PIL import Image
import numpy as np
# import scipy
import matplotlib.pyplot as plt

import cv2

import tianchi_data_loader as tcdl
import tianchi_data_processor as tcdp

def img_to_vid(pathlist, outpath): 
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    out = cv2.VideoWriter(outpath,fourcc, 10.0, (101,101))
    for path in pathlist: 
        frame = cv2.imread(path)
        out.write(frame)
        if os.path.exists(path):

            check = False
            while not check:
                try:
                    os.remove(path)
                except:
                    pass
                else:
                    check = not False

    out.release()
    cv2.destroyAllWindows()

def main(mode, sample_list, mini_batch_size = 500):
    # halo = [0.0, 21.0, 35.0, 99.9]
    # helo = [3002, 3201, 3212, 3078]
    for batch, (idx, attrs, labels) in enumerate(tcdl.iter_mini_batches(mode, sample_list, mini_batch_size=mini_batch_size)):
        x = np.reshape(attrs, (-1, 15, 4, 101, 101))
        x = np.transpose(x, (0,2,1,3,4))
        for k in range(np.shape(x)[0]):
            print('mode: %s, batch: %d, idx: %d'%(mode, batch+1, idx[k]))
            for i in range(4): 
                pathlist = []
                for l in range(15):
                    t_m = x[k,i,l,:,:]
                    img = Image.fromarray(t_m.astype(np.uint8))
                    path = '../vid/%s_%d_%.2f_h%d_t%d.jpg'%(mode, idx[k],labels[k,0],i+1,l+1)
                    img.save(path)
                    pathlist += [path]

                outpath = '../vid/%s_%d_%.2f_h%d.avi' % (mode, idx[k],labels[k,0],i+1)
                img_to_vid(pathlist, outpath)
        # break

def main_2(mode, sample_list, mini_batch_size = 500):
    # halo = [0.0, 21.0, 35.0, 99.9]
    # helo = [3002, 3201, 3212, 3078]
    for batch, (idx, attrs, labels) in enumerate(tcdl.iter_mini_batches(mode, sample_list, mini_batch_size=mini_batch_size)):
        x = np.reshape(attrs, (-1, 15, 4, 101, 101))
        for k in range(np.shape(x)[0]):
            print('mode: %s, batch: %d, idx: %d'%(mode, batch+1, idx[k]))
            pathlist = []
            for i in range(15): 
                h_m = np.zeros((0, 101 * 2))
                for j in range(2): 
                    v_m = np.zeros((101, 0))
                    for l in range(2): 
                        v_m = np.hstack((v_m, x[k,i,2*j+l,:,:]))
                    h_m = np.vstack((h_m, v_m))
                    img = Image.fromarray(h_m.astype(np.uint8))
                    path = '../vid/%s_%d_%.2f_%d.jpg'%(mode, idx[k],labels[k,0],i+1)
                    img.save(path)
                    pathlist += [path]

            outpath = '../vid/%s_%d_%.2f.avi' % (mode, idx[k],labels[k,0])
            img_to_vid(pathlist, outpath)

if __name__ == '__main__': 
    sample_dict = tcdp.random_select_samples()
    mode = 'train'
    main(mode, sample_dict[mode])
    mode = 'testA'
    main(mode, sample_dict[mode])