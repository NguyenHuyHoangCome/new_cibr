import os
import pandas as pd
import cv2
import numpy as np
from my_tools.lfcnn.features_cbir_run import cnn_one_image,feature_extraction_img_809,norm_minmax_feature,loadthamso
import pickle
from sklearn.decomposition import PCA
import  csv
from sklearn.preprocessing import StandardScaler

from sklearn.decomposition import FastICA
from sklearn.decomposition import TruncatedSVD


b_loaded_cnn = False

def dimension_reduction(file_index):
    df = pd.read_csv(file_index)
    X = df[df.columns[1:]]
    imgpath = df['label']
    print(imgpath.shape)
    pca = PCA(n_components=445)
    pcalfCNN = pca.fit_transform(X)

    pickle.dump(pca, open("model/pca_new.pkl", "wb"))
    dflfcnn = pd.DataFrame(pcalfCNN)
    dflfcnn['imgpath'] =imgpath
    dflfcnn.to_csv("lfcnn/filecsv/lfcnn_pca_n.csv",index=False)

def dimension_reduction_TruncatedSVD(file_index):
    df = pd.read_csv(file_index)
    X = df[df.columns[1:]]
    imgpath = df['label']

    ICA = TruncatedSVD(n_components=300, random_state=226)
    X = ICA.fit_transform(X)

    pickle.dump(ICA, open("model/SVD_new.pkl", "wb"))
    dflfcnn = pd.DataFrame(X)
    dflfcnn['imgpath'] =imgpath
    dflfcnn.to_csv("lfcnn/filecsv/lfcnn_SVD_n.csv",index=False)


def index_one(imagepath):
    # pca_reload = pickle.load(open("my_tools/model/pca_new.pkl", 'rb'))
    ica_reload = pickle.load(open("my_tools/model/SVD_new.pkl", 'rb'))

    global vCNN
    v_min, v_max, v_mean, v_var, v_std = loadthamso('my_tools/lfcnn/filecsv/thongke.csv')
    I = cv2.imread(imagepath)
    if I is not None:
        vCNN = cnn_one_image(I)

    vlf = feature_extraction_img_809(I)

    vlf_n=norm_minmax_feature(np.asarray([vlf]),v_min,v_max)[0]

    vlf_cnn= np.concatenate((vlf_n,vCNN), axis=None)
    print(vlf_cnn)

    result_new = ica_reload.transform(vlf_cnn.reshape(1, -1))[0]

    # print(result)
    # #
    # # create the csv writer
    # writer = csv.writer(f)
    #
    # # write a row to the csv file
    # writer.writerow(result_new)
    #
    # # close the file
    # f.close()

    return result_new

# if __name__ == '__main__':
    # giảm chiều dữ liệu
    # dimension_reduction('lfcnn/filecsv/lfcnn.csv')
    # dimension_reduction_FastICA('lfcnn/filecsv/lfcnn.csv')
    #dimension_reduction_TruncatedSVD('lfcnn/filecsv/lfcnn.csv')

    # # test ảnh

    # print(index_one('D:\\HocTapEPU\\CBIR\\CBIRThi\\static\\images\\00QdGZgE5S9Z.jpg'))



