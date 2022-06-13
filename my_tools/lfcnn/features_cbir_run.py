import cv2
import numpy as np

from my_tools.lfcnn.edh import get_edh_37
from my_tools.lfcnn.gcm import get_gcm_81
from my_tools.lfcnn.gwt import gwtfeature
from my_tools.lfcnn.gist import compute_gist_descriptor
from my_tools.lfcnn.lbp import lbpfeature
import pickle

import os

import pandas as pd

from tensorflow.keras.applications.vgg16 import VGG16
from tensorflow.keras.models import Model
from tensorflow.keras.preprocessing.image import load_img
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.applications.vgg16 import preprocess_input


# Ham tao model
def get_extract_model():
    vgg16_model = VGG16(weights="imagenet")
    extract_model = Model(inputs=vgg16_model.inputs, outputs=vgg16_model.get_layer('fc1').output)
    return extract_model


#  hàm lấy vector ảnh
def extract_vector(model, image_path):
    # print("Xu ly : ", image_path)
    img = load_img(image_path, target_size=(224, 224))
    x = img_to_array(img)
    x = np.expand_dims(x, axis=0)
    x = preprocess_input(x)
    flatten = model.predict(x)
    vCNN = list(flatten[0])

    vCNN = np.array(vCNN).reshape(1, -1)

    print("độ dài", vCNN.shape)
    return vCNN


def feature_extraction_img_809(img):
    gwt = gwtfeature(img)  # tự động đổi sang gray bên trong
    lbp = lbpfeature(img)  # tự động đổi sang gray bên trong
    if len(img.shape) == 3:
        Ig = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    else:
        Ig = img.copy()
    edh = get_edh_37(Ig)  # bắt buộc đầu vào là ảnh gray

    if len(img.shape) == 3:
        Ig = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    else:
        Ig = img.copy()
    gist = compute_gist_descriptor(Ig)  # bắt buộc đầu vào là ảnh gray
    if len(img) < 3:
        imgColor = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    else:
        imgColor = img.copy()
    gcm = get_gcm_81(imgColor)  # bắt buộc đầu vào là ảnh mầu BGR, tự động đổi sang HSV bên trong

    vec = np.concatenate((gcm, edh, gwt, lbp, gist), axis=None)
    vec = np.array(vec)
    return vec


header = [i for i in range(4905)]
header_lst = ["label", *header]


def load_models():
    global b_loaded_cnn
    global m_model
    if b_loaded_cnn == False:
        b_loaded_cnn = True
        model = get_extract_model()
        m_model = model
        return model

def feature_extraction_csv(img_folder):
    m_cnn = load_models()
    l_File = []
    l_Label = []
    for folder, subfolders, filenames in os.walk(img_folder):
        for filename in filenames:
            is_jpg = filename.lower().endswith(('.jpg'))
            is_jpeg = filename.lower().endswith(('.jpeg'))
            is_png = filename.lower().endswith(('.png'))
            is_bmp = is_jpeg = filename.lower().endswith(('.bmp'))
            if (is_jpg or is_jpeg or is_png or is_bmp):
                fileFullname = os.path.join(folder, filename)
                lab = "static/images/" + filename
                print("lab",lab)

                l_File.append(fileFullname)
                l_Label.append(lab)
    print(l_File)
    print(l_Label)

    l_vectors = []
    l_cnn_vectors = []
    l_actual_label = []
    for i in range(len(l_File)):
        # for i in range(3):
        fullFileName = l_File[i]
        label = l_Label[i]
        I = cv2.imread(fullFileName)
        if I is None:
            continue
        print(i, fullFileName)
        vector = feature_extraction_img_809(I)
        # nor = normalization(vector, thamso)
        # print(vector)
        l_vectors.append(vector)
        if len(I.shape) < 3:
            imgColor = cv2.cvtColor(I, cv2.GRAY2BGR)
        else:
            imgColor = I
        x = cv2.resize(imgColor, (224, 224))
        x = np.expand_dims(x, axis=0)
        x = preprocess_input(x)

        flatten = m_cnn.predict(x)
        vCNN = list(flatten[0])
        vCNN = np.array(vCNN).reshape(1, -1)[0]
        # print('vCNN:',vCNN)
        l_cnn_vectors.append(vCNN)
        l_actual_label.append(label)

    aLF = np.asarray(l_vectors)
    aCNN = np.asarray(l_cnn_vectors)
    dim = l_vectors[0].shape[0]

    # tìm min,max,mean,var,std mỗi côt
    v_min = np.zeros(dim, dtype='float32')
    v_max = np.zeros(dim, dtype='float32')
    v_mean = np.zeros(dim, dtype='float32')
    v_var = np.zeros(dim, dtype='float32')
    v_std = np.zeros(dim, dtype='float32')

    for i in range(dim):
        v_min[i] = np.min(aLF[:, i])
        v_max[i] = np.max(aLF[:, i])
        v_mean[i] = np.mean(aLF[:, i])
        v_var[i] = np.var(aLF[:, i])
        v_std[i] = np.std(aLF[:, i])

    print(v_min)
    print(v_max)
    print(v_mean)
    print(v_var)
    print(v_std)

    return aLF, aCNN, l_actual_label, v_min, v_max, v_mean, v_var, v_std


def norm_minmax_feature(aLF, v_min, v_max):
    aLF_n = np.zeros((aLF.shape[0], aLF.shape[1]), dtype='float32')
    for k in range(aLF.shape[0]):
        for i in range(aLF.shape[1]):
            if (v_max[i] - v_min[i]) > 0:
                aLF_n[k, i] = (aLF[k, i] - v_min[i]) / (v_max[i] - v_min[i])
            else:
                aLF_n[k, i] = 0.5
    return aLF_n


def lf_cnn_to_csv(csv_feature_filename, csv_param_filename, aLF_n, aCNN, l_label, v_min, v_max, v_mean, v_var, v_std):
    header = [i for i in range(aLF_n.shape[1] + aCNN.shape[1])]
    header_lst = ["label", *header]
    f = open(csv_feature_filename, 'w')
    sz = ""
    for j in range(len(header_lst)):
        sz = sz + str(header_lst[j])
        if j < len(header_lst) - 1:
            sz = sz + ","
    f.write(sz + '\n')

    for k in range(aLF_n.shape[0]):
        v_lf_cnn = np.concatenate((aLF_n[k, :], aCNN[k, :]), axis=None)
        lv_lf_cnn = list(v_lf_cnn)
        print(len(lv_lf_cnn))
        label = l_label[k]
        # writer.writerow([label, * lv_lf_cnn])
        lv = [label, *lv_lf_cnn]

        sz = ""
        for j in range(len(lv)):
            sz = sz + str(lv[j])
            if j < len(lv) - 1:
                sz = sz + ","
        f.write(sz + '\n')
    f.close()

    header_lst = ["min", "max", "mean", "var", "std"]
    f = open(csv_param_filename, 'w')
    # writer = csv.writer(f)
    # writer.writerow(header_lst)
    sz = ""
    for j in range(len(header_lst)):
        sz = sz + header_lst[j]
        if j < len(header_lst) - 1:
            sz = sz + ","
    f.write(sz + '\n')
    for i in range(aLF_n.shape[1]):
        v = np.concatenate((v_min[i], v_max[i], v_mean[i], v_var[i], v_std[i]), axis=None)
        lv = list(v)
        sz = ""
        for j in range(len(lv)):
            sz = sz + str(lv[j])
            if j < len(lv) - 1:
                sz = sz + ","
        f.write(sz + '\n')
    f.close()


def lf_to_csv(csv_feature_filename, csv_param_filename, aLF_n, l_label, v_min, v_max, v_mean, v_var, v_std):
    header = [i for i in range(aLF_n.shape[1])]
    header_lst = ["label", *header]
    f = open(csv_feature_filename, 'w')
    sz = ""
    for j in range(len(header_lst)):
        sz = sz + str(header_lst[j])
        if j < len(header_lst) - 1:
            sz = sz + ","
    f.write(sz + '\n')

    for k in range(aLF_n.shape[0]):
        lv_lf = list(aLF_n[k, :])
        print(len(lv_lf))
        label = l_label[k]
        # writer.writerow([label, * lv_lf_cnn])
        lv = [label, *lv_lf]

        sz = ""
        for j in range(len(lv)):
            sz = sz + str(lv[j])
            if j < len(lv) - 1:
                sz = sz + ","
        f.write(sz + '\n')
    f.close()

    header_lst = ["min", "max", "mean", "var", "std"]
    f = open(csv_param_filename, 'w')
    # writer = csv.writer(f)
    # writer.writerow(header_lst)
    sz = ""
    for j in range(len(header_lst)):
        sz = sz + header_lst[j]
        if j < len(header_lst) - 1:
            sz = sz + ","
    f.write(sz + '\n')
    for i in range(aLF_n.shape[1]):
        v = np.concatenate((v_min[i], v_max[i], v_mean[i], v_var[i], v_std[i]), axis=None)
        lv = list(v)
        sz = ""
        for j in range(len(lv)):
            sz = sz + str(lv[j])
            if j < len(lv) - 1:
                sz = sz + ","
        f.write(sz + '\n')
    f.close()


def cnn_to_csv(csv_cnn_filename, aCNN, l_label):
    header = [i for i in range(aCNN.shape[1])]
    header_lst = ["label", *header]
    f = open(csv_cnn_filename, 'w')
    sz = ""
    for j in range(len(header_lst)):
        sz = sz + str(header_lst[j])
        if j < len(header_lst) - 1:
            sz = sz + ","
    f.write(sz + '\n')

    for k in range(aCNN.shape[0]):
        v_cnn = aCNN[k, :]
        lv_cnn = list(v_cnn)
        print(len(lv_cnn))
        label = l_label[k]
        # writer.writerow([label, * lv_lf_cnn])
        lv = [label, *lv_cnn]

        sz = ""
        for j in range(len(lv)):
            sz = sz + str(lv[j])
            if j < len(lv) - 1:
                sz = sz + ","
        f.write(sz + '\n')
    f.close()


def loadthamso(path_thamso):
    df = pd.read_csv(path_thamso)
    dim = df.shape[0]
    v_min = np.zeros(dim, dtype='float32')
    v_max = np.zeros(dim, dtype='float32')
    v_mean = np.zeros(dim, dtype='float32')
    v_var = np.zeros(dim, dtype='float32')
    v_std = np.zeros(dim, dtype='float32')

    for i in range(dim):
        v_min[i] = df['min'][i]
        v_max[i] = df['max'][i]
        v_mean[i] = df['mean'][i]
        v_var[i] = df['var'][i]
        v_std[i] = df['std'][i]
    return v_min, v_max, v_mean, v_var, v_std


def cnn_one_image(I):
    if b_loaded_cnn == False:
        load_models()

    if len(I.shape) < 3:
        imgColor = cv2.cvtColor(I, cv2.GRAY2BGR)
    else:
        imgColor = I
    x = cv2.resize(imgColor, (224, 224))
    x = np.expand_dims(x, axis=0)
    x = preprocess_input(x)

    flatten = m_model.predict(x)
    vCNN = list(flatten[0])
    vCNN = np.array(vCNN).reshape(1, -1)[0]

    return vCNN


using_test = 1  # =0 neu khong test
b_loaded_cnn = False
# if using_test == 1:

#     #train dữ liệu
#     aLF,aCNN,l_actual_label, v_min,v_max,v_mean,v_var,v_std=feature_extraction_csv('static\images')
#     print('feature_extraction_csv done!')
#     aLF_n=norm_minmax_feature(aLF,v_min,v_max)
#     print('norm_minmax_feature done!')
#     lf_cnn_to_csv("lfcnnnew.csv","thongkenew.csv", aLF_n,aCNN,l_actual_label,v_min,v_max,v_mean,v_var,v_std)
#     print('lf_cnn_to_csv done!')

    # # lf_to_csv("lf.csv","thongke.csv", aLF_n,l_actual_label,v_min,v_max,v_mean,v_var,v_std)
    # print('lf_to_csv done!')
    # cnn_to_csv("cnn.csv",aCNN,l_actual_label)
    # print('cnn_to_csv done!')

    # v_min,v_max,v_mean,v_var,v_std=loadthamso('D:\\HocTapEPU\\CBIR\\CBIRThi\\my_tools\\lfcnn\\filecsv\\thongke.csv')
    # print(v_min)
    # print('<----v_min')
    # print(v_max)
    # print('<----v_max')
    # print(v_mean)
    # print('<----v_mean')
    # print(v_var)
    # print('<----v_var')
    # print(v_std)
    # print('<----v_std')
    # print('loadthamso done!')

    # I = cv2.imread('D:\\HocTapEPU\\CBIR\\CBIRThi\\0nkY0vb5jZ9a - Copy.jpg')
    # if I is not None:
    #     vCNN = cnn_one_image(I)
    #     print(vCNN)
    #     print('<---vCNN')
    #
    #     vlf = feature_extraction_img_809(I)
    #     print(vlf)
    #     print('<---5 low level feature')
    #
    #     vlf_n=norm_minmax_feature(np.asarray([vlf]),v_min,v_max)[0]
    #     print(vlf_n)
    #     print('<---5 low level feature normalized ')
    #
    #     vlf_cnn= np.concatenate((vlf_n,vCNN), axis=None)
    #     print(vlf_cnn)
    #
    #     pca_reload = pickle.load(open("D:\\HocTapEPU\\CBIR\\CBIRThi\\pca.pkl", 'rb'))
    #     result_new = pca_reload.transform(vlf_cnn.reshape(1, -1))
    #
    #     print("result_new",result_new.shape)
    #     print('<---5 low level feature normalized and CNN ')