from edh import get_edh_37
from gcm import get_gcm_81
from gwt import gwtfeature
from gist import compute_gist_descriptor
from lbp import lbpfeature
import numpy as np
import csv
import glob
# from objcls import load_model_cnn, extract_cnn
import pandas as pd

from tensorflow.keras.applications.vgg16 import VGG16
from tensorflow.keras.models import  Model
from keras.preprocessing.image import load_img
from keras.preprocessing.image import img_to_array
from keras.applications.vgg16 import preprocess_input
import numpy as np
import glob
import csv

# Ham tao model
def get_extract_model():
    vgg16_model = VGG16(weights="imagenet")
    extract_model = Model(inputs=vgg16_model.inputs, outputs=vgg16_model.get_layer('fc1').output)
    return extract_model

#  hàm lấy vector ảnh
def extract_vector(model, image_path):
    print("Xu ly : ", image_path)
    img = load_img(image_path, target_size=(224, 224))
    x = img_to_array(img)
    x = np.expand_dims(x, axis=0)
    x = preprocess_input(x)
    flatten = model.predict(x)
    vCNN = list(flatten[0])

    vCNN = np.array(vCNN).reshape(1, -1)
    return vCNN
    return

def feature_extraction_img_809(img_path):

    gwt = gwtfeature(img_path)

    lbp = lbpfeature(img_path)

    edh = get_edh_37(img_path)

    gcm = get_gcm_81(img_path)

    gist = compute_gist_descriptor(img_path)
    print("đã xử lý ảnh ",img_path)

    return np.concatenate((gcm, edh, gwt, lbp, gist), axis=None)


header = [i for i in range(4905)]
header_lst = ["label", *header]


# def feature_extraction_csv(imgpath_folder):
#     with open('datasetlfcnn.csv', 'w') as f:
#         writer = csv.writer(f)
#         writer.writerow(header_lst)
#         for f in glob.glob(imgpath_folder+"\\*"):
#
#             name = f.split("\\")[6]
#             print(name)
#             if name == "bottle" or name == "Bottle":
#
#                 for b in glob.glob("D:\\HocTapEPU\\LabEPU\\5lfcnn\\datamoi\\datatrain\\Bottle\\*"):
#
#                     vector = feature_extraction_img_809(b)
#                     nor = normalization(vector, thamso)
#                     cnn = extract_vector(model,b)
#                     vector_nor_cnn = np.concatenate((nor,cnn), axis=None)
#                     lv_ncnn = list(vector_nor_cnn)
#                     print(len(lv_ncnn))
#
#                     writer.writerow([name, * lv_ncnn])
#                     print("Đã ghi bottle xong vào csv !")
#             if name== "Can" or name == "can":
#                 print("h")
#                 for c in glob.glob("D:\\HocTapEPU\\LabEPU\\5lfcnn\\datamoi\\datatrain\\Can\\*"):
#
#
#                     nor = normalization(vector, thamso)
#                     cnn = extract_vector(model,c)
#                     vector_nor_cnn = np.concatenate((nor,cnn), axis=None)
#                     lv_ncnn = list(vector_nor_cnn)
#                     writer.writerow([name, * lv_ncnn])
#                     print("Đã ghi can xong vào csv !")
#             if name == "milk" or name == "Milk":
#                 print("n")
#                 for m in glob.glob("D:\\HocTapEPU\\LabEPU\\5lfcnn\\datamoi\\datatrain\\Milk\\*"):
#
#                     nor = normalization(vector, thamso)
#                     cnn = extract_vector(model, b)
#                     vector_nor_cnn = np.concatenate((nor, cnn), axis=None)
#                     lv_ncnn = list(vector_nor_cnn)
#                     writer.writerow([name, *lv_ncnn])
#                     print("Đã ghi milk xong vào csv !")
#         print("Done!")



# def loadthamso(path_thamso):
#     df = pd.read_csv(path_thamso)
#     return df.drop("STT", axis=1)
#
# def normalization(vlf,thamsothongke):
#     vlf_nom = []
#
#     for i in range(len(vlf)):
#         nm_min_max = (vlf[i] - thamsothongke['min'][i]) / (thamsothongke['max'][i] - thamsothongke['min'][i])
#         vlf_nom.append(nm_min_max)
#     return np.array(vlf_nom)


# if __name__ == '__main__':
#
#     # đường dẫn folder chứa thư mục train : D:\\HocTapEPU\\LabEPU\\5lfcnn\\datamoi\\datatrain
#     model = get_extract_model()
#
#     thamso = loadthamso('D:\\HocTapEPU\\LabEPU\\5lfcnn\\cbir\\thamsothongke.csv')
#
#     feature_extraction_csv("D:\\HocTapEPU\\LabEPU\\5lfcnn\\datamoi\\datatrain")