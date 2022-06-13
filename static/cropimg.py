import cv2
import glob
from matplotlib import pyplot as plt
import os
import random
import string

#  hàm tạo tên mới
def random_string(letter_count, digit_count):
    str1 = ''.join((random.choice(string.ascii_letters) for x in range(letter_count)))
    str1 += ''.join((random.choice(string.digits) for x in range(digit_count)))

    sam_list = list(str1)  # it converts the string to list.
    random.shuffle(sam_list)  # It uses a random.shuffle() function to shuffle the string.
    final_string = ''.join(sam_list)
    return final_string

#  dẫn thư mục ảnh crop được lưu
path = "D:\HocTapEPU\CBIR\CBIRThi\static\images"

for name in glob.glob('D:\\HocTapEPU\\CBIR\\CBIRThi\\static\\imagestest1 - Copy (7)\\*'):
        print(name)
        I = cv2.imread(name)
        name_img_new = random_string(4, 4) + ".jpg"
        # cv2.imwrite(os.path.join(path, name_img_new), I[100:315,170:260])
        cv2.imwrite(os.path.join(path, name_img_new), I)
