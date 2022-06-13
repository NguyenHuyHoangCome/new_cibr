from fileinput import filename
from flask import Flask, render_template, request , jsonify,redirect,url_for
import time
import os
import cv2
from CenterFace import CenterFace
from my_tools.index import index_one
from my_tools.search import Search


app = Flask(__name__)
#general parameters
UPLOAD_FOLDER = 'static/upload_image'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.jinja_env.variable_start_string = '{['
app.jinja_env.variable_end_string = ']}'
detail12 = {}

@app.route('/')
def index():
    return render_template('test.html' )

@app.post('/upload')
def upload():
    detail = []
    # Saving the Uploaded image in the Upload folder
    file = request.files['image']
    new_file_name = str(
        str(time.time()) + '.jpg'
    )
    file.save(os.path.join(
            app.config['UPLOAD_FOLDER'],new_file_name
        )
    )
    # Trích xuất vectơ đối tượng từ các hình ảnh đã tải lên và thêm vectơ này vào cơ sở dữ liệu của chúng tôi
   
    ##############################################
    a = str(UPLOAD_FOLDER + '/' + new_file_name)
    I = cv2.imread(a)
    frame = I
    h, w = frame.shape[:2]
    landmarks = True
    centerface = CenterFace(landmarks=landmarks)
    if landmarks:
        dets, lms = centerface(frame, h, w, threshold=0.35)
    else:
        dets = centerface(frame, threshold=0.35)
    area = 0
    cropped = None
    list_crop = []
    for det in dets:
        boxes, score = det[:4], det[4]
        cropped = frame[int(boxes[1]):int(boxes[3]), int(boxes[0]):int(boxes[2]), :]
        list_crop.append(cropped)
    if len(list_crop)!=0:
        I = cv2.resize(list_crop[0], (160, 160))
        cv2.imwrite("hoang.jpg",I)
        print("hoang save")

    else:
        pass
    ##############################################
    #features = index_one(str(UPLOAD_FOLDER + '/' + new_file_name) )
    features = index_one("hoang.jpg")
    print(str(UPLOAD_FOLDER + '/' + new_file_name))
    searcher = Search('my_tools/lfcnn/filecsv/lfcnn_SVD_n.csv')
    #searcher = Search('my_tools/lfcnn/filecsv/lfcnn_pca.csv')
    results = searcher.search(features)
    for (score, pathImage) in results:
        detail.append(
        {"image": str(pathImage), "score": str(score)}
    )
    detail12['data']=detail
    #return redirect(url_for('index', filename=str(UPLOAD_FOLDER + '/' + new_file_name)))
    return render_template("test.html", filename=str(UPLOAD_FOLDER + '/' + new_file_name))

@app.route('/api/image', methods=['GET', 'POST'])
def image():
    #print(detail12)
    return jsonify(detail12)

    

if __name__ == '__main__':
    app.run(debug=True)