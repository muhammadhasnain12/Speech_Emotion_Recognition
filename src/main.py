import os
from flask import Flask, request, render_template, send_from_directory, jsonify
import librosa
import numpy as np
import keras
from flask_pymongo import PyMongo
from tensorflow import keras

emotions={
  '01':'neutral',
  '02':'calm',
  '03':'happy',
  '04':'sad',
  '05':'angry',
  '06':'fearful',
  '07':'disgust',
  '08':'surprised'
}

app = Flask(__name__)

# Making connection string with mongo db
app.config["MONGO_URI"] = "mongodb://localhost:27017/emotion_recognition"
mongodb_client = PyMongo(app)
db = mongodb_client.db

# Load Modael
MODEL_PATH = 'models/Emotion_Voice_Detection_Model.h5'

APP_ROOT = os.path.dirname(os.path.abspath(__file__))

# ================= Model prediction code start===============

def emotion_recognition(voice, main_dir):
    lst = []
    for subdir, dirs, files in os.walk(main_dir):
        for file in files:
            X, sample_rate = librosa.load(voice, res_type='kaiser_fast')
            mfccs = np.mean(librosa.feature.mfcc(y=X, sr=sample_rate, n_mfcc=40).T, axis=0)
            file = int(file[7:8]) - 1
            arr = mfccs, file
            lst.append(arr)

    X, y = zip(*lst)
    testing_X = np.asarray(X)
    testing_y = np.asarray(y)
    X_dim = np.expand_dims(testing_X, axis=2)
    print("Ready dimension", X_dim.shape)
    loaded_model = keras.models.load_model(MODEL_PATH, compile = False)
    # record = loaded_model.predict(X_dim)
    prediction = np.argmax(loaded_model.predict(X_dim), axis=-1)
    # print("Record data is ", prediction)

    return prediction[0];
    # =================== Model Prediction code end ========================

# Render main file
@app.route("/")
def index():
    return render_template("index.html")

# Upload student recod in database
@app.route("/upload", methods=["POST"])
def upload():
    # folder_name = request.form['superhero']
    '''
    # this is to verify that folder to upload to exists.
    if os.path.isdir(os.path.join(APP_ROOT, 'files/{}'.format(folder_name))):
        print("folder exist")
    '''
    target = os.path.join(APP_ROOT, 'static/images/')
    print("Already saved data is ", target)
    if not os.path.isdir(target):
        os.mkdir(target)

    studentName = request.form['studentName']
    registrationNumber = request.form['registrationNumber']

    print(request.files.getlist("file"))
    for upload in request.files.getlist("file"):
        # print(upload)
        print("{} is the file name".format(upload.filename))
        filename = upload.filename

        # This is to verify files are supported
        ext = os.path.splitext(filename)[1]
        if (ext == ".wav"):
            print("File supported moving on...")
        else:
            render_template("Error.html", message="Files uploaded are not supported...")

        destination = "/".join([target, filename])
        print("Accept incoming file:", filename)
        print("Save it to:", destination)
        upload.save(destination)

        # ============== calling size detection method ================
        predict = emotion_recognition(destination, target);
        predicted_marks = ''
        if(predict == 1):
            predicted_marks = 5
        if (predict == 2):
            predicted_marks = 6
        if(predict == 3):
            predicted_marks = 4
        if (predict == 4):
            predicted_marks = 7
        if(predict == 5):
            predicted_marks = 2
        if (predict == 6):
            predicted_marks = 1
        if(predict == 7):
            predicted_marks = 8
        if (predict == 8):
            predicted_marks = 7
        print("Predicted value is ", predicted_marks)
        db.emotion_recognition.insert_one({'Student_Name': studentName, 'Registration_number': registrationNumber,
                                           'predicted_emotion': str(predict), 'predicted_marks': predicted_marks})
        print("The original prediction is = ", predict)


    return render_template("complete.html", image_name=filename)


@app.route('/upload/<filename>')
def send_image(filename):
    return send_from_directory("images", filename)


@app.route('/gallery')
def get_gallery():
    col = db["emotion_recognition"]
    x = col.find()
    emotion_record = []
    for emotion in x:
        emotion_record.append(emotion)
    predicted_emotions = emotion_record
    record_length = len(predicted_emotions)
    return render_template("gallery.html", predicted_emotions=predicted_emotions, record_length= record_length)

# === Original code ===
# @app.route('/gallery')
# def get_gallery():
#     image_names = os.listdir('./static/images')
#     print(image_names)
#     return render_template("gallery.html", image_names=image_names)


if __name__ == "__main__":
    app.run(port=4550, debug=True)
