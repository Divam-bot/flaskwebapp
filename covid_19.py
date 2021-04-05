from flask import Flask,render_template, request,redirect,url_for,Response , jsonify
import tensorflow as tf
import  numpy as np
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.layers import Embedding, LSTM, Dense, Dropout, Bidirectional
from tensorflow.keras.preprocessing.text import Tokenizer 
import json , pickle

#from camera import VideoCamera

from models import *

import cv2

covid_19 = Flask(__name__)

covid_19.config["SQLALCHEMY_DATABASE_URI"] = os.getenv("DATABASE_URL")
covid_19.config["SQLALCHEMY_TRACK_MODIFICATIONS"]=False

db.init_app(covid_19)

root_path = os.path.dirname(os.path.abspath(__file__))

# video_stream = VideoCamera()

notes=[]

#details={}
#listd={}
#temp=''

rpsfinder = tf.keras.models.load_model('r_P_s.h5')
sonnetwriter = tf.keras.models.load_model('sonnet_generator_weights.h5')

cam= cv2.VideoCapture(0)
cv2.namedWindow("test")
img_counter = 0


with open("tokenizer.pickle",'rb') as handle:
    tokenizer = pickle.load(handle)
    
#tokenizer = tf.keras.preprocessing.text.tokenizer_from_json(data)


@covid_19.route("/", methods=["GET","POST"])
def index():

    return render_template("index.html" )


# def gen(camera):
#     while True:
#         frame = camera.get_frame()
#         yield (b'--frame\r\n'
#                b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n\r\n')

# @covid_19.route('/video_feed')
# def video_feed():
#     return Response(gen(video_stream))
# #,mimetype='multipart/x-mixed-replace; boundary=frame')

                    


@covid_19.route("/signup" , methods=["GET","POST"])
def signup():
    if(request.method=="POST"):
        name = request.form.get("uname")
        mail = request.form.get("email")
        num =  request.form.get("phone")
        #pass = request.form.get("password")

        #details.update({name:request.form.get("password")})
        #listd.update({name:notes})
        user = Details(uname = name, email=mail , phone=num, password=request.form.get("password"))
        db.session.add(user)
        db.session.commit()
        return redirect(url_for('success'))
    return render_template("signup.html")



@covid_19.route("/success")
def success():
    
    return render_template("success.html")



@covid_19.route("/login" , methods=["GET","POST"])
def login():
    #temp=''
    if(request.method=="POST"):
        p = Details.query.filter_by(uname = request.form.get("uname")).first()

        if(p is None):
            return redirect(url_for('no_user'))

        else:    

            if(request.form.get("password") == p.password ):
                
                return redirect(url_for('apps'))
            else: return("password incorrect")
         

    return render_template("login.html" )    



@covid_19.route("/no_user")
def no_user():
    return render_template("no_user.html")


@covid_19.route("/apps")
def apps():
    return render_template("apps.html")


@covid_19.route("/todolist", methods=["GET","POST"])
def todolist():
    if(request.method == "POST"):
        if( request.form['b1'] == '1'):
            notes.append(request.form.get("note"))
            
        elif( request.form['b1'] == '2'):
            if(request.form.get("note") not in notes):return("<h1>Note not found</h1>")
            notes.remove(request.form.get("note"))
            
      
    return render_template("todolist.html" , notes = notes)

path = os.path.join(root_path, 'static/')
@covid_19.route("/rps",methods=["GET","POST"])
def rps():
    img_counter = 0
    if(request.method == "POST"):
        if( request.form['b1'] == '1'):
            f = request.files['filename']
            destination = "/".join([path,f.filename])
            f.save(destination)
            image_path = destination
            img =  tf.keras.preprocessing.image.load_img(image_path,target_size=(150,150))
            img = tf.keras.preprocessing.image.img_to_array(img)
            img = np.expand_dims(img,axis=0)
            pred = rpsfinder.predict(img)
            return render_template("prediction.html", pred = pred)
        

        elif(request.form['b1']== '2'):
            
            while True:
                ret, frame = cam.read()
                if not ret:
                    print("failed to grab frame")
                    break
                cv2.imshow("test", frame)
            
                k = cv2.waitKey(1)
                if k%256 == 27:
                    # ESC pressed
                    print("Escape hit, closing...")
                    break
                elif k%256 == 32:
                    # SPACE pressed
                    img_name = "opencv_frame_{}.png".format(img_counter)
                    destination = "/".join([path,img_name])
                    cv2.imwrite(destination, frame)
                    print("{} written!".format(img_name))
                    img_counter += 1

            cam.release()
            cv2.destroyAllWindows()
            
            image_path = destination
            img =  tf.keras.preprocessing.image.load_img(image_path,target_size=(150,150))
            img = tf.keras.preprocessing.image.img_to_array(img)
            img = np.expand_dims(img,axis=0)
            pred = rpsfinder.predict(img)
            return render_template("prediction.html", pred = pred)
            
        
    return render_template("rps.html")

@covid_19.route("/sonnet",methods=["GET","POST"])
def sonnet():
    seed_text=""
    
    
    if(request.method=="POST"):
        if(request.form["b"]=='1'):
            seed_text += (request.form.get("sonnet_text"))
            
            for zee in range(100):
                token_list = tokenizer.texts_to_sequences([seed_text])[0]
                token_list = pad_sequences([token_list], maxlen=10, padding='pre')
                predicted = sonnetwriter.predict_classes(token_list, verbose=0)
                output_word = ""
                for word, index in tokenizer.word_index.items():
                    if index == predicted:
                        output_word = word
                        break
                seed_text += " " + output_word
                if(zee%5==0):seed_text+="\n"
    
            #print(seed_text)   
            context = seed_text.split("\n")
            return render_template("sonnetgen.html", context=context)

    
    return render_template("sonnet.html")




@covid_19.route("/totaluser")
def totaluser():
    udetails = Details.query.all()
    return render_template("totaluser.html",udetails = udetails)


covid_19.run(debug=True)