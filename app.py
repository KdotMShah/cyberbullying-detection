from flask import Flask, render_template, request, redirect, url_for, session, flash
import os
from werkzeug.utils import secure_filename
from tensorflow.keras.models import load_model  # Assuming you're using TensorFlow/Keras
from tensorflow.keras.preprocessing import image
from tensorflow.keras.preprocessing.sequence import pad_sequences
import numpy as np
import pickle

app = Flask(__name__)
app.secret_key = 'your_secret_key'  # Used for session management

UPLOAD_FOLDER = 'static/uploads'  # Folder to store uploaded images
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif'}
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Ensure upload folder exists
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

model_2 = load_model('cyberbullying_cnn_model.keras')
model = load_model('sense_media.keras')
with open('tokenizer.pkl', 'rb') as f:
    tokenizer = pickle.load(f)

# Helper function to preprocess images
def preprocess_image(img_path):
    img = image.load_img(img_path, target_size=(224, 224))  # Adjust size to match your model
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    return img_array

# Simple in-memory user store
users = {}

@app.route('/')
def home():
    return render_template('index.html')

# Login Route
@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']
        
        if username in users and users[username] == password:
            session['username'] = username  # Store the username in session
            return redirect(url_for('dashboard'))
        else:
            flash('Invalid username or password', 'error')

    return render_template('login.html')

# Registration Route
@app.route('/register', methods=['GET', 'POST'])
def register():
    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']
        
        if username in users:
            flash('Username already exists. Please choose a different one.', 'error')
        else:
            users[username] = password  # Store the username and password
            flash('Registration successful! Please log in.', 'success')
            return redirect(url_for('login'))
    
    return render_template('register.html')

def allowed_file(filename):
    """Check if the uploaded file has an allowed extension."""
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/dashboard', methods=['GET', 'POST'])
def dashboard():
    if 'username' not in session:
        return redirect(url_for('login'))  # Redirect to login if not logged in

    image_url = None
    if request.method == 'POST':
        if 'image' not in request.files:
            flash('No file part', 'error')
            return redirect(request.url)

        file = request.files['image']

        if file.filename == '':
            flash('No selected file', 'error')
            return redirect(request.url)

        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(file_path)
                        # Check for sensitive content
            try:
                is_sensitive = scan_img(file_path)
                print(is_sensitive)
                if is_sensitive:
                    # Delete the sensitive file
                    os.remove(file_path)
                    flash('The uploaded image contains sensitive content and has been removed.', 'error')
                    return redirect(request.url)
                    
                # Generate image URL only if it passes the check
                image_url = url_for('static', filename=f'uploads/{filename}')
                flash('Image uploaded successfully!', 'success')
            except Exception as e:
                flash(f'Error while processing the image: {e}', 'error')
                return redirect(request.url)            


    return render_template('dashboard.html', username=session['username'], image_url=image_url)

def scan_img(image_url):
            image=preprocess_image(image_url)
            prediction=model.predict(image)
            if prediction[0] < 0.5:
                return True
            else:
                return False

import os

comments = {}

@app.route('/gallery')
def gallery():
    # Get the list of image files in the upload folder
    images = os.listdir(UPLOAD_FOLDER)
    # Filter out the files that are not of the allowed extensions
    images = [img for img in images if allowed_file(img)]
    
    return render_template('gallery.html', images=images, comments=comments)


@app.route('/add_comment/<image_name>', methods=['POST'])
def add_comment(image_name):
    comment = request.form['comment']
    prediction=predict_cyberbullying(comment)

    if prediction=='Not Cyberbullying':
        if image_name in comments:
            comments[image_name].append(comment)
        else:
            comments[image_name] = [comment]
        return redirect(url_for('gallery'))
    else:
        flash('message contains sensitive content')
        return redirect(url_for('gallery'))  # Redirect to gallery or another appropriate page

def predict_cyberbullying(text):

    # Convert the text into sequence
    sequence = tokenizer.texts_to_sequences([text])
    padded_sequence = pad_sequences(sequence, maxlen=100)  # Match the input length during training

    # Predict with the model
    prediction = model_2.predict(padded_sequence)
    print(f"Prediction output: {prediction}")  # Debugging: check output values

    # Classify based on the probability output
    if prediction[0] > 0.5:  # 0.5 as threshold for binary classification
        return 'Cyberbulling'
    else:
        return 'Not Cyberbullying'

# Logout Route
@app.route('/logout')
def logout():
    session.pop('username', None)  # Remove the username from the session
    flash('You have been logged out.', 'success')
    return redirect(url_for('login'))

if __name__ == '__main__':
    app.run(debug=True)