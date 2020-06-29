from flask import Flask, request, jsonify, abort, redirect, url_for, render_template, send_file
import pandas as pd

app = Flask(__name__)
dict = {0: '<img src="/static/setosa.jpg" alt="Image">',
        1: '<img src="/static/versicolor.jpg" alt="Image">', 
        2: '<img src="/static/virginica.jpg" alt="Image">'}
@app.route('/')
def hello_world():
    print(1+5)
    return '<h2>Hello my best friend!</h2>'

if __name__ == '__main__':
    app.run()



from markupsafe import escape

@app.route('/user/<username>')
def show_user_profile(username):
    # show the user profile for that user
    return 'User %s' % escape(username)

def mean(numbers):
    return float(sum(numbers))/ max(len(numbers), 1)

#@app.route('/<nums>')
def mean_nums(nums):
    nums = nums.split(',')
    nums = [float(x) for x in nums]
    print(nums)
    return str(mean(nums))

import numpy as np
import joblib
from sklearn import datasets

knn = joblib.load('knn.pkl')
@app.route('/iris/<params>')


def iris(params):
    params = params.split(',')
    params = [float(x) for x in params]
    params = np.array(params).reshape(1,-1)
    pred = knn.predict(params)

    #print('<img src='+dict[int(pred)]+' alt="Image">')
    return dict[int(pred[0])] 
    #return '<img src="setosa.jpg" alt="Image">'

@app.route('/iris_show')
def show_image():
    
    return '<img src="https://notebooks.azure.com/xukai286/libraries/justmarkham-scikit-learn/raw/images/03_iris.png" alt="User Image">'


@app.route('/iris_post', methods=['POST'])
def add_message():
    try:
        content = request.get_json()
        params = content['flower'].split(',')
        param = [float(x) for x in params]
        params = np.array(params).reshape(1,-1).astype(np.float64)
        pred = knn.predict(params)
        print(pred) # # Do your processing
        pred_class = {'class': str(pred[0])}
        return jsonify(pred_class) 
    except:
        return redirect(url_for('bad_request'))
    #print(5)
    #return(str(param))

@app.route('/bad_request400')

def bad_request():
    return abort(400)


from flask_wtf import FlaskForm
from wtforms import StringField, FileField
from wtforms.validators import DataRequired
from werkzeug.utils import secure_filename
import os

app.config['SECRET_KEY'] = 'any secret string'

class MyForm(FlaskForm):
    name = StringField('name', validators=[DataRequired()])
    file = FileField()

@app.route('/submit', methods=('GET', 'POST'))
def submit():
    form = MyForm()
    if form.validate_on_submit():
        f = form.file.data
        filename=form.name.data + '.csv'
        """f.save(os.path.join(
            filename
        ))"""
        df = pd.read_csv(f, header = None)
        pred = knn.predict(df)
        result = pd.DataFrame(pred)
        result.to_csv(filename, index = False)
        print(pred)
        return send_file(filename,
                     mimetype='text/csv',
                     attachment_filename=filename,
                     as_attachment=True)
    return render_template('submit.html', form=form)


import os
from flask import Flask, flash, request, redirect, url_for
from werkzeug.utils import secure_filename

UPLOAD_FOLDER = ''
ALLOWED_EXTENSIONS = {'txt', 'pdf', 'png', 'jpg', 'jpeg', 'gif'}

#app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/upload', methods=['GET', 'POST'])
def upload_file():
    if request.method == 'POST':
        # check if the post request has the file part
        if 'file' not in request.files:
            flash('No file part')
            return redirect(request.url)
        file = request.files['file']
        # if user does not select file, browser also
        # submit an empty part without filename
        if file.filename == '':
            flash('No selected file')
            return redirect(request.url)
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
            return 'file uploaded'
    return '''
    <!doctype html>
    <title>Upload new File</title>
    <h1>Upload new File</h1>
    <form method=post enctype=multipart/form-data>
      <input type=file name=file>
      <input type=submit value=Upload>
    </form>
    '''