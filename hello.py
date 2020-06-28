from flask import Flask, request, jsonify, abort, redirect, url_for
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

@app.route('/<nums>')
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