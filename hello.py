from flask import Flask
app = Flask(__name__)

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

from sklearn import datasets


@app.route('/iris/<params>')
def iris(params):
    params = params.split(',')
    params = [float(x) for x in params]
    params = np.array(params).reshape(1,-1)
    iris = datasets.load_iris()
    iris_X = iris.data
    iris_y = iris.target
    np.unique(iris_y)
    np.random.seed(0)
    indices = np.random.permutation(len(iris_X))
    iris_X_train = iris_X[indices[:-10]]
    iris_y_train = iris_y[indices[:-10]]
    iris_X_test = iris_X[indices[-10:]]
    iris_y_test = iris_y[indices[-10:]]
    # Create and fit a nearest-neighbor classifier
    from sklearn.neighbors import KNeighborsClassifier
    knn = KNeighborsClassifier()
    knn.fit(iris_X_train, iris_y_train)

    pred = knn.predict(params)


    return str(pred) 
