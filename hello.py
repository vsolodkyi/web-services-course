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