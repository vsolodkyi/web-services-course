from flask import Flask
app = Flask(__name__)

@app.route('/')
def hello_world():
    print(1+5)
    return '<h2>Hello my best friend!</h2>'

if __name__ == '__main__':
    app.run()