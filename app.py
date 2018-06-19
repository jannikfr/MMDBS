from flask import Flask

app = Flask(__name__)


@app.route('/')
def hello_world():
    test = "test"
    return 'Hello World!'


if __name__ == '__main__':
    app.run(debug=True, use_debugger=False, use_reloader=False)
    print("dsjkjsdkds")
