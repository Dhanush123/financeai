from flask import Flask, jsonify
app = Flask(__name__)

@app.route('/')
def hello_world():
    return jsonify('Bleach')

@app.route('/best-stock/<int:num>')
def