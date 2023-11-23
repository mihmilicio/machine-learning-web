from flask import Flask

app = Flask(__name__)
app.config['SECRET_KEY'] = '1786f0c45b7a2c10b359fbed5c0bb4f0'

from iris_ml import routes
