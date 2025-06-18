from flask import Flask
from flask_cors import CORS
import os

def create_app():
    # app = Flask(__name__)
    template_dir = os.path.abspath('templates')  # absolute path from root
    app = Flask(__name__, template_folder=template_dir)

    CORS(app)

    from .routes import main
    # app.register_blueprint(main)
    app.register_blueprint(main, url_prefix="/api")

    return app

