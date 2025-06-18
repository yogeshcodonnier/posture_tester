# from app import create_app

# app = create_app()

# if __name__ == "__main__":
#     app.run(debug=True)

from flask import Flask
from app.routes import bp

app = Flask(__name__)
app.register_blueprint(bp)

@app.route("/")
def home():
    return "Flask is working on Render!"
