from flask import Flask, request
from flask_cors import CORS
from flask_login import current_user
from mongoengine import connect

from config import Config
from routes.assistant_routes import assistant_bp
from routes.auth_routes import auth_bp, login_manager
from tracing import setup_tracing

app = Flask(__name__)
app.config.from_object(Config)

setup_tracing(app)

CORS(
    app,
    supports_credentials=True,
    resources={r"/*": {"origins": app.config["CORS_ORIGINS"]}},
)

connect(host=app.config["MONGO_URI"])
login_manager.init_app(app)


@app.before_request
def require_login():
    if (
        (not current_user.is_authenticated)
        and (request.endpoint not in ["auth.login"])
        and (request.method != "OPTIONS")
    ):
        return login_manager.unauthorized()

    return None


app.register_blueprint(assistant_bp, url_prefix="/assistant")
app.register_blueprint(auth_bp)

if __name__ == "__main__":
    app.run(port=5001, debug=True)
