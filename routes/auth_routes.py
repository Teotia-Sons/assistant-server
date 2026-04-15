import requests
from flask import Blueprint, request
from flask_login import LoginManager, UserMixin, current_user, login_user

from config import Config
from db.models.user import User

auth_bp = Blueprint("auth", __name__)

login_manager = LoginManager()


class UserModel(UserMixin):
    def __init__(self, username):
        self.id = username


@login_manager.user_loader
def load_user(username):
    user = User.objects(username=username).first()
    if user:
        return UserModel(user.username)
    return None


@auth_bp.route("/auth/user", methods=["GET"])
def get_user():
    return {"user": current_user.get_id()}


@auth_bp.route("/auth/login", methods=["POST"])
def login():
    auth_code = request.json.get("code")
    token_url = "https://oauth2.googleapis.com/token"
    token_payload = {
        "code": auth_code,
        "client_id": Config.GOOGLE_OAUTH_CLIENT_ID,
        "client_secret": Config.GOOGLE_OAUTH_CLIENT_SECRET,
        "grant_type": "authorization_code",
        "redirect_uri": "postmessage",
    }

    try:
        token_response = requests.post(token_url, data=token_payload)
        token_response.raise_for_status()
        token_data = token_response.json()
    except requests.exceptions.RequestException:
        return {"message": "Failed to exchange authorization code for token"}, 403

    access_token = token_data.get("access_token")

    userinfo_url = "https://www.googleapis.com/oauth2/v3/userinfo"
    headers = {"Authorization": f"Bearer {access_token}"}

    try:
        userinfo_response = requests.get(userinfo_url, headers=headers)
        userinfo_response.raise_for_status()
        user_google_info = userinfo_response.json()
    except requests.exceptions.RequestException:
        return {"message": "Failed to fetch user information from Google"}, 403

    user_email = user_google_info.get("email")
    user_doc = User.objects(username=user_email).first()
    if not user_doc:
        return {"message": "User is not registered."}, 401

    login_user(UserModel(user_doc.username), remember=True)

    return {"user": current_user.get_id()}, 200
