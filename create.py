import os

from flask import Flask 
from models import *

covid_19 = Flask(__name__)

covid_19.config["SQLALCHEMY_DATABASE_URI"] = os.getenv("DATABASE_URL")
covid_19.config["SQLALCHEMY_TRACK_MODIFICATIONS"]  = False

db.init_app(covid_19)

def main():
	db.create_all()

if __name__ == "__main__":
	with covid_19.app_context():
		main()