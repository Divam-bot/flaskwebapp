import os

from flask import Flask

from flask_sqlalchemy import SQLAlchemy

db = SQLAlchemy()

class Details(db.Model):
	__tablename__ ="Details"
	uname = db.Column(db.String, primary_key=True)
	password = db.Column(db.String, nullable=False)
	email = db.Column(db.String, nullable=False)
	phone = db.Column(db.String, nullable=False) 