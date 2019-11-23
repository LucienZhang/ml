# -*- coding: utf-8 -*-
# @Author: Lucien Zhang
# @Date:   2019-09-19 17:55:24
# @Last Modified by:   Lucien Zhang
# @Last Modified time: 2019-09-22 19:07:18

import os
from flask import Flask, redirect, url_for
from ml.ml_module import ml_api
from ml.config import config


def init_app(app):
    pass


def create_app(config_name=None):
    # create and configure the app
    root_path = os.path.abspath(os.path.dirname(__file__))
    app = Flask(__name__, instance_relative_config=True, root_path=root_path,
                instance_path=os.path.join(root_path, 'instance'))
    if config_name:
        app.config.from_object(config[config_name])
    else:
        app.config.from_pyfile('config.py', silent=True)

    if app.config["DEBUG"]:
        @app.after_request
        def after_request(response):
            response.headers["Cache-Control"] = "no-cache, no-store, must-revalidate, public, max-age=0"
            response.headers["Expires"] = 0
            response.headers["Pragma"] = "no-cache"
            return response

    app.register_blueprint(ml_api, url_prefix='/ml')
    init_app(app)

    @app.route('/')
    def homepage():
        return redirect(url_for('ml_api.home'))

    return app
