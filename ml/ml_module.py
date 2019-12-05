# -*- coding: utf-8 -*-
# @Author: Lucien Zhang
# @Date:   2019-09-19 17:53:51
# @Last Modified by:   Lucien Zhang
# @Last Modified time: 2019-09-22 19:02:24
from flask import Blueprint, render_template, request

from ml.predictor import predict

ml_api = Blueprint('ml_api', __name__, template_folder='templates', static_folder='static')


@ml_api.route('/')
def home():
    return render_template("ml.html")


@ml_api.route('/mnist', methods=['GET', 'POST'])
def mnist():
    if request.method == 'POST':
        image = request.files['predictImg']
        # filename = str(int(time.mktime(time.localtime()))) + '.png'
        # imgurl = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        # predictImg.save(imgurl)
        results, probabilities = predict('mnist', [image])
        return str(results[0])
    else:
        return render_template("mnist.html")
