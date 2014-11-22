from flask import render_template, g, request, redirect

from topmodel.model_data import ModelDataManager
from topmodel import plots
from topmodel.hmetrics import auc
from topmodel.model_data import ModelData
from web import app

import matplotlib.pyplot as plt


@app.route("/")
def home():
    model_data_manager = ModelDataManager(g.file_system)
    return render_template("index.html", models=model_data_manager.list())


@app.route("/compare")
def compare():
    models = request.args.getlist('model[]')
    cached_datas = []
    for path in models:
        model_data = ModelData(g.file_system, path)
        cached_datas.append(model_data.get_metrics(10))

    _, ax = plt.subplots(figsize=(12, 6))
    for name, cached_data in zip(models, cached_datas):
        prc = plots.precision_recall_curve(cached_data, ax=ax, label=name)

    _, ax = plt.subplots(figsize=(12, 6))
    for name, cached_data in zip(models, cached_datas):
        roc = plots.roc_curve(cached_data, ax=ax, label=name)

    context = {
        'precision_recall_curve': prc,
        'roc_curve': roc}

    return render_template("compare.html", **context)


@app.route("/model/<path:path>/")
def training(path):
    model_data = ModelData(g.file_system, path)
    cached_data = model_data.get_metrics(50)
    # print cached_data

    context = {
        'precision_recall_curve': plots.precision_recall_curve(cached_data),
        'roc_curve': plots.roc_curve(cached_data),
        'score_distribution': plots.score_distribution(cached_data[0]),
        'marginal_precision_curve': plots.marginal_precision_curve(cached_data[0]),
        'threshold_graph': plots.thresholds_graph(cached_data[0]),
        'threshold_table': plots.thresholds_table(cached_data[0]),

        'brier': plots.box_brier(cached_data),
        'auc': auc(cached_data[0]['fprs'], cached_data[0]['recalls']),
        'notes': ModelData(g.file_system, path).get_notes(),
        'path': path,
    }
    return render_template("results.html", **context)


@app.route("/model/<path:path>", methods=['DELETE'])
def delete_path(path):
    g.file_system.remove(path)
    return "Success!"


@app.route("/model/<path:path>/notes/", methods=['PUT'])
def update_notes(path):
    ModelData(g.file_system, path).set_notes(request.form['notes'])
    return "Success!"
