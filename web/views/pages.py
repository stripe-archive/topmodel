from flask import render_template, g, request, redirect

from topmodel import plots
from topmodel.hmetrics import auc
from web import app

import matplotlib.pyplot as plt


@app.route("/")
def home():
    model_paths = sorted(g.model_data_manager.models.keys())
    return render_template("index.html", model_paths=model_paths)


@app.route("/compare")
def compare():
    models = request.args.getlist('model[]')
    cached_datas = []
    for path in models:
        model_data = g.model_data_manager.models[path]
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
    model_data = g.model_data_manager.models[path]
    cached_data = model_data.get_metrics(50)

    context = {
        'precision_recall_curve': plots.precision_recall_curve(cached_data),
        'roc_curve': plots.roc_curve(cached_data),
        'score_distribution': plots.score_distribution(cached_data[0]),
        'absolute_score_distribution': plots.absolute_score_distribution(cached_data[0]),
        'marginal_precision_curve': plots.marginal_precision_curve(cached_data[0]),
        'threshold_graph': plots.thresholds_graph(cached_data[0]),
        'threshold_table': plots.thresholds_table(cached_data[0]),

        'auc': auc(cached_data[0]['fprs'], cached_data[0]['recalls']),
        'notes': model_data.get_notes(),
        'model_metadata': model_data.get_metadata(),
        'path': path,
    }
    return render_template("results.html", **context)


@app.route("/model/<path:path>", methods=['DELETE'])
def delete_path(path):
    g.file_system.remove(path)
    return "Success!"


@app.route("/model/<path:path>/notes/", methods=['PUT'])
def update_notes(path):
    model_data = g.model_data_manager.models[path]
    model_data.set_notes(request.form['notes'])
    return "Success!"
