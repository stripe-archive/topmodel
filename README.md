topmodel
========

topmodel is a service for evaluating binary classifiers. It comes with built-in
metrics and comparisons so that you don't have to build your own from scratch.

You can store your data either locally or in S3.

## Metrics

Here are the graphs topmodel will give you for any binary classifier:

#### Precision/recall curve

![Precision/recall curve](http://i.imgur.com/h2aOeS5.png)

#### ROC (Receiver operating characteristic) curve

![ROC curve](http://i.imgur.com/tunfpcu.png)

We also use
[bootstrapping](http://en.wikipedia.org/wiki/Bootstrapping_%28statistics%29#Methods_for_bootstrap_confidence_intervals)
to show the uncertainty on ROC curves and precision/recall curves. Here's an
example:

![ROC curve with bootstrapping](http://i.imgur.com/dc21r9j.png)

#### Marginal precision

The idea here is that among all items with score 0.9, you expect 90% of them to
be in the target group (marked 'True'). This graph compares the expected rate
to the actual rate -- the closer it is to a straight line, the better.

![Marginal precision](http://i.imgur.com/yeqpD8A.png)


#### Score distribution

Plots the distribution of scores for all instances and only for instances
labelled 'True'.

![Score frequencies](http://i.imgur.com/P77AQ5C.png)

## Using topmodel locally

topmodel comes with example data so you can try it out right away. Here's how:

1. Create a virtualenv
1. Install the requirements: `pip install -r requirements.txt`
2. Start a topmodel server:

    ```
    ./topmodel_server.py
    ```
1. topmodel should now be running at [http://localhost:9191](http://localhost:9191).
1. See a page of metrics for some example data at [http://localhost:9191/model/data/test/my_model_name/](http://localhost:9191/model/data/test/my_model_name/)

You can now add new models for evaluation! (see "How to add a model to topmodel" below for more)

## Using topmodel with S3

It's better to store your model data in a S3 bucket, so that you don't lose it. To get this working:

Create a `config.yaml` file:

```
cp config_example.yaml config.yaml
```

and fill it in with the S3 bucket you want to use and your AWS secret key and
access key. topmodel will automatically find models in the bucket as long as
they're named correctly (see "How to add a model to topmodel")

Then start topmodel with

```
./topmodel_server.py --remote
```

## How to add a model to topmodel

1. Create a TSV with columns 'pred_score' and 'actual'. Save it to `your_model_name.tsv`. The columns should be separated by tabs. In each row:
   * `actual` should be 0 or 1 (True/False also work)
   * `pred_score` should be the score the model determined.
   * See the examples in `example_data/`
   * For example:

    ```
    actual	pred-score
    False	0.2
    True	0.8
    True	0.7
    False	0.3
    ```

2. Copy the TSV to S3 at `s3://your-s3-bucket/your_model_name/scores.tsv`, or locally to `data/your_model_name/scores.tsv`
3. You're done! Your model should appear at http://localhost:9191/ if you reload.

## Developing topmodel

We'd love for you to contribute. If you run topmodel with

```
./topmodel_server.py --development
```

it will autoreload code.

There's example data to test on in `data/test`.


## Authors

* Julia Evans [http://twitter.com/b0rk](http://twitter.com/b0rk)
* Chris Wu [http://github.com/cwu](http://github.com/cwu)
* George Hotz [http://geohot.com](http://geohot.com)

## License

Copyright 2014 Stripe, Inc

Licensed under the MIT license.
