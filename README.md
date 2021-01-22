# no-ops-machine-learning
_A PaaS End-to-End ML Setup with Metaflow, Serverless and SageMaker._

- - - -

This repo is an end-to-end model building exercise, showing how to go from a dataset file to 
an endpoint serving predictions through a repeatable, modular and scalable process (DAG-based). 
More importantly, all of this is achieved in minutes from a developer laptop, without 
explicitly deploying/maintaining *any infrastructure*.

For the full context and details about the tutorial, please refer to the blog post (WIP).

## Overview

The project is a regression problem using [Keras](https://www.tensorflow.org/tutorials/keras/regression), 
where we try to predict a numerical target *y* based on a numerical feature *x*. The functions are not 
particularly terse, but we aim for explanatory power over economy. The DAG in `training.py` is the following:

![picture alt](https://cdn-images-1.medium.com/max/1600/1*QrfmZITVzFTTMcLw-s9M9g.jpeg "Regression DAG")

In this project you can see:

* the use of static files, versioned by MetaFlow automatically;
* the use of the "fanout" capability, allowing parallel execution of training steps;
* the use of the "batch" capability, in which local execution and cloud-enhanced computing co-exist seamlessly;
* a GPU-backed deployment and prediction testing in 10 lines of code.

In a nutshell, three main tools are used:

* [Metaflow](https://metaflow.org/), to orchestrate the pipeline in a consistent and scalable manner;
* [SageMaker](https://aws.amazon.com/sagemaker/), to host a trained Keras model and provide real-time inference through an internal AWS endpoint;
* [serverless](https://www.serverless.com/), to configure and deploy an external-facing endpoint (public API) that connects clients 
making GET requests to the model predictions served by the SageMaker endpoint.

Please refer to the blog post for an in-depth discussion about the tools and the philosophy behind
our choices.

## Project Structure

The `flow` folder contains the necessary code to train several models over a target dataset, store the intermediate
results in s3 and then deploying the best performing variant to a SageMaker endpoint.

The `serverless` folder contains the necessary code to launch a lambda-powered endpoint, exposing SageMaker
predictions to client applications.


## How to Run the Full Tutorial

### 0. Prerequisites for the Development Environment

Make sure to have properly configured in your developer machine the following tools:

* AWS credentials to access the relevant cloud services;
* Metaflow, configured to use the [AWS metastore](https://docs.metaflow.org/metaflow-on-aws/metaflow-on-aws);
* [Serverless](https://www.serverless.com/framework/docs/providers/aws/guide/credentials/). 

If you are running the script from a virtual environment, first install 
the packages specified in the `requirements.txt`.

### 1. Create a Dataset

Run the `create_dataset.py` script in the `flow` folder to create a `dataset.txt` file 
with artificial data - the file is imported in the DAG to simulate the 
target dataset.

### 2. Train and Deploy a Model

Cd ino the `flow` folder and run the command: 

`python training.py run`

The Metaflow DAG will run from data loading (first step), to shipping a trained model to a
newly created SageMaker endpoint (last step). 
We suggest to comment the `@batch(gpu=1, memory=80000)` decorator 
for the first run, to verify that the end-to-end local computation is working as expected.

At the end of the DAG, the terminal will print the name of the SageMaker endpoint hosting the
model. Write the name down as it will be used to power the public lambda endpoint.


### 3. Deploy Public Endpoint with Serverless

Cd into the `serverless` folder and run the following command:

`serverless deploy —endpoint regression-138048024289-endpoint`

where `regression-138048024289-endpoint` is the name of the endpoint created by the MetaFlow DAG.

If all goes well, after a minute you should be greeted by a success message:

![picture alt](https://cdn-images-1.medium.com/max/1600/1*U_-c7PrafMlzqKs4Qq6kaw.png "Serverless successful deployment")

By copying and pasting the GET url into the browser, you can get a real-time prediction from 
the model by passing the *x* parameter, e.g.:

`https://my-url.amazonaws.com/dev/predict?x=9.45242`

## License

All code is provided freely "as is" under a standard MIT license.