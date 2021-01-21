# no-ops-machine-learning

This repo is an end-to-end model building exercise, showing how to go from a dataset file to 
an endpoint serving predictions through a repeatable, modular and scalable process (DAG-based). 
More importantly, all of this is achieved from a developer laptop setup, without 
explicitely deploying *any infrastructure*.

For the full context and details about the tutorial, please refer to the blog post (WIP).

## Overall Structure

The `flow` folder contains the necessary code to train several models over a target dataset, store the intermediate
results in s3 and then deploying the best performing variant to a SageMaker endpoint.

The `serverless` folder contains the necessary code to launch a lambda-powered endpoint, exposing SageMaker
predictions to clients.


## How to Run the Full Tutorial

### Prerequisite for the Development Environment


### Create a Dataset



### Train and Deploy a Model


### Deploy Endpoint with Serverless

Cd into the `serverless` folder and run the following command:

`serverless deploy —endpoint regression-138048024289-endpoint`

where `regression-138048024289-endpoint` is the name of the endpoint created by the MetaFlow DAG in `training.py`.

## License

All code is provided freely "as is" under a standard MIT license.