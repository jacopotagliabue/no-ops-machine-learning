"""

This is a one-script MetaFlow project, showing how to go from a dataset file to an endpoint serving predictions through
a repeatable, modular and scalable process (DAG-based). More importantly, all of this is achieved from a developer
laptop setup, without explicitely deploying *any infrastructure*.

The functions are not particularly terse, but we aim for explanatory power over economy: our challenge is to find a
compromise between a toy-example far removed from anything real, and something over-complicated and too tailored to
a specific use case. In this example you can see:

* the use of static file, versioned by MetaFlow automatically;
* the use of the "fanout" capability, allowing parallel execution of steps;
* the use of the "batch" capability, in which local execution and cloud-enhanced computing co-exist seamlessly;
* a GPU-backed deployment and prediction testing in 10 lines of code.

"""

from metaflow import FlowSpec, step, IncludeFile, batch, S3, Parameter
import time
import numpy as np
from io import StringIO
from random import choice


class RegressionModel(FlowSpec):

    # if a static file is part of the flow, it can be called in any downstream process, gets versioned etc.
    # https://docs.metaflow.org/metaflow/data#data-in-local-files
    DATA_FILE = IncludeFile(
        'dataset',
        help='Text File With Regression Numbers',
        is_text=True,
        default='dataset.txt')

    # uri from: https://github.com/aws/deep-learning-containers/blob/master/available_images.md
    DOCKER_IMAGE_URI = Parameter(
        name='sagemaker_image',
        help='AWS Docker Image URI for SageMaker Inference',
        default='763104351884.dkr.ecr.us-west-2.amazonaws.com/tensorflow-inference:2.3.0-gpu-py37-cu102-ubuntu18.04'
    )

    SAGEMAKER_INSTANCE = Parameter(
        name='sagemaker_instance',
        help='AWS Instance to Power SageMaker Inference',
        default='ml.p3.2xlarge'
    )

    @step
    def start(self):
        """

        Read data in, and parallelize model building with two params (in this case, dummy example with learning rate).

        """
        # data is an array of lines from the text file containing the numbers
        raw_data = StringIO(self.DATA_FILE).readlines()
        print("Total of {} rows in the dataset!".format(len(raw_data)))
        # cast strings to float and prepare for training
        self.dataset = [[float(_) for _ in d.strip().split('\t')] for d in raw_data]
        print("Raw data: {}, cleaned data: {}".format(raw_data[0].strip(), self.dataset[0]))
        # store dataset as train and test set
        split_index = int(len(self.dataset) * 0.8)
        self.train_dataset = self.dataset[:split_index]
        self.test_dataset = self.dataset[split_index:]
        print("Training data: {}, test data: {}".format(len(self.train_dataset), len(self.test_dataset)))
        # this is the only MetaFlow-specific part: based on a list of options (here, learning rates)
        # spin up N parallel process, passing the given option to the child process
        self.learning_rates = [0.1, 0.2]
        self.next(self.train_model, foreach='learning_rates')

    @batch(gpu=1, memory=80000)
    @step
    def train_model(self):
        """

        Train a dummy regression model (https://www.tensorflow.org/tutorials/keras/regression) to predict a value
        with the simple y = alpha x + beta model.

        To make sure our GPU setup is working we print some verbose logging (totally unnecessary)

        """
        # this is the CURRENT learning rate in the fan-out
        # each copy of this step in the parallelization will have it's own value
        self.learning_rate = self.input
        # do some specific import
        import tensorflow as tf
        from tensorflow.keras import layers
        import tarfile
        # build the model
        x_train = np.array([[_[0]] for _ in self.train_dataset])
        y_train = np.array([_[1] for _ in self.train_dataset])
        x_test = np.array([[_[0]] for _ in self.test_dataset])
        y_test = np.array([_[1] for _ in self.test_dataset])
        x_model = tf.keras.Sequential([
            layers.Dense(input_shape=[1,], units=1)
        ])
        # print out models for debug
        print(x_model.summary())
        x_model.compile(
            optimizer=tf.optimizers.Adam(learning_rate=self.learning_rate),
            loss='mean_absolute_error')
        history = x_model.fit(x_train, y_train,epochs=100, validation_split=0.2)
        self.hist = history.history
        # store loss for downstream tasks
        self.results = x_model.evaluate(x_test, y_test)
        print("Test set results: {}".format(self.results))
        # save model: IMPORTANT: TF models need to have a version
        # see: https://github.com/aws/sagemaker-python-sdk/issues/1484
        FOLDER_NAME = "regression-model-{}".format(self.learning_rate)
        MODEL_NAME = "{}/1".format(FOLDER_NAME)
        TAR_NAME = 'model-{}.tar.gz'.format(self.learning_rate)
        x_model.save(filepath=MODEL_NAME)
        # zip keras folder to a single file
        with tarfile.open(TAR_NAME, mode="w:gz") as _tar:
            _tar.add(MODEL_NAME, recursive=True)
        # metaflow nice s3 client needs a byte object for the put
        with open(TAR_NAME, "rb") as in_file:
            data = in_file.read()
            with S3(run=self) as s3:
                url = s3.put(TAR_NAME, data)
                print("Message saved at: {}".format(url))
                # save this path for downstream reference!
                self.s3_path = url
        # finally join with the others
        self.next(self.join_runs)

    @step
    def join_runs(self, inputs):
        """
        Join the parallel runs and merge results into a dictionary.
        """
        # merge results (loss) from runs with different learning rates
        self.results_from_runs = {
            inp.learning_rate:
                {
                    'metrics': inp.results,
                    'tar': inp.s3_path
                }
            for inp in inputs}
        print("Current results: {}".format(self.results_from_runs))
        # pick one according to some rule, say smaller loss (here just pick a random one)
        self.best_learning_rate = choice(list(self.results_from_runs.keys()))
        self.best_s3_model_path = self.results_from_runs[self.best_learning_rate]['tar']
        # next, deploy
        self.next(self.deploy)

    @step
    def deploy(self):
        """
        Use SageMaker to deploy the model as a stand-alone, PaaS endpoint, with our choice of the underlying
        Docker image and hardware capabilities.

        Available images for inferences can be chosen from AWS official list:
        https://github.com/aws/deep-learning-containers/blob/master/available_images.md

        Once the endpoint is deployed, you can add a further step with for example behavioral testing, to
        ensure model robustness (e.g. see https://arxiv.org/pdf/2005.04118.pdf). Here, we just "prove" that
        the endpoint is up and running!

        """
        from sagemaker.tensorflow import TensorFlowModel
        # generate a signature for the endpoint, using learning rate and timestamp as a convention
        ENDPOINT_NAME = 'regression-{}-endpoint'.format(int(round(time.time() * 1000)))
        # print out the name, so that we can use it when deploying our lambda
        print("\n\n================\nEndpoint name is: {}\n\n".format(ENDPOINT_NAME))
        # this is the name of the IAM role with SageMaker permissions
        # make sure this role has access to the bucket containing the model tar file!
        IAM_SAGE_ROLE = 'MetaSageMakerRole'
        model = TensorFlowModel(model_data=self.best_s3_model_path,
                               image_uri=self.DOCKER_IMAGE_URI,
                               role=IAM_SAGE_ROLE)
        predictor = model.deploy(initial_instance_count=1,
                                instance_type=self.SAGEMAKER_INSTANCE,
                                endpoint_name=ENDPOINT_NAME)
        # run a small test against the endpoint
        # pick a number for X and check the predicted Y is sensible
        input = {'instances': np.array([[0.57457947234]])}
        # output is on the form {'predictions': [[10.879798]]}
        result = predictor.predict(input)
        print(input, result)
        assert result['predictions'][0][0] > 0
        self.next(self.end)

    @step
    def end(self):
        """
        The final step is empty here, but cleaning operations and/or sending hooks for downstream deployment tasks
        is a natural necessity for machine learning DAGs.

        """
        print('Dag ended!')


if __name__ == '__main__':
    RegressionModel()

