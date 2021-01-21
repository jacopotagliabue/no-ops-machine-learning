from metaflow import FlowSpec, step, retry,  IncludeFile, Parameter, resources, batch, S3
import time
from io import StringIO


class RegressionModel(FlowSpec):

    # if a static file is part of the flow, it can be called in any downstream process, gets versioned etc.
    # https://docs.metaflow.org/metaflow/data#data-in-local-files
    data_file = IncludeFile('dataset',
                            help='Text File With Regression Numbers',
                            is_text=True,
                            default='dataset.txt')

    @step
    def start(self):
        """
        Read data in, and parallelize model building with two params (in this case, dummy example with learning rate).

        :return:
        """
        # data is an array of lines from the text file containing the numbers
        raw_data = StringIO(self.data_file).readlines()
        print("Total of {} rows in the dataset!".format(len(raw_data)))
        # cast strings to float and prepare for training
        self.dataset = [[float(_) for _ in d.strip().split('\t')] for d in raw_data]
        print("Raw data: {}, cleaned data: {}".format(raw_data[0].strip(), self.dataset[0]))
        # this is the only MetaFlow-specific part: based on a list of options (here, learning rates)
        # spin up N parallel process, passing the given option to the child process
        learning_rates = [0.1, 0.2]
        #self.next(self.train_model, foreach='sports')
        self.next(self.end)

    @step
    def end(self):
        print('End')


if __name__ == '__main__':
    RegressionModel()

