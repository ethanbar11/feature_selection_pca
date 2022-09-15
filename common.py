import pandas as pd


class ResultsHandler:
    def __init__(self, ):
        self.current_experiment = None
        self.past_experiments = []
        # self.filename = filename
        # self.f = open(filename, 'w')
        # self.f.write("name, value

    def start_new_experiment(self, dataset, algo_name):
        self.current_experiment = [('dataset', dataset), ('algo_name', algo_name)]
        self.past_experiments.append(self.current_experiment)

    def add_result(self, name, value):
        self.current_experiment.append((name, value))

    def get_pretty_results(self):
        lst = []
        for data in self.past_experiments:
            experiment = {}
            experiment['dataset_name'] = list(filter(lambda x: x[0] == 'dataset', data))[0][1]
            experiment['algo_name'] = list(filter(lambda x: x[0] == 'algo_name', data))[0][1]
            experiment['first_b_precent'] = list(filter(lambda x: x[0] == 'b_precent', data))[0][1]
            experiment['first_c_precent'] = list(filter(lambda x: x[0] == 'c_precent', data))[0][1]
            experiment['last_b_precent'] = list(filter(lambda x: x[0] == 'b_precent', data))[-1][1]
            experiment['last_c_precent'] = list(filter(lambda x: x[0] == 'c_precent', data))[-1][1]
            experiment['w_accuracy'] = list(filter(lambda x: x[0] == 'w_accuracy', data))[-1][1]
            lst.append(experiment)
        df = pd.DataFrame(lst)
        return df

    def save_results(self, filename):
        pretty_results = self.get_pretty_results()
        pretty_results.to_csv(filename, index=False)
