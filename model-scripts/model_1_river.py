import pickle
import os
import numpy as np

from river import compose
from river import linear_model
from river import metrics
from river import preprocessing
from river import feature_extraction
from river import naive_bayes
import joblib

from label_studio_ml.model import LabelStudioMLBase


class StockDataClassifier(LabelStudioMLBase):

    def __init__(self, **kwargs):
        # don't forget to initialize base class...
        super(StockDataClassifier, self).__init__(**kwargs)
        self.from_name, self.info = list(self.parsed_label_config.items())[0]
        self.to_name = self.info['to_name'][0]
        self.value = self.info['inputs'][0]['value']

        # self.model = compose.Pipeline(('tokenize', feature_extraction.BagOfWords(lowercase=False)),
        #                               ('nb', naive_bayes.MultinomialNB(alpha=1)))
        self.load_update_model = joblib.load("D:\Asha\label-studio\model-scripts\model_trained.bin")

    def predict(self, tasks, **kwargs):
        # collect input texts
        # input_texts = []
        # for task in tasks:
        #     input_text = task['data'].get(self.value) or task['data'].get('$Text')
        #     input_texts.append(input_text)
        # print(len(input_texts))
        # print('************************************', end='\n\n')
        # print(input_texts)
        # print('************************************', end='\n\n')
        # print('self.from_name -->', self.from_name)
        # print('************************************', end='\n\n')
        # print('self.info -->', self.info)
        # print('************************************', end='\n\n')
        # print('self.to_name -->', self.to_name)
        # print('************************************', end='\n\n')
        # print('self.value -->', self.value)
        # print('************************************', end='\n\n')

        predictions = []

        for task in tasks:
            input_text = task['data'].get(self.value) or task['data'].get('$Text')

            # get model predictions
            predicted_label = self.load_update_model.predict_one(input_text)
            score = self.load_update_model.predict_proba_one(input_text)
            max_probability = round(max(score.values()) * 100, 2)
            print(f'{score} --> {max_probability}')
            score = str(max_probability)+'%'

            result = [{
                'from_name': self.from_name,
                'to_name': self.to_name,
                'type': 'choices',
                'value': {'choices': [predicted_label]}
            }]

            # expand predictions with their scores for all tasks
            predictions.append({'result': result, 'score': max_probability})

        return predictions

    def fit(self, completions, workdir=None, **kwargs):
        # input_texts = []
        # output_labels, output_labels_idx = [], []
        label2idx = {l: i for i, l in enumerate(self.labels)}

        for completion in completions:
            # get input text from task data
            print(completion)
            if completion['annotations'][0].get('skipped') or completion['annotations'][0].get('was_cancelled'):
                continue

            input_text = completion['data'].get(self.value) or completion['data'].get('$Text')
            # input_texts.append(input_text)

            # get an annotation
            output_label = completion['annotations'][0]['result'][0]['value']['choices'][0]
            print(f'Text -> {input_text} \n Sentimental -> {output_label}')
            # output_labels.append(output_label)
            # output_label_idx = label2idx[output_label]
            # output_labels_idx.append(output_label_idx)

        # new_labels = set(output_labels)
        # if len(new_labels) != len(self.labels):
        #     self.labels = list(sorted(new_labels))
        #     print('Label set has been changed:' + str(self.labels))
        #     label2idx = {l: i for i, l in enumerate(self.labels)}
        #     output_labels_idx = [label2idx[label] for label in output_labels]

        # train the model
        # self.reset_model()
        # self.model.fit(input_texts, output_labels_idx)

        # save output resources
        # model_file = os.path.join(workdir, 'model.pkl')
        # with open(model_file, mode='wb') as fout:
        #     pickle.dump(self.model, fout)

        # train_output = {
        #     'labels': self.labels,
        #     'model_file': model_file
        # }
        # return train_output
