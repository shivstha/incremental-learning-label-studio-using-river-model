import os
import pickle

import joblib
import numpy as np
from label_studio_ml.model import LabelStudioMLBase
from river import (compose, feature_extraction, linear_model, metrics,
                   naive_bayes, preprocessing)


# Date          :: 2021/06/21
# Desc          :: Class for inheriting LabelStudioMLBase backend for predicting and training tasks
class StockDataClassifier(LabelStudioMLBase):

    def __init__(self, **kwargs):
        super(StockDataClassifier, self).__init__(**kwargs)
        self.from_name, self.info = list(self.parsed_label_config.items())[0]
        self.to_name = self.info['to_name'][0]
        self.value = self.info['inputs'][0]['value']

        self.labels = self.info['labels']
        self.load_update_model = joblib.load("./model_trained.bin")

    def predict(self, tasks, **kwargs):
        predictions = []

        for task in tasks:
            input_text = task['data'].get(self.value) or task['data'].get('$Text')

            # get model predictions
            predicted_label = self.load_update_model.predict_one(input_text)
            score = self.load_update_model.predict_proba_one(input_text)
            max_probability = round(max(score.values()) * 100, 2)
            print(f'{score} --> {max_probability}')
            score = str(max_probability) + '%'

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
        for completion in completions:
            # get input text from task data
            print(completion)
            if completion['annotations'][0].get('skipped') or completion['annotations'][0].get('was_cancelled'):
                continue

            input_text = completion['data'].get(self.value) or completion['data'].get('$Text')

            # get an annotation
            output_label = completion['annotations'][0]['result'][0]['value']['choices'][0]
            print(f'Text -> {input_text} \n Sentimental -> {output_label}')
            print(f'before learning probability --> {self.load_update_model.predict_proba_one(input_text)}')
            self.load_update_model.learn_one(input_text, output_label)
            print(f'after learning probability --> {self.load_update_model.predict_proba_one(input_text)}')

        model_file = os.path.join(workdir, 'model_trained_human.bin')

        joblib.dump(self.load_update_model, model_file)

        train_output = {
            'model_file': model_file
        }
        return train_output
