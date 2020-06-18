from allennlp.models.archival import load_archive
from allennlp.predictors import Predictor
from allennlp.common.util import import_submodules
import logging


from flask import Flask, request, jsonify


def _get_predictor(archive_path, predictor) -> Predictor:
    archive = load_archive(archive_path)
    return Predictor.from_archive(archive, predictor)


class RiskPredictor(object):
    def __init__(self, archive_path):
        self._pri_predictor = _get_predictor(archive_path, 'text_classifier')
        self._sec_predictors = dict()

    def add_predictor(self, label, model_path):
        self._sec_predictors[label] = _get_predictor(model_path, 'text_classifier')

    def predict_json(self, json_data):
        # logging.info(f'预测')
        pri_prediction = self._pri_predictor.predict_json(json_data)
        print(pri_prediction)
        sec_predictor = self._sec_predictors.get(pri_prediction['label'], None)
        print(sec_predictor)
        sec_prediction = sec_predictor.predict_json(json_data) if sec_predictor else {'label': '', 'probs': [0]}
        return {'pri_label': pri_prediction['label'],
                'pri_prob': max(pri_prediction['probs']),
                'sec_label': sec_prediction['label'],
                'sec_prob': max(sec_prediction['probs'])}


def make_app(predictor):
    app = Flask(__name__)

    @app.route('/predict', methods=['POST'])
    def processing():
        sentences = request.form.get('content').split('\n')
        sentences.append(request.form.get('title'))
        result = {'pri_label': set(), 'sec_label': set()}
        for sentence in sentences:
            prediction = predictor.predict_json({'sentence': sentence})
            result['pri_label'].add(prediction['pri_label'])
            sec_label = prediction['sec_label']
            if sec_label != '':
                result['sec_label'].add(prediction['sec_label'])
        result['pri_label'] = list(result['pri_label'])
        result['sec_label'] = list(result['sec_label'])
        return jsonify(result)
    return app


if __name__ == '__main__':
    logging.basicConfig(format="%(asctime)s %(name)s:%(levelname)s:%(message)s",
                        datefmt='%Y-%m-%d %H:%M:%S',
                        level=logging.WARNING)

    import_submodules('AllenFrame')

    logging.info(f'实例化')
    path = '~/workplace/news_risk/model/i/model.tar.gz'
    predictor = RiskPredictor(path)
    predictor.add_predictor('企业经营相关', '~/workplace/news_risk/model/i3/model.tar.gz')
    predictor.add_predictor('产品质量相关', '~/workplace/news_risk/model/i6/model.tar.gz')
    predictor.add_predictor('企业管理相关', '~/workplace/news_risk/model/i4/model.tar.gz')

    logging.info(f'创建app')
    app = make_app(predictor)
    app.run(host="0.0.0.0", port=8001)
