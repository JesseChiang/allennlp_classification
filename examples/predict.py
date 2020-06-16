from allennlp.models.archival import load_archive
from allennlp.predictors import Predictor
from allennlp.common.util import import_submodules
import logging


from flask import Flask, request, jsonify


def _get_predictor(archive_path, predictor) -> Predictor:
    archive = load_archive(archive_path)
    return Predictor.from_archive(archive, predictor)


class I6Predictor(object):
    def __init__(self):
        archive_path = 'C:\\Users\\jesse\\Documents\\Datago\\model\\i6\\model.tar.gz'
        self.predictor = _get_predictor(archive_path, 'text_classifier')

    def predict_json(self, json_data):
        logging.info(f'预测')
        prediction = self.predictor.predict_json(json_data)
        return {'label': prediction['label'], 'prob': max(prediction['probs'])}


def make_app(predictor):
    app = Flask(__name__)

    @app.route('/')
    @app.route('/predict', methods=['POST'])
    def processing():
        data = {'sentence': request.form.get('sentence')}
        print(data)
        result = predictor.predict_json(data)
        return jsonify(result)

    return app


if __name__ == '__main__':
    logging.basicConfig(format="%(asctime)s %(name)s:%(levelname)s:%(message)s",
                        datefmt='%Y-%m-%d %H:%M:%S', level=logging.INFO)

    import_submodules('AllenFrame')

    logging.info(f'实例化')
    predictor = I6Predictor()
    logging.info(f'创建app')
    app = make_app(predictor)

    app.run(host="0.0.0.0", port=8001)
