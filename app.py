from flask import Flask, request
from transformers import TFDistilBertForSequenceClassification, DistilBertTokenizerFast

from analyzer import Analyzer

model = TFDistilBertForSequenceClassification.from_pretrained("Maheshwaranr/bert-analyzer")
tokenizer = DistilBertTokenizerFast.from_pretrained('distilbert-base-uncased')

app = Flask(__name__)


@app.route('/')
def index():
    return 'Hi there! everything working fine..'


@app.route('/ping', methods=['POST', 'GET'])
def ping():
    req = request.json
    return f"This is a post-request ping check dear {req['name']} and everything works fine here as well"


@app.route('/analyze', methods=['POST', 'GET'])
def analyze():
    req = request.json
    reviews, resp = None, {}

    if req:
        reviews = req['reviews']
    if reviews:
        bert_analyzer = Analyzer(
            model=model,
        )
        labels = bert_analyzer.analyze(
            reviews,
            tokenizer=tokenizer,
        )
        resp = bert_analyzer.generate_response(labels)
    return response.json(resp)


app.run()
