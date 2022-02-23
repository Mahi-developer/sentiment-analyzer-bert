from sanic import Sanic, response
from transformers import TFDistilBertForSequenceClassification, DistilBertTokenizerFast

from analyzer import Analyzer

model = TFDistilBertForSequenceClassification.from_pretrained("Maheshwaranr/bert-analyzer")
tokenizer = DistilBertTokenizerFast.from_pretrained('distilbert-base-uncased')

app = Sanic(
    name=__name__
)


@app.route('/')
async def index(request):
    return response.text('Hi there! everything working fine..')


@app.route('/ping', methods=['POST', 'GET'])
async def index(request):
    return response.text(
        f"This is a post-request ping check dear {request.json['name']} and everything works fine here as well"
    )


@app.route('/analyze', methods=['POST', 'GET'])
async def index(request):
    req = request.json
    reviews, resp = None, {}

    if req:
        reviews = req['reviews']
    if reviews:
        bert_analyzer = Analyzer(
            model=model,
        )
        labels = await bert_analyzer.analyze(
            reviews,
            tokenizer=tokenizer,
        )
        resp = await bert_analyzer.generate_response(labels)
    return response.json(resp)


app.run()
