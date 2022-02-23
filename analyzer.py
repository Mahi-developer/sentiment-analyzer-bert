from datetime import datetime
import tensorflow as tf


class Analyzer:

    def __init__(self, model):
        self.model = model
        self.created_dtm = datetime.now()
        self.modified_dtm = None

    async def analyze(self, reviews, tokenizer):
        predict_input = tokenizer(
            reviews,
            max_length=256,
            padding=True,
            truncation=True,
            return_tensors='tf'
        )

        tf_output = self.model(predict_input)

        tf_prediction = tf.nn.softmax(tf_output[0], axis=-1)
        label = tf.argmax(tf_prediction, axis=1)
        label = label.numpy()
        self.modified_dtm = datetime.now()
        return label

    async def generate_response(self, labels):
        total_reviews = len(labels)
        positive_reviews = (labels == 1).sum()
        negative_reviews = (labels == 0).sum()
        response = {
            "meta-data": {
              "created_dtm": str(self.created_dtm),
              "completed_dtm": str(self.modified_dtm)
            },
            "total_processed": int(total_reviews),
            "positive": int(positive_reviews),
            "negative": int(negative_reviews)
        }
        return response
