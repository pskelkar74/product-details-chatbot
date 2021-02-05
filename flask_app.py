from flask import Flask, request
from flask_cors import CORS
import os
from transformers import DistilBertTokenizer, DistilBertForQuestionAnswering
import torch

app = Flask(__name__)
CORS(app)
app.config['SECRET_KEY'] = 'secret'

SENTENCE_DELIMITER = "|"
PARAGRAPH_DELIMITER = "\n"
MATCH_THRESHOLD = 0.5
FAILURE_MESSAGE = "Sorry :(\nWe could not find an answer for your query"

tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased',return_token_type_ids = True)
model = DistilBertForQuestionAnswering.from_pretrained('distilbert-base-uncased-distilled-squad')

@app.route('/api/v1/query', methods={'GET'})
def getQuery():
    print("- QUERY API -")

    product = request.args.get('product')
    query = request.args.get('q')

    print("Query = " +query)
    print("Product = " +product)

    fp = open(os.getcwd() + "/inputs/" +product +".txt")
    
    corpus = fp.read()
    sentences = corpus.split(SENTENCE_DELIMITER)

    max_match = 0
    top_match = str()

    for context in sentences:
        # Pass to model as context
        encoding = tokenizer.encode_plus(query, context)

        input_ids, attention_mask = encoding["input_ids"], encoding["attention_mask"]

        start_scores, end_scores = model(torch.tensor([input_ids]), attention_mask=torch.tensor([attention_mask]))

        ans_tokens = input_ids[torch.argmax(start_scores) : torch.argmax(end_scores)+1]
        answer_tokens = tokenizer.convert_ids_to_tokens(ans_tokens , skip_special_tokens=True)
        answer_tokens_to_string = tokenizer.convert_tokens_to_string(answer_tokens)

    

        # Find max match
        # If match > max_match, save sentence
        


    if max_match < MATCH_THRESHOLD:
        return FAILURE_MESSAGE

    elif top_match == "":
        return FAILURE_MESSAGE

    else:
        return top_match

    # return "Query Result"

# Match whole corpus as context

# Match each paragraph


if __name__ == '__main__':
    app.run(debug=True)