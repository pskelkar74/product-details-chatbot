from flask import Flask, request, jsonify
from flask_cors import CORS
import os
from transformers import DistilBertTokenizer, DistilBertForQuestionAnswering, pipeline
import torch

app = Flask(__name__)
CORS(app)
app.config['SECRET_KEY'] = 'secret'

SENTENCE_DELIMITER = "\n"
PARAGRAPH_DELIMITER = "\n"
MATCH_THRESHOLD = 0.5
FAILURE_MESSAGE = "Sorry :(\nWe could not find an answer for your query"

tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased',return_token_type_ids = True)
model = DistilBertForQuestionAnswering.from_pretrained('distilbert-base-uncased-distilled-squad')
nlp = pipeline("question-answering", model=model, tokenizer=tokenizer)

# Match sentence by sentence
@app.route('/api/sentencematch/query', methods={'GET'})
def getQuery():
    print("- QUERY API -")

    product = request.args.get('product')
    query = request.args.get('q')

    print("Query = " +query)
    print("Product = " +product)

    fp = open(os.getcwd() + "/data/" +product +".txt")
    
    corpus = fp.read()
    sentences = corpus.split(SENTENCE_DELIMITER)

    print("Found " +str(len(sentences)) +" contexts")

    sentences = [i.strip() for i in sentences if i] 

    max_match = 0
    top_match = str()

    matches = dict()

    for context in sentences:
        # Pass to model as context
        result = nlp(question=query, context=context, model=model, tokenizer=tokenizer)
        print("Calculated result for context " ,context.index)

        matches[result['answer']] = result['score']

        # Find max match
        # If match > max_match, save sentence
        if result['score'] > MATCH_THRESHOLD and max_match < result['score']:
            max_match = result['score']
            top_match = result['answer']
        
    print("MATCHES AND SCORES : \n" +str(matches))

    if top_match == "":
        print("ANSWER = NOT FOUND")
        return jsonify(
            answer=FAILURE_MESSAGE,
            confidence=max_match
        )

    else:
        print("ANSWER = " +top_match +"\n" +"CONFIDENCE = " +str(max_match))
        return jsonify(
            answer=top_match,
            confidence=max_match
        )


# Match whole corpus as context
@app.route('/api/corpusmatch/query', methods={'GET'})
def getQueryv2():
    print("- QUERY API -")

    product = request.args.get('product')
    query = request.args.get('q')

    print("Query = " +query)
    print("Product = " +product)

    fp = open(os.getcwd() + "/data/" +product +".txt")
    
    corpus = fp.read()

    max_match = 0
    top_match = str()

    result = nlp(question=query, context=corpus)
    max_match = result['score']

    if result['score'] > MATCH_THRESHOLD:
        top_match = result['answer']

    if top_match == "":
        print("ANSWER = NOT FOUND\nCONFIDENCE = " +str(max_match))
        return jsonify(
            answer=FAILURE_MESSAGE,
            confidence=max_match
        )

    else:
        print("ANSWER = " +top_match +"\n" +"CONFIDENCE = " +str(max_match))
        return jsonify(
            answer=top_match,
            confidence=max_match
        )


# todo : cluster sentences into paragraphs
# todo : make a paragraph-by-paragraph API

if __name__ == '__main__':
    app.run(debug=True)