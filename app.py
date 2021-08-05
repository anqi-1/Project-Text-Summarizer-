from transformers import pipeline
import re
from flask import Flask, render_template, request, jsonify
import pickle
import nltk
import numpy as np
from nltk import tokenize
nltk.download('punkt')


app = Flask(__name__)

# load model
#model = pickle.load(open('forest_taxi1.pkl', 'rb'))


def generate_abstractiveSummary(text):
    from transformers import PegasusTokenizer, PegasusForConditionalGeneration

    model = PegasusForConditionalGeneration.from_pretrained(
        'google/pegasus-xsum')
    tokenizer = PegasusTokenizer.from_pretrained('google/pegasus-xsum')

    inputs = tokenizer([text], max_length=100, return_tensors='pt')

    # Generate Summary
    summary_ids = model.generate(inputs['input_ids'])
    p_summary = [tokenizer.decode(
        g, skip_special_tokens=True, clean_up_tokenization_spaces=False) for g in summary_ids]
    return p_summary[0]


def generate_extractiveSummary(text):
    # split into sentences
    sentences = tokenize.sent_tokenize(text)

    # lower caps
    lst = [s.lower() for s in sentences]

    # remove non-digit char
    new_lst = [re.sub('[^A-z]', ' ', s) for s in lst]

    # split list of sentences into list of list of words
    words = [tokenize.word_tokenize(s) for s in new_lst]
    # words

    # word_freq
    word_freq = {}
    for sent in words:
        for word in sent:
            if word in word_freq:
                word_freq[word] += 1
            else:
                word_freq[word] = 1

    # get weight of each sentence
    sent_weight = []
    for sent in words:
        weight = 0
        for word in sent:
            weight += word_freq[word]
        sent_weight.append(weight)

    # argsort, get top two sentences and join tgt
    indice = (-np.array(sent_weight)).argsort()[:2]
    empty_str = ""
    for i in list(indice):
        empty_str += sentences[i] + " "

    return empty_str


@app.route("/")
def home():
    return render_template('index.html')


@app.route('/summary', methods=['POST'])
def extractive_summary():

    text = request.form['input_text']
    summaryE = generate_extractiveSummary(text)
    summaryA = generate_abstractiveSummary(text)

    return render_template('index.html', title="Extractive Summarizer", original_text=text, output_summary1=summaryE, output_summary2=summaryA)


if __name__ == "__main__":
    app.run(debug=True)
