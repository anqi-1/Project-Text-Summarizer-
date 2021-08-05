from transformers import pipeline
import re
from flask import Flask, render_template, requests
import nltk
import numpy as np
from nltk import tokenize
nltk.download('punkt')


app = Flask(__name__)


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



@app.route("/")
def home():
    def streamed_proxy():
        r = requests.get(my_path_to_server01, stream=True)
        return Response(r.iter_content(chunk_size=10*1024),
                    content_type=r.headers['Content-Type'])
    return render_template('index.html')



@app.route('/summary', methods=['POST'])
def extractive_summary():

    text = request.form['input_text']
    #summaryE = generate_extractiveSummary(text)
    summaryA = generate_abstractiveSummary(text)

    return render_template('index.html', title="Extractive Summarizer", original_text=text, output_summary2=summaryA)


if __name__ == "__main__":
    app.run(debug=True)
