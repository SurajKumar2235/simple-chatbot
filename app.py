from flask import Flask, jsonify, render_template, request
import numpy as np
import nltk 
import string
import random
import warnings
warnings.simplefilter('ignore')
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

## import and read corpos


nltk.download('punkt') # punct tokenizer
nltk.download('wordnet')# dictionary


f=open('chatbot.txt','r',errors='ignore')
raw_doc=f.read()
raw_doc=raw_doc.lower()
sent_token = nltk.sent_tokenize(raw_doc) #converting doc to list of sentences
word_token = nltk.word_tokenize(raw_doc) #converting doc to list of tokenozer

lemnt=nltk.stem.WordNetLemmatizer()

def lemmatize(words):
    return [lemnt.lemmatize(word) for word in words]


remove_punct_dict=dict((ord(punct),None) for punct in string.punctuation)

def LemNormilazation(text):
    return lemmatize(nltk.word_tokenize(text.lower().translate(remove_punct_dict)))


GREET_INPUTS=('hello','hi','greetings','sup',"what's up","hey",'hi')

GREET_RESPONSE=['hi','hey',
                '*nods*',
                "hi there",
                'hello',
                'i am glad',
                'how are you?',

                ]
def greet(sentence):
    for word in sentence.split():
        if word.lower() in GREET_INPUTS:
            return random.choice(GREET_RESPONSE)
        
vectorizer = TfidfVectorizer(tokenizer=LemNormilazation, stop_words='english')
Tfidf = vectorizer.fit_transform(sent_token)

def response(user_input):
    chat_response=''
    user_input = LemNormilazation(user_input)
    user_input.append(user_input[-1])
    tfidf_values = vectorizer.transform(user_input)
    cosine_sim_value=cosine_similarity(tfidf_values[-1], Tfidf)
    idx = cosine_sim_value.argsort()[0][-2]
    flat = cosine_sim_value.flatten()
    flat.sort()
    req_tfidf=flat[-2]

    if req_tfidf==0:
        chat_response=chat_response+"I am sorry! I don't understand you"
        return chat_response
    else:
        chat_response=chat_response+sent_token[idx]

        return chat_response






app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/get_response', methods=['POST'])
def get_response():
    user_input = request.form['user_input']
    
    if user_input not in ['bye', 'tata', "let's talk later"]:
        if user_input == 'thanks' or user_input == "thank you":
            return jsonify(response='ok thanks for the conversation')
        else:
            res = greet(user_input)
            if res:
                return jsonify(response=res)
            else:
                sent_token.append(user_input)
                word_token.extend(nltk.word_tokenize(user_input))
                final_words = list(set(word_token))
                return jsonify(response=response(user_input))
    else:
        return jsonify(response='bye')

if __name__ == '__main__':
    app.run(debug=True)
