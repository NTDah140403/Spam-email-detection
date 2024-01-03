from fastapi import FastAPI
import joblib
import uvicorn 
from mangum import Mangum
import dill
import string
from nltk import PorterStemmer,WordNetLemmatizer, corpus
from nltk.tokenize import word_tokenize
import nltk
nltk.data.path.append("nltkdata")


app = FastAPI(
    title="Spam email detection Model API",
    description="A simple API that use NLP model to detect spam emails",
    version="0.1",
)
handler = Mangum(app)
with open( 'vocab_ham.pkl', "rb") as f:
    vocab_ham = dill.load(f)
with open('vocab_spam.pkl', "rb") as f:
    vocab_spam = dill.load(f)

with open( "model.pkl", "rb") as f:
    model = dill.load(f)


def preprocesser(text,stem= True):
    stopwords = corpus.stopwords.words("english")
    stopwords.append('\n')
    ps = PorterStemmer()
    wn = WordNetLemmatizer()
    nonP_text = "".join([char for char in text if char not in string.punctuation])
    tok_text = word_tokenize(nonP_text)
    noStop_text = [word for word in tok_text if word not in stopwords]
    if stem == True:
        stem_text = [ps.stem(word) for word in noStop_text]
        return stem_text
    lama_text = [wn.lemmatize(word) for word in noStop_text]
    return lama_text

def common_words(text, is_spam=True):
    tok =  set(preprocesser(text))
    res = []
    if is_spam:
        dict_token = {it : vocab_spam.get(it) for it in tok}
    else:
        dict_token = {it : vocab_ham.get(it) for it in tok}
    sorted_dict_token = sorted(dict_token.items(), key = lambda item: 0 if item[1] is None else item[1],reverse=True)
    if len(sorted_dict_token) < 10:
        for i in range(len(sorted_dict_token)):
            if sorted_dict_token[i][1]:
                res.append(sorted_dict_token[i])    
        return res
    for i in range(10):
        if sorted_dict_token[i][1]:
            res.append(sorted_dict_token[i])    
    return res

@app.get("/")
def root():
    return {"messages":"spam_email_model"}
@app.get("/detect_spam_email/{email}")
def detect_spam(email: str):
    tokens = [preprocesser(email)]
    res = model.predict(tokens)[0]
    words = { x[0]:x[1] for x in common_words(email,is_spam=res)}
    return {"prediction": int(res), "words": words}

if __name__ == "__main__":
   uvicorn.run(app, host="0.0.0.0", port=8080)