# import pandas as pd
# import numpy as np
import string
import re
from nltk import PorterStemmer,WordNetLemmatizer, corpus




def preprocessing(text,stem= True):
    stopwords = corpus.stopwords.words("english")
    stopwords.append('\n')
    ps = PorterStemmer()
    wn = WordNetLemmatizer()
    nonP_text = "".join([char for char in text if char not in string.punctuation])
    tok_text = re.split("\W+", nonP_text)
    noStop_text = [word for word in tok_text if word not in stopwords]
    if stem == True:
        stem_text = [ps.stem(word) for word in noStop_text]
        return stem_text
    lama_text = [wn.lemmatize(word) for word in noStop_text]
    return lama_text
if __name__ == "__main__":
    pass
    # print(preprocessing("job posting - apple-iss research center,content - length : 3386 apple-iss research center a us $ 10 million joint venture between apple computer inc . and the institute of systems science of the national university of singapore , located in singapore , is looking for : a senior speech scientist - - - - - - - - - - - - - - - - - - - - - - - - - the successful candidate will have research expertise in computational linguistics , including natural language processing and * * english * * and * * chinese * * statistical language modeling . knowledge of state-of - the-art corpus-based n - gram language models , cache language models , and part-of - speech language models are required . a text - to - speech project leader - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - the successful candidate will have research expertise expertise in two or more of the following areas : computational linguistics , including natural language parsing , lexical database design , and statistical language modeling ; text tokenization and normalization ; prosodic analysis . substantial knowledge of the phonology , syntax , and semantics of chinese is required . knowledge of acoustic phonetics and / or speech signal processing is desirable . both candidates will have a phd with at least 2 to 4 years of relevant work experience , or a technical msc degree with at least 5 to 7 years of experienc e . very strong software engineering skills , including design and implementation , and productization are required in these positions . knowledge of c , c + + and unix are preferred . a unix & c programmer - - - - - - - - - - - - - - - - - - - - we are looking for an experienced unix & c programmer , preferably with good industry experience , to join us in breaking new frontiers . strong knowledge of unix tools ( compilers , linkers , make , x - windows , e - mac , . . . ) and experience in matlab required . sun and silicon graphic experience is an advantage . programmers with less than two years industry experience need not apply . these positions include interaction with scientists in the national university of singapore , and with apple s speech research and productization efforts located in cupertino , california . attendance and publication in international scientific / engineering conferences is encouraged . benefits include an internationally competitive salary , housing subsidy , and relocation expenses . _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ send a complete resume , enclosing personal particulars , qualifications , experience and contact telephone number to : mr jean - luc lebrun center manager apple - iss research center , institute of systems science heng mui keng terrace , singapore 0511 tel : ( 65 ) 772-6571 fax : ( 65 ) 776-4005 email : jllebrun @ iss . nus . sg"))