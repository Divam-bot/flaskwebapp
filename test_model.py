import tensorflow as tf
import  numpy as np
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.layers import Embedding, LSTM, Dense, Dropout, Bidirectional
from tensorflow.keras.preprocessing.text import Tokenizer
import json , io
import pickle
tokenizer = Tokenizer()
sonnetwriter = tf.keras.models.load_model('sonnet_generator_weights.h5')

data = open("sonnets.txt").read()
corpus = data.lower().split("\n")
     
print(corpus[1:10])

tokenizer.fit_on_texts(corpus)
#print(tokenizer.word_index)



# saving
with open('tokenizer.pickle', 'wb') as handle:
    pickle.dump(tokenizer, handle, protocol=pickle.HIGHEST_PROTOCOL)
    

'''
seed_text =input()
print([seed_text])
for _ in range(100):
    token_list = tokenizer.texts_to_sequences([seed_text])[0]
    print(token_list)
    token_list = pad_sequences([token_list], maxlen=10, padding='pre')
    predicted = sonnetwriter.predict_classes(token_list, verbose=0)
    output_word = ""
    for word, index in tokenizer.word_index.items():
        if index == predicted:
            output_word = word
            break
    seed_text += " " + output_word


    #print(seed_text)
   # print(tokenizer.word_index)
    
'''