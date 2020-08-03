# this python module is to predict writing 
# based on the previously traing module 

# prepair a seed text string 
# Reading in files as a string text
def read_file(filepath):  
    with open(filepath) as f:
        str_text = f.read()
    return str_text

read_file('dick_chapters.txt')

#Tokenize and Clean Text
import spacy
nlp = spacy.load('en',disable=['parser', 'tagger','ner'])
​nlp.max_length = 1198623

# remove punctuations
def separate_punc(doc_text):
    return [token.text.lower() for token in nlp(doc_text) if token.text not in '\n\n \n\n\n!"-#$%&()--.*+,-/:;<=>?@[\\]^_`{|}~\t\n ']

d = read_file('dick_chapters.txt')
tokens = separate_punc(d)

#Create Sequences of Tokens
# one of sequences can be used as a seed text string
train_len = 50+1 
​text_sequences = []
​
for i in range(train_len, len(tokens)):
    # Grab train_len# amount of characters
    seq = tokens[i-train_len:i]
    # Add to list of sequences
    text_sequences.append(seq)

#predict writing
from pickle import load
from keras.models import load_model
from keras.preprocessing.sequence import pad_sequences

# import previously trained module and tokenizer
model = load_model('model_30.h5')
tokenizer = load(open("model_30", "rb"))

def generate_text(model, tokenizer, seq_len, seed_text, num_gen_words):    
    # Final Output
    output_text = []
    # Intial Seed text string
    input_text = seed_text
    # Create num_gen_words
    for i in range(num_gen_words):
        # Take the input text string and encode it to a sequence
        encoded_text = tokenizer.texts_to_sequences([input_text])[0]
        # Pad sequences to our trained rate (50 words in the video)
        pad_encoded = pad_sequences([encoded_text], maxlen=seq_len, truncating='pre')
        # Predict Class Probabilities for each word
        pred_word_ind = model.predict_classes(pad_encoded, verbose=0)[0]
        # Grab word
        pred_word = tokenizer.index_word[pred_word_ind] 
        # Update the sequence of input text (shifting one over with the new word)
        input_text += ' ' + pred_word
        output_text.append(pred_word)        
    # Make it look like a sentence.
    return ' '.join(output_text)

#take a random seed sequence
random_seed_text = text_sequences[0]
seed_text = ' '.join(random_seed_text)
generate_text(model,tokenizer,seq_len,seed_text=seed_text,num_gen_words=50)
# predictive writing of 50 words:
#"was a little years ago or randolphs and be have a early deal in the middle of the room and was the harpooneer was a whale 's blanket as the great arched parliament tapered colour the counterpane and a purse and a purse and a purse and a purse and"
