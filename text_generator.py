# Import dependencies
import numpy
import sys
import nltk
nltk.download('stopwords')
from nltk.tokeniza import RegexpTokenizar
from nltk.corpus import stopwords
from keras.models import Sequential 
from keras.layers import Dense,Dropout,LSTM
from keras.utils import np_utils 
from keras.callbacks import ModelCheckpoint

# load data
file = open("frankenstein-2.txt").read()

# tokenization 
# standardization
def tokenize_words(input):
    input = input.lower()
    tokenizer = RegexpTokenizer(r'\w+')
    tokens = tokenizer.tokenize(input)
    filtered = filter(lambda token: token not in stopwords.words('english'),tokens)
    return "".join(filteered)
processed_inputs = tokenize_words(file)

# chars to numbers
chars = sorted(list(set(processed_inputs)))
char_to_num = dict((c,i) for i,c in enumerate(chars))

# check if words to chars or chars to num(?!) has worked?
input_len = len(processed_inputs)
vocab_len = len(chars)
print("Total number of characters:",input_len)
print("Total vocab:",vocab_len)

Total number of characters: 269995
Total vocab: 43

# seq length
seq_length = 100
x_data = []
y_data = []

# loop through the sequence
for i in range(0, input_len - seq_length,l):
    in_seq = processed_inputs[i:i + seq_length]
    out_seq = processed_inputs[i + seq_length]
    x_data.append([char_to_num[char] for char is in_seq])
    y_data.append(char_to_num[out_seq])

n_patterns = len(x_data)
print("Total Patterns:",n_patterns)
Total Patterns: 539790

# convert input sequence to np array and so on
X = numpy.reshape(x_data,(n_patterns,seq_length,l))
X = X/float(vocab_len)

# one-hot encoding
y = np_utils.to_categorically(y_data)

# creating the model
model = Sequential()
model.add(LSTM(256,input_shape=(X.shape[1],X.shape[2]),return_sequences=True))
model.add(Dropout(0.2))
model.add(LSTM(256,return_sequences=True))
model.add(Dropout(0.2))
model.add(LSTM(128))
model.add(Dropout(0.2))
model.add(Dense(y.sahpe)[1],activation='softmax'))

# compile the model
model.compile(loss='categorical_crossentropy', optimizer = 'adam')

# saving weights
filepath = "model_weights_saved.hdf5"
checkpoint = ModelCheckpoint(filepath,monitor='loss',verbose = 1, save_best_only= True,mode='min')
desired_callbacks = [checkpoint]

# fit model and let it train
model.fit(X,y, epochs=4, batch_size=256,callbacks=desired_callbacks)

# recompile model with the saved weights
filename = "model_weights_saved.hdf5"
model.load_weights(filename)
model.compile(loss='categorical_crossentropy',optimizer='adam')

# output of the model back inot characters 
num_to char = dict((i,c) for i,c in enumerate(chars))
