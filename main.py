import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, TimeDistributed, Activation, Dense
import string

DATA_DIR = 'jane.txt'
SEQ_LENGTH = 100
HIDDEN_DIM = 700
LAYER_NUM = 3
BATCH_SIZE = 12

data = open(DATA_DIR, 'r', encoding='latin-1').read()

valid_characters = string.ascii_letters + ".,! -'" + string.digits
character_to_int = {}
int_to_character = {}
for index in range(len(valid_characters)):
    character = valid_characters[index]
    character_to_int[character] = index
    int_to_character[index] = character

training_string = ""
for character in data:
    if character in valid_characters:
        training_string += character
    elif character == '\n':
        training_string += ' '

while True:
    if "  " in training_string:
        training_string = training_string.replace("  ", ' ')
    else:
        break

target_string = training_string[1:] + training_string[0]

X = []
y = []
for i in range(0, len(training_string), SEQ_LENGTH):
    training_sequence = training_string[i:(i + SEQ_LENGTH)]
    integer_training_sequence = [character_to_int[value] for value in training_sequence]
    input_sequence = np.zeros((SEQ_LENGTH, len(valid_characters)))
    if len(integer_training_sequence) == SEQ_LENGTH:
        for j in range(SEQ_LENGTH):
            input_sequence[j][integer_training_sequence[j]] = 1.
    X.append(input_sequence)

    y_sequence = target_string[i:(i + SEQ_LENGTH)]
    print(training_sequence, '|', y_sequence)
    y_sequence_ix = [character_to_int[value] for value in y_sequence]
    target_sequence = np.zeros((SEQ_LENGTH, len(valid_characters)))
    if len(y_sequence_ix) == SEQ_LENGTH:
        for j in range(SEQ_LENGTH):
            target_sequence[j][y_sequence_ix[j]] = 1.
    y.append(target_sequence)

X = np.reshape(X, (-1, SEQ_LENGTH, len(valid_characters)))
y = np.reshape(y, (-1, SEQ_LENGTH, len(valid_characters)))

model = Sequential()
model.add(LSTM(HIDDEN_DIM, input_shape=(None, len(valid_characters)), return_sequences=True))
for i in range(LAYER_NUM - 1):
    model.add(LSTM(HIDDEN_DIM, return_sequences=True))
model.add(TimeDistributed(Dense(len(valid_characters))))
model.add(Activation('softmax'))
model.compile(loss="categorical_crossentropy", optimizer="adam")


def generate_text(model, length):
    ix = [np.random.randint(len(valid_characters))]
    y_char = [int_to_character[ix[-1]]]
    X = np.zeros((1, length, len(valid_characters)))
    for i in range(length):
        X[0, i, :][ix[-1]] = 1
        print(int_to_character[ix[-1]], end="")
        ix = np.argmax(model.predict(np.array(X[:, :i + 1, :]))[0], 1)
        y_char.append(int_to_character[ix[-1]])
    return ''.join(y_char)

GENERATE_LENGTH = 20
nb_epoch = 0
while True:
    model.fit(X, y, batch_size=BATCH_SIZE, verbose=1, epochs=1)
    nb_epoch += 1
    generate_text(model, GENERATE_LENGTH)
    if nb_epoch % 10 == 0:
        model.save_weights('checkpoint_{}_epoch_{}.hdf5'.format(HIDDEN_DIM, nb_epoch))
