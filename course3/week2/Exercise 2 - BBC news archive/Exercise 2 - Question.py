import csv
import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
import numpy as np

vocab_size = 1000
embedding_dim = 16
max_length = 120
trunc_type = 'post'
padding_type = 'post'
oov_tok = '<OOV>'
training_portion = .8

sentences = []
labels = []
stopwords = [ "a", "about", "above", "after", "again", "against", "all", "am", "an", "and", "any", "are", "as", "at", "be", "because", "been", "before", "being", "below", "between", "both", "but", "by", "could", "did", "do", "does", "doing", "down", "during", "each", "few", "for", "from", "further", "had", "has", "have", "having", "he", "he'd", "he'll", "he's", "her", "here", "here's", "hers", "herself", "him", "himself", "his", "how", "how's", "i", "i'd", "i'll", "i'm", "i've", "if", "in", "into", "is", "it", "it's", "its", "itself", "let's", "me", "more", "most", "my", "myself", "nor", "of", "on", "once", "only", "or", "other", "ought", "our", "ours", "ourselves", "out", "over", "own", "same", "she", "she'd", "she'll", "she's", "should", "so", "some", "such", "than", "that", "that's", "the", "their", "theirs", "them", "themselves", "then", "there", "there's", "these", "they", "they'd", "they'll", "they're", "they've", "this", "those", "through", "to", "too", "under", "until", "up", "very", "was", "we", "we'd", "we'll", "we're", "we've", "were", "what", "what's", "when", "when's", "where", "where's", "which", "while", "who", "who's", "whom", "why", "why's", "with", "would", "you", "you'd", "you'll", "you're", "you've", "your", "yours", "yourself", "yourselves" ]
print(len(stopwords))

# read the csv file
sentences = []
labels = []
with open('./bbc-text.csv') as csvfile:
    csv_reader = csv.reader(csvfile, delimiter=",")
    first_line = True
    for row in csv_reader:
        if first_line:
            first_line = False
        else:
            labels.append(row[0])
            sentence = row[1]
            # remove the stop words from this sentence
            for word in stopwords:
                token = " " + word + " "
                sentence = sentence.replace(token, " ")
                sentence = sentence.replace("  ", " ")
            sentences.append(sentence)

print(len(labels))
print(len(sentences))
print(sentences[0][0:10])

# create training and test data
train_size = int(len(sentences) * training_portion)
train_sentences = sentences[0: train_size]
train_labels = labels[0: train_size]

validation_sentences = sentences[train_size:]
validation_labels = labels[train_size:]

print(train_size)
print(len(train_sentences))
print(len(train_labels))
print(len(validation_sentences))
print(len(validation_labels))

# create the tokenizer
tokenizer = Tokenizer(num_words=vocab_size, oov_token=oov_tok)
tokenizer.fit_on_texts(train_sentences)
word_index = tokenizer.word_index

train_sequences = tokenizer.texts_to_sequences(train_sentences)
train_padded = pad_sequences(train_sequences, maxlen=max_length, padding=padding_type, truncating=trunc_type)

print(len(train_sequences[0]))
print(len(train_padded[0]))

print(len(train_sequences[1]))
print(len(train_padded[1]))

print(len(train_sequences[10]))
print(len(train_padded[10]))

# find what the validation padded sequence should end up looking like
validation_sequences = tokenizer.texts_to_sequences(validation_sentences)
validation_padded = pad_sequences(validation_sequences, padding=padding_type, maxlen=max_length)

print(len(validation_sequences))
print(validation_padded.shape)

# create a new tokenizer for the labels (the categories)
label_tokenizer = Tokenizer()
label_tokenizer.fit_on_texts(labels)
label_index = label_tokenizer.word_index


training_label_seq = np.array(label_tokenizer.texts_to_sequences(train_labels))
validation_label_seq = np.array(label_tokenizer.texts_to_sequences(validation_labels))

print(training_label_seq[0])
print(training_label_seq[1])
print(training_label_seq[2])
print(training_label_seq.shape)

print(validation_label_seq[0])
print(validation_label_seq[1])
print(validation_label_seq[2])
print(validation_label_seq.shape)

# define the model
model = tf.keras.Sequential([
    tf.keras.layers.Embedding(vocab_size, embedding_dim, input_length=max_length),
    tf.keras.layers.GlobalAveragePooling1D(),
    tf.keras.layers.Dense(24, activation='relu'),
    # there are a total of 6 categories (labels)
    tf.keras.layers.Dense(6, activation='softmax')
])
model.compile(loss='sparse_categorical_crossentropy',optimizer='adam',metrics=['accuracy'])
model.summary()

num_epochs = 30
history = model.fit(
    train_padded, # x
    training_label_seq, # y
    epochs=num_epochs,
    validation_data=(validation_padded, validation_label_seq)
)

print("Prediction:")
sentence = ["berlin hails european cinema organisers say this year s berlin film festival  which opens on thursday with period epic man to man  will celebrate a revitalised european cinema.  of the 21 films in competition for the golden and silver bear awards  more than half are from europe with france particularly well represented. festival director dieter kosslick says this strong showing signals  a new consciousness for european films .  they re on an incredible winning streak   he told the reuters agency.  this isn t to say there aren t any good american films   he continued.  it s just that there are more good european films.   however  mr kosslick refused to accept that widespread opposition to the iraq war had turned audiences against hollywood imports.  there is no anti-american mood   he said. some 350 films will be screened at this year s festival  with a further 300 shown at the european film market that runs alongside it. more than a dozen celebrities are scheduled to attend  among them will smith  kevin spacey and keanu reeves. but mr kosslick says more would be coming had the academy awards not been brought forward to 27 february.  i m not worried that we won t be able to fill the red carpet with stars   he said  though he admitted the festival may be moved to january next year to avoid a similar clash. the 10-day berlinale runs until 20 february", 
"federer defeated nadal in an excellent championship tennis match",
"profits slide at india s dr reddy profits at indian drugmaker dr reddy s fell 93% as research costs rose and sales flagged.  the firm said its profits were 40m rupees ($915 000; £486 000) for the three months to december on sales which fell 8% to 4.7bn rupees. dr reddy s has built its reputation on producing generic versions of big-name pharmaceutical products. but competition has intensified and the firm and the company is short on new product launches. the most recent was the annoucement in december 2000 that it had won exclusive marketing rights for a generic version of the famous anti-depressant prozac from its maker  eli lilly. it also lost a key court case in march 2004  banning it from selling a version of pfizer s popular hypertension drug norvasc in the us. research and development of new drugs is continuing apace  with r&d spending rising 37% to 705m rupees - a key cause of the decrease in profits alongside the fall in sales. patents on a number of well-known products are due to run out in the near future  representing an opportunity for dr reddy  whose shares are listed in new york  and other indian generics manufacturers.  sales in dr reddy s generics business fell 8.6% to 966m rupees. another staple of the the firm s business  the sale of ingredients for drugs  also performed poorly. sales were down more than 25% from the previous year to 1.4bn rupees in the face of strong competition both at home  and in the us and europe. dr reddy s indian competitors are gathering strength although they too face heavy competitive pressures."]
sequences = tokenizer.texts_to_sequences(sentence)
padded = pad_sequences(sequences, maxlen=max_length, padding=padding_type, truncating=trunc_type)
results = model.predict(padded)
for i in results:
    for j in i:
        print(np.format_float_positional(j, trim='-'), end=" ")
    print("\n")

print(label_index)
print(set(labels))