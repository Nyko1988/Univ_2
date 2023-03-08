import pandas as pd
import numpy as np
import nltk
from collections import Counter
import tokenizer as tokenizer
from nltk.tokenize import TreebankWordTokenizer
from nltk.corpus import stopwords
from collections import Counter
import text
from collections import OrderedDict

# one-hot vector
# sentence = "The domain of a predicate variable is the collection of all possible values that the variable may take."
# token_sentence = sentence.split()
# number_tokens = len(token_sentence)
# vocab = sorted(set(token_sentence))
# vocab_size = len(vocab)
# print(number_tokens, vocab_size)
# onehot_vectors = np.zeros((number_tokens, vocab_size), str)
# for i, word in enumerate(token_sentence):
#     onehot_vectors[i, vocab.index(word)] = 1
# print(pd.DataFrame(onehot_vectors, columns=vocab))


# print("Bag-of-words vector for a lot of sentences")
# sentences = "The domain of a predicate variable is the collection of all possible values that the variable may take. \n"
# sentences += "Construction was done mostly by local masons and carpenters. \n"
# sentences += "That may be true or false depending on the values of these variables.\n"
# sentences += "Predicate logic extends propositional logic"
#
# corpus = {}
# for i, sent in enumerate(nltk.tokenize.sent_tokenize(sentences)):
#     corpus['sent{}'.format(i)] = dict((tok, 1) for tok in nltk.tokenize.word_tokenize(sent))
# df = pd.DataFrame(corpus).fillna(0).astype(int)
# print(df)
#
# df = df.T
# print(df)

# sentence = text.text
# tokenizer = TreebankWordTokenizer()
# text_tokens = tokenizer.tokenize(sentence.lower())
# tokens_without_sw = [word for word in text_tokens if not word in stopwords.words()]
# bag_of_words = Counter(tokens_without_sw)
# print(bag_of_words)
#
# document_vector = []
# doc_len = len(tokens_without_sw)
# tokens_counts = Counter(tokens_without_sw)
# for key, value in tokens_counts.most_common():
#     document_vector.append(round(value / doc_len, 4))
# print(document_vector)
docs = [text.text_1_sent, text.text_2_sent, text.text_3_sent, text.text_4_sent]
print(docs)
doc_tokens = []
tokenizer2 = TreebankWordTokenizer()
for doc in docs:
    doc_tokens += [sorted(tokenizer2.tokenize(doc.lower()))]
print(len(doc_tokens[0]))
all_doc_tokens = sum(doc_tokens, [])
print(len(all_doc_tokens))
lexicon = sorted(set(all_doc_tokens))
print(len(lexicon))
print(lexicon)
zero_vector = OrderedDict((token, 0) for token in lexicon)
print(zero_vector, len(zero_vector), sep='\n')
