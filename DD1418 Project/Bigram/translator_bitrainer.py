import math
import os
import argparse
import time
from collections import defaultdict
import nltk
from halo import Halo


class BigramTrainer(object):
    def __init__(self, filenames):

        self.__sources = filenames  # De filer som ska läsas igenom

        self.tokens = list()
        self.__vocab = set()

        # BigramTrainer inits följer nedan
        # En dictionary används för att kunna få index för respektive ord.
        self.index = dict()
        # En dictionary för att få ordet för respektive index.
        self.word = dict()

        # Unigram-count för varje ord
        self.unigram_count = defaultdict(int)  # No KeyError. Creates Nil value=0

        # Bigram-count för ordpar
        self.bigram_count = defaultdict(lambda: defaultdict(int))

        # Trigram-count som i slutändan inte användes i vår implementation
        self.trigram_count = defaultdict(lambda: defaultdict(lambda: defaultdict(int)))

        # Index för tidigare ord. Används av process_token() för att komma ihåg vilka ord som kom innan.
        # Nödvändigt för att kunna beräkna bigram-count och unigram-count.
        self.last_index = -1
        self.last_index2 = -1

        # Antal unika ord i de träningskorpus som lästs in.
        self.unique_words = 0

        # Totalt antal ord i träninskorpus som lästs in.
        self.total_words = 0

    def process_files(self, n_files):
        """
        Går igenom n_files(en integer) stycken filer som den hämtar från self.__sources.
        För varje fil gör den tokens av orden.

        När den tokeniserat n_files filer så kallar den på process_token() för varje token.
        """
        files_read = 0  # Hur många filer den gått igenom hittills.
        for fname in self.__sources:
            if files_read < n_files:
                with open(fname, encoding='utf8', errors='ignore') as f:
                    text = reader = str(f.read())
                try:
                    self.tokens += nltk.word_tokenize(
                        text)
                except LookupError:
                    nltk.download('punkt')
                    self.tokens += nltk.word_tokenize(text)
                files_read += 1
            else:
                break
        for token in self.tokens:
            self.process_token(token)

    def process_token(self, token):
        """
        Går igenom alla tokens och uppdaterar unigram-, bigram- och trigram-count.
        Beräknar även totalt antal ord och antal unika ord.
        """
        # Nästan samma kod som användes i Language Models-laborationen.
        # Har adderat trigram_count.

        if token not in self.index.keys():
            self.__vocab.update(token)
            self.index[token] = len(self.index)
        index = self.index[token]
        self.word[index] = token

        self.unigram_count[token] += 1

        if self.last_index != -1:
            prev_word = self.word[self.last_index]
            self.bigram_count[prev_word][token] += 1
            if self.last_index2 != -1:
                prev_word2 = self.word[self.last_index2]
                self.trigram_count[prev_word2][prev_word][token] += 1   # I slutändan användes inte trigram_count.

        self.last_index2 = self.last_index
        self.last_index = self.index[token]

        self.unique_words = len(self.word)
        self.total_words += 1

    def stats(self):
        """
        Returnerar två listor som används för att skapa 2 filer:
        1. En fil för varje ord med respektive index samt antal förekomster (unigram count).
        2. En fil med beräknade log-bigram-sannolikheter: index1, index2 och log-sannolikhet.
        """
        rows_to_print1 = []
        rows_to_print2 = []

        # YOUR CODE HERE
        rows_to_print1.append(str(self.unique_words) + " " + str(self.total_words))

        for key in self.index:  # key is a unique word
            # Print index of word, word, unigram_count of word
            rows_to_print1.append(str(self.index[key]) + " " + str(key) + " " + str(self.unigram_count[key]))

        for key1 in self.bigram_count:
            for key2 in self.bigram_count[key1]:
                bigram_count = self.bigram_count[key1][key2]
                i_word1 = self.index[key1]
                i_word2 = self.index[key2]

                if bigram_count == 0:
                    prob = 0
                else:
                    prob = math.log(bigram_count / self.unigram_count[key1])
                rows_to_print2.append(str(i_word1) + " " + str(i_word2) + " " + f"{prob:.15f}")

        return rows_to_print1, rows_to_print2


if __name__ == '__main__':
    n_files = 200  # Ange hur många av de inlästa filerna som ska användas till beräkning av bigram-sannolikheter.
    # Placeringen i början är för lätt åtkomst att ändra.


    # Läser in de filer som förutsätts ha rengjorts av translator_textstäd.py
    dir_name = "textfiler (kopia)"
    filenames = [os.path.join(dir_name, fn) for fn in os.listdir(dir_name)]

    # spinner har tagits från koden som gavs i Random Indexing-laborationen.
    # Endast för att utsmycka terminal-outputen
    spinner = Halo(spinner='arrow3')
    bi = BigramTrainer(filenames)  # Initierar konstruktorn och sparar filerna i objektet bi.

    spinner.start(text="Processing files...")
    start = time.time()
    bi.process_files(
        n_files)  # Filerna som sparats i bi läses igenom och tokeniseras. Läser endast in n_files antal filer
    spinner.succeed(
        text="Processed files in {}s. Number of files: {}".format(round(time.time() - start, 2), n_files))



    spinner.start(text="Calculating bigram probabilities and statistics...")
    start = time.time()
    word_stats, biprobs = bi.stats() # Två listor med sträng-rader som ska sparas i två filer.
    spinner.succeed(text="Done calculating in {}s.".format(round(time.time() - start, 2)))

    spinner.start(text="Writing probabilities and statistics to file...")
    with open("word_stats_bi.txt", mode='w', encoding='utf-8') as f:
        # En fil för index, ord, unigram count
        for row in word_stats:
            f.write(row + '\n')
    with open("bigram_probs.txt", mode='w', encoding='utf-8') as f:
        # En fil för log-bigram-sannolikheter och dess index för orden.
        for row in biprobs:
            f.write(row + '\n')

    spinner.succeed(text="Done saving probabilities and statistics in {}s.".format(round(time.time() - start, 2)))
