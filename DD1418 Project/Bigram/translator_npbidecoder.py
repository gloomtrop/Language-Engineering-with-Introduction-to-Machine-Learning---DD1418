import numpy as np
import codecs
import math
from halo import Halo
import time


class ViterbiBigramDecoder(object):
    """
    Denna klass implementerar Viterbi-avkodning med bigram sannolikheter för att korrigera felvalda ord.
    Kod från laboration HMM har återanvänts i hög utsträckning.
    """

    def read_model(self, filename):

        """
        Nödvändigt för att kunna veta vilket index som tillhör vilket ord.
        Läser in filen word_stats_bi.txt som innehåller ord, respektive index samt unigram-count.
        """
        with codecs.open(filename, 'r', 'utf-8') as f:
            self.unique_words, self.total_words = map(int, f.readline().strip().split(' '))
            self.index[" "] = self.unique_words  # START_END
            self.word[self.index[" "]] = " "  # START_END
            self.START_END = self.unique_words  # Eftersom index börjar på 0 så kan START_END få index motsvarande
            # antal unika ord

            for _ in range(self.unique_words):
                # Läser igenom alla rader i word_stats_bi.txt
                word_index, word, uni_count = f.readline().strip().split(" ")
                word_index = int(word_index)
                self.word[word_index] = word
                self.index[word] = word_index
                self.unicount[word_index] = int(uni_count)

    def init_a(self, filename):
        """
        Läser in bigram-sannolikheter från filen bigram_probs.txt
        """
        self.a = np.zeros((self.unique_words + 1, self.unique_words + 1), dtype='double')  # Initierar matrisens storlek

        for i in range(len(self.a) - 1):  # -1 för att START_END inte finns med i unicount
            self.a[i, :] = np.log(1 / (self.unicount[i] + self.unique_words + 1))  # Ger varje bigram en sannolikhet med
            # Laplace-smoothing. +1 för att ta hänsyn till START_END

        with codecs.open(filename, 'r', 'utf-8') as f:
            for line in f:
                i, j, d = [func(x) for func, x in zip([int, int, float], line.strip().split(' '))]

                # Måste "räkna sannolikheten bakåt" för att kunna justera sannolikheten med Laplace-smoothing
                d = math.exp(d) # Vanlig sannolikhet (jämfört med log-sannolikhet)
                d = d * self.unicount[i]    # Bigram-count

                # Laplace-smoothing och spara sedan sannolikheten i matris A
                self.a[i][j] = math.log((d + 1) / (self.unicount[i] + self.unique_words + 1))

    # ------------------------------------------------------

    def init_b(self):
        """
        Initierar observationssannolikheter för alla ord till ett lågt värde.
        """

        self.b = np.zeros((self.unique_words + 1, self.unique_words + 1), dtype='double')
        self.b[:, :] = -float("inf")  # log=-inf ger sannolikhet≈0

    def adjust_b(self, word, synonyms):
        """
        Justerar observationssannolikheter. Om ordet anges som potentiell översättning av t.ex. bab.la så bör
        sannolikheten öka. Vi har valt att fördela observationssannolikheterna för de "mer sannolika orden" likformigt.
        Dvs dela 100% sannolikhet på antal ord.

        Man bör egentligen start om programmet efter varje mening eftersom dessa förändringar inte nollställs.
        """
        syno = [w for w in synonyms]
        if len(synonyms) > 0:
            for w in synonyms:
                if w in self.index.keys():
                    pass
                else:
                    syno.remove(w)
                    print(w, "synonym does not exist in vocabulary. Removed")  # Större träningskorpus minskar detta
                    # behov. Nödvändigt för att koden inte ska krascha om ett ord inte finns med.
            s_index = [self.index[x] for x in syno]
            word_index = self.index[word]
            for index in s_index:
                self.b[word_index][index] = math.log(
                    1 / len(s_index))  # Fördelning av observationssannolikheter på orden.

    # ------------------------------------------------------

    def viterbi(self, s):
        """
        Viterbi-avkodning och slutligen används en bakåtpekar-matris för att returnera den mest sannolika översättningen.
        """

        s = s.split()  # Gör om meningen till en lista med alla ord i meningen.
        s.append(" ")  # Lägg på " " representerar START_END
        index = [self.index[x] for x in s]  # Ta ut alla index från meningen.

        # Initierar viterbi-matris och bakåtpekar-matris
        self.v = np.zeros((len(s), self.unique_words + 1))
        self.v[:, :] = -float("inf")
        self.backptr = np.zeros((len(s) + 1, self.unique_words + 1), dtype='int')

        self.backptr[0, :] = self.START_END  # Lägger till START_END i meningens början
        self.v[0, :] = self.a[self.START_END, :] + self.b[index[0], :]  # START_END har i A-matrisen
        # log-sannolikhet 0 för alla ord som följer eftersom vi inte har beräknat någon bigram-sannolikhet
        # med START_END-symbol(inte bra).

        for t in range(len(s) - 1):  # Beräkna sekventiellt för varje ord, från ord nr2 till och med START_END i s
            t = t + 1
            # print(t, "out of", len(s) - 1)
            for k in range(self.unique_words + 1):  # För alla möjliga ord
                probs = (self.v[t - 1, :] + self.a[:, k] + self.b[index[t], k]).tolist()
                self.backptr[t, k] = probs.index(max(probs))
                self.v[t, k] = max(probs)



        # Genomgång av resultat med hjälp bakåtpekare
        last_prob_list = self.v[-2, :].tolist()  # Spara sista kolumnen i viterbi-matrisen (hoppa över START_END-kolumnen, därav -2)
        index_max_last = last_prob_list.index(max(last_prob_list))  # Hittar index för max-värdet i sista kolumnen.
        result = self.word[index_max_last]  # Resultatet börjar med (dvs ligger sist i slutresultatet)
        # ordet som hör till indexet som funnits på raden ovan.

        k = self.backptr[-3, index_max_last]
        t = len(self.backptr) - 4
        while t >= 0:
            # Går igenom bakåtpekar-matrisen och lägger till ett ord i taget.
            word = self.word[k]
            k = self.backptr[t, k]
            result = word + " " + result  # Lägg på ord i början av resultatet.
            t -= 1
        return result.strip()

    # ------------------------------------------------------

    def __init__(self, filename=None):
        """
        Konstruktorn initierar olika attribut som används i koden.
        Om A- och B-matriserna finns på fil så läses de in för att spara tid. Annars konstrueras de och sparas
        sedan till fil för framtida användning vilket sparar lite tid.
        """

        self.unique_words = 0
        self.total_words = 0
        self.START_END = self.unique_words
        self.unicount = dict()

        self.word = dict()
        self.index = dict()

        self.v = None

        # A-matrisen innehåller bigram-sannolikheter
        self.a = np.zeros((self.unique_words + 1, self.unique_words + 1), dtype='double')

        # B-matrisen innehåller observationssannolikheter
        self.b = np.zeros((self.unique_words + 1, self.unique_words + 1), dtype='double')

        # Blir senare en matris med bakåtpekare för den mest sannolika ordföljden.
        self.backptr = None

        self.read_model("word_stats_bi.txt")  # Läser framförallt in ord och respektive index.

        # Kollar om A-matrisen finns som fil. Annars skapas den och sparas som fil för framtiden.
        # OBS! Stor fil! Med 29000 unika ord blir filen ca 7GB
        try:
            a_file = open("a_matrix.npy", "rb")
            self.a = np.load(a_file)
        except IOError:
            self.init_a(filename)
            with open("a_matrix.npy", "wb") as f:
                np.save(f, self.a)
                print("A saved to file")

        # Kollar om B-matrisen finns som fil. Annars skapas den och sparas som fil för framtiden.
        # OBS! Stor fil! Med 29000 unika ord blir filen ca 7GB
        try:
            b_file = open("b_matrix.npy", "rb")
            self.b = np.load(b_file)
        except IOError:
            self.init_b()
            with open("b_matrix.npy", "wb") as f:
                np.save(f, self.b)
                print("B saved to file")

    # ------------------------------------------------------


def main():
    spinner = Halo(spinner='arrow3')
    spinner.start(text="Initializing A and B matrices...")
    start = time.time()
    d = ViterbiBigramDecoder("bigram_probs.txt")
    spinner.succeed(
        text="Completed initialization in: {}".format(round(time.time() - start, 2)))

    while True:
        """
        OBS! While True tillåter att testa flera meningar utan att starta om programmet men B-matrisen nollställs inte!
        Bör egentligen bara användas till olika tester av koden.
        """
        inp = True
        while inp:
            inp = False
            translation = input("Enter the english sentence: ").lower() #Den mening som ska förbättras

            for word in translation.split():
                # För varje ord i meningen bör man ange förslag på andra ord.
                # Annars kommer inte ordet ändras eftersom det givna ordet får observationssannolikhet 1 vilket
                # i sin tur kommer dominera eventuella bigram-sannolikheter.
                try:
                    d.index[word]

                except KeyError:
                    inp = True
                    print(f"\"{word}\" not in vocabulary. Try another sentence.")

        for word in translation.split():
            # Set observation probabilities
            synonyms = input(f"Enter other words for \"{word}\": ").lower().split()
            for s in synonyms:
                if s == word:
                    synonyms.remove(s)
                    print("Don't enter the same word. Additional",word,"has been removed.")
            synonyms.append(word)
            d.adjust_b(word, synonyms)

        # Beräkning av viterbi-matris och genomgång av bakåtpekare
        spinner.start(text="Doing Viterbi Decoding for sentence...")
        start = time.time()
        result = d.viterbi(translation)
        spinner.succeed(
            text="Completed viterbi decoding in: {}".format(round(time.time() - start, 2)))
        print(result) #Slutligen skrivs resultatet ut.


if __name__ == "__main__":
    main()
