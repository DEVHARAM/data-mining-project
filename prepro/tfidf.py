from konlpy.tag import Twitter
from sklearn.feature_extraction.text import CountVectorizer
import numpy as np
import math
import csv

twitter = Twitter()
count = CountVectorizer()
np.set_printoptions(precision=2)

emoticons = ["!", "@", "#", "$", "%", "^", "&", "*", "(", ")", "-", "=", "_", "+", "~", ",", ".", "?", "/", ">", "<",
             "\t"]

readLine = []
words = []


# In csv, element : [word, TF-IDF value, TF(word frequency)]
def make_csv():
    with open("simple.txt", "r", encoding="utf8") as In:
        words_count = {}
        read = In.read()
        for emoticon in emoticons:
            read = read.replace(emoticon, "")
        words = twitter.morphs(read)
        for word in words:
            if word in words_count:
                words_count[word] += 1
            else:
                words_count[word] = 1
        print(words_count)

        # init df dictionary
        df = {}
        for word in words_count.keys():
            df[word] = 0

        # get df value
        LineWords = []
        readLines = read.split('\n')
        for readLine in readLines:
            LineWords = twitter.morphs(readLine)
            for word in words_count.keys():
                if word in LineWords:
                    df[word] += 1

        # calculate TF-IDF
        tf = {}
        idf = {}
        tfidf = {}
        for word in words_count:
            tf[word] = words_count[word] / len(words_count)
            idf[word] = math.log10(len(readLines) / (1 + df[word]))
            tfidf[word] = tf[word] * idf[word]

        # clean data
        del tfidf['\n']
        del tfidf['\n\n']
        del tfidf['\ufeff']
        del tfidf['0']
        del tfidf['2']
        del tfidf['1']
        sorted_tfidf = sorted(tfidf, key=lambda k: tfidf[k], reverse=True)
        print(sorted_tfidf)

        # write csv
        with open("TF-IDF.csv", "w", encoding="utf8", newline='\n') as out_tfidf:
            wr = csv.writer(out_tfidf)
            for i in sorted_tfidf:
                wr.writerow([i, tfidf[i], words_count[i]])


def make_txt(threshold):
    with open("result.txt", "r", encoding="utf8") as In:
        with open("TF-IDF.csv", "r", encoding="utf8", newline='\n') as In_tfidf:
            with open("first.txt", "w", encoding="utf8") as Out:
                read = In.read()
                read_words = twitter.morphs(read)

                tfidf_words = [0 for _ in range(threshold)]
                r = csv.reader(In_tfidf)

                # store tfidf words
                i = 0
                for row in r:
                    tfidf_words[i] = (row[0], row[2])
                    i += 1
                    if i >= threshold:
                        break
                print(tfidf_words)

                # delete tfidf words
                for word, count in tfidf_words:
                    try:
                        print(word, count)
                        for i in range(0, int(count)):
                            read_words.remove(word)
                    except ValueError:
                        pass

                write = ' '.join(read_words)
                Out.write(write)


make_csv()
make_txt(100)

