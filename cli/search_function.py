import json
import string
from nltk.stem import PorterStemmer

stemmer = PorterStemmer()
stopwordslist = []

with open("data/stopwords.txt", "r") as file:
    stopwordslist = file.read().splitlines()

def text_preprocessing(text):
    translation_table = str.maketrans("","", string.punctuation)
    x = text.lower().translate(translation_table)
    x = " ".join(set(x.split(" ")) - set(stopwordslist))
    x = " ".join(stemmer.stem(word) for word in x.split(" "))
    return x

def search_function(query):
    with open('data/movies.json', 'r') as file:
            data = json.load(file)

    movieslist = data["movies"]

    query = text_preprocessing(query)
    querylist = query.split(" ")

    filteredlist = []
    movieids = set()

    for movie in movieslist:
        movietitlelist = text_preprocessing(movie["title"]).split(" ")
        # if len(set(querylist) & set(movietitlelist)) != 0 :
        #     filteredlist.append(movie)
        for word in querylist:
            if any(word in item for item in movietitlelist):
                    if movie["id"] not in movieids :
                        filteredlist.append(movie) 
                        movieids.add(movie["id"])
                    

    sortedlist = sorted(filteredlist, key = lambda x: x["id"])

    # set_of_movie_tuples = {tuple(d.items()) for d in sortedlist}
    # list_of_unique_movies = [dict(t) for t in set_of_movie_tuples]

    return sortedlist
