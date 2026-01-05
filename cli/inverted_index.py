import pickle, json
import os
from search_function import text_preprocessing

class InvertedIndex:

    def __init__(self):
        self.index = {}  # dictionary mapping tokens (strings) to sets of document IDs (integers).
        self.docmap = {} # dictionary mapping document IDs to their full document objects.

    def __add_document(self, doc_id, text):
        tokenized_input_text = text_preprocessing(text).split(" ") # list of input text split into tokens
        for item in tokenized_input_text:
            if item in self.index.keys():
                self.index[item].add(doc_id)
            else:
                self.index[item] = {doc_id}
                
    def get_documents(self, term):
        return sorted(list(self.index[text_preprocessing(term)]))

    def build(self):
        with open('data/movies.json', 'r') as file:
            data = json.load(file)
        
        for m in data["movies"]:
            self.__add_document(m["id"],f"{m['title']} {m['description']}")
            self.docmap[m["id"]] = m

    def save(self):
        os.makedirs('cache', exist_ok=True)
        with open('cache/index.pkl', 'wb') as file:
            pickle.dump(self.index, file) 
        with open('cache/docmap.pkl', 'wb') as file:
            pickle.dump(self.docmap, file) 

    def load(self):
        with open('cache/docmap.pkl', 'rb') as file1:
            docmap = pickle.load(file1)   
        with open('cache/index.pkl', 'rb') as file2:
            index = pickle.load(file2)  

        return docmap, index