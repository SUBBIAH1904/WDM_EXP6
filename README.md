### EX6 Information Retrieval Using Vector Space Model in Python

### DATE: 28-09-2024

### AIM: To implement Information Retrieval Using Vector Space Model in Python.

### Description: 
<div align = "justify">
Implementing Information Retrieval using the Vector Space Model in Python involves several steps, including preprocessing text data, constructing a term-document matrix, 
calculating TF-IDF scores, and performing similarity calculations between queries and documents. Below is a basic example using Python and libraries like nltk and 
sklearn to demonstrate Information Retrieval using the Vector Space Model.

### Procedure:
1. Define sample documents.
2. Preprocess text data by tokenizing, removing stopwords, and punctuation.
3. Construct a TF-IDF matrix using TfidfVectorizer from sklearn.
4. Define a search function that calculates cosine similarity between a query and documents based on the TF-IDF matrix.
5. Execute a sample query and display the search results along with similarity scores.

### Program:

```python
import numpy as np
import pandas as pd

class BooleanRetrieval:
    def _init_(self):
        self.index = {}
        self.documents_matrix = None

    def index_document(self, doc_id, text):
        terms = text.lower().split()
        print("Document -", doc_id, terms)

        for term in terms:
            if term not in self.index:
                self.index[term] = set()
            self.index[term].add(doc_id)

    def create_documents_matrix(self, documents):
        terms = list(self.index.keys())
        num_docs = len(documents)
        num_terms = len(terms)

        self.documents_matrix = np.zeros((num_docs, num_terms), dtype=int)

        for i, (doc_id, text) in enumerate(documents.items()):
            doc_terms = text.lower().split()
            for term in doc_terms:
                if term in self.index:
                    term_id = terms.index(term)
                    self.documents_matrix[i, term_id] = 1

    def print_documents_matrix_table(self):
        df = pd.DataFrame(self.documents_matrix, columns=self.index.keys())
        print(df)

    def print_all_terms(self):
        print("All terms in the documents:")
        print(list(self.index.keys()))

    def boolean_search(self, query):
        query_terms = query.lower().split()
        results = set()  # Initialize as empty set to accumulate results
        current_set = None  # Current set to handle 'or' logic

        i = 0
        while i < len(query_terms):
            term = query_terms[i]

            if term == 'or':
                if current_set is not None:
                    results.update(current_set)
                current_set = None  # Reset current set for the next term
            elif term == 'and':
                i += 1
                continue  # 'and' is implicit, move to next term
            elif term == 'not':
                i += 1
                if i < len(query_terms):
                    not_term = query_terms[i]
                    if not_term in self.index:
                        not_docs = self.index[not_term]
                        if current_set is None:
                            current_set = set(range(1, len(documents) + 1))  # All doc IDs
                        current_set.difference_update(not_docs)
            else:
                if term in self.index:
                    term_docs = self.index[term]
                    if current_set is None:
                        current_set = term_docs.copy()
                    else:
                        current_set.intersection_update(term_docs)
                else:
                    current_set = set()  # If the term doesn't exist, it results in an empty set

            i += 1

        # Update results with the last processed set
        if current_set is not None:
            results.update(current_set)

        return sorted(results)

if _name_ == "_main_":
    indexer = BooleanRetrieval()

    documents = {
        1: "Python is a programming language",
        2: "Information retrieval deals with finding information",
        3: "Boolean models are used in information retrieval"
    }

    for doc_id, text in documents.items():
        indexer.index_document(doc_id, text)

    indexer.create_documents_matrix(documents)
    indexer.print_documents_matrix_table()
    indexer.print_all_terms()

    query = input("Enter your boolean query: ")
    results = indexer.boolean_search(query)
    if results:
        print(f"Results for '{query}': {results}")
    else:
        print("No results found for the query.")
```
### Output:

![11](https://github.com/user-attachments/assets/08014843-6078-4d47-a188-0a245790bfcd)


### Result:
Thus, the implementation of Information Retrieval Using Vector Space Model in Python is executed successfully.

