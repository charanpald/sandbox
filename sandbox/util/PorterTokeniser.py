
import string

#Tokenise the documents                 
class PorterTokeniser(object):
    def __init__(self):
        try:
            import Stemmer 
            self.stemmer = Stemmer.Stemmer('english')
        except ImportError: 
           print("Stemmer module not found")     
        self.minWordLength = 2
     
    def __call__(self, doc):
        doc = doc.lower().encode('utf-8').strip()
        doc = doc.translate(string.maketrans("",""), string.punctuation).decode("utf-8")
        try: 
            import Stemmer        
            tokens =  [self.stemmer.stemWord(t) for t in doc.split()]  
        except ImportError: 
            tokens =  [t for t in doc.split()] 
            
        return [token for token in tokens if len(token) >= self.minWordLength]