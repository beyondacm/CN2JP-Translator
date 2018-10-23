from config import *

flatten = lambda l: [item for sublist in l for item in sublist]

def prepare_sequence(seq, to_index):
    idxs = list(map(lambda w: to_index[w] if to_index.get(w) is not None else to_index["<UNK>"], seq))
    return Variable(LongTensor(idxs))

# Turn a Unicode string to plain ASCII, thanks to http://stackoverflow.com/a/518232/2809427
def unicode_to_ascii(s):
    return ''.join(
        c for c in unicodedata.normalize('NFD', s)
        if unicodedata.category(c) != 'Mn'
    )

# Lowercase, trim, and remove non-letter characters
def normalize_string(s):
    s = unicode_to_ascii(s.lower().strip())
    s = re.sub(r"([,.!?])", r" \1 ", s)
    # s = re.sub(r"[^a-zA-Z,.!?]+", r" ", s)
    s = re.sub(r"\s+", r" ", s).strip()
    return s

class Train_Data(object):

    def __init__(self):

        self.MIN_LENGTH = 1 
        self.MAX_LENGTH = 16 
        self.corpus = self.load_corpus()
        print(len(self.corpus))

        self.X_r, self.y_r = self.get_raw_pairs()
        print(len(self.X_r), len(self.y_r))
        print(self.X_r[0], self.y_r[0])

        self.source_vocab = list(set(flatten(self.X_r)))
        self.target_vocab = list(set(flatten(self.y_r)))
        print(len(self.source_vocab), len(self.target_vocab))

        self.source2index = self.get_source2index()
        self.index2source = self.get_index2source()
        self.target2index = self.get_target2index()
        self.index2target = self.get_index2target()
        print(len(self.source2index), len(self.target2index))

        self.X_p, self.y_p = self.get_pro_pairs()
        print(len(self.X_p), len(self.y_p))
        pass
    
    def load_corpus(self):
        corpus = open('../cn-jp.txt', 'r', encoding='utf-8').readlines()
        # corpus = open('../fra-eng.txt', 'r', encoding='utf-8').readlines()
        return corpus

    def get_raw_pairs(self):
        X_r, y_r = [],[]
        for parallel in self.corpus:
            if "\t" not in parallel:
                continue    

            so,ta = parallel[:-1].split('\t')
            if so.strip() == "" or ta.strip() == "": 
                continue

            normalized_so = normalize_string(so).split()
            normalized_ta = normalize_string(ta).split()
             
            if len(normalized_so) >= self.MIN_LENGTH and len(normalized_so) <= self.MAX_LENGTH \
            and len(normalized_ta) <= self.MAX_LENGTH:
            # and len(normalized_ta) >= MIN_LENGTH 
                X_r.append(normalized_so)
                y_r.append(normalized_ta)
            
        return X_r, y_r
    
    def get_source2index(self):
        source2index = {'<PAD>': 0, '<UNK>': 1, '<s>': 2, '</s>': 3}
        for vo in self.source_vocab:
            if source2index.get(vo) is None:
                source2index[vo] = len(source2index)
        return source2index
        pass

    def get_index2source(self):
        index2source = {v:k for k, v in self.source2index.items()}
        return index2source
        pass
        
    def get_target2index(self):
        target2index = {'<PAD>': 0, '<UNK>': 1, '<s>': 2, '</s>': 3}
        for vo in self.target_vocab:
            if target2index.get(vo) is None:
                target2index[vo] = len(target2index)
        return target2index 
        pass

    def get_index2target(self):
        index2target = {v:k for k, v in self.target2index.items()}
        return index2target
        pass

    def get_pro_pairs(self):
        X_p, y_p = [], []
        for so, ta in zip(self.X_r, self.y_r):
            X_p.append(prepare_sequence(so + ['</s>'], self.source2index).view(1, -1))
            y_p.append(prepare_sequence(ta + ['</s>'], self.target2index).view(1, -1))
        return X_p, y_p   

    def get_train_data(self):
        train_data = list(zip(self.X_p, self.y_p))
        return train_data


class Test_Data():

    def __init__(self):
        self.Train_Data = Train_Data()
        self.MIN_LENGTH = self.Train_Data.MIN_LENGTH 
        self.MAX_LENGTH = self.Train_Data.MAX_LENGTH 
        self.corpus = self.load_corpus()
        print(len(self.corpus))

        self.X_r, self.y_r = self.get_raw_pairs()
        print(len(self.X_r), len(self.y_r))
        print(self.X_r[0], self.y_r[0])
        
        self.source2index = self.Train_Data.source2index
        self.target2index = self.Train_Data.target2index
        print(len(self.source2index), len(self.target2index))

        self.X_p, self.y_p = self.get_pro_pairs()
        print(len(self.X_p), len(self.y_p))
        pass

    def load_corpus(self):
        # corpus = open('../fra-eng.txt', 'r', encoding='utf-8').readlines()
        corpus = open('../code-question-test.txt', 'r', encoding='utf-8').readlines()
        return corpus

    def get_raw_pairs(self):
        X_r, y_r = [],[]
        for parallel in self.corpus:
            if "\t" not in parallel:
                continue    

            so,ta = parallel[:-1].split('\t')
            if so.strip() == "" or ta.strip() == "": 
                continue

            normalized_so = normalize_string(so).split()
            normalized_ta = normalize_string(ta).split()
    
            if len(normalized_so) >= self.MIN_LENGTH and len(normalized_so) <= self.MAX_LENGTH \
            and len(normalized_ta) <= self.MAX_LENGTH:
            # and len(normalized_ta) >= MIN_LENGTH 
                X_r.append(normalized_so)
                y_r.append(normalized_ta)
        return X_r, y_r

    def get_pro_pairs(self):
        X_p, y_p = [], []
        for so, ta in zip(self.X_r, self.y_r):
            X_p.append(prepare_sequence(so + ['</s>'], self.source2index).view(1, -1))
            y_p.append(prepare_sequence(ta + ['</s>'], self.target2index).view(1, -1))
        return X_p, y_p   

    def get_test_data(self):
        test_data = list(zip(self.X_p, self.y_p))
        return test_data

# train = Train_Data()   
# print('========================')
# test = Test_Data()
