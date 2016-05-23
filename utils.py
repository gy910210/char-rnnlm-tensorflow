import codecs
import json
import numpy as np

class VocabularyLoader(object):
    def load_vocab(self, vocab_file, encoding):
        with codecs.open(vocab_file, 'r', encoding=encoding) as f:
            vocab_index_dict = json.load(f)
        index_vocab_dict = {}
        vocab_size = 0
        for char, index in vocab_index_dict.iteritems():
            index_vocab_dict[index] = char
            vocab_size += 1
        self.vocab_index_dict = vocab_index_dict
        self.index_vocab_dict = index_vocab_dict
        self.vocab_size = vocab_size
        return vocab_index_dict, index_vocab_dict, vocab_size
        
    def create_vocab(self, text):
        unique_chars = list(set(text))
        self.vocab_size = len(unique_chars)
        self.vocab_index_dict = {}
        self.index_vocab_dict = {}
        for i, char in enumerate(unique_chars):
            self.vocab_index_dict[char] = i
            self.index_vocab_dict[i] = char
            
    def save_vocab(self, vocab_file, encoding):
        with codecs.open(vocab_file, 'w', encoding=encoding) as f:
            json.dump(self.vocab_index_dict, f, indent=2, sort_keys=True)
            

class BatchGenerator(object):
    def __init__(self, vocab_index_dict, text, batch_size, seq_length):
        self.batch_size = batch_size
        self.seq_length = seq_length
        self.tensor = np.array(list(map(vocab_index_dict.get, text)))
        self.create_batches()
        self.reset_batch_pointer()
        
    def reset_batch_pointer(self):
        self.pointer = 0
        
    def create_batches(self):
        self.num_batches = int(self.tensor.size / (self.batch_size * self.seq_length))
        
        # When the data (tesor) is too small, let's give them a better error message
        if self.num_batches==0:
            assert False, "Not enough data. Make seq_length and batch_size small."
        
        self.tensor = self.tensor[:self.num_batches * self.batch_size * self.seq_length]
        xdata = self.tensor
        ydata = np.copy(self.tensor)
        ydata[:-1] = xdata[1:]
        ydata[-1] = xdata[0]
        self.x_batches = np.split(xdata.reshape(self.batch_size, -1), self.num_batches, 1)
        self.y_batches = np.split(ydata.reshape(self.batch_size, -1), self.num_batches, 1)
        
    def next_batch(self):
        x, y = self.x_batches[self.pointer], self.y_batches[self.pointer]
        self.pointer += 1
        return x, y
   
# util functions
def batche2string(batch, index_vocab_dict):
    return ''.join(list(map(index_vocab_dict.get, batch)))