import json
from nltk_utils import tokenize,stem,bag_of_words
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

with open('intents.json', 'r') as f:
    intents = json.load(f)

all_words = []
tags = []
xy = []

for intent in intents['intents']:
    tag = intent['tag']
    # add tags to list
    tags.append(tag)
    for pattern in intent['patterns']:
        # tokenize each word in the sentence
        w =  tokenize(pattern)
        # add to our words list like (['Hi'], 'greeting'), (['Hey'], 'greeting'),
        all_words.extend(w)
        # add to xy pair  
        xy.append((w,tag))

# stem and lower each word        
ignore_words = ['?', '.', '!']
all_words = [stem(w) for w in all_words if w not in ignore_words]
# remove duplicates and sort
all_words = sorted(set(all_words))
tags = sorted(set(tags))

print(len(xy), "patterns")
print(len(tags), "tags:", tags)
print(len(all_words), "unique stemmed words:", all_words) 

# create training data

x_train = []
y_train =[]

for (pattern_sentence, tag) in xy:
    # X: bag of words for each pattern_sentence
    bag = bag_of_words(pattern_sentence, all_words)
    x_train.append(bag)
     # y: PyTorch CrossEntropyLoss needs only class labels, not one-hot
    label = tags.index(tag)
    y_train.append(label)
    
x_train = np.array(x_train)
y_train = np.array(y_train)


class ChatDataset(Dataset):
    
    def __ini__(self):
        self.n_samples = len(x_train)
        self.x_data = x_train
        self.y_data = y_train
        
        # support indexing such that dataset[i] can be used to get i-th sample
        
        def __getitem__(self):
            return self.x_data[index],self.y_data[index]
        # we can call len(dataset) to return the size
        def __len__(self):
            return self.n_samples

dataset = ChatDataset()
train_loader = DataLoader(dataset=dataset, batch_size=batch_size, shuffle=True, num_workers=0)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')  