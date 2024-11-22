import pandas as pd
import numpy as np
import torch

def get_data_loaders(batch_size):
    splits = {'train': 'train.csv', 'test': 'test.csv'}
    df_train = pd.read_csv("hf://datasets/gxb912/large-twitter-tweets-sentiment/" + splits["train"])
    df_test = pd.read_csv("hf://datasets/gxb912/large-twitter-tweets-sentiment/" + splits["test"])
    
    # STEP 1: Remove NULL values from the train, validation and test datasets
    df_train = df_train.dropna()
    df_test = df_test.dropna()
    
    # now, let us encode the labels assuming positive -> 0, irrelevant/neutral -> 1, negative -> 2
    # STEP 2: ONE HOT ENCODING
    
    def encode_labels(label_string):
        if label_string == 1:
            label_string = "0"
        elif label_string == 0:
            label_string = "2"
        else:
            label_string = "1"
        return label_string
    
    # apply encoding on label column of dataframe
    train_labels = np.array(list(map(encode_labels, list(df_train['sentiment']))), dtype = np.float32)
    test_labels = np.array(list(map(encode_labels, list(df_test['sentiment']))), dtype = np.float32)
        
    # for passing sentences properly to models, they need to be parsed properly within lists
    # STEP 3: Parse the train and test datasets sentences as strings
    
    df_train['text'] = df_train['text'].astype("string")
    df_test['text'] = df_test['text'].astype("string")
    
    train_sentences = np.array(df_train['text'])
    test_sentences = np.array(df_test['text'])
    
    # split test sentences into validation and test data
    num_valid = 15000
    # selecting 15000 random indices between [0, len(df_test)] without replacements
    validdata_indices = np.random.choice(len(df_test), num_valid, False)
    # compute remaining indices
    rem_indices = np.setdiff1d(np.arange(len(df_test)), validdata_indices)
    
    validation_sentences, test_sentences = test_sentences[validdata_indices], test_sentences[rem_indices]
    validation_labels, test_labels = test_labels[validdata_indices], test_labels[rem_indices]
    
    # remove all the empty [] sentences from train_sentences and corresponding labels,to remove abnormalities
    
    empty_or_not = [True if len(l)==0 else False for l in train_sentences]
    train_sentences, train_labels = np.delete(train_sentences, empty_or_not), np.delete(train_labels, empty_or_not)
    
    empty_or_not = [True if len(l)==0 else False for l in validation_sentences]
    validation_sentences, validation_labels = np.delete(validation_sentences, empty_or_not), np.delete(validation_labels, empty_or_not)
    
    empty_or_not = [True if len(l)==0 else False for l in test_sentences]
    test_sentences, test_labels = np.delete(test_sentences, empty_or_not), np.delete(test_labels, empty_or_not)
    
    # STEP 4: Create PyTorch dataloaders of sentences with sampler
    
    class SentenceDataset(torch.utils.data.Dataset):
        def __init__(self, sentences, labels):
            self.sentences = sentences
            self.labels = labels
            
        def __getitem__(self, index):
            return self.sentences[index], self.labels[index]
        
        def __len__(self):
            return len(self.sentences)

    train_dataset = SentenceDataset(train_sentences, train_labels)
    validation_dataset = SentenceDataset(validation_sentences, validation_labels)
    test_dataset = SentenceDataset(test_sentences, test_labels)
    
    # creating the Pytorch dataloaders from the training, validation and test datasets, with samplers to randomly select sentences
    train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size = batch_size, sampler = torch.utils.data.RandomSampler(train_dataset, False, len(train_dataset)))
    validation_dataloader = torch.utils.data.DataLoader(validation_dataset, batch_size = batch_size, sampler = torch.utils.data.RandomSampler(validation_dataset, False, len(validation_dataset)))
    test_dataloader = torch.utils.data.DataLoader(test_dataset, batch_size = batch_size, sampler = torch.utils.data.RandomSampler(test_dataset, False, len(test_dataset)))

    return (train_dataloader, validation_dataloader, test_dataloader)