from io import open
import unicodedata
import string
import re
import random
import torch
import torch.nn as nn
import time
import math
import matplotlib.pyplot as plt
plt.switch_backend('agg')
import matplotlib.ticker as ticker
import numpy as np

from torch import optim
import torch.nn.functional as F

from model import EncoderRNN, BiLSTM, AttnDecoderRNN, prepareData, evaluate

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

input_lang, output_lang, supertag_lang, pairs = prepareData('ref', 'para', 'train')
teacher_forcing_ratio = 0.5
hidden_size = 256


encoder1 = EncoderRNN(input_lang.n_words, hidden_size).to(device)
encoder1.load_state_dict(torch.load('12-5-19/encoder_step_75000.pt'))
encoder1.eval()

supertag_encoder1 = BiLSTM(supertag_lang.n_words, hidden_size).to(device)
supertag_encoder1.load_state_dict(torch.load('12-5-19/supertage_encoder_step_75000.pt'))
supertag_encoder1.eval()

attn_decoder1 = AttnDecoderRNN(hidden_size, output_lang.n_words, dropout_p=0.1).to(device)
attn_decoder1.load_state_dict(torch.load('12-5-19/decoder_step_75000.pt'))
attn_decoder1.eval()

test_input, test_output, test_supertags, test_pairs = prepareData('test-ref', 'test-para', 'test')


with open('12-5-19/test_output.txt', 'w') as f:
    for pair in test_pairs:
        output_words, attentions = evaluate(encoder1, supertag_encoder1, attn_decoder1, pair[0], pair[2])
        output_sentence = ' '.join(output_words)
        f.write(output_sentence + '\n')
