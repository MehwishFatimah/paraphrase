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

if __name__ == '__main__':
    input_lang, output_lang, supertag_lang, pairs = prepareData('ref', 'para', test=False)
    teacher_forcing_ratio = 0.5
    hidden_size = 256


    encoder1 = EncoderRNN(input_lang.n_words, hidden_size).to(device)
    encoder1.load_state_dict(torch.load('3-11-20/encoder_step_250000.pt'))
    encoder1.eval()

    # supertag_encoder1 = BiLSTM(supertag_lang.n_words, hidden_size).to(device)
    supertag_encoder1 = EncoderRNN(supertag_lang.n_words, hidden_size).to(device)

    supertag_encoder1.load_state_dict(torch.load('3-11-20/supertag_encoder_step_250000.pt'))
    supertag_encoder1.eval()

    attn_decoder1 = AttnDecoderRNN(hidden_size, output_lang.n_words, dropout_p=0.1).to(device)
    attn_decoder1.load_state_dict(torch.load('3-11-20/decoder_step_250000.pt'))
    attn_decoder1.eval()

    test_input, test_output, test_supertags, test_pairs = prepareData('test-ref', 'test-para', test=True, openNMT=False)


    with open('3-11-20/linear-output.txt', 'w') as f:
        for pair in test_pairs:
            output_words, attentions = evaluate(encoder1, supertag_encoder1, attn_decoder1, pair[0], pair[2], input_lang, supertag_lang, output_lang)
            output_sentence = ' '.join(output_words)
            f.write(output_sentence + '\n')
