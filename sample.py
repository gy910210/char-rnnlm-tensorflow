import json
import os, sys
import logging

import numpy as np
import tensorflow as tf
from model import CharRNNLM
from config import config_sample
from utils import VocabularyLoader

def main():
    args = config_sample()
    
    logging.basicConfig(stream=sys.stdout,
                        format='%(asctime)s %(levelname)s:%(message)s', 
                        level=logging.INFO, datefmt='%I:%M:%S')
    
    # Prepare parameters.
    with open(os.path.join(args.init_dir, 'result.json'), 'r') as f:
        result = json.load(f)
    params = result['params']
    best_model = result['best_model']
    best_valid_ppl = result['best_valid_ppl']
    if 'encoding' in result:
        args.encoding = result['encoding']
    else:
        args.encoding = 'utf-8'
    args.vocab_file = os.path.join(args.init_dir, 'vocab.json')
    vocab_loader = VocabularyLoader()
    vocab_loader.load_vocab(args.vocab_file, args.encoding)
    
    logging.info('best_model: %s\n', best_model)
    
    # Create graphs
    graph = tf.Graph()
    with graph.as_default():
        with tf.name_scope('evaluation'):
            model = CharRNNLM(is_training=False, infer=True, **params)
            saver = tf.train.Saver(name='model_saver')

    if args.seed >= 0:
        np.random.seed(args.seed)
    with tf.Session(graph=graph) as session:
        saver.restore(session, best_model) 
        sample = model.sample_seq(session, args.length, args.start_text, vocab_loader,
                                  max_prob=args.max_prob)
        print('Sampled text is:\n\n%s' % sample)

if __name__ == '__main__':
    main()