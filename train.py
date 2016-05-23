import codecs
import json
import logging
import os
import shutil
import sys

import numpy as np
import tensorflow as tf
from model import CharRNNLM
from config import config
from utils import VocabularyLoader, BatchGenerator, batche2string

TF_VERSION = int(tf.__version__.split('.')[1])

def main():
    args = config()
    
    # Specifying location to store model, best model and tensorboard log.
    args.save_model = os.path.join(args.output_dir, 'save_model/model')
    args.save_best_model = os.path.join(args.output_dir, 'best_model/model')
    args.tb_log_dir = os.path.join(args.output_dir, 'tensorboard_log/')
    args.vocab_file = ''
    
    # Create necessary directories.
    if len(args.init_dir) != 0:
        args.output_dir = args.init_dir
    else:
        if os.path.exists(args.output_dir):
            shutil.rmtree(args.output_dir)
        for paths in [args.save_model, args.save_best_model, args.tb_log_dir]:
            os.makedirs(os.path.dirname(paths))
        
    logging.basicConfig(stream=sys.stdout,
                        format='%(asctime)s %(levelname)s:%(message)s', 
                        level=logging.INFO, datefmt='%I:%M:%S')
    
    print('=' * 60)
    print('All final and intermediate outputs will be stored in %s/' % args.output_dir)
    print('=' * 60 + '\n')
    
    if args.debug:
        logging.info('args are:\n%s', args)
        
    if len(args.init_dir) != 0:
        with open(os.path.join(args.init_dir, 'result.json'), 'r') as f:
            result = json.load(f)
        params = result['params']
        args.init_model = result['latest_model']
        best_model = result['best_model']
        best_valid_ppl = result['best_valid_ppl']
        if 'encoding' in result:
            args.encoding = result['encoding']
        else:
            args.encoding = 'utf-8'
        args.vocab_file = os.path.join(args.init_dir, 'vocab.json')
    else:
        params = {'batch_size': args.batch_size,
                  'num_unrollings': args.num_unrollings,
                  'hidden_size': args.hidden_size,
                  'max_grad_norm': args.max_grad_norm,
                  'embedding_size': args.embedding_size,
                  'num_layers': args.num_layers,
                  'learning_rate': args.learning_rate,
                  'model': args.model,
                  'dropout': args.dropout,
                  'input_dropout': args.input_dropout}
        best_model = ''
    logging.info('Parameters are:\n%s\n', json.dumps(params, sort_keys=True, indent=4))
    
    # Read and split data.
    logging.info('Reading data from: %s', args.data_file)
    with codecs.open(args.data_file, 'r', encoding=args.encoding) as f:
        text = f.read()
    
    if args.test:
        text = text[:50000]
    logging.info('Number of characters: %s', len(text))
    
    if args.debug:
        logging.info('First %d characters: %s', 10, text[:10])
    
    logging.info('Creating train, valid, test split')
    train_size = int(args.train_frac * len(text))
    valid_size = int(args.valid_frac * len(text))
    test_size = len(text) - train_size - valid_size
    train_text = text[:train_size]
    valid_text = text[train_size:train_size + valid_size]
    test_text = text[train_size + valid_size:]
    
    vocab_loader = VocabularyLoader()
    if len(args.vocab_file) != 0:
        vocab_loader.load_vocab(args.vocab_file, args.encoding)
    else:
        logging.info('Creating vocabulary')
        vocab_loader.create_vocab(text)
        vocab_file = os.path.join(args.output_dir, 'vocab.json')
        vocab_loader.save_vocab(vocab_file, args.encoding)
        logging.info('Vocabulary is saved in %s', vocab_file)
        args.vocab_file = vocab_file
    
    params['vocab_size'] = vocab_loader.vocab_size
    logging.info('Vocab size: %d', vocab_loader.vocab_size)
    
    # Create batch generators.
    batch_size = params['batch_size']
    num_unrollings = params['num_unrollings']
    
    train_batches = BatchGenerator(vocab_loader.vocab_index_dict, train_text, batch_size, num_unrollings)
    valid_batches = BatchGenerator(vocab_loader.vocab_index_dict, valid_text, batch_size, num_unrollings)
    test_batches = BatchGenerator(vocab_loader.vocab_index_dict, test_text, batch_size, num_unrollings)
    
    if args.debug:
        logging.info('Test batch generators')
        x, y = train_batches.next_batch()
        logging.info((str(x[0]), str(batche2string(x[0], vocab_loader.index_vocab_dict))))
        logging.info((str(y[0]), str(batche2string(y[0], vocab_loader.index_vocab_dict))))
        
    # Create graphs
    logging.info('Creating graph')
    graph = tf.Graph()
    with graph.as_default():
        with tf.name_scope('training'):
            train_model = CharRNNLM(is_training=True, infer=False, **params)
        tf.get_variable_scope().reuse_variables()
        with tf.name_scope('validation'):
            valid_model = CharRNNLM(is_training=False, infer=False, **params)
        with tf.name_scope('evaluation'):
            test_model = CharRNNLM(is_training=False, infer=False, **params)
            saver = tf.train.Saver(name='model_saver')
            best_model_saver = tf.train.Saver(name='best_model_saver')
    
    logging.info('Start training\n')
    
    result = {}
    result['params'] = params
    result['vocab_file'] = args.vocab_file
    result['encoding'] = args.encoding
    
    try:
        with tf.Session(graph=graph) as session:
            # Version 8 changed the api of summary writer to use
            # graph instead of graph_def.
            if TF_VERSION >= 8:
                graph_info = session.graph
            else:
                graph_info = session.graph_def
            
            train_writer = tf.train.SummaryWriter(args.tb_log_dir + 'train/', graph_info)
            valid_writer = tf.train.SummaryWriter(args.tb_log_dir + 'valid/', graph_info)
            
            # load a saved model or start from random initialization.
            if len(args.init_model) != 0:
                saver.restore(session, args.init_model)
            else:
                tf.initialize_all_variables().run()
            
            for epoch in range(args.num_epochs):
                logging.info('=' * 19 + ' Epoch %d ' + '=' * 19 + '\n', epoch)
                logging.info('Training on training set')
                # training step
                ppl, train_summary_str, global_step = train_model.run_epoch(session, train_batches, is_training=True,
                                      verbose=args.verbose, freq=args.progress_freq)
                # record the summary
                train_writer.add_summary(train_summary_str, global_step)
                train_writer.flush()
                # save model
                saved_path = saver.save(session, args.save_model,
                                        global_step=train_model.global_step)
                                        
                logging.info('Latest model saved in %s\n', saved_path)
                logging.info('Evaluate on validation set')
                
                valid_ppl, valid_summary_str, _ = valid_model.run_epoch(session, valid_batches, is_training=False,
                                      verbose=args.verbose, freq=args.progress_freq)
                
                # save and update best model
                if (len(best_model) == 0) or (valid_ppl < best_valid_ppl):
                    best_model = best_model_saver.save(session, args.save_best_model, 
                                                       global_step=train_model.global_step)
                    best_valid_ppl = valid_ppl
                valid_writer.add_summary(valid_summary_str, global_step)
                valid_writer.flush()
                
                logging.info('Best model is saved in %s', best_model)
                logging.info('Best validation ppl is %f\n', best_valid_ppl)
                
                result['latest_model'] = saved_path
                result['best_model'] = best_model
                # Convert to float because numpy.float is not json serializable.
                result['best_valid_ppl'] = float(best_valid_ppl)
                
                result_path = os.path.join(args.output_dir, 'result.json')
                if os.path.exists(result_path):
                    os.remove(result_path)
                with open(result_path, 'w') as f:
                    json.dump(result, f, indent=2, sort_keys=True)
                
            logging.info('Latest model is saved in %s', saved_path)
            logging.info('Best model is saved in %s', best_model)
            logging.info('Best validation ppl is %f\n', best_valid_ppl)
            
            logging.info('Evaluate the best model on test set')
            saver.restore(session, best_model)
            test_ppl, _, _ = test_model.run_epoch(session, test_batches, is_training=False,
                                      verbose=args.verbose, freq=args.progress_freq)
            result['test_ppl'] = float(test_ppl)
    finally:
        result_path = os.path.join(args.output_dir, 'result.json')
        if os.path.exists(result_path):
            os.remove(result_path)
        with open(result_path, 'w') as f:
            json.dump(result, f, indent=2, sort_keys=True)
    

if __name__ == '__main__':
    main()