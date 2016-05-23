import argparse
import logging
import numpy as np

def config():
    parser = argparse.ArgumentParser()
    
    # Data and vocabulary file
    parser.add_argument('--data_file', type=str,
                        default='data/tiny_shakespeare.txt',
                        help='data file')

    parser.add_argument('--encoding', type=str,
                        default='utf-8',
                        help='the encoding of the data file.')
    
    # Parameters for saving models.
    parser.add_argument('--output_dir', type=str, default='output',
                        help=('directory to store final and'
                              ' intermediate results and models.'))
    
    # Parameters to configure the neural network.
    parser.add_argument('--hidden_size', type=int, default=128,
                        help='size of RNN hidden state vector')
    parser.add_argument('--embedding_size', type=int, default=0,
                        help='size of character embeddings, 0 for one-hot')
    parser.add_argument('--num_layers', type=int, default=2,
                        help='number of layers in the RNN')
    parser.add_argument('--num_unrollings', type=int, default=10,
                        help='number of unrolling steps.')
    parser.add_argument('--model', type=str, default='lstm',
                        help='which model to use (rnn, lstm or gru).')
    
    # Parameters to control the training.
    parser.add_argument('--num_epochs', type=int, default=50,
                        help='number of epochs')
    parser.add_argument('--batch_size', type=int, default=20,
                        help='minibatch size')
    parser.add_argument('--train_frac', type=float, default=0.9,
                        help='fraction of data used for training.')
    parser.add_argument('--valid_frac', type=float, default=0.05,
                        help='fraction of data used for validation.')
    # test_frac is computed as (1 - train_frac - valid_frac).
    parser.add_argument('--dropout', type=float, default=0.0,
                        help='dropout rate, default to 0 (no dropout).')

    parser.add_argument('--input_dropout', type=float, default=0.0,
                        help=('dropout rate on input layer, default to 0 (no dropout),'
                              'and no dropout if using one-hot representation.'))
    
    # Parameters for gradient descent.
    parser.add_argument('--max_grad_norm', type=float, default=5.,
                        help='clip global grad norm')
    parser.add_argument('--learning_rate', type=float, default=5e-3,
                        help='initial learning rate')

    # Parameters for logging.
    parser.add_argument('--progress_freq', type=int, default=100,
                        help=('frequency for progress report in training and evalution.'))
    parser.add_argument('--verbose', type=int, default=0,
                        help=('whether to show progress report in training and evalution.'))
    
    # Parameters to feed in the initial model and current best model.
    parser.add_argument('--init_model', type=str,
                        default='', help=('initial model'))
    parser.add_argument('--best_model', type=str,
                        default='', help=('current best model'))
    parser.add_argument('--best_valid_ppl', type=float,
                        default=np.Inf, help=('current valid perplexity'))
    
    # Parameters for using saved best models.
    parser.add_argument('--init_dir', type=str, default='',
                        help='continue from the outputs in the given directory')
    
    # Parameters for debugging.
    parser.add_argument('--debug', dest='debug', action='store_true',
                        help='show debug information')
    parser.set_defaults(debug=False)

    # Parameters for unittesting the implementation.
    parser.add_argument('--test', dest='test', action='store_true',
                        help=('use the first 1000 character to as data to test the implementation'))
    parser.set_defaults(test=False)
    
    args = parser.parse_args()
    
    return args