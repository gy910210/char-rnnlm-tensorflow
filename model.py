import logging
import time
import numpy as np
import tensorflow as tf
from tensorflow.models.rnn import rnn

logging.getLogger('tensorflow').setLevel(logging.WARNING)

class CharRNNLM(object):
    def __init__(self, is_training, batch_size, num_unrollings, vocab_size,
                 hidden_size, max_grad_norm, embedding_size, num_layers,
                 learning_rate, model, dropout=0.0, input_dropout=0.0, infer=False):
        self.batch_size = batch_size
        self.num_unrollings = num_unrollings
        if infer:
            self.batch_size = 1
            self.num_unrollings = 1
        self.hidden_size = hidden_size
        self.vocab_size = vocab_size
        self.max_grad_norm = max_grad_norm
        self.num_layers = num_layers
        self.embedding_size = embedding_size
        self.model = model
        self.dropout = dropout
        self.input_dropout = input_dropout
        if embedding_size <= 0:
            self.input_size = vocab_size
            self.input_dropout = 0.0
        else:
            self.input_size = embedding_size
        
        self.input_data = tf.placeholder(tf.int64, [self.batch_size, self.num_unrollings], name='inputs')
        self.targets =  tf.placeholder(tf.int64, [self.batch_size, self.num_unrollings], name='targets')
        
        if self.model == 'rnn':
            cell_fn = tf.nn.rnn_cell.BasicRNNCell
        elif self.model == 'lstm':
            cell_fn = tf.nn.rnn_cell.BasicLSTMCell
        elif self.model == 'gru':
            cell_fn = tf.nn.rnn_cell.GRUCell
        
        params = {'input_size': self.input_size}
        if self.model == 'lstm':
            params['forget_bias'] = 0.0
        cell = cell_fn(self.hidden_size, **params)
        
        cells = [cell]
        params['input_size'] = self.hidden_size
        for i in range(self.num_layers-1):
            higher_layer_cell = cell_fn(self.hidden_size, **params)
            cells.append(higher_layer_cell)
        
        if is_training and self.dropout > 0:
            cells = [tf.nn.rnn_cell.DropoutWrapper(cell, output_keep_prob=1.0-self.dropout) for cell in cells]
        
        multi_cell = tf.nn.rnn_cell.MultiRNNCell(cells)
        
        with tf.name_scope('initial_state'):
            self.zero_state = multi_cell.zero_state(self.batch_size, tf.float32)
            self.initial_state = tf.placeholder(tf.float32, [self.batch_size, multi_cell.state_size], 'initial_state')
        
        with tf.name_scope('embedding_layer'):
            if embedding_size > 0:
                self.embedding = tf.get_variable('embedding', [self.vocab_size, self.embedding_size])
            else:
                self.embedding = tf.constant(np.eye(self.vocab_size), dtype=tf.float32)
            inputs = tf.nn.embedding_lookup(self.embedding, self.input_data)
            if is_training and self.input_dropout > 0:
                inputs = tf.nn.dropout(inputs, 1-self.input_dropout)
        
        with tf.name_scope('slice_inputs'):
            # num_unrollings * (batch_size, embedding_size), the format of rnn inputs.
            sliced_inputs = [tf.squeeze(input_, [1]) for input_ in tf.split(1, self.num_unrollings, inputs)]
            
        outputs, final_state = rnn.rnn(multi_cell, sliced_inputs, initial_state=self.initial_state)
        self.final_state = final_state
        
        with tf.name_scope('flatten_outputs'):
            flat_outputs = tf.reshape(tf.concat(1, outputs), [-1, hidden_size])
        
        with tf.name_scope('flatten_targets'):
            flat_targets = tf.reshape(tf.concat(1, self.targets), [-1])
        
        with tf.variable_scope('softmax') as sm_vs:
            softmax_w = tf.get_variable('softmax_w', [hidden_size, vocab_size])
            softmax_b = tf.get_variable('softmax_b', [vocab_size])
            self.logits = tf.matmul(flat_outputs, softmax_w) + softmax_b
            self.probs = tf.nn.softmax(self.logits)
            
        with tf.name_scope('loss'):
            loss = tf.nn.sparse_softmax_cross_entropy_with_logits(self.logits, flat_targets)
            self.mean_loss = tf.reduce_mean(loss)
            
        with tf.name_scope('loss_montor'):
            count = tf.Variable(1.0, name='count')
            sum_mean_loss = tf.Variable(1.0, name='sum_mean_loss')
            
            self.reset_loss_monitor = tf.group(sum_mean_loss.assign(0.0), 
                                               count.assign(0.0), name='reset_loss_monitor')
            self.update_loss_monitor = tf.group(sum_mean_loss.assign(sum_mean_loss+self.mean_loss),
                                                count.assign(count+1), name='update_loss_monitor')
            
            with tf.control_dependencies([self.update_loss_monitor]):
                self.average_loss = sum_mean_loss / count
                self.ppl = tf.exp(self.average_loss)
                            
            average_loss_summary = tf.scalar_summary('average loss', self.average_loss)
            ppl_summary = tf.scalar_summary('perplexity', self.ppl)
            
        self.summaries = tf.merge_summary([average_loss_summary, ppl_summary], name='loss_monitor')
        
        self.global_step = tf.get_variable('global_step', [], initializer=tf.constant_initializer(0.0))
        
        self.learning_rate = tf.constant(learning_rate)
        
        if is_training:
            tvars = tf.trainable_variables()
            grads, _ = tf.clip_by_global_norm(tf.gradients(self.mean_loss, tvars), self.max_grad_norm)
            optimizer = tf.train.AdamOptimizer(self.learning_rate)
            self.train_op = optimizer.apply_gradients(zip(grads, tvars), global_step=self.global_step)
    
    
    def run_epoch(self, session, batch_generator, is_training, verbose=0, freq=10):
        epoch_size = batch_generator.num_batches
        
        if verbose > 0:
            logging.info('epoch_size: %d', epoch_size)
            logging.info('data_size: %d', batch_generator.seq_length)
            logging.info('num_unrollings: %d', self.num_unrollings)
            logging.info('batch_size: %d', self.batch_size)
        
        if is_training:
            extra_op = self.train_op
        else:
            extra_op = tf.no_op()
        
        state = self.zero_state.eval()
        self.reset_loss_monitor.run()
        batch_generator.reset_batch_pointer()
        start_time = time.time()
        for step in range(epoch_size):
            x, y = batch_generator.next_batch()
            
            ops = [self.average_loss, self.ppl, self.final_state, extra_op,
                   self.summaries, self.global_step, self.learning_rate]
            
            feed_dict = {self.input_data: x, self.targets: y, self.initial_state: state}
            
            results = session.run(ops, feed_dict)
            average_loss, ppl, final_state, _, summary_str, global_step, lr = results
            
            if (verbose > 0) and ((step+1) % freq == 0):
                logging.info('%.1f%%, step:%d, perplexity: %.3f, speed: %.0f words',
                             (step + 1) * 1.0 / epoch_size * 100, step, ppl,
                             (step + 1) * self.batch_size * self.num_unrollings / (time.time() - start_time))
        logging.info("Perplexity: %.3f, speed: %.0f words per sec",
                     ppl, (step + 1) * self.batch_size * self.num_unrollings / (time.time() - start_time))
                     
        return ppl, summary_str, global_step
    
    def sample_seq(self):
        pass