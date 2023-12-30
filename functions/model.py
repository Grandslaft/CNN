import numpy as np
import pickle
from .layers import *

class Main_Model:
    
    # connecting all the layers
    def __init__(self, layers = None, file2load=None):
        
        if file2load is not None:
            self.load_layers(file2load)
            return
        
        elif len(layers) < 2:
            raise ValueError("Not enough layers. Should be at least 2")
        
        self.layers = [layers[0]]
        
        for layer in layers[1:]:
            self.layers[-1].add_layer(layer)
            if layer is not None and layer not in self.layers:
                self.layers.append(layer)
        
        self.early_stop = None
        
    # @calculate_execution_time
    def forward(self, image):
        # forward propagation through all the layers
        for layer in self.layers:
            image = layer.forward(image)
        return image
    
    # @calculate_execution_time
    def backward(self, gradient, learning_rate):
        # backward propagation through all the layers
        for layer in reversed(self.layers):
            gradient = layer.backward(gradient, learning_rate)

    def train(self, X_train, y_train, X_val = None, y_val = None, epochs=10, learning_rate=0.1, lr_decay=0, step=50, val_step=100, verbose=False):
        # extracting amount of data
        data_amount = X_train.shape[0]
        # how many digits to be expected
        num_digits = len(str(data_amount))
        # saving initial learning rate for possible decaying if lr_decay is different from 0
        learning_rate_0 = learning_rate
        
        if verbose and self.early_stop is None:
            self.history = {
                "accuracy": {},     # accuracy on train data
                "loss": {},     # loss on train data
                'val_accuracy': {},     # accuracy on val data
                'val_loss': {},     # accuracy on val data
            }
            
        for epoch in range(epochs):
            accuracies = []
            losses = []
            
            # array for accuracies and losses on the train data
            self.history['accuracy'][epoch] = []
            self.history['loss'][epoch] = []
                
            if X_val is not None:
                # array for accuracies and losses on the validation data
                self.history['val_accuracy'][epoch] = []
                self.history['val_loss'][epoch] = []   
            
            for i, X in enumerate(X_train):
                # forward prop output for image
                output = self.forward(X)
                
                # calculating accuracy and loss
                accuracy = output[np.argmax(y_train[i])]
                loss = -np.sum(y_train[i] * np.log(output))
                accuracies.append(accuracy)
                losses.append(loss)
                
                if X_val is None:
                    self.history['accuracy'][epoch].append(accuracy)
                    self.history['loss'][epoch].append(loss)
                
                # calculating gradient for the backprop
                gradient = np.copy(output) - y_train[i]
                # starting backprop
                self.backward(gradient, learning_rate)
                
                # decay learning rate
                learning_rate = (1/(1+lr_decay*((epoch + 1) * (i + 1)))) * learning_rate_0
                
                if i % step == 0:
                    # average accuracy and loss on the {step} window
                    avg_accuracy = np.mean(accuracies)
                    avg_loss = np.mean(losses)
                    
                    if X_val is not None:
                        val_accuracy, val_loss = self.val_pred(X_val, y_val)
                        self.history['val_accuracy'][epoch].append(val_accuracy)
                        self.history['val_loss'][epoch].append(val_loss)
                    
                    accuracies = []
                    losses = []
                    
                    if self.early_stop is not None:
                        if self.EarlyStopping(epoch, i, avg_accuracy, avg_loss, step, X_val, y_val):
                            self.output = np.array(output)
                            return
                    
                    if i > val_step:
                        if X_val is not None:
                            val_accuracy, val_loss = self.history['val_accuracy'][epoch][-1], self.history['val_loss'][epoch][-1]
                            
                            print(f'[Image {i:{num_digits}d}]: ' +
                                f'Avg Acc: {avg_accuracy:7.2%} | ' +
                                f'Avg Loss {avg_loss:5.2f} | ' +
                                f'Val Acc: {val_accuracy:7.2%} | '+
                                f'Val Loss: {val_loss:5.2f}')
                        else:
                            print(f'[Image {i:{num_digits}d}]: ' +
                                  f'Avg Acc: {avg_accuracy:7.2%} | ' +
                                  f'Avg Loss {avg_loss:5.2f}')
                          
    # function to make predictions on the validation data
    def val_pred(self, X_val, y_val):
        accuracies = []
        losses = []
        
        for i, X in enumerate(X_val):
            current_output = self.forward(X)
            
            accuracy = current_output[np.argmax(y_val[i])]
            loss = -np.sum(y_val[i] * np.log(current_output))
            accuracies.append(accuracy)
            losses.append(loss)
            
        return np.mean(accuracies), np.mean(losses)
    
    # predict function
    def predict(self, X):
        return np.array([np.argmax(self.forward(x), 0) for x in X])
    
    # early stop algorithm
    def EarlyStop(self, monitor="loss", min_delta=.1, min_monitor = 1, patience=3, restore_best_layers=False):
        self.early_stop = {
            "monitor": monitor,     # What value to monitor to make early stop
            "min_delta": min_delta,     # min difference when we think that values don't change enough to continue execution
            "min_monitor": min_monitor,     # min value for monitor, until then early stop won't happen,
                                            # for example we can set this value for accuracy smth like 0.8 
                                            # and early stop won't kick in until accuracy 0.8 has not reached
            "patience": patience,           # window in which we will check values for {min_delta}
            "restore_best_layers": restore_best_layers,     # Restore best weights for all layers {bool}
            'LossOrAcc': True if 'loss' in monitor else False
        }
        
        self.history = {
            "accuracy": {},     # accuracy on train data
            "loss": {},     # loss on train data
            'val_accuracy': {},     # accuracy on val data
            'val_loss': {},     # accuracy on val data
            "global_minmax_index": (0, 0),     # global min or max {monitor} index, depends whether it's loss or accuracy
            "best_layers": []       # best weights for all the layers will be saved here
        }
    
    # process of early stopping
    def EarlyStopping(self, epoch, im_i, accuracy, loss, step, X_val=None, y_val=None):
        
        # transform into packages
        im_i = int(im_i/step) - 1
        
        # if we have enough data to to get data in {patience} window
        if ((epoch + 1) * (im_i + 1)) > self.early_stop['patience']:
            
            # If current monitor metric is bigger/less than global max we update global max to this value
            if self.early_stop['LossOrAcc']:
                if self.history[self.early_stop['monitor']][epoch][-1] >= self.history[self.early_stop['monitor']][self.history['global_minmax_index'][0]][self.history['global_minmax_index'][1]]:
                    self.history['global_minmax_index'] = (epoch, im_i)
            else:
                if self.history[self.early_stop['monitor']][epoch][-1] <= self.history[self.early_stop['monitor']][self.history['global_minmax_index'][0]][self.history['global_minmax_index'][1]]:
                    self.history['global_minmax_index'] = (epoch, im_i)
                    
            # and if we set {restore_best_layers} to True
            if self.early_stop['restore_best_layers'] and (epoch, im_i) == self.history['global_minmax_index']:
                # we save all the crucial data
                self.history['best_layers'] = []
                for layer in self.layers:
                    if layer.__class__.__name__ == 'Conv':
                        self.history['best_layers'].append(layer.filters)
                    elif layer.__class__.__name__ == 'MaxPool':
                        self.history['best_layers'].append(None)
                    elif layer.__class__.__name__ == 'FCL':
                        self.history['best_layers'].append((layer.weights, layer.biases))
            
            if self.early_stop['LossOrAcc']:
                # extracting local max monitor metric in {patience} window
                max_local_loss = max(self.history[self.early_stop['monitor']][epoch][-self.early_stop['patience']:])
                # calculating difference between global min and local max
                difference = abs(max_local_loss - self.history[self.early_stop['monitor']][self.history['global_minmax_index'][0]][self.history['global_minmax_index'][1]])
                
                # if max_local_loss is lesser then the set {min_monitor} and difference is less than {min_delta}
                if max_local_loss <= self.early_stop['min_monitor'] and difference < self.early_stop['min_delta']:
                    # if we set {restore_best_layers} to True than we restore best weights
                    if self.early_stop['restore_best_layers']:
                        for i, layer in enumerate(self.layers):
                            if layer.__class__.__name__ == 'Conv':
                                layer.filters = self.history['best_layers'][i]
                            elif layer.__class__.__name__ == 'MaxPool':
                                continue
                            elif layer.__class__.__name__ == 'FCL':
                                layer.weights, layer.biases = self.history['best_layers'][i]
                    # and now we stop the training process
                    return True
            else:
                # extracting local min monitor metric in {patience} window
                min_local_accuracy = min(self.history[self.early_stop['monitor']][epoch][-self.early_stop['patience']:])
                # calculating difference between global max and local min
                difference = abs(min_local_accuracy - self.history[self.early_stop['monitor']][self.history['global_minmax_index'][0]][self.history['global_minmax_index'][1]])
                
                # if min_local_accuracy is bigger then the set {min_monitor} and difference is less than {min_delta}
                if min_local_accuracy >= self.early_stop['min_monitor'] and difference < self.early_stop['min_delta']:
                    # if we set {restore_best_layers} to True than we restore best weights
                    if self.early_stop['restore_best_layers']:
                        for i, layer in enumerate(self.layers):
                            if layer.__class__.__name__ == 'Conv':
                                layer.filters = self.history['best_layers'][i]
                            elif layer.__class__.__name__ == 'MaxPool':
                                continue
                            elif layer.__class__.__name__ == 'FCL':
                                layer.weights, layer.biases = self.history['best_layers'][i]
                    # and now we stop the training process
                    return True
        return False
    
    # same as described in EarlyStopping
    def restore_best_layers(self):
        if self.early_stop['restore_best_layers']:
            for i, layer in enumerate(self.layers):
                if layer.__class__.__name__ == 'Conv':
                    layer.filters = self.history['best_layers'][i]
                elif layer.__class__.__name__ == 'MaxPool':
                    continue
                elif layer.__class__.__name__ == 'FCL':
                    layer.weights, layer.biases = self.history['best_layers'][i]
                    
    def save_layers(self, file_name):
        layers = []
        for layer in self.layers:
            if layer.__class__.__name__ == 'Conv':
                layers.append(dict(
                    name = 'Conv',
                    num_filters = layer.num_filters,
                    size = layer.size,
                    stride = layer.stride,
                    activation = layer.activation.name,
                    filters = layer.filters
                ))
            elif layer.__class__.__name__ == 'MaxPool':
                layers.append(dict(
                    name = 'MaxPool',
                    size = layer.size,
                    stride = layer.stride,
                ))
            elif layer.__class__.__name__ == 'FCL':
                layers.append(dict(
                    name = 'FCL',
                    weights = layer.weights,
                    biases = layer.biases,
                    activation = layer.activation.name
                ))
            
            with open(f"{file_name}.text", "wb") as file:
                pickle.dump([layer for layer in layers], file)
    
    def load_layers(self, file_name):
        layers = []
        
        with open(f"{file_name}.text", "rb") as file:
            layers2load = [layer for layer in pickle.load(file)]
                    
        for layer in layers2load:
            if layer['name'] == 'Conv':
                tmp = Conv(
                    num_filters = layer['num_filters'],
                    size = layer['size'],
                    stride = layer['stride'],
                    activation = layer['activation'],
                )
                tmp.filters = layer['filters']
                layers.append(tmp)
            elif layer['name'] == 'MaxPool':
                layers.append(
                    MaxPool(
                        size = layer['size'],
                        stride = layer['stride'],
                    )
                )
            elif layer['name'] == 'FCL':
                tmp = FCL(activation = layer['activation'])
                tmp.weights = layer['weights']
                tmp.biases = layer['biases']
                layers.append(tmp)
        
        self.__init__(layers)