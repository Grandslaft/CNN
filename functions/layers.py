import numpy as np
from activation_functions import Activation
from time_wrapper import calculate_execution_time

class Conv:

    def __init__(self, num_filters = 3, size = 3, stride = 1, activation = 'relu'):
        # pointer to the next layer
        self.next_layer = None
        # num of filters
        self.num_filters = num_filters
        # filter size
        self.size = size
        # step through image when applying filter 
        self.stride = stride
        # activation function
        self.activation = Activation(activation)

        # filters
        self.filters = np.random.randn(num_filters, size, size) / 10
    
    # creation of the pointer self.next_layer
    def add_layer(self, child):
        self.next_layer = child
        return self
    
    # @calculate_execution_time
    def forward(self, image):
        # saving last input for back propagation
        self.last_input = image
        # pixels of the image (supposed to be 1:1 image)
        input_dimension = image.shape[1]
        # size of the images on the output
        output_dimension = int((input_dimension - self.size) / self.stride) + 1
        # output array
        output = np.zeros((self.filters.shape[0], output_dimension, output_dimension))
        
        # for all filters
        for f in range(self.num_filters):
            current_y = output_y = 0
            
            while current_y + self.size <= input_dimension:
                current_x = output_x = 0
                
                while current_x + self.size <= input_dimension:
                    # going through all the pixels if stride is 1 and 
                    # finding all the features of the image using filters
                    # patch of the image with the shape of filter self.size
                    patch = image[:, current_y:current_y + self.size, current_x:current_x + self.size]
                    # assigning value to each pixel after applying filter
                    output[f, output_y, output_x] = np.sum(self.filters[f] * patch)
                    
                    current_x += self.stride
                    output_x += 1
                    
                current_y += self.stride
                output_y += 1
        
        # return filtered image
        output = self.last_output = self.activation.function(output)
        return output
    
    # @calculate_execution_time
    def backward(self, out_prev, lr):
        # clipping value in the input if value are extreme big or small,
        # so it won't return NaN values while calculating back-propagation
        out_prev = np.clip(out_prev, -1e+5, 1e+5)
        input_dimension = self.last_input.shape[1]
        
        # computing the gradient of the loss function with respect to the input
        out_prev = out_prev * self.last_output
        
        # output array
        out_next = np.zeros(self.last_input.shape)
        # value for filter changes aka deltas
        dfilt = np.zeros(self.filters.shape)
        
        for f in range(self.filters.shape[0]):
            current_y = output_y = 0
            
            while current_y + self.size <= input_dimension:
                current_x = output_x = 0
                
                while current_x + self.size <= input_dimension:
                    # extracting patch
                    patch = self.last_input[:, current_y:current_y + self.size, current_x:current_x + self.size]
                    # convolution of the patch and the previous layer’s gradient,
                    # adding the result to the filter’s gradient
                    dfilt[f] += np.sum(out_prev[f, output_y, output_x] * patch, axis=0)
                    # adding the result to the input’s gradient
                    out_next[:, current_y:current_y + self.size, current_x:current_x + self.size] += out_prev[f, output_y, output_x] * self.filters[f]
                    
                    current_x += self.stride
                    output_x += 1
                    
                current_y += self.stride
                output_y += 1
                
        # updating filters (lr -- learning rate)
        self.filters -= lr * dfilt
        return out_next
    
class MaxPool:
    
    def __init__(self, size = 2, stride = 2):
        # pointer to the next layer
        self.next_layer = None
        # step through image when using Pool
        self.stride = stride
        # Pool size or patch size, where we will look for max value
        self.size = size

    def add_layer(self, child):
        self.next_layer = child
        return self
    
    # @calculate_execution_time            
    def forward(self, image):
        self.last_input = image
        
        # extracting image shape where:
        #   first is color channels of the image 
        #   second is the height of the image 
        #   third is the width of the image
        c_channels, h_prev, w_prev = image.shape
        # correcting for the output shape
        h = int((h_prev - self.size) / self.stride) + 1
        w = int((w_prev - self.size) / self.stride) + 1
        
        # output array with downscaled image
        downscaled = np.zeros((c_channels, h, w))
        
        for c in range(c_channels):
            current_y = output_y = 0
            
            while current_y + self.size <= h_prev:
                current_x = output_x = 0
                
                while current_x + self.size <= w_prev:
                    # going through patches we find max values in them and assign 
                    # them to each current pixel in the output array
                    patch = image[c, current_y:current_y + self.size, current_x:current_x + self.size]
                    downscaled[c, output_y, output_x] = np.max(patch)
                    
                    current_x += self.stride
                    output_x += 1
                    
                current_y += self.stride
                output_y += 1
                
        return downscaled
    # @calculate_execution_time
    def backward(self, out_prev, learning_rate):
        # clipping value in the input if value are extreme big or small,
        # so it won't return NaN values while calculating back-propagation
        out_prev = np.clip(out_prev, -1e+5, 1e+5)
        
        # output array
        out_next = np.zeros(self.last_input.shape)
        
        # saving shape of the last layer's input in forward to pass it to the previous layer
        c_channels, shape, _ = self.last_input.shape
        
        for c in range(c_channels):
            current_y = output_y = 0
            while current_y + self.size <= shape:
                current_x = output_x = 0
                while current_x + self.size <= shape:
                    # extracting a patch of the last input
                    patch = self.last_input[c, current_y:current_y + self.size, current_x:current_x + self.size]
                    # finding the max value index in the pool window ignoring NaN values 
                    (x, y) = np.unravel_index(np.nanargmax(patch), patch.shape)
                    # assigning value in the correct pixel, all the others pixels will be zeros
                    out_next[c, current_y + x, current_x + y] += out_prev[c, output_y, output_x]
                    
                    current_x += self.stride
                    output_x += 1
                current_y += self.stride
                output_y += 1
        return out_next

class FCL:
    def __init__(self, n_inputs, n_neurons, activation):
        self.next_layer = None
        # Setting weights and biases as random variables with 
        # a normal distribution divided by 10
        self.weights = np.random.randn(n_inputs, n_neurons) / 10
        self.biases = np.random.randn(n_neurons) / 10
        
        self.activation = Activation(activation)

    def add_layer(self, child):
        self.next_layer = child
        return self
    
    # @calculate_execution_time
    def forward(self, image):
        self.last_input_shape = image.shape
        # flatten given dataset, like if we were given 10x10 image it'll be flatten to 100 array
        image = image.flatten()
        output = np.dot(image, self.weights) + self.biases
        self.last_input = image
        self.last_output = output
        return self.activation.function(output)
    
    # @calculate_execution_time
    def backward(self, out_prev, learning_rate):
        # clipping value in the input if value are extreme big or small,
        # so it won't return NaN values while calculating back-propagation
        out_prev = np.clip(out_prev, -1e+5, 1e+5)
        
        # if this is last layer then we just pass output as dW
        if self.next_layer is None:
            dW = out_prev[:, np.newaxis]
        # else we pass output multiplied by last.output derivative
        else:
            dW = out_prev[:, np.newaxis] * self.activation.derivative(self.last_output)[:, np.newaxis]
        
        # calculation of deltas for updating weights
        dW = (dW * self.last_input[np.newaxis, :]).T
        db = np.copy(out_prev)
        out_next = self.weights @ db
        
        # updating weights
        self.weights -= learning_rate * dW
        self.biases -= learning_rate * db
        return out_next.reshape(self.last_input_shape)