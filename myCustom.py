"""
Custom stuff that is specific to the galaxy contest
"""

import theano
import theano.tensor as T
import numpy as np



class SpiralArmsOnlyDivGalaxyOutputLayer(object):
    """
    divisive normalisation, optimised for performance.
    """
    def __init__(self, input_layer):
        self.input_layer = input_layer
        self.input_shape = self.input_layer.get_output_shape()
        self.params = []
        self.bias_params = []
        self.mb_size = self.input_layer.mb_size

        self.target_var = T.matrix() # variable for the labels

        self.question_slices = [slice(0, 7)]#, slice(1, 7)]

        # self.scaling_factor_indices = [None, [1], [1, 4], [1, 4], [1, 4], None, [0], [13], [1, 3], [1, 4, 7], [1, 4, 7]]
        # indices of all the probabilities that scale each question.

        self.normalisation_mask = theano.shared(self.generate_normalisation_mask())
        # self.scaling_mask = theano.shared(self.generate_scaling_mask())

        # sequence of scaling steps to be undertaken.
        # First element is a slice indicating the values to be scaled. Second element is an index indicating the scale factor.
        # these have to happen IN ORDER else it doesn't work correctly.
#rescaling irrelevant
        #self.scaling_sequence = [
         #   (slice(3, 5), 1), # I: rescale Q2 by A1.2
         #   (slice(5, 13), 4), # II: rescale Q3, Q4, Q5 by A2.2
         #   (slice(15, 18), 0), # III: rescale Q7 by A1.1
         #   (slice(18, 25), 13), # IV: rescale Q8 by A6.1
         #   (slice(25, 28), 3), # V: rescale Q9 by A2.1
         #   (slice(28, 37), 7), # VI: rescale Q10, Q11 by A4.1
       # ]


    def generate_normalisation_mask(self):
        """
        when the clipped input is multiplied by the normalisation mask, the normalisation denominators are generated.
        So then we can just divide the input by the normalisation constants (elementwise).
        """
        mask = np.zeros((7, 7), dtype='float32')
        for s in self.question_slices:
            mask[s, s] = 1.0
        return mask

    # def generate_scaling_mask(self):
    #     """
    #     This mask needs to be applied to the LOGARITHM of the probabilities. The appropriate log probs are then summed,
    #     which corresponds to multiplying the raw probabilities, which is what we want to achieve.
    #     """
    #     mask = np.zeros((37, 37), dtype='float32')
    #     for s, factor_indices in zip(self.question_slices, self.scaling_factor_indices):
    #         if factor_indices is not None:
    #             mask[factor_indices, s] = 1.0
    #     return mask

    def answer_probabilities(self, *args, **kwargs):
        """
        normalise the answer groups for each question.
        """
        input = self.input_layer.output(*args, **kwargs)
        input_clipped = T.maximum(input, 0) # T.clip(input, 0, 1) # T.maximum(input, 0)

        normalisation_denoms = T.dot(input_clipped, self.normalisation_mask) + 1e-12 # small constant to prevent division by 0
        input_normalised = input_clipped / normalisation_denoms

        return input_normalised
        # return [input_normalised[:, s] for s in self.question_slices]

    # def weighted_answer_probabilities(self, *args, **kwargs):
    #     answer_probabilities = self.answer_probabilities(*args, **kwargs)
        
    #     log_scale_factors = T.dot(T.log(answer_probabilities), self.scaling_mask)
    #     scale_factors = T.exp(T.switch(T.isnan(log_scale_factors), -np.inf, log_scale_factors)) # need NaN shielding here because 0 * -inf = NaN.

    #     return answer_probabilities * scale_factors

    def weighted_answer_probabilities(self, *args, **kwargs):
        probs = self.answer_probabilities(*args, **kwargs)

        # go through the rescaling sequence in order (6 steps)
        #for probs_slice, scale_idx in self.scaling_sequence:
         #   probs = T.set_subtensor(probs[:, probs_slice], probs[:, probs_slice] * probs[:, scale_idx].dimshuffle(0, 'x'))

        return probs

    def predictions(self, normalisation=True, *args, **kwargs):
        return self.weighted_answer_probabilities(*args, **kwargs)

    def predictions_no_normalisation(self, *args, **kwargs):
        """
        Predict without normalisation. This can be used for the first few chunks to find good parameters.
        """
        input = self.input_layer.output(*args, **kwargs)
        input_clipped = T.clip(input, 0, 1) # clip on both sides here, any predictions over 1.0 are going to get normalised away anyway.
        return input_clipped

    def error(self, normalisation=True, *args, **kwargs):
        if normalisation:
            predictions = self.predictions(*args, **kwargs)
        else:
            predictions = self.predictions_no_normalisation(*args, **kwargs)
        error = T.mean((predictions - self.target_var) ** 2)
	sqrtError =T.sqrt(error)
        return sqrtError

