import warnings
import numpy as np
import keras.backend as T
from keras.metrics import categorical_accuracy, mean_squared_error
from ellipse_fit import get_ellipse_kaggle_par
from keras.callbacks import Callback
import os


def get_pool_flags(pooled, unpooled):
    warnings.warn(
        'at the moment only implemented for (2,2) pool layers')
    pooled = np.concatenate((pooled, pooled), axis=3)
    pooled = np.concatenate((pooled, pooled), axis=4)

    return np.equal(unpooled, pooled)


def get_maxout_flags(mo_output, mo_input, weights):
    warnings.warn(
        'at the moment only implemented for maxout with 2 dense layers')

    w, b = weights

    input_ = np.dot(mo_input, w) + b

    pooled = np.reshape(np.concatenate(
        (mo_output, mo_output), axis=-1), newshape=input_.shape)
    return np.equal(input_, pooled)


class weight_history(Callback):
    def __init__(self, layername='conv_0', submodelname='main_seq',
                 path='weight_history', **kwargs):
        self.layername = layername
        self.submodelname = submodelname
        self.path = path
        if not os.path.isdir(self.path):
            os.mkdir(self.path)
        super(weight_history, self).__init__(**kwargs)

    def save(self):
        layer = self.model.get_layer(
            self.submodelname).get_layer(self.layername)
        weight = layer.get_weights()
        np.save(self.path + '/weights_of_' +
                self.layername + '_' + str(self.counter) + '.npy', weight)
        self.counter += 1

        del weight
        del layer

    def on_train_begin(self, logs={}):
        self.counter = 0
        self.save()

    def on_batch_end(self, batch, logs={}):
        self.save()


def lr_function(e, lrs):
    for i in xrange(e, -1, -1):
        if i in lrs:
            return float(lrs[i])


def kaggle_MultiRotMergeLayer_output(x, num_views=2):
    # TODO: stop using mb_size argument! needed for events mod batch_size!=0!
    # build check?
    input_shape = T.shape(x)
    input_ = x
    mb_size = input_shape[0] / 4 / num_views
    # split out the 4* dimension
    input_r = input_.reshape((4 * num_views, mb_size, T.prod(input_shape[1:])))

    def output_shape(lx): return (
        lx[0] // 4 // num_views, (lx[1] * lx[2] * lx[3] * 4 * num_views))
    return input_r.transpose(1, 0, 2).reshape(output_shape(input_shape))


# TODO anyway to get 'num_views' when used as output_shape function? ->
# use a lambda function? use functools.partial would work too
def kaggle_MultiRotMergeLayer_output_shape(input_shape, num_views=2):
    #       #size = T.prod(input_shape[1:]) * (4 * num_views)
    size = (input_shape[1] * input_shape[2] * input_shape[3]) * (4 * num_views)
    return (input_shape[0] // 4 // num_views, size)


def kaggle_input(x, part_size=45, n_input_var=2, include_flip=False, random_flip=False):
    parts = []
    for i in range(0, n_input_var):
        input_ = x[i]
        ps = part_size  # shortcut

        if include_flip:
            input_representations = [input_, input_[
                :, :, :, ::-1]]  # regular and flipped
        elif random_flip:
            input_representations = [input_] if np.random.binomial(1, 0.5) else [
                input_[:, :, :, ::-1]]
        else:
            input_representations = [input_]  # just regular

        for input_rep in input_representations:
            part0 = input_rep[:, :, :ps, :ps]  # 0 degrees
            part1 = input_rep[:, :, :ps, :-ps - 1:-
                              1].dimshuffle(0, 1, 3, 2)  # 90 degrees
            part2 = input_rep[:, :, :-ps - 1:-1, :-ps - 1:-1]  # 180 degrees
            part3 = input_rep[:, :, :-ps - 1:-1,
                              :ps].dimshuffle(0, 1, 3, 2)  # 270 degrees
            parts.extend([part0, part1, part2, part3])

    return T.concatenate(parts, axis=0)


# unfortunately keras2 does not support dicts as output for metrics anymore...
def kaggle_sliced_accuracy(y_true, y_pred, slice_weights=[1.] * 11):
    question_slices = [slice(0, 3), slice(3, 5), slice(5, 7), slice(7, 9), slice(9, 13), slice(13, 15),
                       slice(15, 18), slice(18, 25), slice(25, 28), slice(28, 31), slice(31, 37)]

    accuracy_slices = [categorical_accuracy(
        y_true[:, question_slices[i]], y_pred[:, question_slices[i]]) * slice_weights[i] for i in range(len(question_slices))]
    accuracy_slices = T.cast(accuracy_slices, 'float32')
    return {'sliced_accuracy_mean': T.mean(accuracy_slices), 'sliced_accuracy_std':  T.std(accuracy_slices)}


def sliced_accuracy_mean(y_true, y_pred, slice_weights=[1.] * 11):
    return kaggle_sliced_accuracy(y_true, y_pred, slice_weights)['sliced_accuracy_mean']


def sliced_accuracy_std(y_true, y_pred, slice_weights=[1.] * 11):
    return kaggle_sliced_accuracy(y_true, y_pred, slice_weights)['sliced_accuracy_std']


def mse(y_true, y_pred):
    return mean_squared_error(y_true, y_pred)


def rmse(y_true, y_pred):
    return T.sqrt(mean_squared_error(y_true, y_pred))


def simple_unsupervised_loss(y_true, y_pred):
    warnings.warn('Do not use this unsupervised loss.')
    predmax = T.max(y_pred, axis=-1, keepdims=True)
    unsupervised_y = T.cast(T.equal(y_pred, predmax),
                            T.floatx())
    unsupervised_y = T.softmax(unsupervised_y)

    return mse(unsupervised_y, y_pred)


# in keras 2 weights cant be set directly on layer creation. constant
# initilizer are now available. orth_style will need custom initilizer


def dense_weight_init_values(n_inputs, n_outputs, nb_feature=None, w_std=0.001,
                             b_init_val=0.01, orth_style=False):
    if type(n_inputs) != tuple and type(n_inputs) == int:
        n_inputs = (n_inputs,)
    if type(n_outputs) != tuple and type(n_outputs) == int:
        n_outputs = (n_outputs,)
    W_shape = n_inputs + n_outputs
    if nb_feature:
        W_shape = (nb_feature,) + W_shape
        n_outputs = (nb_feature,) + n_outputs
    weights = (np.random.randn(*W_shape).astype(np.float32) * w_std,
               np.ones(n_outputs).astype(np.float32) * b_init_val)
    if orth_style:
        if len(n_inputs) != 1 or len(n_inputs) != 1:
            raise ValueError(
                'For orthagonal matrixes in weight init the numer of input and output dimension must be one')
        if nb_feature:
            weights = (weights[0] + np.repeat(
                [np.eye(n_inputs[0], n_outputs[0],
                        dtype=np.float32) - w_std / 2], nb_feature, 0),
                weights[1])
        else:
            weights = (weights[0] + np.eye(n_inputs[0], n_outputs[0],
                                           dtype=np.float32) - w_std / 2,
                       weights[1])
    return weights


# TODO categorised and LL loss not updated
def OptimisedDivGalaxyOutput(x, mb_size=None, normalised=True, categorised=False):
    input_layer = x
    if not mb_size:
        mb_size = input_layer.shape[0]
    params = []
    bias_params = []

    question_slices = [slice(0, 3), slice(3, 5), slice(5, 7), slice(7, 9), slice(9, 13), slice(13, 15),
                       slice(15, 18), slice(18, 25), slice(25, 28), slice(28, 31), slice(31, 37)]

    normalisation_mask = T.variable(
        generate_normalisation_mask(question_slices))
    # self.scaling_mask = theano.shared(self.generate_scaling_mask())

    # sequence of scaling steps to be undertaken.
    # First element is a slice indicating the values to be scaled. Second element is an index indicating the scale factor.
    # these have to happen IN ORDER else it doesn't work correctly.
    scaling_sequence = [
        (slice(3, 5), 1),  # I: rescale Q2 by A1.2
        (slice(5, 13), 4),  # II: rescale Q3, Q4, Q5 by A2.2
        (slice(15, 18), 0),  # III: rescale Q7 by A1.1
        (slice(18, 25), 13),  # IV: rescale Q8 by A6.1
        (slice(25, 28), 3),  # V: rescale Q9 by A2.1
        (slice(28, 37), 7),  # VI: rescale Q10, Q11 by A4.1
    ]

    if normalised:
        return predictions(input_layer, normalisation_mask, categorised, mb_size, scaling_sequence)
    else:
        return predictions_no_normalisation(input_layer)


def predictions(input_layer, normalisation_mask, categorised, mb_size, scaling_sequence, *args, **kwargs):
    return weighted_answer_probabilities(input_layer, normalisation_mask, categorised, mb_size, scaling_sequence, *args, **kwargs)


def predictions_no_normalisation(input_layer, *args, **kwargs):
    """
    Predict without normalisation. This can be used for the first few chunks to find good parameters.
    """
    return T.clip(input_layer, 0, 1)  # clip on both sides here, any predictions over 1.0 are going to get normalised away anyway.


def generate_normalisation_mask(question_slices):
    """
    when the clipped input is multiplied by the normalisation mask, the normalisation denominators are generated.
    So then we can just divide the input by the normalisation constants (elementwise).
    """
    mask = np.zeros((37, 37), dtype='float32')
    for s in question_slices:
        mask[s, s] = 1.0
    return mask


'''
def error(self, normalisation=True, *args, **kwargs):
        if normalisation:
            predictions = self.predictions(*args, **kwargs)
        else:
            predictions = self.predictions_no_normalisation(*args, **kwargs)
        error = T.mean((predictions - self.target_var) ** 2)
        return error
'''


def weighted_answer_probabilities(input_layer, normalisation_mask, categorised, mb_size, scaling_sequence, *args, **kwargs):
    probs = answer_probabilities(
        input_layer, normalisation_mask, categorised, mb_size, *args, **kwargs)
    # go through the rescaling sequence in order (6 steps)
    output = []
    output.append(probs[:, 0:3])
    if not categorised:
        # prob_scale = T.ones(T.shape(probs))
        for probs_slice, scale_idx in scaling_sequence:
           # probs = T.set_subtensor(probs[:, probs_slice], probs[:,
           # probs_slice] * probs[:, scale_idx].dimshuffle(0, 'x'))
            output.append(probs[:, probs_slice] *
                          probs[:, scale_idx].dimshuffle(0, 'x'))
            if probs_slice == slice(5, 13):
                output.append(probs[:, 13:15])
            # T.batch_set_value(probs[:, probs_slice], probs[:, probs_slice] *
            # probs[:, scale_idx].dimshuffle(0, 'x'))
        output = T.concatenate(output)
    else:
        output = probs

    return output


def answer_probabilities(x, normalisation_mask, categorised, mb_size, *args, **kwargs):
    """
    normalise the answer groups for each question.
    """
    input_ = T.reshape(x, (mb_size, 37))  # not needed but keep as saveguard
    input_clipped = T.maximum(input_, 0)

    # small constant to prevent division by 0
    normalisation_denoms = T.dot(input_clipped, normalisation_mask) + 1e-7
    input_normalised = input_clipped / normalisation_denoms

    '''
	if categorised:
		output=[]
		for k in xrange(0,mb_size):
		    input_q = [
   			input_normalised[k][0:3], # 1.1 - 1.3,
    			input_normalised[k][3:5], # 2.1 - 2.2
    			input_normalised[k][5:7], # 3.1 - 3.2
    			input_normalised[k][7:9], # 4.1 - 4.2
    			input_normalised[k][9:13], # 5.1 - 5.4
    			input_normalised[k][13:15], # 6.1 - 6.2
    			input_normalised[k][15:18], # 7.1 - 7.3
    			input_normalised[k][18:25], # 8.1 - 8.7
    			input_normalised[k][25:28], # 9.1 - 9.3
    			input_normalised[k][28:31], # 10.1 - 10.3
    			input_normalised[k][31:37], # 11.1 - 11.6
		    ]
		    z_v=[]
		    for i in xrange(0,len(input_q)):
			z=1.
			z = z * (1.-T.greater(input_q[0][2],0.6))
			if i==1: z = z * T.greater(input_q[0][1],0.6)
			if i==2: z = z * T.greater(input_q[1][1],0.6)
			if i==3: z = z * T.greater(input_q[1][1],0.6)
			if i==4: z = z * T.greater(input_q[1][1],0.6)
			if i==6: z = z * T.greater(input_q[0][0],0.6)
			if i==9: z = z * T.greater(input_q[4][0],0.6)
			if i==10: z = z * T.greater(input_q[4][0],0.6)
			for j in xrange(0,len(input_q)):
				z_v.append(z)
		    output.append(T.dot(input_normalised[k],z))
		# print output
		return output
		# input_normaised = sum(input_q,[]) #flattens lists of lists
		# input_normalised[0:3]=input_q[0]
    		# input_normalised[3:5]=input_q[1]
    		# input_normalised[5:7]=input_q[2]
    		# input_normalised[7:9]=input_q[3]
    		# input_normalised[9:13]=input_q[4]
    		# input_normalised[13:15]=input_q[5]
    		# input_normalised[15:18]=input_q[6]
    		# input_normalised[18:25]=input_q[7]
    		# input_normalised[25:28]=input_q[8]
    		# input_normalised[28:31]=input_q[9]
    		# input_normalised[31:37]=input_q[10]
	'''
    return input_normalised
    # return [input_normalised[:, s] for s in self.question_slices]


'''
# not original, not in use
def sqrtError(self, normalisation=True, *args, **kwargs):
	return  T.sqrt(self.error(normalisation=True, *args, **kwargs))

# not original

def error_weighted(self,weight, normalisation=True, *args, **kwargs):
        if normalisation:
            predictions = self.predictions(*args, **kwargs)
        else:
            predictions = self.predictions_no_normalisation(*args, **kwargs)
        error = T.mean(((predictions - self.target_var)*weight) ** 2)
        return error
# not original
# not quadratic like the error!
def ll_error(self, normalisation=True, *args, **kwargs):
	if normalisation:
            predictions = self.predictions(*args, **kwargs)
        else:
            predictions = self.predictions_no_normalisation(*args, **kwargs)
        error =   logloss(self.target_var,predictions)
        return error
'''


def input_generator(train_gen):
    for chunk in train_gen:
        if not chunk:
            print 'WARNING: data input generator yielded ' + str(chunk)
            + ', something went wrong'
        chunk_data, chunk_length = chunk
        y_chunk = chunk_data.pop()  # last element is labels.
        xs_chunk = chunk_data

        # need to transpose the chunks to move the 'channels' dimension up
        xs_chunk = [x_chunk.transpose(0, 3, 1, 2) for x_chunk in xs_chunk]

        l0_input_var = xs_chunk[0]
        l0_45_input_var = xs_chunk[1]
        l6_target_var = y_chunk

        yield ([l0_input_var, l0_45_input_var], l6_target_var)


def ellipse_par_gen(train_gen, num_par=3):
    for img, target in input_generator(train_gen):
        yield (np.array(map(lambda x: x, get_ellipse_kaggle_par(input_=img[0],
                                                                num_par=num_par)) if len(
            np.shape(img[0])) >= 4 else get_ellipse_kaggle_par(input_=img[0],
                                                               num_par=num_par)),
            target)
