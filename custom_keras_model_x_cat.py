from custom_keras_model_x_cat_x_maxout import kaggle_x_cat_x_maxout


class kaggle_x_cat(kaggle_x_cat_x_maxout):
    def __init__(self, *args,
                 **kwargs):

        super(kaggle_x_cat_x_maxout, self).__init__(
            *args,
            **kwargs)

    def init_models(self, final_units=3, loss='categorical_crossentropy'):
        return super(kaggle_x_cat, self).init_models(self,
                                                     final_units=final_units,
                                                     loss=loss,
                                                     n_maxout_layers=2)
