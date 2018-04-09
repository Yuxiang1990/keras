import keras.backend as K

def focal_loss(gamma=2, alpha=2):
    def focal_loss_fixed(y_true, y_pred):
        if (K.backend() == "tensorflow"):
            import tensorflow as tf
            pt = tf.where(tf.equal(y_true, 1), y_pred, 1 - y_pred)
            return -K.sum(alpha * K.pow(1. - pt, gamma) * K.log(pt))
        if (K.backend() == "theano"):
            import theano.tensor as T
            pt = T.where(T.eq(y_true, 1), y_pred, 1 - y_pred)
            return -K.sum(alpha * K.pow(1. - pt, gamma) * K.log(pt))
    return focal_loss_fixed

class instanceloss(object):
    def __init__(self, beta=1., smooth=1., alpha=5.):
        self.beta = beta
        self.smooth = smooth
        self.alpha = alpha
        self.__name__ = 'dice_loss_' + str(int(beta * 100))

    def __call__(self, y_true, y_pred):
        bb = self.beta * self.beta
        y_true_f = K.batch_flatten(y_true)
        y_pred_f = K.batch_flatten(y_pred)
        intersection = (1 + bb) * K.sum(y_true_f * y_pred_f, axis=-1)
        union = bb * K.sum(y_true_f, axis=-1) + K.sum(y_pred_f, axis=-1)
        return -(intersection + self.smooth) / (union + self.alpha * self.smooth)
    
instance_loss = instanceloss(beta=1., smooth=1., alpha=5.)


class DiceLoss(object):
    def __init__(self, beta=1., smooth=1.):
        self.beta = beta  # the more beta, the more recall
        self.smooth = smooth
        if beta == 1:
            self.__name__ = 'dice_loss'
        else:
            self.__name__ = 'dice_loss_' + str(int(beta * 100))

    def __call__(self, y_true, y_pred):
        bb = self.beta * self.beta
        y_true_f = K.flatten(y_true)
        y_pred_f = K.flatten(y_pred)
        intersection = (1 + bb) * K.sum(y_true_f * y_pred_f)
        union = bb * K.sum(y_true_f) + K.sum(y_pred_f)
        return -(intersection + self.smooth) / (union + self.smooth)


dice_loss = DiceLoss(beta=1.)

def iou(y_true, y_pred):
    y_true_f = K.batch_flatten(y_true)
    y_pred_f = K.batch_flatten(y_pred)
    intersection = K.sum(K.round(K.clip(y_true_f * y_pred_f, 0, 1)), axis=-1)
    union = K.sum(K.round(K.clip(y_true_f, 0, 1)), axis=-1) + \
        K.sum(K.round(K.clip(y_true_f, 0, 1)), axis=-1)
    score = (intersection + K.epsilon()) / (union + K.epsilon())
    return score

class Sensitivity:
    def __init__(self, thresh):
        self.thresh = 10**(-thresh)
        self.__name__ = 'sensi_' + str(thresh)

    def __call__(self, y_true, y_pred):
        y_true_f = K.batch_flatten(y_true)
        y_pred_f = K.batch_flatten(y_pred)
        target = K.sum(y_true_f * y_pred_f, axis=-1) > self.thresh
        target_count = K.sum(K.cast(target, K.floatx()))
        total_area = K.sum(y_true_f, axis=-1) > K.epsilon()
        truth_count = K.sum(K.cast(total_area, K.floatx()))
        return target_count / truth_count


def precision(y_true, y_pred):
    '''Calculates the precision, a metric for multi-label classification of
    how many selected items are relevant.
    '''
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
    precision = true_positives / (predicted_positives + K.epsilon())
    return precision


def recall(y_true, y_pred):
    '''Calculates the recall, a metric for multi-label classification of
    how many relevant items are selected.
    '''
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
    recall = true_positives / (possible_positives + K.epsilon())
    return recall


def fbeta_score(y_true, y_pred, beta):
    '''Calculates the F score, the weighted harmonic mean of precision and recall.
    This is useful for multi-label classification, where input samples can be
    classified as sets of labels. By only using accuracy (precision) a model
    would achieve a perfect score by simply assigning every class to every
    input. In order to avoid this, a metric should penalize incorrect class
    assignments as well (recall). The F-beta score (ranged from 0.0 to 1.0)
    computes this, as a weighted mean of the proportion of correct class
    assignments vs. the proportion of incorrect class assignments.
    With beta = 1, this is equivalent to a F-measure. With beta < 1, assigning
    correct classes becomes more important, and with beta > 1 the metric is
    instead weighted towards penalizing incorrect class assignments.
    '''
    if beta < 0:
        raise ValueError('The lowest choosable beta is zero (only precision).')

    # If there are no true positives, fix the F score at 0 like sklearn.
    if K.sum(K.round(K.clip(y_true, 0, 1))) == 0:
        return 0

    p = precision(y_true, y_pred)
    r = recall(y_true, y_pred)
    bb = beta ** 2
    fbeta_score = (1 + bb) * (p * r) / (bb * p + r + K.epsilon())
    return fbeta_score


def fmeasure(y_true, y_pred):
    '''Calculates the f-measure, the harmonic mean of precision and recall.
    '''
    return fbeta_score(y_true, y_pred, beta=1)
