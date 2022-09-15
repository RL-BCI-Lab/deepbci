from pdb import set_trace 

import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import tensorflow_probability as tfp

"""
Code here is adapted from https://github.com/google-research/google-research/blob/46536766dff97d679d55d02711d45405140016a1/representation_similarity/Demo.ipynb
"""

def cca(X, Y):
    """Compute the mean squared CCA correlation (R^2_{CCA}).

    Args:
        X (tf.tensor): Tensor of shape (data samples, features)
        Y (tf.tensor): Tensor of shape (data samples, features)

    Returns:
        The mean squared CCA correlations between X and Y.
    """
    qx, _ = tf.linalg.qr(X)
    qy, _ = tf.linalg.qr(Y)
    dimx = X.get_shape().as_list()[1]
    dimy = Y.get_shape().as_list()[1]
    demoninator = tf.dtypes.cast(tf.math.minimum(dimx, dimy), tf.float32)

    return tf.pow(tf.norm(tf.matmul(tf.transpose(qx), qy)), 2) / demoninator

def linear_kernel(x):
    """Compute Gram (kernel) matrix for a linear kernel.

    Args:
        x (tf.tensor): Data matrix to be used by linear kernel.

    Returns:
        A Gram matrix of shape (data samples, data samples)
    """
    return tf.matmul(x, tf.transpose(x))


def rbf_kernel(x, threshold=1.0):
    """Compute Gram (kernel) matrix for an RBF kernel.

    Args:
    x (tf.tensor): Data matrix to be used by rbf kernel.
    
    threshold: Fraction of median Euclidean distance to use as RBF kernel
      bandwidth (this is the heuristic used use in the CKA paper).

    Returns:
        A Gram matrix of shape (data samples, data samples)
    """
    dot_products =  tf.matmul(x, tf.transpose(x))
    sq_norms = tf.linalg.diag_part(dot_products)
    sq_distances = -2 * dot_products + tf.reshape(sq_norms, [-1, 1]) + tf.reshape(sq_norms, [1, -1])
    sq_median_distance = tfp.stats.percentile(sq_distances, 50., interpolation='midpoint')
    return  tf.math.exp(-sq_distances / (2 * threshold ** 2 * sq_median_distance))

def cka(gram_x, gram_y):
    """ Compute centered kernel alignment

    @inproceedings{pmlr-v97-kornblith19a,
        title = {Similarity of Neural Network Representations Revisited},
        author = {Kornblith, Simon and Norouzi, Mohammad and Lee, Honglak and Hinton, Geoffrey},
        booktitle = {Proceedings of the 36th International Conference on Machine Learning},
        pages = {3519--3529},
        year = {2019},
        volume = {97},
        month = {09--15 Jun},
        publisher = {PMLR}
    }
    
    Args:
        gram_x (tf.tensor): A tensor Gram matrix of shape (data samples, data samples).
        gram_y (tf.tensor): A tensor Gram matrix of shape (data samples, data samples).

    Returns:
        CKA score
    """
    # XX.TH
    gram_x_centered = centering(gram_x)
    # YY.TH
    gram_y_centered = centering(gram_y)
    # Note the following is equal to tr(KHLH) where the scaling of HSCI is
    # not needed as it cancels out in the CKA equation. See Eq. (4) in the 
    # CKA paper.
    scaled_hsic = tf.einsum('ij,ji->', gram_x_centered, gram_y_centered)

    # sqrt(XX.TH)
    norm_x = tf.sqrt(tf.linalg.trace(tf.matmul(gram_x_centered, gram_x_centered)))
    # sqrt(XY.TH)
    norm_y = tf.sqrt(tf.linalg.trace(tf.matmul(gram_y_centered, gram_y_centered)))

    return scaled_hsic / (norm_x * norm_y)

def centering(gram):
    """ Centers a matrix by subtracting the mean of each column (axis=0).

        Args:
            gram (tf.tesnor): A Gram matrix of shape (data samples, data samples)
    """
    return gram - tf.math.reduce_mean(gram, axis=0)

def cka_compare_all_layers(X_layers, Y_layers):
    """ Computes CKA scores for two networks for all layer activation pairs

        Args:
            X_layers (list): A list of tf.tensors which contain the activations
                of a given layer. Each tensor should be of shape 
                (data samples, neuron activations).

            Y_layers (list): A list of tf.tensors which contain the activations
                of a given layer. Each tensor should be of shape 
                (data samples, neuron activations).
    """
    layer_scores = np.zeros([len(X_layers), len(Y_layers)])
    X_iter = np.arange(len(X_layers),)[::-1]
    Y_iter = np.arange(len(Y_layers))

    for X, i in zip(X_layers, X_iter):
        scores = []
        for Y, j in zip(Y_layers, Y_iter):
            score = cka(linear_kernel(X), linear_kernel(Y))
            layer_scores[i, j] = score.numpy()

    return layer_scores

def get_network_similarity_plot(layer_scores, layer1_names, layer2_names):
    """ Creates heat map based on network similarity scores.

        Args:
            layer_scores (np.ndarray): NumPy array containing layer scores. 
                Where axis 0 corresponds to layers in model 2 and axis 1 
                corresponds to layers in model 1.

            layer1_names (list): Labels for layers in model 1 or 
                axis 1 of layer_scores.

            layer2_names (list): Labels for layers in model 2 or 
                axis 0 of layer_scores.
    """
    fig, ax = plt.subplots(figsize=(20,20))
    im = ax.imshow(layer_scores)

    ax.set_xticks(np.arange(len(layer1_names)))
    ax.set_yticks(np.arange(len(layer2_names)))

    ax.set_xticklabels(layer1_names)
    ax.set_yticklabels(layer2_names[::-1])

    plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
             rotation_mode="anchor")

    for i in range(len(layer1_names)):
        for j in range(len(layer2_names)):
            text = ax.text(j, i, np.round(layer_scores[i, j], 2), ha="center", va="center", color="w")

    cbar = ax.figure.colorbar(im, ax=ax,)

    ax.set_title("CKA")
    fig.tight_layout()
    
    return fig, ax

