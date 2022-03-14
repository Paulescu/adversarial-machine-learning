# from pdb import set_trace as stop

import numpy as np
import matplotlib.pyplot as plt


def plot(x, x_adv, x_grad, epsilon, x_label, x_prob, x_adv_label, x_adv_prob):
    """"""

    # figure, ax = plt.subplots(1,3, figsize=(18,8))
    figure, ax = plt.subplots(1,3, figsize=(12,8))
    ax[0].imshow(x)
    ax[0].set_title('Original', fontsize=16)

    ax[1].imshow(x_grad)
    ax[1].set_title('Perturbation', fontsize=16)
    ax[1].set_yticklabels([])
    ax[1].set_xticklabels([])
    ax[1].set_xticks([])
    ax[1].set_yticks([])

    ax[2].imshow(x_adv)
    ax[2].set_title('New', fontsize=16)

    ax[0].axis('off')
    ax[2].axis('off')

    ax[0].text(1.1,0.5, "+{}*".format(round(epsilon,3)), size=15, ha="center",
               transform=ax[0].transAxes)

    ax[0].text(0.5,-0.13, "Prediction: {}\n Probability: {:.0%}".format(x_label, x_prob), size=12, ha="center",
               transform=ax[0].transAxes)

    ax[1].text(1.1,0.5, " = ", size=15, ha="center", transform=ax[1].transAxes)

    ax[2].text(0.5,-0.13, "Prediction: {}\n Probability: {:.0%}".format(x_adv_label, x_adv_prob), size=12, ha="center",
               transform=ax[2].transAxes)


    plt.show()


def plot_single(x):
    """"""
    # figure, ax = plt.subplots(1,3, figsize=(18,8))
    figure, ax = plt.subplots(1,3, figsize=(12,8))
    ax[0].imshow(x)
    ax[0].set_title('Clean Example', fontsize=16)

    plt.show()


def visualize_OLD(x, x_adv, x_grad, epsilon):
    """"""
    # get predicted label and confidence of x
    x_predictions = get_predictions(x)
    x_label = x_predictions.label
    x_prob = x_predictions.confidence

    # get predicted label and confidence of x_adv
    x_adv_predictions = get_predictions(x_adv)
    x_adv_label = x_adv_predictions.label
    x_adv_prob = x_adv_predictions.confidence

    # transform x into its raw form for visualization
    x = x.squeeze(0) # remove batch dimension: B x C x H x W ==> C x H x W
    x = inverse_preprocess(x)
    x = np.transpose( x , (1,2,0)) # C x H x W ==> H x W x C
    x = np.clip(x, 0, 1)

    # transform x_adv into its raw form for visualization
    x_adv = x_adv.squeeze(0)
    x_adv = inverse_preprocess(x_adv)
    x_adv = np.transpose( x_adv , (1,2,0))   # C X H X W  ==>   H X W X C
    x_adv = np.clip(x_adv, 0, 1)

    # transform grad into its raw form for visualization
    x_grad = x_grad.squeeze(0).numpy()
    x_grad = np.transpose(x_grad, (1,2,0))
    x_grad = np.clip(x_grad, 0, 1)

    # ----------------
    # start plotting
    # ----------------
    figure, ax = plt.subplots(1,3, figsize=(18,8))
    ax[0].imshow(x)
    ax[0].set_title('Clean Example', fontsize=20)

    ax[1].imshow(x_grad)
    ax[1].set_title('Perturbation', fontsize=20)
    ax[1].set_yticklabels([])
    ax[1].set_xticklabels([])
    ax[1].set_xticks([])
    ax[1].set_yticks([])


    ax[2].imshow(x_adv)
    ax[2].set_title('Adversarial Example', fontsize=20)

    ax[0].axis('off')
    ax[2].axis('off')

    ax[0].text(1.1,0.5, "+{}*".format(round(epsilon,3)), size=15, ha="center",
               transform=ax[0].transAxes)

    ax[0].text(0.5,-0.13, "Prediction: {}\n Probability: {}".format(x_label, x_prob), size=15, ha="center",
               transform=ax[0].transAxes)

    ax[1].text(1.1,0.5, " = ", size=15, ha="center", transform=ax[1].transAxes)

    ax[2].text(0.5,-0.13, "Prediction: {}\n Probability: {}".format(x_adv_label, x_adv_prob), size=15, ha="center",
               transform=ax[2].transAxes)


    plt.show()