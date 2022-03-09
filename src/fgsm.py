from typing import Tuple, List, Optional

import torch
from torch.autograd import Variable
import torch.nn as nn

from src.model import inverse_preprocess


def fast_gradient_sign(
    model: nn.Module,
    x: torch.Tensor,
    eps: float,
    output_type: Optional[str] = 'tensor'
) -> Tuple[torch.Tensor, torch.Tensor]:
    """"""
    assert output_type in {'tensor', 'rgb'}, 'Invalid output_type!'

    # tensor to variable so we can compute gradients with respect to it.
    img_variable = Variable(x, requires_grad=True)

    # forward pass on the original image
    output = model.forward(img_variable)

    # predicted class
    y_true = torch.max(output.data, 1)[1][0].item()
    target = Variable(torch.LongTensor([y_true]), requires_grad=False)

    # loss
    loss_fn = torch.nn.CrossEntropyLoss()
    loss = loss_fn(output, target)

    # compute gradient wrt each variable (requires_grad=True)
    # which you can later access with "var.grad.data"
    loss.backward(retain_graph=True)

    # sign of the gradient wrt input image
    x_grad = torch.sign(img_variable.grad.data)

    # FGSM
    x_adversarial = img_variable.data + eps * x_grad

    if output_type == 'tensor':
        return x_adversarial, x_grad
    else:
        return inverse_preprocess(x_adversarial), inverse_preprocess(x_grad)

def iterative_fast_gradient_sign(
    model: nn.Module,
    x_: torch.Tensor,
    epsilon,
    n_steps: int,
    alpha: float
# ) -> List[Tuple[torch.Tensor, torch.Tensor]]:
) -> Tuple[torch.Tensor, torch.Tensor]:
    """"""
    x = x_.clone().detach()

    for step in range(n_steps):

        # one step using basic FGSM
        x_adv, grad = fast_gradient_sign(model, x, alpha)

        # total perturbation
        total_grad = x_adv - x_

        # force total perturbation to be lower than epsilon in
        # absolute value
        total_grad = torch.clamp(total_grad, -epsilon, epsilon)

        # add total perturbation to the original image
        x_adv = x_ + total_grad

        print('Step ', step + 1)
        # visualize(x, x_adv, grad, eps)

        x = x_adv

    return x_adv, total_grad


def iterative_fast_gradient_sign_(
    model: nn.Module,
    x_: torch.Tensor,
    epsilon,
    n_steps: int,
    alpha: float
    # ) -> List[Tuple[torch.Tensor, torch.Tensor]]:
) -> Tuple[torch.Tensor, torch.Tensor]:
    """"""
    x = x_.clone().detach()

    for step in range(n_steps):

        # one step using basic FGSM
        x_adv, grad = fast_gradient_sign(model, x, alpha)

        # total perturbation
        total_grad = x_adv - x_

        # force total perturbation to be lower than epsilon in
        # absolute value
        total_grad = torch.clamp(total_grad, -epsilon, epsilon)

        # add total perturbation to the original image
        x_adv = x_ + total_grad

        x = x_adv

        yield inverse_preprocess(x_adv), inverse_preprocess(grad)
