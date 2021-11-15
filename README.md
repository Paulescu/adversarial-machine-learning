# Breaking deep learning models with adversarial examples 😈

#### Table of Contents  
1. [What is this repo about?](#what-is-this-repo-about)     
2. [Quick setup](#quick-setup)     
3. [Adversarial attacks](#adversarial-attacks)     
4. [Contact me](#contact-me)  
    


## What is this repo about?

PyTorch code to generate adversarial examples that break Inception V3.

If you do not know what adversarial examples are go check my [blog post](http://datamachines.xyz/2021/07/05/adversarial-examples-to-break-deep-learning-models/).

## Quick setup

You need Python >= 3.7 and a tool to install the exact packages you need for this
code to run as expected.

Two popular tools are virtualenv (a bit old) and [poetry](https://python-poetry.org/) (my favourite).

Instructions for virtualenv:
```
...cd into the root directory...
$ pip install virtualenv
$ virtualenv venv
$ source venv/bin/activate
$ pip install -r requirements.txt
```

Instructions for Poetry:

```
.. install poetry if you haven't
$ curl -sSL https://raw.githubusercontent.com/python-poetry/poetry/master/get-poetry.py | python -

.. cd into the root directory of the repo
$ poetry install
```

## Adversarial attacks

We use the vanilla fast gradient sign method

![](images/fgsm_formula.png)


And its iterative version.

![](images/ifgsm_formula.png)


In the code you can find the "magic" that transforms a nice puppy into a paper towel.

![](images/step_1_to_9.png)


## Contact me

If you want to learn more about real-world ML topics and become a better data scientist

👉 [Subscribe](http://datamachines.xyz/subscribe) to the datamachines newsletter.


Have a great day

Pau