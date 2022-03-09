<div align="center">
<h1>Breaking deep learning models with adversarial examples 😈 </h1>
<i><h3>Or maybe it's just that beautiful things are so easily broken by the world.</h3></i>
<h4>-- Cassandra Clare, City of Fallen Angels</h4>
</div>

![](http://datamachines.xyz/wp-content/uploads/2021/06/camaleon-2048x1524.jpg)

## Table of Contents

1. [What is this repo about?](#what-is-this-repo-about)     
2. [Quick setup](#quick-setup)   
3. [Fast Gradient Sign method](#fast-gradient-sign-method)
4. [Notebook]()
5. [Live stremlit app]()
6. [Contact me](#contact-me)  
    

## What is this repo about?

PyTorch code and [streamlit app]() that demonstrate how easy it is to break deep learning models in computer vision.
More precisely, Inception V3.

If you do not know what adversarial examples are go check my blog post.

[📝 Adversarial examples to break deep learning models](http://datamachines.xyz/2021/07/05/adversarial-examples-to-break-deep-learning-models/)  

Also available in [Medium](https://towardsdatascience.com/adversarial-examples-to-break-deep-learning-models-e7f543833eae)

## Quick setup

[TODO]

## Fast Gradient Sign method

We use the vanilla fast gradient sign method

![](images/fgsm_formula.png)


And its iterative version.

![](images/ifgsm_formula.png)


In the code you can find the "magic" that transforms a nice puppy into a paper towel.

![](images/step_1_to_9.jpg)


## Contact me

If you want to learn more about real-world ML topics and become a better data scientist

👉 [Subscribe](http://datamachines.xyz/subscribe) to the datamachines newsletter.

[![image](images/subscribe.jpg)](http://datamachines.xyz/subscribe)