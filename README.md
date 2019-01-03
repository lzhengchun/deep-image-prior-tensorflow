# deep-image-prior-tensorflow
A simple yet working implementation of Deep Image Prior paper [https://arxiv.org/pdf/1711.10925.pdf](https://arxiv.org/pdf/1711.10925.pdf) using tensorflow. 

Currently I only implemented the denoising use case.

Denoising to remove the image compression artifact.

<img src="img-prior-in/snail.jpg" alt="Drawing" style="width: 300px;"/> <img src="out/denoised-it7500.png" alt="Drawing" style="width: 300px;"/>

Results after every 1000 iterations, [!] the last one is not always the best one:

<img src="out/denoised-it000.png" alt="Drawing" style="width: 100px;"/>
<img src="out/denoised-it1000.png" alt="Drawing" style="width: 100px;"/>
<img src="out/denoised-it2000.png" alt="Drawing" style="width: 100px;"/>
<img src="out/denoised-it3000.png" alt="Drawing" style="width: 100px;"/>
<img src="out/denoised-it4000.png" alt="Drawing" style="width: 100px;"/>
<img src="out/denoised-it5000.png" alt="Drawing" style="width: 100px;"/>
<img src="out/denoised-it6000.png" alt="Drawing" style="width: 100px;"/>
<img src="out/denoised-it6500.png" alt="Drawing" style="width: 100px;"/>
