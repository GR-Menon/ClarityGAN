# ClarityGAN

Image Dehazing using CycleGANs.  
</br>
Code Reference : https://github.com/aladdinpersson/Machine-Learning-Collection/tree/master/ML/Pytorch/GANs/CycleGAN  
</br>
REalistic Single Image DEhazing dataset (RESIDE) :

- [Official](https://sites.google.com/view/reside-dehaze-datasets/reside-standard)
- [Kaggle](https://www.kaggle.com/datasets/balraj98/indoor-training-set-its-residestandard)

Kaggle notebooks:

- [Regular Training](https://www.kaggle.com/code/gautamrmenon/claritygan-training)
- [Mixed Precision Training](https://www.kaggle.com/code/gautamrmenon/claritygan-mixed-precision-training)

Model Quantization notebooks:

- [Quantization Aware Training](https://www.kaggle.com/code/gautamrmenon/claritygan-quantization-aware-training)
- [Static / Dynamic Quantization](https://www.kaggle.com/code/gautamrmenon/claritygan-quantization-static-dynamic/notebook)

### Results

- Dehazed
  Images ![dehazed_images](https://github.com/GR-Menon/ClarityGAN/assets/98706887/80e87459-57a0-42f9-9bd4-e598884ce587)
- Haze Generated
  Images ![haze_generated_images](https://github.com/GR-Menon/ClarityGAN/assets/98706887/9063ce63-aa12-4e7f-b0cf-aa722cdafeb7)  
  </br>

# Details

### Image-to-Image Translation

This is a class of vision problems where the goal is to learn the mapping between an input image and an output image
using a set of aligned image pairs. CycleGANs introduce an approach for learning to translate an image from source
domain $X$ to a target domain $Y$ in the absence of paired examples. Basically, the model learns a mapping $G \ : \ X \ \rightarrow \ Y$ such that the distribution of images from $G(X)$ is indistinguishable from that of distribution $Y$.  
CycleGANs do not rely on any task-specific, predefined similarity function between the input and output, nor does it assume that the input and output have to lie in the same low-dimensional embedding space.Since this mapping is highly under-constrained, the CycleGAN couples it with an inverse mapping $F \ : \ Y \rightarrow \ X$ and introduce a **cycle-consistency** loss to enforce $F(G(X)) \approx X$.

### Adversarial Loss

Adversarial losses are applied to both mapping functions, $G \ : \ X \rightarrow \ Y$ and $F \ : Y \rightarrow \ X$. For mapping function $G \ : \ X \rightarrow Y$ and its discriminator $D_Y$, the adversarial loss is defined as :       
```math
\mathcal {L}_{GAN}(G, D_Y, X, Y) \ = \ \mathbb {E}_{y \sim p_{data}(y)}[log \ D_Y(y)] \ + \ \mathbb {E}_{x\sim p_{data}(x)}[log(1 \ - \ D_Y(G(x)))]
```  
where $G$ tries to generate images $G(x)$ that look similar to images in domain Y, while $D_Y$ tries to distinguish between translated samples $G(x)$ and real samples $y$.

### Cycle Consistency Loss

Even with the use of adversarial losses, given sufficient capacity, a network can learn to produce outputs that match the target domain's distribution, without learning to map each individual image to a specific desired output. Instead, it could map the same set of input images to any random permutation of images in the target domain, still matching the overall distribution.

Cycle consistency means that if you start with an image $x$ from domain $X$, transform it to domain $Y$ using $G$, and then transform it back to domain $X$ using $F$, we should get back the original image $x$.

Forward Consistency : $\ x \rightarrow G(x) \rightarrow F(G(x)) \approx x$

Backward Consistency : $\ y \rightarrow F(y) \rightarrow G(F(y)) \approx y$

Cycle Consistency Loss is defined as :  
```math
\mathcal{L}_{cyc}(G,F) \ = \ \mathbb{E}_{x \sim p_{data}(x)}\big[ ||F(G(x)) \ - \ x||_1 \big] \ + \ \mathbb{E}_{y \sim p_{data}{y}}\big[ ||G(F(y)) \ - \ y||_1\big]
```

### VGG Loss

VGG Loss is a type of content loss introduced in [Perceptual Losses for Real-Time Style Transfer and Super-Resolution](https://paperswithcode.com/paper/perceptual-losses-for-real-time-style). It is an alternative to pixel-wise losses, and it attempts to be closer to perceptual similarity loss.
Its based on the ReLU activation of the pre-trained VGG19 network.

# References

[Unpaired Image-to-Image Translation using Cycle-Consistent Adversarial Networks](https://arxiv.org/pdf/1703.10593)

```markdown
@misc{zhu2020unpairedimagetoimagetranslationusing,
title={Unpaired Image-to-Image Translation using Cycle-Consistent Adversarial Networks},
author={Jun-Yan Zhu and Taesung Park and Phillip Isola and Alexei A. Efros},
year={2020},
eprint={1703.10593},
archivePrefix={arXiv},
primaryClass={cs.CV},
url={https://arxiv.org/abs/1703.10593},
}
```

[Cycle-Dehaze: Enhanced CycleGAN for Single Image Dehazing](https://arxiv.org/pdf/1805.05308)

```tex
@misc{engin2018cycledehazeenhancedcyclegansingle,
      title={Cycle-Dehaze: Enhanced CycleGAN for Single Image Dehazing},
      author={Deniz Engin and Anıl Genç and Hazım Kemal Ekenel},
      year={2018},
      eprint={1805.05308},
      archivePrefix={arXiv},
      primaryClass={cs.CV},
      url={https://arxiv.org/abs/1805.05308},
}
```

[VGG Loss](https://paperswithcode.com/method/vgg-loss)
