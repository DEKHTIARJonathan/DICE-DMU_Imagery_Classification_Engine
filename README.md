# DICE - DMU Imagery Classification Engine

[![License: CC BY-SA 4.0](https://img.shields.io/badge/License-CC%20BY--SA%204.0-lightgrey.svg)](http://creativecommons.org/licenses/by-sa/4.0/)

This repository contains a Tensorflow implementation of the DICE application.

A Deep Convolutional Neural Network model, [Inception-V3](http://arxiv.org/abs/1512.00567) developed by Szegedy et al., pretrained on the [ImageNet](http://www.image-net.org/) dataset has been used for this application. The final layers have been retrained using a transfer learning method, due to the limited amount of data available in the DMU-Net dataset. This process gives much more accurate results and less overfitting on small dataset.

## Hardware Requirements

NVidia GPU cards **GTX Titan X** have been used to accelerate the computation. The final layers fine-tuning process should be running with a powerful CPU in an acceptable period of time (expect a few hours of computation).

## Installation

This process will not be deeply explored. Please refer to the tensorflow documentation for the installation of the library and GPU CUDA drivers, if necessary.

We recommend using the [Anaconda Python](https://www.continuum.io/downloads) environment. It will make the installation a lot easier.

The following libraries are mandatory to run the project:
- **tensorflow:** version => 1.1.0
- **numpy:** version => 1.12.1
- **matplotlib:** version => 2.0.1

We recommend using the latest versions available for each library, in case of trouble please check the version and try installing the versions mentionned above:

```python
import tensorflow as tf
import numpy as np
import matplotlib as mt

print("tensorflow version:", tf.__version__)
print("numpy version:",      np.__version__)
print("matplotlib version:", mt.__version__)
```

## Where to get the Dataset used for this study ?

Please visit the following links and add the data to the **"data/"** folder:
- **Github Repository:** https://github.com/DEKHTIARJonathan/dmu-net.org
- **Official Website:** https://www.dmu-net.org

## Cite This Work
*DEKHTIAR Jonathan, DURUPT Alexandre, BRICOGNE Matthieu, EYNARD Benoit, ROWSON Harvey and KIRITSIS Dimitris* (2017). <br>
Deep Machine Learning for Big Data Engineering Applications - Survey, Opportunities and Case Study.
```
@article {DEKHTIAR2017:DMUNet,
    author = {DEKHTIAR, Jonathan and DURUPT, Alexandre and BRICOGNE, Matthieu and EYNARD, Benoit and ROWSON, Harvey and KIRITSIS, Dimitris},
    title  = {Deep Machine Learning for Big Data Engineering Applications - Survey, Opportunities and Case Study},
    month  = {jan},
    year   = {2017}
}
```

## Open Source Licence - Creative Commons:

### You are free to:

- **Share** — copy and redistribute the material in any medium or format
- **Adapt** — remix, transform, and build upon the material for any purpose, even commercially.

*The licensor cannot revoke these freedoms as long as you follow the license terms.*

### Under the following terms:

- **Attribution** — You must give appropriate credit, provide a link to the license, and indicate if changes were made. You may do so in any reasonable manner, but not in any way that suggests the licensor endorses you or your use.
- **ShareAlike** — If you remix, transform, or build upon the material, you must distribute your contributions under the same license as the original.
 - **No additional restrictions** — You may not apply legal terms or technological measures that legally restrict others from doing anything the license permits.

## Maintainer

* **Lead Developer:** Jonathan DEKHTIAR
* **Contact:** [contact@jonathandekhtiar.eu](mailto:contact@jonathandekhtiar.eu)
* **Twitter:** [@born2data](https://twitter.com/born2data)
* **LinkedIn:** [JonathanDEKHTIAR](https://fr.linkedin.com/in/jonathandekhtiar)
* **Personal Website:** [JonathanDEKHTIAR](http://www.jonathandekhtiar.eu)
* **RSS Feed:** [FeedCrunch.io](https://www.feedcrunch.io/@dataradar/)
* **Tech. Blog:** [born2data.com](http://www.born2data.com/)
* **Github:** [DEKHTIARJonathan](https://github.com/DEKHTIARJonathan)

## Contacts

* **Jonathan DEKHTIAR:** [contact@jonathandekhtiar.eu](mailto:contact@jonathandekhtiar.eu)
* **Alexandre DURUPT:** [alexandre.durupt@utc.fr](mailto:alexandre.durupt@utc.fr)
* **Matthieu BRICOGNE:** [matthieu.bricogne@utc.fr](mailto:matthieu.bricogne@utc.fr)
* **Benoit EYNARD:** [benoit.eynard@utc.fr](mailto:benoit.eynard@utc.fr)
* **Harvey ROWSON:** [rowson@deltacad.fr](mailto:rowson@deltacad.fr)
* **Dimitris KIRITSIS:** [dimitris.kiritsis@epfl.ch](mailto:dimitris.kiritsis@epfl.ch)

