# Generating High-fidelity, Synthetic   Time Series Datasets with DoppelGANger

**[[paper (arXiv)](http://arxiv.org/abs/1909.13403)]**
**[[code](https://github.com/fjxmlzn/DoppelGANger)]**


**Authors:** [Zinan Lin](http://www.andrew.cmu.edu/user/zinanl/), [Alankar Jain](https://www.linkedin.com/in/alankar-jain-5835ab5a/), [Chen Wang](https://wangchen615.github.io/), [Giulia Fanti](https://www.andrew.cmu.edu/user/gfanti/), [Vyas Sekar](https://users.ece.cmu.edu/~vsekar/)

**Abstract:** Limited data access is a substantial barrier to data-driven networking research and development. Although many organizations are motivated to share data, privacy concerns often prevent the sharing of proprietary data, including between teams in the same organization and with outside stakeholders (e.g., researchers, vendors). Many researchers have therefore proposed synthetic data models, most of which have not gained traction because of their narrow scope. In this work, we present DoppelGANger, a synthetic data generation framework based on generative adversarial networks (GANs). DoppelGANger is designed to work on time series datasets with both continuous features (e.g. traffic measurements) and discrete ones (e.g., protocol name). Modeling time series and mixed-type data is known to be difficult; DoppelGANger circumvents these problems through a new conditional architecture that isolates the generation of metadata from time series, but uses metadata to strongly influence time series generation. We demonstrate the efficacy of DoppelGANger on three real-world datasets. We show that DoppelGANger achieves up to 43% better fidelity than baseline models, and captures structural properties of data that baseline methods are unable to learn. Additionally, it gives data holders an easy mechanism for protecting attributes of their data without substantial loss of data utility. 

---
Please visit https://github.com/fjxmlzn/DoppelGANger for details.