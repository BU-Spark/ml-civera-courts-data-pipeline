**Dataset labeling techniques**:

Interactive visual labeling vs active learning

- <https://link.springer.com/article/10.1631/FITEE.1900549>
- Active learning algorithms can automatically determine a subset of data instances which labels would provide useful input ; interactive visual labeling provide effective visual overviews where the analyst can simultaneously explore the dataset and select items to label

GAN-BERT: Generative Adversarial Learning for Robust Text Classification with a Bunch of Labeled Examples

- <https://paperswithcode.com/paper/gan-bert-generative-adversarial-learning-for>
- BERT + SS-GAN
- HighLight: good performance with limited annotated data
- BERT
  - Classical NLP model
- Semi-Supervised Generative Adversarial Networks
  - <https://ronan.collobert.com/pub/2012_deeplearning_springer.pdf>
- Structure:![](Aspose.Words.8ca64f96-a83b-4da2-add2-2809c6d9d185.001.png)
- Loss:
- Generator(G):
  - the difference between generated representations with the real data representation
  - whether the fake data can be identified by D
- Discriminator(D):
- Label the real data
- Identify the generated data

**Dataset Label Classification**:

Dual Contrastive Learning: Text Classification via Label-Aware Data Augmentation

- <https://arxiv.org/pdf/2201.08702v1.pdf>
- Utilizing contrastive learning on labeled data through simultaneously learning features of input samples and parameters of classifiers in the same space
- Simultaneously learn two kinds of representations with pre-trained language models

**Open Source Projects:**

Open-source Neural Hierarchical Multi-label Text Classification Toolkit:

<https://github.com/Tencent/NeuralNLP-NeuralClassifier>

- Designed for quick implementation of neural models for hierarchical multi-label classification task
- Provides a variety of text encoders, such as FastText, TextCNN, TextRNN, RCNN, VDCNN, DPCNN, DRNN, AttentiveConvNet and Transformer encoder
- Supports other text classification scenarios, including binary-class and multi-class classification and is built on PyTorch

GAN-BERT: Generative Adversarial Learning for Robust Text Classification with a Bunch of Labeled Examples

- <https://paperswithcode.com/paper/gan-bert-generative-adversarial-learning-for>
- <https://github.com/crux82/ganbert>
