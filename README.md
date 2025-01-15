# Awesome Domain Generalization
[![Awesome](https://cdn.rawgit.com/sindresorhus/awesome/d7305f38d29fed78fa85652e3a63e154dd8e8829/media/badge.svg)](https://github.com/sindresorhus/awesome)

This repository is a collection of awesome things about **domain generalization**, including papers, code, etc.

If you would like to contribute to our repository or have any questions/advice, see [Contributing & Contact](#contributing--contact).

# Contents
- [Awesome Domain Generalization](#awesome-domain-generalization)
- [Contents](#contents)
- [Papers](#papers)
  - [Survey](#survey)
  - [Theory \& Analysis](#theory--analysis)
  - [Dataset](#dataset)
  - [Domain Generalization](#domain-generalization)
    - [Domain Alignment-Based Methods](#domain-alignment-based-methods)
    - [Data Augmentation-Based Methods](#data-augmentation-based-methods)
    - [Meta-Learning-Based Methods](#meta-learning-based-methods)
    - [Ensemble Learning-Based Methods](#ensemble-learning-based-methods)
    - [Self-Supervised Learning-Based Methods](#self-supervised-learning-based-methods)
    - [Disentangled Representation Learning-Based Methods](#disentangled-representation-learning-based-methods)
    - [Regularization-Based Methods](#regularization-based-methods)
    - [Normalization-Based Methods](#normalization-based-methods)
    - [Information-Based Methods](#information-based-methods)
    - [Causality-Based Methods](#causality-based-methods)
    - [Inference-Time-Based Methods](#inference-time-based-methods)
    - [Neural Architecture Search-based Methods](#neural-architecture-search-based-methods)
  - [Single Domain Generalization](#single-domain-generalization)
  - [Semi/Weak/Un-Supervised Domain Generalization](#semiweakun-supervised-domain-generalization)
  - [Open/Heterogeneous Domain Generalization](#openheterogeneous-domain-generalization)
  - [Federated Domain Generalization](#federated-domain-generalization)
  - [Source-free Domain Generalization](#source-free-domain-generalization)
  - [Applications](#applications)
    - [Person Re-Identification](#person-re-identification)
    - [Face Recognition \& Anti-Spoofing](#face-recognition--anti-spoofing)
  - [Related Topics](#related-topics)
    - [Life-Long Learning](#life-long-learning)
- [Publications](#publications)
- [Datasets](#datasets)
- [Libraries](#libraries)
- [Lectures \& Tutorials \& Talks](#lectures--tutorials--talks)
- [Other Resources](#other-resources)
- [Contributing \& Contact](#contributing--contact)
- [Acknowledgements](#acknowledgements)

# Papers
> We list papers, implementation code (the unofficial code is marked with *), etc, in the order of year and from journals to conferences. Note that some papers may fall into multiple categories.

## Survey
- Generalizing to Unseen Domains: A Survey on Domain Generalization [[IJCAI 2021](https://arxiv.53yu.com/pdf/2103.03097)] [[Slides](http://jd92.wang/assets/files/DGSurvey-ppt.pdf)] [155]
- Domain Generalization in Vision: A Survey [[TPAMI 2022](https://arxiv.org/abs/2103.02503)] [3]

## Theory & Analysis
> We list the papers that either provide inspiring theoretical analyses or conduct extensive empirical studies for domain generalization.

- A Generalization Error Bound for Multi-Class Domain Generalization [[arXiv 2019](https://arxiv.org/pdf/1905.10392)] [123]
- Domain Generalization by Marginal Transfer Learning [[JMLR 2021](https://www.jmlr.org/papers/volume22/17-679/17-679.pdf)] [[Code](https://github.com/aniketde/DomainGeneralizationMarginal)] (**MTL**) [188]
- The Risks of Invariant Risk Minimization [[ICLR 2021](https://arxiv.org/pdf/2010.05761)] [196]
- In Search of Lost Domain Generalization [[ICLR 2021](https://arxiv.org/pdf/2007.01434.pdf?fbclid=IwAR1YkUXkIhC6fhr6eI687zBXo_W2tTjjTAFnyjEWvmq4gQKon_4pIDbTnQ4)] [134]
- The Many Faces of Robustness: A Critical Analysis of Out-of-Distribution Generalization [[ICCV 2021](https://openaccess.thecvf.com/content/ICCV2021/papers/Hendrycks_The_Many_Faces_of_Robustness_A_Critical_Analysis_of_Out-of-Distribution_ICCV_2021_paper.pdf)] [[Code](https://github.com/hendrycks/imagenet-r)] [135]
- An Empirical Investigation of Domain Generalization with Empirical Risk Minimizers [[NeurIPS 2021](https://proceedings.neurips.cc/paper/2021/file/ecf9902e0f61677c8de25ae60b654669-Paper.pdf)] [[Code](https://github.com/facebookresearch/domainbed_measures)] [198]
- Towards a Theoretical Framework of Out-Of-Distribution Generalization [[NeurIPS 2021](https://proceedings.neurips.cc/paper/2021/file/c5c1cb0bebd56ae38817b251ad72bedb-Paper.pdf)] [199]
- Out-of-Distribution Generalization in Kernel Regression [[NeurIPS 2021](https://proceedings.neurips.cc/paper/2021/file/691dcb1d65f31967a874d18383b9da75-Paper.pdf)] [205]
- Quantifying and Improving Transferability in Domain Generalization [[NeurIPS 2021](https://proceedings.neurips.cc/paper/2021/file/5adaacd4531b78ff8b5cedfe3f4d5212-Paper.pdf)] [[Code](https://github.com/Gordon-Guojun-Zhang/Transferability-NeurIPS2021)] (**Transfer**) [206]
- OoD-Bench: Quantifying and Understanding Two Dimensions of Out-of-Distribution Generalization [[CVPR 2022](https://openaccess.thecvf.com/content/CVPR2022/papers/Ye_OoD-Bench_Quantifying_and_Understanding_Two_Dimensions_of_Out-of-Distribution_Generalization_CVPR_2022_paper.pdf)] [[Code](https://github.com/ynysjtu/ood_bench)] (**OoD-Bench**) [214]

## Dataset
- Free Viewpoint Action Recognition Using Motion History Volumes [[CVIU 2006](https://hal.inria.fr/docs/00/54/46/29/PDF/cviu_motion_history_volumes.pdf)] (**IXMAS dataset**) [39]
- Geodesic flow kernel for unsupervised domain adaptation [[CVPR 2012](http://openaccess.thecvf.com/content_iccv_2013/papers/Fang_Unbiased_Metric_Learning_2013_ICCV_paper.pdf)] (**Office-Caltech dataset**) [32]
- Unbiased Metric Learning: On the Utilization of Multiple Datasets and Web Images for Softening Bias [[ICCV 2013](https://openaccess.thecvf.com/content_iccv_2013/papers/Fang_Unbiased_Metric_Learning_2013_ICCV_paper.pdf)] (**VLCS dataset**) [16]
- Domain Generalization for Object Recognition with Multi-Task Autoencoders [[ICCV 2015](http://openaccess.thecvf.com/content_iccv_2015/papers/Ghifary_Domain_Generalization_for_ICCV_2015_paper.pdf)] [[Code](https://github.com/Emma0118/mate)] (**MTAE**, **Rotated MNIST dataset**) [6]
- Scalable Person Re-identification: A Benchmark [[ICCV 2015](https://www.cv-foundation.org/openaccess/content_iccv_2015/papers/Zheng_Scalable_Person_Re-Identification_ICCV_2015_paper.pdf)] (**Market-1501 dataset**) [46]
- The Cityscapes Dataset for Semantic Urban Scene Understanding [[CVPR 2016](https://openaccess.thecvf.com/content_cvpr_2016/papers/Cordts_The_Cityscapes_Dataset_CVPR_2016_paper.pdf)] (**Cityscapes dataset**) [44]
- The SYNTHIA Dataset: A Large Collection of Synthetic Images for Semantic Segmentation of Urban Scenes [[CVPR 2016](https://www.cv-foundation.org/openaccess/content_cvpr_2016/papers/Ros_The_SYNTHIA_Dataset_CVPR_2016_paper.pdf)] (**SYNTHIA dataset**) [42]
- Playing for Data: Ground Truth from Computer Games [[ECCV 2016](https://linkspringer.53yu.com/chapter/10.1007/978-3-319-46475-6_7)] (**GTA5 dataset**) [43]
- Performance Measures and a Data Set for Multi-target, Multi-camera Tracking [[ECCV 2016](https://linkspringer.53yu.com/chapter/10.1007/978-3-319-48881-3_2)] (**Duke dataset**) [47]
- VisDA: The Visual Domain Adaptation Challenge [[arXiv 2017](https://arxiv.org/pdf/1710.06924.pdf)] (**VisDA-17 dataset**) [36]
- Deep Hashing Network for Unsupervised Domain Adaptation [[CVPR 2017](https://openaccess.thecvf.com/content_cvpr_2017/papers/Venkateswara_Deep_Hashing_Network_CVPR_2017_paper.pdf)] (**OfficeHome dataset**) [20]
- Deeper, Broader and Artier Domain Generalization [[ICCV 2017](https://openaccess.thecvf.com/content_ICCV_2017/papers/Li_Deeper_Broader_and_ICCV_2017_paper.pdf)] [[Code](https://dali-dl.github.io/project_iccv2017.html)] (**PACS dataset**) [2]
- Learning Multiple Visual Domains with Residual Adapters [[NeurIPS 2017](https://proceedings.neurips.cc/paper/2017/file/e7b24b112a44fdd9ee93bdf998c6ca0e-Paper.pdf)] (**Visual Decathlon (VD) dataset**) [38]
- Recognition in Terra Incognita [[ECCV 2018](https://openaccess.thecvf.com/content_ECCV_2018/papers/Beery_Recognition_in_Terra_ECCV_2018_paper.pdf)] (**Terra Incognita dataset**) [45]
- Invariant Risk Minimization [[arXiv 2019](https://arxiv.53yu.com/pdf/1907.02893.pdf;)] [[Code](https://github.com/facebookresearch/InvariantRiskMinimization)] (**IRM**, **Colored MNIST dataset**) [165]
- Learning Robust Representations by Projecting Superficial Statistics Out [[ICLR 2019](https://arxiv.53yu.com/pdf/1903.06256)] [[Code](https://github.com/HaohanWang/HEX)] (**HEX**, **ImageNet-Sketch dataset**) [35]
- Benchmarking Neural Network Robustness to Common Corruptions and Perturbations [[ICLR 2019](https://arxiv.org/pdf/1903.12261.pdf?ref=https://githubhelp.com)] (**CIFAR-10-C / CIFAR-100-C / ImageNet-C dataset**) [37]
- Moment Matching for Multi-Source Domain Adaptation [[ICCV 2019](https://openaccess.thecvf.com/content_ICCV_2019/papers/Peng_Moment_Matching_for_Multi-Source_Domain_Adaptation_ICCV_2019_paper.pdf)] [[Code](http://ai.bu.edu/M3SDA/)] (**DomainNet dataset**) [33]
- Learning to Generate Novel Domains for Domain Generalization [[ECCV 2020](https://arxiv.org/pdf/2007.03304)] [[Code](https://github.com/mousecpn/L2A-OT)] (**L2A-OT**, **Digits-DG dataset**) [28]
- Domain Adaptive Ensemble Learning [[TIP 2021](https://arxiv.53yu.com/pdf/2003.07325)] [[Code](https://github.com/KaiyangZhou/Dassl.pytorch)] (**mini-DomainNet dataset**) [34]
- Towards Non-IID Image Classification A Dataset and Baselines [[PR 2021](https://arxiv.org/pdf/1906.02899)] (**NICO dataset**) [108]
- NICO++ Towards Better Benchmarking for Domain Generalization [[arXiv 2022](https://arxiv.org/pdf/2204.08040)] (**NICO++ dataset**) [183]
- MetaShift: A Dataset of Datasets for Evaluating Contextual Distribution Shifts and Training Conflicts [[ICLR 2022](https://arxiv.org/pdf/2202.06523)] [[Code](https://github.com/Weixin-Liang/MetaShift)] (**MetaShift dataset**) [213]

## Domain Generalization
> To address the dataset/domain shift problem [[109]](https://www.sciencedirect.com/science/article/pii/S0031320311002901?casa_token=qIu5tyPmlgQAAAAA:IDLcYED3jzUGsissKY_EuDLQTMCkGQrEWoAq542Cbcd4FKQinvp78Wgb6jhRiSLqGdQCvcifwprz)) [[110](http://proceedings.mlr.press/v97/recht19a/recht19a.pdf))] [[111](https://link.springer.com/content/pdf/10.1007/s10994-009-5152-4.pdf))] [[112]](https://proceedings.neurips.cc/paper/2020/file/d8330f857a17c53d217014ee776bfd50-Paper.pdf), domain generalization [[113](https://proceedings.neurips.cc/paper/2011/file/b571ecea16a9824023ee1af16897a582-Paper.pdf)] aims to learn a model from source domain(s) and make it generalize well to unknown target domains.

### Domain Alignment-Based Methods
> Domain alignment-based methods aim to minimize divergence between source domains for learning domain-invariant representations.

- Domain Generalization via Invariant Feature Representation [[ICML 2013](http://proceedings.mlr.press/v28/muandet13.pdf)] [[Code](https://github.com/krikamol/dg-dica)] (**DICA**) [65]
- Domain-Adversarial Training of Neural Networks [[JMLR 2016](https://www.jmlr.org/papers/volume17/15-239/15-239.pdf)] [[Code](https://graal.ift.ulaval.ca/dann/)] (**DANN**) [226]
- Learning Attributes Equals Multi-Source Domain Generalization [[CVPR 2016](https://www.cv-foundation.org/openaccess/content_cvpr_2016/papers/Gan_Learning_Attributes_Equals_CVPR_2016_paper.pdf)] (**UDICA**) [120]
- Robust Domain Generalisation by Enforcing Distribution Invariance [[IJCAI 2016](https://eprints.qut.edu.au/115382/15/Erfani2016IJCAI.pdf)] (**ESRand**) [66]
- Scatter Component Analysis A Unified Framework for Domain Adaptation and Domain Generalization [[TPAMI 2017](https://arxiv.53yu.com/pdf/1510.04373)] (**SCA**) [67]
- Unified Deep Supervised Domain Adaptation and Generalization [[ICCV 2017](https://openaccess.thecvf.com/content_ICCV_2017/papers/Motiian_Unified_Deep_Supervised_ICCV_2017_paper.pdf)] [[Code](https://github.com/samotiian/CCSA)] (**CCSA**) [71]
- Beyond Domain Adaptation: Unseen Domain Encapsulation via Universal Non-volume Preserving Models [[arXiv 2018](https://arxiv.53yu.com/pdf/1812.03407)] (**UNVP**) [166]
- Domain Generalization via Conditional Invariant Representation [[AAAI 2018](https://ojs.aaai.org/index.php/AAAI/article/view/11682/11541)] (**CIDG**) [68]
- Domain Generalization with Adversarial Feature Learning [[CVPR 2018](https://openaccess.thecvf.com/content_cvpr_2018/papers/Li_Domain_Generalization_With_CVPR_2018_paper.pdf)] [[Code](https://github.com/YuqiCui/MMD_AAE)] (**MMD-AAE**) [76]
- Deep Domain Generalization via Conditional Invariant Adversarial Networks [[ECCV 2018](http://openaccess.thecvf.com/content_ECCV_2018/papers/Ya_Li_Deep_Domain_Generalization_ECCV_2018_paper.pdf)] (**CIDDG, CDANN**) [77]
- Generalizing to Unseen Domains via Distribution Matching [[arXiv 2019](https://arxiv.53yu.com/pdf/1911.00804)] [[Code](https://github.com/belaalb/G2DM)] (**G2DM**) [81]
- Image Alignment in Unseen Domains via Domain Deep Generalization [[arXiv 2019](https://arxiv.org/pdf/1905.12028)] (**DeGIA**) [169]
- Multi-Adversarial Discriminative Deep Domain Generalization for Face Presentation Attack Detection [[CVPR 2019](http://openaccess.thecvf.com/content_CVPR_2019/papers/Shao_Multi-Adversarial_Discriminative_Deep_Domain_Generalization_for_Face_Presentation_Attack_Detection_CVPR_2019_paper.pdf)] [[Code](https://github.com/rshaojimmy/CVPR2019-MADDoG)] (**MADDG**) [78]
- Generalizable Feature Learning in the Presence of Data Bias and Domain Class Imbalance with Application to Skin Lesion Classification [[MICCAI 2019](https://www.cs.sfu.ca/~hamarneh/ecopy/miccai2019d.pdf)] [72]
- Domain Generalization via Model-Agnostic Learning of Semantic Features [[NeurIPS 2019](https://proceedings.neurips.cc/paper/2019/file/2974788b53f73e7950e8aa49f3a306db-Paper.pdf)] [[Code](https://github.com/biomedia-mira/masf)] (**MASF**) [18]
- Adversarial Invariant Feature Learning with Accuracy Constraint for Domain Generalization [[ECMLPKDD 2019](https://arxiv.53yu.com/pdf/1904.12543)] [[Code](https://github.com/akuzeee/AFLAC)] (**AFLAC**) [84]
- Feature Alignment and Restoration for Domain Generalization and Adaptation [[arXiv 2020](https://arxiv.org/pdf/2006.12009)] (**FAR**) [189]
- Representation via Representations: Domain Generalization via Adversarially Learned Invariant Representations [[arXiv 2020](https://arxiv.53yu.com/pdf/2006.11478)] (**RVR**) [82]
- Correlation-aware Adversarial Domain Adaptation and Generalization [[PR 2020](https://arxiv.53yu.com/pdf/1911.12983)] [[Code](https://github.com/mahfujur1/CA-DA-DG)] (**CAADA**) [80]
- Domain Generalization Using a Mixture of Multiple Latent Domains [[AAAI 2020](https://ojs.aaai.org/index.php/AAAI/article/view/6846/6700)] [[Code](https://github.com/mil-tokyo/dg_mmld)] [83]
- Single-Side Domain Generalization for Face Anti-Spoofing [[CVPR 2020](http://openaccess.thecvf.com/content_CVPR_2020/papers/Jia_Single-Side_Domain_Generalization_for_Face_Anti-Spoofing_CVPR_2020_paper.pdf)] [[Code](https://github.com/taylover-pei/SSDG-CVPR2020)] (**SSDG**) [79]
- Scanner Invariant Multiple Sclerosis Lesion Segmentation from MRI [[ISBI 2020](https://arxiv.53yu.com/pdf/1910.10035)] [85]
- Respecting Domain Relations: Hypothesis Invariance for Domain Generalization [[ICPR 2020](https://arxiv.53yu.com/pdf/2010.07591)] (**HIR**) [74]
- Domain Generalization via Multidomain Discriminant Analysis [[UAI 2020](http://proceedings.mlr.press/v115/hu20a/hu20a.pdf)] [[Code](https://github.com/amber0309/Multidomain-Discriminant-Analysis)] (**MDA**) [70]
- Domain Generalization for Medical Imaging Classification with Linear-Dependency Regularization [[NeurIPS 2020](https://proceedings.neurips.cc/paper/2020/file/201d7288b4c18a679e48b31c72c30ded-Paper.pdf)] [[Code](https://github.com/wyf0912/LDDG)] (**LDDG**) [75]
- Domain Generalization via Entropy Regularization [[NeurIPS 2020](https://proceedings.neurips.cc/paper/2020/file/b98249b38337c5088bbc660d8f872d6a-Paper.pdf)] [[Code](https://github.com/sshan-zhao/DG_via_ER)] [86]
- Iterative Feature Matching: Toward Provable Domain Generalization with Logarithmic Environments [[arXiv 2021](https://arxiv.org/pdf/2106.09913)] [192]
- Semi-Supervised Domain Generalization in RealWorld: New Benchmark and Strong Baseline [[arXiv 2021](https://arxiv.org/pdf/2111.10221)] [179]
- Collaborative Semantic Aggregation and Calibration for Separated Domain Generalization [[arXiv 2021](https://arxiv.org/pdf/2110.06736)] [[Code](https://github.com/junkunyuan/CSAC)] (**CSAC**) [161]
- Multi-Domain Adversarial Feature Generalization for Person Re-Identification [[TIP 2021](https://ieeexplore.ieee.org/iel7/83/9263394/09311771.pdf)] (**MMFA-AAE**) [144]
- Scale Invariant Domain Generalization Image Recapture Detection [[ICONIP 2021](https://arxiv.org/pdf/2110.03496)] (**SADG**) [177]
- Domain Generalization under Conditional and Label Shifts via Variational Bayesian Inference [[IJCAI 2021](https://arxiv.org/pdf/2107.10931)] (**VBCLS**) [195]
- Domain Generalization using Causal Matching [[ICML 2021](http://proceedings.mlr.press/v139/mahajan21b/mahajan21b.pdf)] [[Code](https://github.com/microsoft/robustdg)] (**MatchDG**) [73]
- Generalization on Unseen Domains via Inference-Time Label-Preserving Target Projections [[CVPR 2021](http://openaccess.thecvf.com/content/CVPR2021/papers/Pandey_Generalization_on_Unseen_Domains_via_Inference-Time_Label-Preserving_Target_Projections_CVPR_2021_paper.pdf)] [[Code](https://github.com/yys-Polaris/InferenceTimeDG)] [118]
- Progressive Domain Expansion Network for Single Domain Generalization [[CVPR 2021](http://openaccess.thecvf.com/content/CVPR2021/papers/Li_Progressive_Domain_Expansion_Network_for_Single_Domain_Generalization_CVPR_2021_paper.pdf)] [[Code](https://github.com/lileicv/PDEN)] (**PDEN**) [141]
- Confidence Calibration for Domain Generalization Under Covariate Shift [[ICCV 2021](https://openaccess.thecvf.com/content/ICCV2021/papers/Gong_Confidence_Calibration_for_Domain_Generalization_Under_Covariate_Shift_ICCV_2021_paper.pdf)] [133]
- On Calibration and Out-of-domain Generalization [[NeurIPS 2021](https://proceedings.neurips.cc/paper/2021/file/118bd558033a1016fcc82560c65cca5f-Paper.pdf)] [154]
- Domain-invariant Feature Exploration for Domain Generalization [[TMLR 2022](https://arxiv.org/pdf/2207.12020)] [[Code](https://github.com/jindongwang/transferlearning/tree/master/code/DeepDG)] (**DIFEX**) [209]
- Cross-Domain Ensemble Distillation for Domain Generalization [[ECCV 2022](https://arxiv.org/pdf/2211.14058)] (**XDED**) [94]


### Data Augmentation-Based Methods
> Data augmentation-based methods augment original data and train the model on the generated data to improve model robustness.

- Certifying Some Distributional Robustness with Principled Adversarial Training [[arXiv 2017](https://arxiv.53yu.com/pdf/1710.10571.pdf])] [[Code](https://github.com/duchi-lab/certifiable-distributional-robustness)] [52]
- Generalizing across Domains via Cross-Gradient Training [[ICLR 2018](https://arxiv.53yu.com/pdf/1804.10745)] [[Code](https://github.com/vihari/crossgrad)] (**CrossGrad**) [53]
- Generalizing to Unseen Domains via Adversarial Data Augmentation [[NeurIPS 2018](https://proceedings.neurips.cc/paper/2018/file/1d94108e907bb8311d8802b48fd54b4a-Paper.pdf)] [[Code](https://github.com/ricvolpi/generalize-unseen-domains)] [25]
- Staining Invariant Features for Improving Generalization of Deep Convolutional Neural Networks in Computational Pathology [[Frontiers in Bioengineering and Biotechnology 2019](https://www.frontiersin.org/articles/10.3389/fbioe.2019.00198/full?&utm_source=Email_to_authors_&utm_medium=Email&utm_content=T1_11.5e1_author&utm_campaign=Email_publication&field=&journalName=Frontiers_in_Bioengineering_and_Biotechnology&id=474781)] [26]
- Multi-component Image Translation for Deep Domain Generalization [[WACV 2019](https://arxiv.53yu.com/pdf/1812.08974)] [[Code](https://github.com/mahfujur1/mit-DG)] [167]
- Domain Generalization by Solving Jigsaw Puzzles [[CVPR 2019](https://openaccess.thecvf.com/content_CVPR_2019/papers/Carlucci_Domain_Generalization_by_Solving_Jigsaw_Puzzles_CVPR_2019_paper.pdf)] [[Code](https://github.com/fmcarlucci/JigenDG)] (**JiGen**) [98]
- Addressing Model Vulnerability to Distributional Shifts Over Image Transformation Sets [[ICCV 2019](https://openaccess.thecvf.com/content_ICCV_2019/papers/Volpi_Addressing_Model_Vulnerability_to_Distributional_Shifts_Over_Image_Transformation_Sets_ICCV_2019_paper.pdf)] [[Code](https://github.com/ricvolpi/domain-shift-robustness)] [21]
- Domain Randomization and Pyramid Consistency: Simulation-to-Real Generalization Without Accessing Target Domain Data [[ICCV 2019](http://openaccess.thecvf.com/content_ICCV_2019/papers/Yue_Domain_Randomization_and_Pyramid_Consistency_Simulation-to-Real_Generalization_Without_Accessing_Target_ICCV_2019_paper.pdf)] [[Code](https://github.com/xyyue/DRPC)] [62]
- Hallucinating Agnostic Images to Generalize Across Domains [[ICCV workshop 2019](https://arxiv.53yu.com/pdf/1808.01102)] [[Code](https://github.com/fmcarlucci/ADAGE)] [63]
- Improve Unsupervised Domain Adaptation with Mixup Training [[arXiv 2020](https://arxiv.org/pdf/2001.00677)] [[Code*](https://github.com/facebookresearch/DomainBed)] (**Mixup**) [227]
- Improving the Generalizability of Convolutional Neural Network-Based Segmentation on CMR Images [[Frontiers in Cardiovascular Medicine 2020](https://www.frontiersin.org/articles/10.3389/fcvm.2020.00105/full)] [24]
- Generalizing Deep Learning for Medical Image Segmentation to Unseen Domains via Deep Stacked Transformation [[TMI 2020](https://www.ncbi.nlm.nih.gov/pmc/articles/pmc7393676/)] (**BigAug**) [23]
- Deep Domain-Adversarial Image Generation for Domain Generalisation [[AAAI 2020](https://ojs.aaai.org/index.php/AAAI/article/download/7003/6857)] [[Code](https://github.com/KaiyangZhou/Dassl.pytorch)] (**DDAIG**) [55]
- Towards Universal Representation Learning for Deep Face Recognition [[CVPR 2020](http://openaccess.thecvf.com/content_CVPR_2020/papers/Shi_Towards_Universal_Representation_Learning_for_Deep_Face_Recognition_CVPR_2020_paper.pdf)] [[Code](https://github.com/MatyushinMA/uni_rep_deep_faces)] [22]
- Heterogeneous Domain Generalization via Domain Mixup [[ICASSP 2020](https://arxiv.org/pdf/2009.05448)] [[Code](https://github.com/wyf0912/MIXALL)] [128]
- Learning to Generate Novel Domains for Domain Generalization [[ECCV 2020](https://arxiv.org/pdf/2007.03304)] [[Code](https://github.com/mousecpn/L2A-OT)] (**L2A-OT**, **Digits-DG dataset**) [28]
- Learning from Extrinsic and Intrinsic Supervisions for Domain Generalization [[ECCV 2020](https://arxiv.53yu.com/pdf/2007.09316)] [[Code](https://github.com/emma-sjwang/EISNet)] (**EISNet**) [99]
- Towards Recognizing Unseen Categories in Unseen Domains [[ECCV 2020](https://arxiv.53yu.com/pdf/2007.12256.pdf?ref=https://githubhelp.com)] [[Code](https://github.com/mancinimassimiliano/CuMix)] (**CuMix**) [57]
- Rethinking Domain Generalization Baselines [[ICPR 2020](https://arxiv.53yu.com/pdf/2101.09060)]
- More is Better: A Novel Multi-view Framework for Domain Generalization [[arXiv 2021](https://arxiv.org/pdf/2112.12329)] [184]
- Semi-Supervised Domain Generalization with Stochastic StyleMatch [[arXiv 2021](https://arxiv.53yu.com/pdf/2106.00592)] [[Code](https://github.com/KaiyangZhou/ssdg-benchmark)] (**StyleMatch**) [54]
- Better Pseudo-label Joint Domain-aware Label and Dual-classifier for Semi-supervised Domain Generalization [[arXiv 2021](https://arxiv.53yu.com/pdf/2110.04820)] [156]
- Out-of-domain Generalization from a Single Source: A Uncertainty Quantification Approach [[arXiv 2021](https://arxiv.53yu.com/pdf/2108.02888)] [151]
- Towards Principled Disentanglement for Domain Generalization [[arXiv 2021](https://arxiv.org/pdf/2111.13839)] [[Code](https://github.com/hlzhang109/DDG)] (**DDG**) [170]
- MixStyle Neural Networks for Domain Generalization and Adaptation [[arXiv 2021](https://arxiv.53yu.com/pdf/2107.02053)] [[Code](https://github.com/KaiyangZhou/mixstyle-release)] (**MixStyle**) [58]
- VideoDG: Generalizing Temporal Relations in Videos to Novel Domains [[TPAMI 2021](https://arxiv.org/pdf/1912.03716)] [[Code](https://github.com/thuml/VideoDG)] (**APN**) [197]
- Domain Generalization by Marginal Transfer Learning [[JMLR 2021](https://www.jmlr.org/papers/volume22/17-679/17-679.pdf)] [[Code](https://github.com/aniketde/DomainGeneralizationMarginal)] [188]
- Domain Generalisation with Domain Augmented Supervised Contrastive Learning [[AAAI Student Abstract 2021](https://www.aaai.org/AAAI21Papers/SA-197.LeHS.pdf)] (**DASCL**) [139]
- DecAug: Out-of-Distribution Generalization via Decomposed Feature Representation and Semantic Augmentation [[AAAI 2021](https://arxiv.org/pdf/2012.09382)] [[Code](https://github.com/HaoyueBaiZJU/DecAug)] (**DecAug**) [171]
- Domain Generalization with Mixstyle [[ICLR 2021](https://arxiv.53yu.com/pdf/2104.02008)] [[Code](https://github.com/KaiyangZhou/mixstyle-release)] (**MixStyle**) [56]
- Robust and Generalizable Visual Representation Learning via Random Convolutions [[ICLR 2021](https://arxiv.53yu.com/pdf/2007.13003)] [[Code](https://github.com/wildphoton/RandConv)] (**RC**) [59]
- Learning to Learn Single Domain Generalization [[CVPR 2020](https://openaccess.thecvf.com/content_CVPR_2020/papers/Qiao_Learning_to_Learn_Single_Domain_Generalization_CVPR_2020_paper.pdf)] [[Code](https://github.com/joffery/M-ADA)] (**M-ADA**) [27]
- FSDR: Frequency Space Domain Randomization for Domain Generalization [[CVPR 2021](http://openaccess.thecvf.com/content/CVPR2021/papers/Huang_FSDR_Frequency_Space_Domain_Randomization_for_Domain_Generalization_CVPR_2021_paper.pdf)] [[Code](https://github.com/jxhuang0508/FSDR)] (**FSDR**) [115]
- FedDG: Federated Domain Generalization on Medical Image Segmentation via Episodic Learning in Continuous Frequency Space [[CVPR 2021](http://openaccess.thecvf.com/content/CVPR2021/papers/Liu_FedDG_Federated_Domain_Generalization_on_Medical_Image_Segmentation_via_Episodic_CVPR_2021_paper.pdf)] [[Code](https://github.com/liuquande/FedDG-ELCFS)] (**FedDG**) [147]
- Uncertainty-guided Model Generalization to Unseen Domains [[CVPR 2021](http://openaccess.thecvf.com/content/CVPR2021/papers/Qiao_Uncertainty-Guided_Model_Generalization_to_Unseen_Domains_CVPR_2021_paper.pdf)] [[Code](https://github.com/joffery/UMGUD)] [168]
- Continual Adaptation of Visual Representations via Domain Randomization and Meta-learning [[CVPR 2021](http://openaccess.thecvf.com/content/CVPR2021/papers/Volpi_Continual_Adaptation_of_Visual_Representations_via_Domain_Randomization_and_Meta-Learning_CVPR_2021_paper.pdf)] (**Meta-DR**) [153]
- A Fourier-Based Framework for Domain Generalization [[CVPR 2021](https://openaccess.thecvf.com/content/CVPR2021/papers/Xu_A_Fourier-Based_Framework_for_Domain_Generalization_CVPR_2021_paper.pdf)] [[Code](https://github.com/MediaBrain-SJTU/FACT)] (**FACT**) [160]
- Open Domain Generalization with Domain-Augmented Meta-Learning [[CVPR 2021](http://openaccess.thecvf.com/content/CVPR2021/papers/Shu_Open_Domain_Generalization_with_Domain-Augmented_Meta-Learning_CVPR_2021_paper.pdf)] [[Code](https://github.com/thuml/OpenDG-DAML)] (**DAML**) [119]
- A Simple Feature Augmentation for Domain Generalization [[ICCV 2021](https://openaccess.thecvf.com/content/ICCV2021/papers/Li_A_Simple_Feature_Augmentation_for_Domain_Generalization_ICCV_2021_paper.pdf)] (**SFA**) [142]
- Universal Cross-Domain Retrieval Generalizing Across Classes and Domains [[ICCV 2021](https://openaccess.thecvf.com/content/ICCV2021/papers/Paul_Universal_Cross-Domain_Retrieval_Generalizing_Across_Classes_and_Domains_ICCV_2021_paper.pdf)] [[Code](https://github.com/mvp18/UCDR)] (**SnMpNet**) [150]
- Feature Stylization and Domain-aware Contrastive Learning for Domain Generalization [[MM 2021](https://dl.acm.org/doi/pdf/10.1145/3474085.3475271)] [137]
- Adversarial Teacher-Student Representation Learning for Domain Generalization [[NeurIPS 2021](https://proceedings.neurips.cc/paper/2021/file/a2137a2ae8e39b5002a3f8909ecb88fe-Paper.pdf)] [203]
- Model-Based Domain Generalization [[NeurIPS 2021](https://proceedings.neurips.cc/paper/2021/file/a8f12d9486cbcc2fe0cfc5352011ad35-Paper.pdf)] [[Code](https://github.com/arobey1/mbdg)] (**MBDG**) [200]
- Optimal Representations for Covariate Shift [[ICLR 2022](https://arxiv.org/pdf/2201.00057)] [[Code](https://github.com/ryoungj/optdom)] (**CAD**) [223]
- Label-Efficient Domain Generalization via Collaborative Exploration and Generalization [[MM 2022](https://arxiv.org/abs/2208.03644)] [[Code](https://github.com/junkunyuan/CEG)] (**CEG**) [211]


### Meta-Learning-Based Methods
> Meta-learning-based methods train the model on a meta-train set and improve its performance on a meta-test set for boosting out-of-domain generalization ability.

- Learning to Generalize: Meta-Learning for Domain Generalization [[AAAI 2018](https://www.aaai.org/ocs/index.php/AAAI/AAAI18/paper/viewFile/16067/16547)] [[Code](https://github.com/HAHA-DL/MLDG)] (**MLDG**) [1]
- MetaReg: Towards Domain Generalization using Meta-Regularization [[NeurIPS 2018](https://proceedings.neurips.cc/paper/2018/file/647bba344396e7c8170902bcf2e15551-Paper.pdf)] [[Code*](https://github.com/elliotbeck/MetaReg_PyTorch)] (**MetaReg**) [4]
- Feature-Critic Networks for Heterogeneous Domain Generalisation [[ICML 2019](http://proceedings.mlr.press/v97/li19l/li19l.pdf)] [[Code](https://github.com/liyiying/Feature_Critic)] (**Feature-Critic**) [5]
- Episodic Training for Domain Generalization [[ICCV 2019](https://openaccess.thecvf.com/content_ICCV_2019/papers/Li_Episodic_Training_for_Domain_Generalization_ICCV_2019_paper.pdf)] [[Code](https://github.com/HAHA-DL/Episodic-DG)] (**Epi-FCR**) [7]
- Domain Generalization via Model-Agnostic Learning of Semantic Features [[NeurIPS 2019](https://proceedings.neurips.cc/paper/2019/file/2974788b53f73e7950e8aa49f3a306db-Paper.pdf)] [[Code](https://github.com/biomedia-mira/masf)] (**MASF**) [18]
- Domain Generalization via Semi-supervised Meta Learning [[arXiv 2020](https://arxiv.org/pdf/2009.12658)] [[Code](https://github.com/hosseinshn/DGSML)] (**DGSML**) [127]
- Frustratingly Simple Domain Generalization via Image Stylization [[arXiv 2020](https://arxiv.53yu.com/pdf/2006.11207)] [[Code](https://github.com/GT-RIPL/DomainGeneralization-Stylization)] [60]
- Domain Generalization for Named Entity Boundary Detection via Metalearning [[TNNLS 2020](https://ieeexplore.ieee.org/abstract/document/9174763/)] (**METABDRY**) [124]
- Learning to Learn Single Domain Generalization [[CVPR 2020](https://openaccess.thecvf.com/content_CVPR_2020/papers/Qiao_Learning_to_Learn_Single_Domain_Generalization_CVPR_2020_paper.pdf)] [[Code](https://github.com/joffery/M-ADA)] (**M-ADA**) [27]
- Learning to Learn with Variational Information Bottleneck for Domain Generalization [[ECCV 2020](https://arxiv.org/pdf/2007.07645)] (**MetaVIB**) [15]
- Sequential Learning for Domain Generalization [[ECCV workshop 2020](https://arxiv.org/pdf/2004.01377)] (**S-MLDG**) [14]
- Shape-Aware Meta-Learning for Generalizing Prostate MRI Segmentation to Unseen Domains [[MICCAI 2020](https://arxiv.org/pdf/2007.02035)] [[Code](https://github.com/liuquande/SAML)] (**SAML**) [17]
- More is Better: A Novel Multi-view Framework for Domain Generalization [[arXiv 2021](https://arxiv.org/pdf/2112.12329)] [184]
- Few-Shot Classification in Unseen Domains by Episodic Meta-Learning Across Visual Domains [[ICIP 2021](https://arxiv.org/pdf/2112.13539)] (**x-EML**) [180]
- Meta-Learned Feature Critics for Domain Generalized Semantic Segmentation [[ICIP 2021](https://arxiv.org/pdf/2112.13538)] [185]
- MetaNorm: Learning to Normalize Few-Shot Batches Across Domains [[ICLR 2021](https://openreview.net/pdf?id=9z_dNsC4B5t)] [[Code](https://github.com/YDU-AI/MetaNorm)] (**MetaNorm**) [19]
- Learning to Generalize Unseen Domains via Memory-based Multi-Source Meta-Learning for Person Re-Identification [[CVPR 2021](https://openaccess.thecvf.com/content/CVPR2021/papers/Zhao_Learning_to_Generalize_Unseen_Domains_via_Memory-based_Multi-Source_Meta-Learning_for_CVPR_2021_paper.pdf)] [[Code](https://github.com/HeliosZhao/M3L)] (**M3L**) [12]
- Uncertainty-guided Model Generalization to Unseen Domains [[CVPR 2021](http://openaccess.thecvf.com/content/CVPR2021/papers/Qiao_Uncertainty-Guided_Model_Generalization_to_Unseen_Domains_CVPR_2021_paper.pdf)] [[Code](https://github.com/joffery/UMGUD)] [168]
- Continual Adaptation of Visual Representations via Domain Randomization and Meta-learning [[CVPR 2021](http://openaccess.thecvf.com/content/CVPR2021/papers/Volpi_Continual_Adaptation_of_Visual_Representations_via_Domain_Randomization_and_Meta-Learning_CVPR_2021_paper.pdf)] (**Meta-DR**) [153]
- Meta Batch-Instance Normalization for Generalizable Person Re-Identification [[CVPR 2021](https://openaccess.thecvf.com/content/CVPR2021/papers/Choi_Meta_Batch-Instance_Normalization_for_Generalizable_Person_Re-Identification_CVPR_2021_paper.pdf)] [[Code](https://github.com/bismex/MetaBIN)] (**MetaBIN**) [13]
- Open Domain Generalization with Domain-Augmented Meta-Learning [[CVPR 2021](http://openaccess.thecvf.com/content/CVPR2021/papers/Shu_Open_Domain_Generalization_with_Domain-Augmented_Meta-Learning_CVPR_2021_paper.pdf)] [[Code](https://github.com/thuml/OpenDG-DAML)] (**DAML**) [119]
- On Challenges in Unsupervised Domain Generalization [[NeurIPS workshop 2021](https://proceedings.mlr.press/v181/narayanan22a/narayanan22a.pdf)] [178]
- Exploiting Domain-Specific Features to Enhance Domain Generalization [[NeurIPS 2021](https://proceedings.neurips.cc/paper/2021/file/b0f2ad44d26e1a6f244201fe0fd864d1-Paper.pdf)] [[Code](https://github.com/manhhabui/mDSDI)] (**mDSDI**) [202]

### Ensemble Learning-Based Methods
> Ensemble learning-based methods mainly train a domain-specific model on each source domain, and then draw on collective wisdom to make accurate prediction.

- Exploiting Low-Rank Structure from Latent Domains for Domain Generalization [[ECCV 2014](https://linkspringer.53yu.com/content/pdf/10.1007/978-3-319-10578-9_41.pdf)] [87]
- Visual recognition by learning from web data: A weakly supervised domain generalization approach [[CVPR 2015](https://openaccess.thecvf.com/content_cvpr_2015/papers/Niu_Visual_Recognition_by_2015_CVPR_paper.pdf)] [89]
- Multi-View Domain Generalization for Visual Recognition [[ICCV 2015](http://openaccess.thecvf.com/content_iccv_2015/papers/Niu_Multi-View_Domain_Generalization_ICCV_2015_paper.pdf)] (**MVDG**) [88]
- Deep Domain Generalization With Structured Low-Rank Constraint [[TIP 2017](https://par.nsf.gov/servlets/purl/10065328)] [91]
- Visual Recognition by Learning From Web Data via Weakly Supervised Domain Generalization [[TNNLS 2017](https://bcmi.sjtu.edu.cn/home/niuli/paper/Visual%20Recognition%20by%20Learning%20From%20Web%20Data%20via%20Weakly%20Supervised%20Domain%20Generalization.pdf)] [121]
- Robust Place Categorization with Deep Domain Generalization [[IEEE Robotics and Automation Letters 2018](https://arxiv.53yu.com/pdf/1805.12048)] [[Code](https://github.com/mancinimassimiliano/caffe)] (**COLD**) [97]
- Multi-View Domain Generalization Framework for Visual Recognition [[TNNLS 2018](http://openaccess.thecvf.com/content_iccv_2015/papers/Niu_Multi-View_Domain_Generalization_ICCV_2015_paper.pdf)] [122]
- Domain Generalization with Domain-Specific Aggregation Modules [[GCPR 2018](https://arxiv.53yu.com/pdf/1809.10966)] (**D-SAMs**) [92]
- Best Sources Forward: Domain Generalization through Source-Specific Nets [[ICIP 2018](https://arxiv.53yu.com/pdf/1806.05810)] [90]
- Batch Normalization Embeddings for Deep Domain Generalization [[arXiv 2020](https://arxiv.53yu.com/pdf/2011.12672)] (**BNE**) [96]
- DoFE: Domain-oriented Feature Embedding for Generalizable Fundus Image Segmentation on Unseen Datasets [[TMI 2020](https://arxiv.53yu.com/pdf/2010.06208)] (**DoFE**) [93]
- MS-Net: Multi-Site Network for Improving Prostate Segmentation with Heterogeneous MRI Data [[TMI 2020](https://arxiv.53yu.com/pdf/2002.03366)] [[Code](https://github.com/liuquande/MS-Net)] (**MS-Net**) [95]
- Generalized Convolutional Forest Networks for Domain Generalization and Visual Recognition [[ICLR 2020](https://openreview.net/pdf?id=H1lxVyStPH)] (**GCFN**) [126]
- Learning to Optimize Domain Specific Normalization for Domain Generalization [[ECCV 2020](https://arxiv.53yu.com/pdf/1907.04275)] (**DSON**) [94]
- Class-conditioned Domain Generalization via Wasserstein Distributional Robust Optimization [[ICLR workshop 2021](https://arxiv.org/pdf/2109.03676)] [175]
- Domain and Content Adaptive Convolution for Domain Generalization in Medical Image Segmentation [[arXiv 2021](https://arxiv.org/pdf/2109.05676)] (**DCAC**) [176]
- Dynamically Decoding Source Domain Knowledge for Unseen Domain Generalization [[arXiv 2021](https://www.researchgate.net/profile/Karthik-Nandakumar-3/publication/355142270_Dynamically_Decoding_Source_Domain_Knowledge_For_Unseen_Domain_Generalization/links/61debe18034dda1b9ef16fc6/Dynamically-Decoding-Source-Domain-Knowledge-For-Unseen-Domain-Generalization.pdf)] (**D2SDK**) [174]
- Domain Adaptive Ensemble Learning [[TIP 2021](https://arxiv.53yu.com/pdf/2003.07325)] [[Code](https://github.com/KaiyangZhou/Dassl.pytorch)] (**mini-DomainNet dataset**) [34]
- Generalizable Person Re-identification with Relevance-aware Mixture of Experts [[CVPR 2021](https://openaccess.thecvf.com/content/CVPR2021/papers/Dai_Generalizable_Person_Re-Identification_With_Relevance-Aware_Mixture_of_Experts_CVPR_2021_paper.pdf)] (**RaMoE**) [187]
- Learning Transferrable and Interpretable Representations for Domain Generalization [[MM 2021](https://dl.acm.org/doi/pdf/10.1145/3474085.3475488)] (**DTN**) [131]
- Embracing the Dark Knowledge: Domain Generalization Using Regularized Knowledge Distillation [[MM 2021](https://arxiv.53yu.com/pdf/2110.04820)] (**KDDG**) [157]
- TransMatcher: Deep Image Matching Through Transformers for Generalizable Person Re-identification [[NeurIPS 2021](https://proceedings.neurips.cc/paper/2021/file/0f49c89d1e7298bb9930789c8ed59d48-Paper.pdf)] [[Code](https://github.com/ShengcaiLiao/QAConv)] (**TransMatcher**) [208]
- Cross-Domain Ensemble Distillation for Domain Generalization [[ECCV 2022](https://arxiv.org/pdf/2211.14058)] (**XDED**) [94]


### Self-Supervised Learning-Based Methods
> Self-supervised learning-based methods improve model generalization by solving some pretext tasks with data itself.

- Domain Generalization for Object Recognition with Multi-Task Autoencoders [[ICCV 2015](http://openaccess.thecvf.com/content_iccv_2015/papers/Ghifary_Domain_Generalization_for_ICCV_2015_paper.pdf)] [[Code](https://github.com/Emma0118/mate)] (**MTAE**, **Rotated MNIST dataset**) [6]
- Domain Generalization by Solving Jigsaw Puzzles [[CVPR 2019](https://openaccess.thecvf.com/content_CVPR_2019/papers/Carlucci_Domain_Generalization_by_Solving_Jigsaw_Puzzles_CVPR_2019_paper.pdf)] [[Code](https://github.com/fmcarlucci/JigenDG)] (**JiGen**) [98]
- Improving Out-Of-Distribution Generalization via Multi-Task Self-Supervised Pretraining [[arXiv 2020](https://arxiv.53yu.com/pdf/2003.13525)] [102]
- Generalized Convolutional Forest Networks for Domain Generalization and Visual Recognition [[ICLR 2020](https://openreview.net/pdf?id=H1lxVyStPH)] (**GCFN**) [126]
- Learning from Extrinsic and Intrinsic Supervisions for Domain Generalization [[ECCV 2020](https://arxiv.53yu.com/pdf/2007.09316)] [[Code](https://github.com/emma-sjwang/EISNet)] (**EISNet**) [99]
- Zero Shot Domain Generalization [[BMVC 2020](https://arxiv.53yu.com/pdf/2008.07443)] [[Code](https://github.com/aniketde/ZeroShotDG)] [100]
- Out-of-domain Generalization from a Single Source: A Uncertainty Quantification Approach [[arXiv 2021](https://arxiv.53yu.com/pdf/2108.02888)] [151]
- Self-Supervised Learning Across Domains [[TPAMI 2021](https://arxiv.53yu.com/pdf/2007.12368)] [[Code](https://github.com/silvia1993/Self-Supervised_Learning_Across_Domains)] [101]
- Multi-Domain Adversarial Feature Generalization for Person Re-Identification [[TIP 2021](https://ieeexplore.ieee.org/iel7/83/9263394/09311771.pdf)] (**MMFA-AAE**) [144]
- Scale Invariant Domain Generalization Image Recapture Detection [[ICONIP 2021](https://arxiv.org/pdf/2110.03496)] (**SADG**) [177]
- Domain Generalisation with Domain Augmented Supervised Contrastive Learning [[AAAI Student Abstract 2021](https://www.aaai.org/AAAI21Papers/SA-197.LeHS.pdf)]
- Progressive Domain Expansion Network for Single Domain Generalization [[CVPR 2021](http://openaccess.thecvf.com/content/CVPR2021/papers/Li_Progressive_Domain_Expansion_Network_for_Single_Domain_Generalization_CVPR_2021_paper.pdf)] [[Code](https://github.com/lileicv/PDEN)] (**PDEN**) [141]
- FedDG: Federated Domain Generalization on Medical Image Segmentation via Episodic Learning in Continuous Frequency Space [[CVPR 2021](http://openaccess.thecvf.com/content/CVPR2021/papers/Liu_FedDG_Federated_Domain_Generalization_on_Medical_Image_Segmentation_via_Episodic_CVPR_2021_paper.pdf)] [[Code](https://github.com/liuquande/FedDG-ELCFS)] (**FedDG**) [147]
- Boosting the Generalization Capability in Cross-Domain Few-shot Learning via Noise-enhanced Supervised Autoencoder [[ICCV 2021](https://openaccess.thecvf.com/content/ICCV2021/papers/Liang_Boosting_the_Generalization_Capability_in_Cross-Domain_Few-Shot_Learning_via_Noise-Enhanced_ICCV_2021_paper.pdf)] (**NSAE**) [194]
- A Style and Semantic Memory Mechanism for Domain Generalization [[ICCV 2021](http://openaccess.thecvf.com/content/ICCV2021/papers/Chen_A_Style_and_Semantic_Memory_Mechanism_for_Domain_Generalization_ICCV_2021_paper.pdf)] (**STEAM**) [130]
- SelfReg: Self-Supervised Contrastive Regularization for Domain Generalization [[ICCV 2021](http://openaccess.thecvf.com/content/ICCV2021/papers/Kim_SelfReg_Self-Supervised_Contrastive_Regularization_for_Domain_Generalization_ICCV_2021_paper.pdf)] (**SelfReg**) [138]
- Domain Generalization for Mammography Detection via Multi-style and Multi-view Contrastive Learning [[MICCAI 2021](https://arxiv.org/pdf/2111.10827)] [[Code](https://github.com/lizheren/MSVCL_MICCAI2021)] (**MSVCL**) [172]
- Feature Stylization and Domain-aware Contrastive Learning for Domain Generalization [[MM 2021](https://dl.acm.org/doi/pdf/10.1145/3474085.3475271)] [137]
- Adversarial Teacher-Student Representation Learning for Domain Generalization [[NeurIPS 2021](https://proceedings.neurips.cc/paper/2021/file/a2137a2ae8e39b5002a3f8909ecb88fe-Paper.pdf)]
- Domain Generalization via Contrastive Causal Learning [[arXiv 2022](https://arxiv.org/abs/2210.02655)] (**CCM**) [212]
- Towards Unsupervised Domain Generalization [[CVPR 2022](https://openaccess.thecvf.com/content/CVPR2022/papers/Zhang_Towards_Unsupervised_Domain_Generalization_CVPR_2022_paper.pdf)] (**DARLING**) [69]
- Unsupervised Domain Generalization by Learning a Bridge Across Domains [[CVPR 2022](https://openaccess.thecvf.com/content/CVPR2022/papers/Harary_Unsupervised_Domain_Generalization_by_Learning_a_Bridge_Across_Domains_CVPR_2022_paper.pdf)] [[Code](https://github.com/leokarlin/BrAD)] (**BrAD**) [182]

### Disentangled Representation Learning-Based Methods
> Disentangled representation learning-based methods aim to disentangle domain-specific and domain-invariant parts from source data, and then adopt the domain-invariant one for inference on the target domains.

- Undoing the Damage of Dataset Bias [[ECCV 2012](https://linkspringer.53yu.com/content/pdf/10.1007/978-3-642-33718-5_12.pdf)] [[Code](https://github.com/adikhosla/undoing-bias)] [103]
- Deeper, Broader and Artier Domain Generalization [[ICCV 2017](https://openaccess.thecvf.com/content_ICCV_2017/papers/Li_Deeper_Broader_and_ICCV_2017_paper.pdf)] [[Code](https://dali-dl.github.io/project_iccv2017.html)] [2]
- DIVA: Domain Invariant Variational Autoencoders [[ICML workshop 2019](http://proceedings.mlr.press/v121/ilse20a/ilse20a.pdf)] [[Code](https://github.com/AMLab-Amsterdam/DIVA)] (**DIVA**) [107]
- Efficient Domain Generalization via Common-Specific Low-Rank Decomposition [[ICML 2020](http://proceedings.mlr.press/v119/piratla20a/piratla20a.pdf)] [[Code](https://github.com/vihari/CSD)] (**CSD**) [105]
- Cross-Domain Face Presentation Attack Detection via Multi-Domain Disentangled Representation Learning [[CVPR 2020](http://openaccess.thecvf.com/content_CVPR_2020/papers/Wang_Cross-Domain_Face_Presentation_Attack_Detection_via_Multi-Domain_Disentangled_Representation_Learning_CVPR_2020_paper.pdf)] [106]
- Learning to Balance Specificity and Invariance for In and Out of* Domain Generalization [[ECCV 2020](https://arxiv.53yu.com/pdf/2008.12839)] [[Code](https://github.com/prithv1/DMG)] (**DMG**) [104]
- Towards Principled Disentanglement for Domain Generalization [[arXiv 2021](https://arxiv.org/pdf/2111.13839)] [[Code](https://github.com/hlzhang109/DDG)] (**DDG**) [170]
- Meta-Learned Feature Critics for Domain Generalized Semantic Segmentation [[ICIP 2021](https://arxiv.org/pdf/2112.13538)] [185]
- DecAug: Out-of-Distribution Generalization via Decomposed Feature Representation and Semantic Augmentation [[AAAI 2021](https://arxiv.org/pdf/2012.09382)] [[Code](https://github.com/HaoyueBaiZJU/DecAug)] (**DecAug**) [171]
- Robustnet: Improving Domain Generalization in Urban-Scene Segmentation via Instance Selective Whitening [[CVPR 2021](https://openaccess.thecvf.com/content/CVPR2021/papers/Choi_RobustNet_Improving_Domain_Generalization_in_Urban-Scene_Segmentation_via_Instance_Selective_CVPR_2021_paper.pdf)] [[Code](https://github.com/shachoi/RobustNet)] (**RobustNet**) [193]
- Reducing Domain Gap by Reducing Style Bias [[CVPR 2021](https://openaccess.thecvf.com/content/CVPR2021/papers/Nam_Reducing_Domain_Gap_by_Reducing_Style_Bias_CVPR_2021_paper.pdf)] [[Code](https://github.com/hyeonseobnam/sagnet)] (**SagNet**)  [230]
- Shape-Biased Domain Generalization via Shock Graph Embeddings [[ICCV 2021](https://openaccess.thecvf.com/content/ICCV2021/papers/Narayanan_Shape-Biased_Domain_Generalization_via_Shock_Graph_Embeddings_ICCV_2021_paper.pdf)] [149]
- Domain-Invariant Disentangled Network for Generalizable Object Detection [[ICCV 2021](https://openaccess.thecvf.com/content/ICCV2021/papers/Lin_Domain-Invariant_Disentangled_Network_for_Generalizable_Object_Detection_ICCV_2021_paper.pdf)] [143]
- Domain Generalization via Feature Variation Decorrelation [[MM 2021](https://dl.acm.org/doi/pdf/10.1145/3474085.3475311)] [146]
- Exploiting Domain-Specific Features to Enhance Domain Generalization [[NeurIPS 2021](https://proceedings.neurips.cc/paper/2021/file/b0f2ad44d26e1a6f244201fe0fd864d1-Paper.pdf)] [[Code](https://github.com/manhhabui/mDSDI)] (**mDSDI**) [202]
- Variational Disentanglement for Domain Generalization [[TMLR 2022](https://arxiv.org/pdf/2109.05826)] (**VDN**) [210]
- Intra-Source Style Augmentation for Improved Domain Generalization [[WACV 2023](https://arxiv.org/pdf/2210.10175.pdf)] (**ISSA**) [215]

### Regularization-Based Methods
> Regularization-based methods leverage regularization terms to prevent the overfitting, or design optimization strategies to guide the training.

- Generalizing from Several Related Classification Tasks to a New Unlabeled Sample [[NeurIPS 2011](https://proceedings.neurips.cc/paper/2011/file/b571ecea16a9824023ee1af16897a582-Paper.pdf)] [113]
- MetaReg: Towards Domain Generalization using Meta-Regularization [[NeurIPS 2018](https://proceedings.neurips.cc/paper/2018/file/647bba344396e7c8170902bcf2e15551-Paper.pdf)] [[Code*](https://github.com/elliotbeck/MetaReg_PyTorch)] (**MetaReg**) [4]
- Invariant Risk Minimization [[arXiv 2019](https://arxiv.53yu.com/pdf/1907.02893.pdf;)] [[Code](https://github.com/facebookresearch/InvariantRiskMinimization)] (**IRM**, **Colored MNIST dataset**) [165]
- Learning Robust Representations by Projecting Superficial Statistics Out [[ICLR 2019](https://arxiv.53yu.com/pdf/1903.06256)] [[Code](https://github.com/HaohanWang/HEX)] (**HEX**, **ImageNet-Sketch dataset**) [35]
- Distributionally Robust Neural Networks for Group Shifts On the Importance of Regularization for Worst-Case Generalization [[ICLR 2020](https://arxiv.org/pdf/1911.08731)] [[Code](https://github.com/kohpangwei/group_DRO)] (**DroupDRO**) [218]
- Self-challenging Improves Cross-Domain Generalization [[ECCV 2020](https://arxiv.53yu.com/pdf/2007.02454)] [[Code](https://github.com/DeLightCMU/RSC)] (**RSC**) [64]
- Energy-based Out-of-distribution Detection [[NeurIPS 2020](https://proceedings.neurips.cc/paper/2020/file/f5496252609c43eb8a3d147ab9b9c006-Paper.pdf)] [[Code](https://github.com/xieshuqin/Energy-OOD)] [181]
- When Can We Formulate the Out-of-Distribution Generalization Problem as an Invariance Problem? [[arXiv 2021](https://openreview.net/pdf?id=FzGiUKN4aBp)] [[Code*](https://github.com/facebookresearch/DomainBed)] (**IGA**) [219]
- Learning Representations that Support Robust Transfer of Predictors  [[arXiv 2021](https://arxiv.org/pdf/2110.09940)] [[Code](https://github.com/Newbeeer/TRM)] (**TRM**) [220]
- SAND-mask: An Enhanced Gradient Masking Strategy for the Discovery of Invariances in Domain Generalization [[arXiv 2021](https://arxiv.org/pdf/2106.02266)] [[Code*](https://github.com/facebookresearch/DomainBed)] (**SANDMask**)  [222]
- Out-of-Distribution Generalization via Risk Extrapolation [[ICML 2021](http://proceedings.mlr.press/v139/krueger21a/krueger21a.pdf)] (**VREx**) [190]
- Learning Explanations that are Hard to Vary [[ICLR 2021](https://arxiv.org/pdf/2009.00329)] [[Code*](https://github.com/facebookresearch/DomainBed)] (**ANDMask**) [221]
- A Fourier-Based Framework for Domain Generalization [[CVPR 2021](https://openaccess.thecvf.com/content/CVPR2021/papers/Xu_A_Fourier-Based_Framework_for_Domain_Generalization_CVPR_2021_paper.pdf)] [[Code](https://github.com/MediaBrain-SJTU/FACT)] (**FACT**) [160]
- Domain Generalization via Gradient Surgery [[ICCV 2021](https://openaccess.thecvf.com/content/ICCV2021/papers/Mansilla_Domain_Generalization_via_Gradient_Surgery_ICCV_2021_paper.pdf)] [[Code](https://github.com/lucasmansilla/DGvGS)] (**Agr**) [148]
- SelfReg: Self-Supervised Contrastive Regularization for Domain Generalization [[ICCV 2021](http://openaccess.thecvf.com/content/ICCV2021/papers/Kim_SelfReg_Self-Supervised_Contrastive_Regularization_for_Domain_Generalization_ICCV_2021_paper.pdf)] (**SelfReg**) [138]
- Embracing the Dark Knowledge: Domain Generalization Using Regularized Knowledge Distillation [[MM 2021](https://arxiv.53yu.com/pdf/2110.04820)]
- Model-Based Domain Generalization [[NeurIPS 2021](https://proceedings.neurips.cc/paper/2021/file/a8f12d9486cbcc2fe0cfc5352011ad35-Paper.pdf)] [[Code](https://github.com/arobey1/mbdg)] (**MBDG**) [200]
- Swad: Domain Generalization by Seeking Flat Minima [[NeurIPS 2021](https://proceedings.neurips.cc/paper/2021/file/bcb41ccdc4363c6848a1d760f26c28a0-Paper.pdf)] [[Code](https://github.com/khanrc/swad)] (**SWAD**) [201]
- Training for the Future: A Simple Gradient Interpolation Loss to Generalize Along Time [[NeurIPS 2021](https://proceedings.neurips.cc/paper/2021/file/a02ef8389f6d40f84b50504613117f88-Paper.pdf)] [[Code](https://github.com/anshuln/Training-for-the-Future)] (**GI**) [204]
- Adaptive Risk Minimization: Learning to Adapt to Domain Shift [[NeurIPS 2021](https://proceedings.neurips.cc/paper/2021/file/c705112d1ec18b97acac7e2d63973424-Paper.pdf)] [[Code](https://github.com/henrikmarklund/arm)] (**ARM**) [228]
- Gradient Starvation: A Learning Proclivity in Neural Networks [[NeurIPS 2021](https://proceedings.neurips.cc/paper/2021/file/0987b8b338d6c90bbedd8631bc499221-Paper.pdf)] [[Code*](https://github.com/facebookresearch/DomainBed)] (**SD**) [225]
- Quantifying and Improving Transferability in Domain Generalization [[NeurIPS 2021](https://proceedings.neurips.cc/paper/2021/file/5adaacd4531b78ff8b5cedfe3f4d5212-Paper.pdf)] [[Code](https://github.com/Gordon-Guojun-Zhang/Transferability-NeurIPS2021)] [206]
- Gradient Matching for Domain Generalization [[ICLR 2022](https://arxiv.org/pdf/2104.09937)] [[Code](https://github.com/YugeTen/fish)] (**Fish**) [224]
- Fishr: Invariant Gradient Variances for Our-of-distribution Generalization [[ICML 2022](https://arxiv.org/pdf/2109.02934)] [[Code](https://github.com/alexrame/fishr)] (**Fishr**) [173]
- Global-Local Regularization Via Distributional Robustness [[AISTATS 2023]](https://arxiv.org/abs/2203.00553) [[Code](https://github.com/VietHoang1512/GLOT)] (**GLOT**) [231]


### Normalization-Based Methods
> Normalization-based methods calibrate data from different domains by normalizing them with their statistic.

- Deep CORAL: Correlation Alignment for Deep Domain Adaptation [[ECCV 2016](https://arxiv.org/pdf/1607.01719)] [[Code](https://github.com/facebookresearch/DomainBed)] (**CORAL**) [229]
- Batch Normalization Embeddings for Deep Domain Generalization [[arXiv 2020](https://arxiv.53yu.com/pdf/2011.12672)] (**BNE**) [96]
- Learning to Optimize Domain Specific Normalization for Domain Generalization [[ECCV 2020](https://arxiv.53yu.com/pdf/1907.04275)] (**DSON**) [94]
- MetaNorm: Learning to Normalize Few-Shot Batches Across Domains [[ICLR 2021](https://openreview.net/pdf?id=9z_dNsC4B5t)] [[Code](https://github.com/YDU-AI/MetaNorm)] (**MetaNorm**) [19]
- Meta Batch-Instance Normalization for Generalizable Person Re-Identification [[CVPR 2021](https://openaccess.thecvf.com/content/CVPR2021/papers/Choi_Meta_Batch-Instance_Normalization_for_Generalizable_Person_Re-Identification_CVPR_2021_paper.pdf)] [[Code](https://github.com/bismex/MetaBIN)] (**MetaBIN**) [13]
- - Learning to Generalize Unseen Domains via Memory-based Multi-Source Meta-Learning for Person Re-Identification [[CVPR 2021](https://openaccess.thecvf.com/content/CVPR2021/papers/Zhao_Learning_to_Generalize_Unseen_Domains_via_Memory-based_Multi-Source_Meta-Learning_for_CVPR_2021_paper.pdf)] [[Code](https://github.com/HeliosZhao/M3L)] (**M3L**) [12]
- Adversarially Adaptive Normalization for Single Domain Generalization [[CVPR 2021](https://openaccess.thecvf.com/content/CVPR2021/papers/Fan_Adversarially_Adaptive_Normalization_for_Single_Domain_Generalization_CVPR_2021_paper.pdf)]  (**ASR**) [116]
- Collaborative Optimization and Aggregation for Decentralized Domain Generalization and Adaptation [[ICCV 2021](https://openaccess.thecvf.com/content/ICCV2021/papers/Wu_Collaborative_Optimization_and_Aggregation_for_Decentralized_Domain_Generalization_and_Adaptation_ICCV_2021_paper.pdf)] (**COPDA**) [159]
- Domain Generalization through Audio-Visual Relative Norm Alignment in First Person Action Recognition [[WACV 2022](https://openaccess.thecvf.com/content/WACV2022/papers/Planamente_Domain_Generalization_Through_Audio-Visual_Relative_Norm_Alignment_in_First_Person_WACV_2022_paper.pdf)] (**RNA-Net**) [186]

### Information-Based Methods
> Information-based methods utilize techniques of information theory to realize domain generalization.

- Learning to Learn with Variational Information Bottleneck for Domain Generalization [[ECCV 2020](https://arxiv.org/pdf/2007.07645)] (**MetaVIB**) [15]
- Progressive Domain Expansion Network for Single Domain Generalization [[CVPR 2021](http://openaccess.thecvf.com/content/CVPR2021/papers/Li_Progressive_Domain_Expansion_Network_for_Single_Domain_Generalization_CVPR_2021_paper.pdf)] [[Code](https://github.com/lileicv/PDEN)] (**PDEN**) [141]
- Learning To Diversify for Single Domain Generalization [[ICCV 2021](https://openaccess.thecvf.com/content/ICCV2021/papers/Wang_Learning_To_Diversify_for_Single_Domain_Generalization_ICCV_2021_paper.pdf)] [[Code](https://github.com/BUserName/Learning)] [158]
- Invariance Principle Meets Information Bottleneck for Out-Of-Distribution Generalization [[NeurIPS 2021](https://proceedings.neurips.cc/paper/2021/file/1c336b8080f82bcc2cd2499b4c57261d-Paper.pdf)] [[Code](https://github.com/ahujak/IB-IRM)] (**IB-IRM**) [207]
- Exploiting Domain-Specific Features to Enhance Domain Generalization [[NeurIPS 2021](https://proceedings.neurips.cc/paper/2021/file/b0f2ad44d26e1a6f244201fe0fd864d1-Paper.pdf)] [[Code](https://github.com/manhhabui/mDSDI)] (**mDSDI**) [202]
- Invariant Information Bottleneck for Domain Generalization [[AAAI 2022](https://arxiv.org/pdf/2106.06333)] [[Code](https://github.com/Luodian/IIB/tree/IIB)] (**IIB**) [140]

### Causality-Based Methods
> Causality-based methods analyze and address the domain generalization problem from a causal perspective.

- Invariant Risk Minimization [[arXiv 2019](https://arxiv.53yu.com/pdf/1907.02893.pdf;)] [[Code](https://github.com/facebookresearch/InvariantRiskMinimization)] (**IRM**, **Colored MNIST dataset**) [165]
- Learning Domain-Invariant Relationship with Instrumental Variable for Domain Generalization [[arXiv 2021](https://arxiv.org/pdf/2110.01438)] (**IV-DG**) [163]
- A Causal Framework for Distribution Generalization [[TPAMI 2021](https://arxiv.org/pdf/2006.07433)] [[Code](https://runesen.github.io/NILE/)] (**NILE**) [191]
- Domain Generalization using Causal Matching [[ICML 2021](http://proceedings.mlr.press/v139/mahajan21b/mahajan21b.pdf)] [[Code](https://github.com/microsoft/robustdg)] (**MatchDG**) [73]
- Deep Stable Learning for Out-of-Distribution Generalization [[CVPR 2021](http://openaccess.thecvf.com/content/CVPR2021/papers/Zhang_Deep_Stable_Learning_for_Out-of-Distribution_Generalization_CVPR_2021_paper.pdf)] [[Code](https://github.com/xxgege/StableNet)] (**StableNet**) [117]
- Out-of-Distribution Generalization via Risk Extrapolation [[ICML 2021](http://proceedings.mlr.press/v139/krueger21a/krueger21a.pdf)] [[Code](https://github.com/facebookresearch/DomainBed)] (**VREx**) [217]
- A Style and Semantic Memory Mechanism for Domain Generalization [[ICCV 2021](http://openaccess.thecvf.com/content/ICCV2021/papers/Chen_A_Style_and_Semantic_Memory_Mechanism_for_Domain_Generalization_ICCV_2021_paper.pdf)] (**STEAM**) [130]
- Learning Causal Semantic Representation for Out-of-Distribution Prediction [[NeurIPS 2021](https://proceedings.neurips.cc/paper/2021/file/310614fca8fb8e5491295336298c340f-Paper.pdf)] [[Code](https://github.com/changliu00/causal-semantic-generative-model)] (**CSG-ind**) [145]
- Recovering Latent Causal Factor for Generalization to Distributional Shifts [[NeurIPS 2021](https://proceedings.neurips.cc/paper/2021/file/8c6744c9d42ec2cb9e8885b54ff744d0-Paper.pdf)] [[Code](https://github.com/wubotong/LaCIM)] (**LaCIM**) [152]
- On Calibration and Out-of-domain Generalization [[NeurIPS 2021](https://proceedings.neurips.cc/paper/2021/file/118bd558033a1016fcc82560c65cca5f-Paper.pdf)]
- Invariance Principle Meets Information Bottleneck for Out-Of-Distribution Generalization [[NeurIPS 2021](https://proceedings.neurips.cc/paper/2021/file/1c336b8080f82bcc2cd2499b4c57261d-Paper.pdf)] [[Code](https://github.com/ahujak/IB-IRM)] (**IB-ERM**, **IB-IRM**) [207]
- Domain Generalization via Contrastive Causal Learning [[arXiv 2022](https://arxiv.org/abs/2210.02655)] (**CCM**) [212]
- Invariant Causal Mechanisms through Distribution Matching [[arXiv 2022](https://arxiv.org/pdf/2206.11646)] [[Code*](https://github.com/facebookresearch/DomainBed)] (**CausIRL-CORAL**, **CausIRL-MMD**) [216]
- Invariant Information Bottleneck for Domain Generalization [[AAAI 2022](https://arxiv.org/pdf/2106.06333)] [[Code](https://github.com/Luodian/IIB/tree/IIB)] (**IIB**) [140]

### Inference-Time-Based Methods
> Inference-time-based methods leverage the unlabeled target data, which is available at inference-time, to improve generalization performance without further model training.

- Generalization on Unseen Domains via Inference-Time Label-Preserving Target Projections [[CVPR 2021](http://openaccess.thecvf.com/content/CVPR2021/papers/Pandey_Generalization_on_Unseen_Domains_via_Inference-Time_Label-Preserving_Target_Projections_CVPR_2021_paper.pdf)] [[Code](https://github.com/yys-Polaris/InferenceTimeDG)] [118]
- Adaptive Methods for Real-World Domain Generalization [[CVPR 2021](http://openaccess.thecvf.com/content/CVPR2021/papers/Dubey_Adaptive_Methods_for_Real-World_Domain_Generalization_CVPR_2021_paper.pdf)] [[Code](https://github.com/abhimanyudubey/GeoYFCC)] (**DA-ERM**) [132]
- Test-Time Classifier Adjustment Module for Model-Agnostic Domain Generalization [[NeurIPS 2021](https://proceedings.neurips.cc/paper/2021/file/1415fe9fea0fa1e45dddcff5682239a0-Paper.pdf)] [[Code](https://github.com/matsuolab/T3A)] (**T3A**) [136]

### Neural Architecture Search-based Methods
> Neural architecture search-based methods aim to dynamically tune the network architecture to improve out-of-domain generalization.

- NAS-OoD Neural Architecture Search for Out-of-Distribution Generalization [[ICCV 2021](https://openaccess.thecvf.com/content/ICCV2021/papers/Bai_NAS-OoD_Neural_Architecture_Search_for_Out-of-Distribution_Generalization_ICCV_2021_paper.pdf)] (**NAS-OoD**) [129]

## Single Domain Generalization
> The goal of single domain generalization task is to improve model performance on unknown target domains by using data from only one source domain.

- Learning to Learn Single Domain Generalization [[CVPR 2020](https://openaccess.thecvf.com/content_CVPR_2020/papers/Qiao_Learning_to_Learn_Single_Domain_Generalization_CVPR_2020_paper.pdf)] [[Code](https://github.com/joffery/M-ADA)] (**M-ADA**) [27]
- Out-of-domain Generalization from a Single Source: A Uncertainty Quantification Approach [[arXiv 2021](https://arxiv.53yu.com/pdf/2108.02888)] [151]
- Uncertainty-guided Model Generalization to Unseen Domains [[CVPR 2021](http://openaccess.thecvf.com/content/CVPR2021/papers/Qiao_Uncertainty-Guided_Model_Generalization_to_Unseen_Domains_CVPR_2021_paper.pdf)] [[Code](https://github.com/joffery/UMGUD)] [168]
- Adversarially Adaptive Normalization for Single Domain Generalization [[CVPR 2021](https://openaccess.thecvf.com/content/CVPR2021/papers/Fan_Adversarially_Adaptive_Normalization_for_Single_Domain_Generalization_CVPR_2021_paper.pdf)]  (**ASR**) [116]
- Progressive Domain Expansion Network for Single Domain Generalization [[CVPR 2021](http://openaccess.thecvf.com/content/CVPR2021/papers/Li_Progressive_Domain_Expansion_Network_for_Single_Domain_Generalization_CVPR_2021_paper.pdf)] [[Code](https://github.com/lileicv/PDEN)] (**PDEN**) [141]
- Learning To Diversify for Single Domain Generalization [[ICCV 2021](https://openaccess.thecvf.com/content/ICCV2021/papers/Wang_Learning_To_Diversify_for_Single_Domain_Generalization_ICCV_2021_paper.pdf)] [[Code](https://github.com/BUserName/Learning)] [158]
- Intra-Source Style Augmentation for Improved Domain Generalization [[WACV 2023](https://arxiv.org/pdf/2210.10175.pdf)] (**ISSA**) [215]

## Semi/Weak/Un-Supervised Domain Generalization
> Semi/weak-supervised domain generalization assumes that a part of the source data is unlabeled, while unsupervised domain generalization assumes no training supervision.

- Visual recognition by learning from web data: A weakly supervised domain generalization approach [[CVPR 2015](https://openaccess.thecvf.com/content_cvpr_2015/papers/Niu_Visual_Recognition_by_2015_CVPR_paper.pdf)] [89]
- Visual Recognition by Learning From Web Data via Weakly Supervised Domain Generalization [[TNNLS 2017](https://bcmi.sjtu.edu.cn/home/niuli/paper/Visual%20Recognition%20by%20Learning%20From%20Web%20Data%20via%20Weakly%20Supervised%20Domain%20Generalization.pdf)] [121]
- Domain Generalization via Semi-supervised Meta Learning [[arXiv 2020](https://arxiv.org/pdf/2009.12658)] [[Code](https://github.com/hosseinshn/DGSML)] (**DGSML**) [127]
- Deep Semi-supervised Domain Generalization Network for Rotary Machinery Fault Diagnosis under Variable Speed [[IEEE Transactions on Instrumentation and Measurement 2020](https://www.researchgate.net/profile/Yixiao-Liao/publication/341199775_Deep_Semisupervised_Domain_Generalization_Network_for_Rotary_Machinery_Fault_Diagnosis_Under_Variable_Speed/links/613f088201846e45ef450a0a/Deep-Semisupervised-Domain-Generalization-Network-for-Rotary-Machinery-Fault-Diagnosis-Under-Variable-Speed.pdf)] (**DSDGN**) [125]
- Semi-Supervised Domain Generalization with Stochastic StyleMatch [[arXiv 2021](https://arxiv.53yu.com/pdf/2106.00592)] [[Code](https://github.com/KaiyangZhou/ssdg-benchmark)] (**StyleMatch**) [54]
- Better Pseudo-label Joint Domain-aware Label and Dual-classifier for Semi-supervised Domain Generalization [[arXiv 2021](https://arxiv.53yu.com/pdf/2110.04820)] [156]
- Semi-Supervised Domain Generalization in RealWorld: New Benchmark and Strong Baseline [[arXiv 2021](https://arxiv.org/pdf/2111.10221)] [179]
- On Challenges in Unsupervised Domain Generalization [[NeurIPS workshop 2021](https://proceedings.mlr.press/v181/narayanan22a/narayanan22a.pdf)] [178]
- Domain-Specific Bias Filtering for Single Labeled Domain Generalization [[IJCV 2022](https://arxiv.org/pdf/2110.00726)] [[Code](https://github.com/junkunyuan/DSBF)] (**DSBF**) [162]
- Towards Unsupervised Domain Generalization [[CVPR 2022](https://openaccess.thecvf.com/content/CVPR2022/papers/Zhang_Towards_Unsupervised_Domain_Generalization_CVPR_2022_paper.pdf)] (**DARLING**) [69]
- Unsupervised Domain Generalization by Learning a Bridge Across Domains [[CVPR 2022](https://openaccess.thecvf.com/content/CVPR2022/papers/Harary_Unsupervised_Domain_Generalization_by_Learning_a_Bridge_Across_Domains_CVPR_2022_paper.pdf)] [[Code](https://github.com/leokarlin/BrAD)] (**BrAD**) [182]
- Label-Efficient Domain Generalization via Collaborative Exploration and Generalization [[MM 2022](https://arxiv.org/abs/2208.03644)] [[Code](https://github.com/junkunyuan/CEG)] (**CEG**) [211]

## Open/Heterogeneous Domain Generalization
> Open/heterogeneous domain generalization assumes the label space of one domain is different from that of another domain.

- Feature-Critic Networks for Heterogeneous Domain Generalisation [[ICML 2019](http://proceedings.mlr.press/v97/li19l/li19l.pdf)] [[Code](https://github.com/liyiying/Feature_Critic)] (**Feature-Critic**) [5]
- Episodic Training for Domain Generalization [[ICCV 2019](https://openaccess.thecvf.com/content_ICCV_2019/papers/Li_Episodic_Training_for_Domain_Generalization_ICCV_2019_paper.pdf)] [[Code](https://github.com/HAHA-DL/Episodic-DG)] (**Epi-FCR**) [7]
- Towards Recognizing Unseen Categories in Unseen Domains [[ECCV 2020](https://arxiv.53yu.com/pdf/2007.12256.pdf?ref=https://githubhelp.com)] [[Code](https://github.com/mancinimassimiliano/CuMix)] (**CuMix**) [57]
- Heterogeneous Domain Generalization via Domain Mixup [[ICASSP 2020](https://arxiv.org/pdf/2009.05448)] [[Code](https://github.com/wyf0912/MIXALL)] [128]
- Open Domain Generalization with Domain-Augmented Meta-Learning [[CVPR 2021](http://openaccess.thecvf.com/content/CVPR2021/papers/Shu_Open_Domain_Generalization_with_Domain-Augmented_Meta-Learning_CVPR_2021_paper.pdf)] [[Code](https://github.com/thuml/OpenDG-DAML)] (**DAML**) [119]
- Universal Cross-Domain Retrieval Generalizing Across Classes and Domains [[ICCV 2021](https://openaccess.thecvf.com/content/ICCV2021/papers/Paul_Universal_Cross-Domain_Retrieval_Generalizing_Across_Classes_and_Domains_ICCV_2021_paper.pdf)] [[Code](https://github.com/mvp18/UCDR)] (**SnMpNet**) [150]


## Federated Domain Generalization
> Federated domain generalization assumes that source data is distributed and can not be fused for data privacy protection.

- Collaborative Semantic Aggregation and Calibration for Separated Domain Generalization [[arXiv 2021](https://arxiv.org/pdf/2110.06736)] [[Code](https://github.com/junkunyuan/CSAC)] (**CSAC**) [161]
- FedDG: Federated Domain Generalization on Medical Image Segmentation via Episodic Learning in Continuous Frequency Space [[CVPR 2021](http://openaccess.thecvf.com/content/CVPR2021/papers/Liu_FedDG_Federated_Domain_Generalization_on_Medical_Image_Segmentation_via_Episodic_CVPR_2021_paper.pdf)] [[Code](https://github.com/liuquande/FedDG-ELCFS)] (**FedDG**) [147]
- Collaborative Optimization and Aggregation for Decentralized Domain Generalization and Adaptation [[ICCV 2021](https://openaccess.thecvf.com/content/ICCV2021/papers/Wu_Collaborative_Optimization_and_Aggregation_for_Decentralized_Domain_Generalization_and_Adaptation_ICCV_2021_paper.pdf)] (**COPDA**) [159]

## Source-free Domain Generalization
> Source-free domain generalization aims to improve model's generalization capability to arbitrary unseen domains without exploiting any source domain data.

- PromptStyler: Prompt-driven Style Generation for Source-free Domain Generalization [[ICCV 2023](https://arxiv.org/abs/2307.15199)] [[Project Page](https://PromptStyler.github.io/)] (**PromptStyler**) [231]

## Applications
### Person Re-Identification
- Deep Domain-Adversarial Image Generation for Domain Generalisation [[AAAI 2020](https://ojs.aaai.org/index.php/AAAI/article/download/7003/6857)] [[Code](https://github.com/KaiyangZhou/Dassl.pytorch)]
- Learning to Generate Novel Domains for Domain Generalization [[ECCV 2020](https://arxiv.org/pdf/2007.03304)] [[Code](https://github.com/mousecpn/L2A-OT)] (**L2A-OT**, **Digits-DG dataset**) [28]
- Learning Generalisable Omni-Scale Representations for Person Re-Identification [[TPAMI 2021](https://arxiv.org/pdf/1910.06827)] [[Code](https://github.com/KaiyangZhou/deep-person-reid)] [114]
- Multi-Domain Adversarial Feature Generalization for Person Re-Identification [[TIP 2021](https://ieeexplore.ieee.org/iel7/83/9263394/09311771.pdf)] (**MMFA-AAE**) [144]
- Domain Generalization with Mixstyle [[ICLR 2021](https://arxiv.53yu.com/pdf/2104.02008)] [[Code](https://github.com/KaiyangZhou/mixstyle-release)] (**MixStyle**) [56]
- Learning to Generalize Unseen Domains via Memory-based Multi-Source Meta-Learning for Person Re-Identification [[CVPR 2021](https://openaccess.thecvf.com/content/CVPR2021/papers/Zhao_Learning_to_Generalize_Unseen_Domains_via_Memory-based_Multi-Source_Meta-Learning_for_CVPR_2021_paper.pdf)] [[Code](https://github.com/HeliosZhao/M3L)] (**M3L**) [12]
- Meta Batch-Instance Normalization for Generalizable Person Re-Identification [[CVPR 2021](https://openaccess.thecvf.com/content/CVPR2021/papers/Choi_Meta_Batch-Instance_Normalization_for_Generalizable_Person_Re-Identification_CVPR_2021_paper.pdf)] [[Code](https://github.com/bismex/MetaBIN)] (**MetaBIN**) [13]
- Generalizable Person Re-identification with Relevance-aware Mixture of Experts [[CVPR 2021](https://openaccess.thecvf.com/content/CVPR2021/papers/Dai_Generalizable_Person_Re-Identification_With_Relevance-Aware_Mixture_of_Experts_CVPR_2021_paper.pdf)] (**RaMoE**) [187]
- TransMatcher: Deep Image Matching Through Transformers for Generalizable Person Re-identification [[NeurIPS 2021](https://proceedings.neurips.cc/paper/2021/file/0f49c89d1e7298bb9930789c8ed59d48-Paper.pdf)] [[Code](https://github.com/ShengcaiLiao/QAConv)] (**TransMatcher**) [208]

### Face Recognition & Anti-Spoofing
- Multi-Adversarial Discriminative Deep Domain Generalization for Face Presentation Attack Detection [[CVPR 2019](http://openaccess.thecvf.com/content_CVPR_2019/papers/Shao_Multi-Adversarial_Discriminative_Deep_Domain_Generalization_for_Face_Presentation_Attack_Detection_CVPR_2019_paper.pdf)] [[Code](https://github.com/rshaojimmy/CVPR2019-MADDoG)] (**MADDG**) [78]
- Towards Universal Representation Learning for Deep Face Recognition [[CVPR 2020](http://openaccess.thecvf.com/content_CVPR_2020/papers/Shi_Towards_Universal_Representation_Learning_for_Deep_Face_Recognition_CVPR_2020_paper.pdf)] [[Code](https://github.com/MatyushinMA/uni_rep_deep_faces)] [22]
- Cross-Domain Face Presentation Attack Detection via Multi-Domain Disentangled Representation Learning [[CVPR 2020](http://openaccess.thecvf.com/content_CVPR_2020/papers/Wang_Cross-Domain_Face_Presentation_Attack_Detection_via_Multi-Domain_Disentangled_Representation_Learning_CVPR_2020_paper.pdf)] [106]
- Single-Side Domain Generalization for Face Anti-Spoofing [[CVPR 2020](http://openaccess.thecvf.com/content_CVPR_2020/papers/Jia_Single-Side_Domain_Generalization_for_Face_Anti-Spoofing_CVPR_2020_paper.pdf)] [[Code](https://github.com/taylover-pei/SSDG-CVPR2020)] (**SSDG**) [79]

## Related Topics
### Life-Long Learning
- Sequential Learning for Domain Generalization [[ECCV workshop 2020](https://arxiv.org/pdf/2004.01377)] (**S-MLDG**) [14]
- Continual Adaptation of Visual Representations via Domain Randomization and Meta-learning [[CVPR 2021](http://openaccess.thecvf.com/content/CVPR2021/papers/Volpi_Continual_Adaptation_of_Visual_Representations_via_Domain_Randomization_and_Meta-Learning_CVPR_2021_paper.pdf)] (**Meta-DR**) [153]

# Publications

| Top Conference  |  Papers  |
|  ----  | ----  |
|  before 2014  |  **CVPR:** [8], [11]; **ICCV:** [16], [41]; **NeurIPS:** [31], [113]; **ECCV:** [32], [87], [103]; **ICML:** [65]  |
|  2015  |  **CVPR:** [89]; **ICML:** [30]; **ICCV:** [6], [46], [88]  |
|  2016  |  **CVPR:** [42], [44], [120]; IJCAI: [66]; **ECCV:** [43], [47], [229]  |
|  2017  |  **CVPR:** [20]; **ICCV:** [2], [71]; **NeurIPS:** [38]  |
|  2018  |  **ICLR:** [1], [68]; **ICLR:** [53]; **CVPR:** [76]; **ECCV:** [45], [77]; **NeurIPS:** [4], [25]  |
|  2019  |  **ICLR:** [35], [37]; **CVPR:** [78], [98]; **ICML:** [5], [107], [110]; **ICCV:** [7], [21], [33], [62], [63]; **NeurIPS:** [18]  |
|  2020  |  **ICLR:** [55], [83], [218]; **ICLR:** [126]; **CVPR:** [22], [27], [79], [106]; **ICML:** [105]; **ECCV:** [14], [15], [28], [57], [64], [94], [99], [104]; **NeurIPS:** [75], [86], [112], [181]  |
|  2021  |  **ICLR:** [19], [56], [59], [134], [175], [196]; **ICLR:** [139], [171], [221]; **CVPR:** [12], [13], [115], [116], [117], [118], [119], [132], [141], [147], [153], [160], [168], [187], [193]; IJCAI: [155], [195], [230]; **ICML:** [73], [190], [217]; **ICCV:** [129], [130], [133], [135], [138], [142], [143], [148], [149], [150], [158], [159], [194]; **MM:**  [131], [137], [146], [157]; **NeurIPS:** [136], [145], [152], [154], [198], [199], [200], [201], [202], [203], [204], [205], [206], [207], [208], [228], [225]  |
|  2022  |  **AAAI:** [140]; **ICLR:** [213], [224]; **CVPR:** [69], [182], [214]; **ICML**: [173]; **MM:** [211]  |
|  2023  |  **WACV:** [215]; **ICLR:** [223]; **ICCV:** [231]  |

| Top Journal  |  Papers  |
|  ----  | ----  |
|  before 2017  | **IJCV:** [9], [10]; **JMLR:** [226]  |
|  2017  |  **TPAMI:** [67]; **TIP:** [91]  |
|  2021|  **TIP:** [34], [144]; **TPAMI:**  [101], [114], [191], [197]; **JMLR:** [188]  |
|  2022 | **TMLR:** [209], [210]; **IJCV:** [162] |

| arXiv  |  Papers  |
|  ----  | ----  |
|  before 2014  |  [40]  |
|  2017  |  [36], [52]  |
|  2018  |  [166]  |
|  2019  |  [81], [123], [165], [169]  |
|  2020  |  [60], [82], [96], [102], [127], [189], [227]  |
|  2021  |  [3], [54], [58], [151], [156], [161], [163], [170], [174], [176], [178], [179], [184], [192], [219], [222]  |
|  2022  |  [183], [212], [216], [220]  |

|  Else  |  Papers  |
|  ----  |  ----  |
|  before 2018  |  [29], [39], [48], [49], [50], [51], [90], [92], [97], [109], [111], [121], [122]  |
|  2019  |  [26], [72], [84], [167]  |
|  2020  |  [17], [23], [24], [61], [70], [74], [80], [85], [93], [95], [100], [124], [125], [128], [164]  |
|  2021  |  [108], [172], [177], [180], [185]  |
|  2022  |  [186]  |

# Datasets
> Evaluations on the following datasets often follow leave-one-domain-out protocol: randomly choose one domain to hold out as the target domain, while the others are used as the  source domain(s).
>
| Datasets (download link) | Description | Related papers |
| :---- | :----: | :----: |
| **Colored MNIST** [[165]](https://arxiv.53yu.com/pdf/1907.02893.pdf) | Handwritten digit recognition; 3 domains: {0.1, 0.3, 0.9}; 70,000 samples of dimension (2, 28, 28); 2 classes | [82], [138], [140], [149], [152], [154], [165], [171], [173], [190], [200], [202], [214], [216], [217], [219], [220], [222], [224] |
| **Rotated MNIST** [[6]](http://openaccess.thecvf.com/content_iccv_2015/papers/Ghifary_Domain_Generalization_for_ICCV_2015_paper.pdf) ([original](https://github.com/Emma0118/mate)) | Handwritten digit recognition; 6 domains with rotated degree: {0, 15, 30, 45, 60, 75}; 7,000 samples of dimension (1, 28, 28); 10 classes | [5], [6], [15], [35], [53], [55], [63], [71], [73], [74], [76], [77], [86], [90], [105], [107], [138], [140], [170], [173], [202], [204], [206], [216], [222], [224] |
| **Digits-DG** [[28]](https://arxiv.org/pdf/2007.03304) | Handwritten digit recognition; 4 domains: {MNIST [[29]](http://lushuangning.oss-cn-beijing.aliyuncs.com/CNN%E5%AD%A6%E4%B9%A0%E7%B3%BB%E5%88%97/Gradient-Based_Learning_Applied_to_Document_Recognition.pdf), MNIST-M [[30](http://proceedings.mlr.press/v37/ganin15.pdf)], SVHN [[31](https://research.google/pubs/pub37648.pdf)], SYN [[30](http://proceedings.mlr.press/v37/ganin15.pdf)]}; 24,000 samples; 10 classes | [21], [25], [27], [28], [34], [35], [55], [59], [63], [94], [98], [116], [118], [130], [141], [142], [146], [151], [153], [157], [158], [159], [160], [166], [168], [179], [189], [203], [209], [210] |
| **VLCS** [[16]](http://openaccess.thecvf.com/content_iccv_2013/papers/Fang_Unbiased_Metric_Learning_2013_ICCV_paper.pdf) ([1](https://drive.google.com/uc?id=1skwblH1_okBwxWxmRsp9_qi15hyPpxg8); or [original](https://www.mediafire.com/file/7yv132lgn1v267r/vlcs.tar.gz/file)) | Object recognition; 4 domains: {Caltech [[8]](http://www.vision.caltech.edu/publications/Fei-FeiCompVIsImageU2007.pdf), LabelMe [[9]](https://idp.springer.com/authorize/casa?redirect_uri=https://link.springer.com/content/pdf/10.1007/s11263-007-0090-8.pdf&casa_token=n3w4Sen-huAAAAAA:sJY2dHreDGe2V4KE9jDehftM1W-Sn1z8bqeF_WK8Q9t4B0dFk5OXEAlIP7VYnr8UfiWLAOPG7dK0ZveYWs8), PASCAL [[10]](https://idp.springer.com/authorize/casa?redirect_uri=https://link.springer.com/content/pdf/10.1007/s11263-009-0275-4.pdf&casa_token=Zb6LfMuhy_sAAAAA:Sqk_aoTWdXx37FQjUFaZN9ZMQxrUhqO2S_HbOO2a9BKtejW7CMekg-3PDVw6Yjw7BZqihyjP0D_Y6H2msBo), SUN [[11]](https://dspace.mit.edu/bitstream/handle/1721.1/60690/Oliva_SUN%20database.pdf?sequence=1&isAllowed=y)}; 10,729 samples of dimension (3, 224, 224); 5 classes; about 3.6 GB | [2], [6], [7], [14], [15], [18], [60], [61], [64], [67], [68], [70], [71], [74], [76], [77], [81], [83], [86], [91], [98], [99], [101], [102], [103], [117], [118], [126], [127], [131], [132], [136], [138], [140], [142], [145], [146], [148], [149], [161], [170], [173], [174], [184], [190], [195], [199], [201], [202], [203], [209], [216], [217], [222], [223], [224], [231] |
| **Office31+Caltech** [[32]](https://linkspringer.53yu.com/content/pdf/10.1007/978-3-642-15561-1_16.pdf) ([1](https://drive.google.com/file/d/14OIlzWFmi5455AjeBZLak2Ku-cFUrfEo/view)) | Object recognition; 4 domains: {Amazon, Webcam, DSLR, Caltech}; 4,652 samples in 31 classes (office31) or 2,533 samples in 10 classes (office31+caltech); 51 MB | [6], [35], [67], [68], [70], [71], [80], [91], [96], [119], [131], [167] |
| **OfficeHome** [[20]](http://openaccess.thecvf.com/content_cvpr_2017/papers/Venkateswara_Deep_Hashing_Network_CVPR_2017_paper.pdf) ([1](https://drive.google.com/uc?id=1uY0pj7oFsjMxRwaD3Sxy0jgel0fsYXLC); or [original](https://drive.google.com/file/d/0B81rNlvomiwed0V1YUxQdC1uOTg/view?resourcekey=0-2SNWq0CDAuWOBRRBL7ZZsw)) | Object recognition; 4 domains: {Art, Clipart, Product, Real World}; 15,588 samples of dimension (3, 224, 224); 65 classes; 1.1 GB | [19], [54], [28], [34], [55], [58], [60], [61], [64], [80], [92], [94], [98], [101], [118], [126], [130], [131], [132], [133], [137], [138], [140], [146], [148], [156], [159], [160], [162], [163], [167], [173], [174], [178], [179], [182], [184], [189], [190], [199], [201], [202], [203], [206], [211], [212], [214], [216], [217], [220], [222], [223], [224], [230], [231] |
| **PACS** [[2]](https://openaccess.thecvf.com/content_ICCV_2017/papers/Li_Deeper_Broader_and_ICCV_2017_paper.pdf) ([1](https://drive.google.com/uc?id=1JFr8f805nMUelQWWmfnJR3y4_SYoN5Pd); or [original](https://drive.google.com/drive/folders/0B6x7gtvErXgfUU1WcGY5SzdwZVk?resourcekey=0-2fvpQY_QSyJf2uIECzqPuQ)) | Object recognition; 4 domains: {photo, art_painting, cartoon, sketch}; 9,991 samples of dimension (3, 224, 224); 7 classes; 174 MB | [1], [2], [4], [5], [14], [15], [18], [19], [34], [54], [28], [35], [55], [56], [57], [58], [59], [60], [61], [64], [69], [73], [77], [80], [81], [82], [83], [84], [86], [90], [92], [94], [96], [98], [99], [101], [102], [104], [105], [116], [117], [118], [127], [129], [130], [131], [132], [136], [137], [138], [139], [140], [142], [145], [146], [148], [149], [153], [156], [157], [158], [159], [160], [161], [162], [163], [167], [170], [171], [173], [174], [178], [179], [180], [182], [184], [189], [190], [195], [199], [200], [201], [202], [203], [206], [209], [210], [211], [212], [214], [216], [217], [220], [222], [223], [224], [230], [231] |
| **DomainNet** [[33](https://openaccess.thecvf.com/content_ICCV_2019/papers/Peng_Moment_Matching_for_Multi-Source_Domain_Adaptation_ICCV_2019_paper.pdf)] ([clipart](http://csr.bu.edu/ftp/visda/2019/multi-source/groundtruth/clipart.zip), [infograph](http://csr.bu.edu/ftp/visda/2019/multi-source/infograph.zip), [painting](http://csr.bu.edu/ftp/visda/2019/multi-source/groundtruth/painting.zip), [quick-draw](http://csr.bu.edu/ftp/visda/2019/multi-source/quickdraw.zip), [real](http://csr.bu.edu/ftp/visda/2019/multi-source/real.zip), and [sketch](http://csr.bu.edu/ftp/visda/2019/multi-source/sketch.zip); or [original](http://ai.bu.edu/M3SDA/)) | Object recognition; 6 domains: {clipart, infograph, painting, quick-draw, real, sketch}; 586,575 samples of dimension (3, 224, 224); 345 classes; 1.2 GB + 4.0 GB + 3.4 GB + 439 MB + 5.6 GB + 2.5 GB | [34], [57], [69], [104], [119], [130], [131], [132], [133], [138], [140], [150], [173], [182], [189], [201], [202], [203], [216], [222], [223], [224], [230], [231] |
| **mini-DomainNet** [[34]](https://arxiv.53yu.com/pdf/2003.07325) | Object recognition; a smaller and less noisy version of DomainNet; 4 domains: {clipart, painting, real, sketch}; 140,006 samples | [34], [130], [156], [157], [210] |
**ImageNet-Sketch** [[35]](https://arxiv.53yu.com/pdf/1903.06256) | Object recognition; 2 domains: {real, sketch}; 50,000 samples | [64] |
**VisDA-17** [[36](https://arxiv.53yu.com/pdf/1710.06924)] | Object recognition; 3 domains of synthetic-to-real generalization; 280,157 samples | [119], [182] |
**CIFAR-10-C** / **CIFAR-100-C** / **ImageNet-C** [[37]](https://arxiv.53yu.com/pdf/1903.12261.pdf?ref=https://githubhelp.com) ([original](https://github.com/hendrycks/robustness/)) | Object recognition; the test data are damaged by 15 corruptions (each with 5 intensity levels) drawn from 4 categories (noise, blur, weather, and digital); 60,000/60,000/1.3M samples | [27], [69], [74], [116], [141], [151], [168] |
| **Visual Decathlon (VD)** [[38]](https://proceedings.neurips.cc/paper/2017/file/e7b24b112a44fdd9ee93bdf998c6ca0e-Paper.pdf) | Object/action/handwritten/digit recognition; 10 domains from the combination of 10 datasets; 1,659,142 samples | [5], [7], [128] |
**IXMAS** [[39]](https://hal.inria.fr/docs/00/54/46/29/PDF/cviu_motion_history_volumes.pdf) | Action recognition; 5 domains with 5 camera views, 10 subjects, and 5 actions; 1,650 samples | [7], [14], [67], [76] |
**SYNTHIA** [[42]](https://www.cv-foundation.org/openaccess/content_cvpr_2016/papers/Ros_The_SYNTHIA_Dataset_CVPR_2016_paper.pdf) | Semantic segmentation; 15 domains with 4 locations and 5 weather conditions; 2,700 samples | [27], [62], [115], [141], [151], [185], [193]  |
**GTA5-Cityscapes** [[43]](https://linkspringer.53yu.com/chapter/10.1007/978-3-319-46475-6_7), [[44]](http://openaccess.thecvf.com/content_cvpr_2016/papers/Cordts_The_Cityscapes_Dataset_CVPR_2016_paper.pdf) | Semantic segmentation; 2 domains of synthetic-to-real generalization; 29,966 samples | [62], [115], [185], [193]  |
**Cityscapes-ACDC** [[44]](http://openaccess.thecvf.com/content_cvpr_2016/papers/Cordts_The_Cityscapes_Dataset_CVPR_2016_paper.pdf) ([original](https://acdc.vision.ee.ethz.ch/overview))  | Semantic segmentation; real life domain shifts, ACDC contains four different weather conditions: rain, fog, snow, night | [215]  |
**Terra Incognita (TerraInc)** [[45]](http://openaccess.thecvf.com/content_ECCV_2018/papers/Beery_Recognition_in_Terra_ECCV_2018_paper.pdf) ([1](https://lilablobssc.blob.core.windows.net/caltechcameratraps/eccv_18_all_images_sm.tar.gz) and [2](https://lilablobssc.blob.core.windows.net/caltechcameratraps/labels/caltech_camera_traps.json.zip); or [original](https://lila.science/datasets/caltech-camera-traps)) | Animal classification; 4 domains captured at different geographical locations: {L100, L38, L43, L46}; 24,788 samples of dimension (3, 224, 224); 10 classes; 6.0 GB + 8.6 MB | [132], [136], [138], [140], [173], [201], [202], [207], [212], [214], [216], [222], [223], [224], [231]  |
**Market-Duke** [[46]](https://www.cv-foundation.org/openaccess/content_iccv_2015/papers/Zheng_Scalable_Person_Re-Identification_ICCV_2015_paper.pdf), [[47]](https://linkspringer.53yu.com/chapter/10.1007/978-3-319-48881-3_2) | Person re-idetification; cross-dataset re-ID; heterogeneous DG with 2 domains; 69,079 samples | [12], [13], [28], [55], [56], [58], [114], [144], [187], [208]  |
<!-- **UCF-HMDB** [[40](https://arxiv.53yu.com/pdf/1212.0402.pdf?ref=https://githubhelp.com)], [[41](https://dspace.mit.edu/bitstream/handle/1721.1/69981/Poggio-HMDB.pdf?sequence=1&isAllowed=y)] | Action recognition | 2 domains with 12 overlapping actions; 3809 samples |  | -->
<!-- **Face** [22] | >5M | 9 | Face recognition | Combination of 9 face datasets |  |
**COMI** [[48](http://www.cbsr.ia.ac.cn/users/jjyan/zhang-icb2012.pdf)], [49], [50], [[51](https://dl.gi.de/bitstream/handle/20.500.12116/18295/183.pdf?sequence=1)] | 8500 | 4 | Face anti-spoofing | Combination of 4 face anti-spoofing datasets |  | -->

# Libraries
> We list the GitHub libraries of domain generalization (sorted by stars).

- [DeepDG (jindongwang)](https://github.com/jindongwang/transferlearning/tree/master/code/DeepDG): Deep Domain Generalization Toolkit.
- [Transfer Learning Library (thuml)](https://github.com/thuml/Transfer-Learning-Library) for Domain Adaptation, Task Adaptation, and Domain Generalization.
- [DomainBed (facebookresearch)](https://github.com/facebookresearch/DomainBed) [134] is a suite to test domain generalization algorithms.
- [Dassl (KaiyangZhou)](https://github.com/KaiyangZhou/Dassl.pytorch): A PyTorch toolbox for domain adaptation, semi-supervised learning, and domain generalization.

# Lectures & Tutorials & Talks
- **(Talk 2021)** Generalizing to Unseen Domains: A Survey on Domain Generalization [155]. [[Video](https://www.bilibili.com/video/BV1ro4y1S7dd/)] [[Slides](http://jd92.wang/assets/files/DGSurvey-ppt.pdf)] *(Jindong Wang (MSRA), in Chinese)*

# Other Resources
- A collection of domain generalization papers organized by  [amber0309](https://github.com/amber0309/Domain-generalization).
- A collection of domain generalization papers organized by [jindongwang](https://github.com/jindongwang/transferlearning/blob/master/doc/awesome_paper.md#domain-generalization).
- A collection of papers on domain generalization, domain adaptation, causality, robustness, prompt, optimization, generative model, etc, organized by [yfzhang114](https://github.com/yfzhang114/Generalization-Causality).
- Adaptation and Generalization Across Domains in Visual Recognition with Deep Neural Networks [[PhD 2020, Kaiyang Zhou (University of Surrey)](https://openresearch.surrey.ac.uk/esploro/outputs/doctoral/Adaptation-and-Generalization-Across-Domains-in/99513024202346)] [164]

# Contributing & Contact
Feel free to contribute to our repository.

- If you woulk like to *correct mistakes*, please do it directly;
- If you would like to *add/update papers*, please finish the following tasks (if necessary):
    1. Find the max index (current max: **[231]**, not used: none), and create a new one.
    2. Update [Publications](#publications).
    3. Update [Papers](#papers).
    4. Update [Datasets](#datasets).
- If you have any *questions or advice*, please contact us by email (yuanjk@zju.edu.cn) or GitHub issues.

Thank you for your cooperation and contributions!

# Acknowledgements
The designed hierarchy of the [Contents](#contents) is mainly based on [awesome-domain-adaptation](https://github.com/zhaoxin94/awesome-domain-adaptation#unsupervised-da).
- We refer to [3] to design the [Contents](#contents) and the table of [Datasets](#datasets).
