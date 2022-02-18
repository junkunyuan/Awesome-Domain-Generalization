# Awesome Domain Generalization
This repository is a collection of awesome things about **domain generalization**, including papers, code, etc. 

If you would like to contribute to our repository or have any questions/advice, see [Contributing & Contact](#contributing--contact).

# Contents
- [Awesome-Domain-Generalization](#awesome-domain-generalization)
- [Contents](#contents)
- [Papers](#papers)
    - [Survey](#survey)
    - [Theory & Analysis](#theory--analysis)
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
    - [Applications](#applications)
        - [Person Re-Identification](#person-re-identification)
        - [Face Recognition & Anti-Spoofing](#face-recognition--anti-spoofing)
    - [Related Topics](#related-topics)
        - [Life-Long Learning](#life-long-learning)
- [Datasets](#datasets)
- [Libraries](#libraries)
- [Lectures & Tutorials & Talks](#lectures--tutorials--talks)
- [Other Resources](#other-resources)
- [Paper Index](#paper-index)
- [Contributing & Contact](#contributing--contact)
- [Acknowledgements](#acknowledgements)



# Papers
> We list papers, implementation code (the unofficial code is marked with *), etc, in the order of year and from journals to conferences. Note that some papers may fall into multiple categories.

## Survey
- Domain Generalization in Vision: A Survey [[arXiv 2021](https://arxiv.org/abs/2103.02503)]
- Generalizing to Unseen Domains: A Survey on Domain Generalization [[IJCAI 2021](https://arxiv.53yu.com/pdf/2103.03097)] [[Slides](http://jd92.wang/assets/files/DGSurvey-ppt.pdf)]


## Theory & Analysis
> We list the papers that either provide inspiring theoretical analyses or conduct extensive empirical studies for domain generalization.

- A Generalization Error Bound for Multi-Class Domain Generalization [[arXiv 2019](https://arxiv.org/pdf/1905.10392)]
- Domain Generalization by Marginal Transfer Learning [[JMLR 2021](https://www.jmlr.org/papers/volume22/17-679/17-679.pdf)] [[Code](https://github.com/aniketde/DomainGeneralizationMarginal)]
- The Risks of Invariant Risk Minimization [[ICLR 2021](https://arxiv.org/pdf/2010.05761)]
- In Search of Lost Domain Generalization [[ICLR 2021](https://arxiv.org/pdf/2007.01434.pdf?fbclid=IwAR1YkUXkIhC6fhr6eI687zBXo_W2tTjjTAFnyjEWvmq4gQKon_4pIDbTnQ4)]
- The Many Faces of Robustness: A Critical Analysis of Out-of-Distribution Generalization [[ICCV 2021](https://openaccess.thecvf.com/content/ICCV2021/papers/Hendrycks_The_Many_Faces_of_Robustness_A_Critical_Analysis_of_Out-of-Distribution_ICCV_2021_paper.pdf)] [[Code](https://github.com/hendrycks/imagenet-r)]
- An Empirical Investigation of Domain Generalization with Empirical Risk Minimizers [[NeurIPS 2021](https://proceedings.neurips.cc/paper/2021/file/ecf9902e0f61677c8de25ae60b654669-Paper.pdf)] [[Code](https://github.com/facebookresearch/domainbed_measures)]
- Towards a Theoretical Framework of Out-Of-Distribution Generalization [[NeurIPS 2021](https://proceedings.neurips.cc/paper/2021/file/c5c1cb0bebd56ae38817b251ad72bedb-Paper.pdf)]
- Out-of-Distribution Generalization in Kernel Regression [[NeurIPS 2021](https://proceedings.neurips.cc/paper/2021/file/691dcb1d65f31967a874d18383b9da75-Paper.pdf)]
- Quantifying and Improving Transferability in Domain Generalization [[NeurIPS 2021](https://proceedings.neurips.cc/paper/2021/file/5adaacd4531b78ff8b5cedfe3f4d5212-Paper.pdf)] [[Code](https://github.com/Gordon-Guojun-Zhang/Transferability-NeurIPS2021)]


## Domain Generalization
> To address the dataset/domain shift problem [108] [109] [110] [111] [112], domain generalization [113] aims to learn a model from source domain(s) and make it generalize well to unknown target domains.

### Domain Alignment-Based Methods
> Domain alignment-based methods aim to minimize divergence between source domains for learning domain-invariant representations.

- Domain Generalization via Invariant Feature Representation [[ICML 2013](http://proceedings.mlr.press/v28/muandet13.pdf)] [[Code](https://github.com/krikamol/dg-dica)]
- Learning Attributes Equals Multi-Source Domain Generalization [[CVPR 2016](https://www.cv-foundation.org/openaccess/content_cvpr_2016/papers/Gan_Learning_Attributes_Equals_CVPR_2016_paper.pdf)]
- Robust Domain Generalisation by Enforcing Distribution Invariance [[IJCAI 2016](https://eprints.qut.edu.au/115382/15/Erfani2016IJCAI.pdf)]
- Scatter Component Analysis A Unified Framework for Domain Adaptation and Domain Generalization [[TPAMI 2017](https://arxiv.53yu.com/pdf/1510.04373)]
- Unified Deep Supervised Domain Adaptation and Generalization [[ICCV 2017](https://openaccess.thecvf.com/content_ICCV_2017/papers/Motiian_Unified_Deep_Supervised_ICCV_2017_paper.pdf)] [[Code](https://github.com/samotiian/CCSA)]
- Beyond Domain Adaptation: Unseen Domain Encapsulation via Universal Non-volume Preserving Models [[arXiv 2018](https://arxiv.53yu.com/pdf/1812.03407)]
- Domain Generalization via Conditional Invariant Representation [[AAAI 2018](https://ojs.aaai.org/index.php/AAAI/article/view/11682/11541)]
- Domain Generalization with Adversarial Feature Learning [[CVPR 2018](https://openaccess.thecvf.com/content_cvpr_2018/papers/Li_Domain_Generalization_With_CVPR_2018_paper.pdf)] [[Code](https://github.com/YuqiCui/MMD_AAE)]
- Deep Domain Generalization via Conditional Invariant Adversarial Networks [[ECCV 2018](http://openaccess.thecvf.com/content_ECCV_2018/papers/Ya_Li_Deep_Domain_Generalization_ECCV_2018_paper.pdf)]
- Generalizing to Unseen Domains via Distribution Matching [[arXiv 2019](https://arxiv.53yu.com/pdf/1911.00804)] [[Code](https://github.com/belaalb/G2DM)]
- Image Alignment in Unseen Domains via Domain Deep Generalization [[arXiv 2019](https://arxiv.org/pdf/1905.12028)]
- Multi-Adversarial Discriminative Deep Domain Generalization for Face Presentation Attack Detection [[CVPR 2019](http://openaccess.thecvf.com/content_CVPR_2019/papers/Shao_Multi-Adversarial_Discriminative_Deep_Domain_Generalization_for_Face_Presentation_Attack_Detection_CVPR_2019_paper.pdf)] [[Code](https://github.com/rshaojimmy/CVPR2019-MADDoG)]
- Generalizable Feature Learning in the Presence of Data Bias and Domain Class Imbalance with Application to Skin Lesion Classification [[MICCAI 2019](https://www.cs.sfu.ca/~hamarneh/ecopy/miccai2019d.pdf)]
- Domain Generalization via Model-Agnostic Learning of Semantic Features [[NeurIPS 2019](https://proceedings.neurips.cc/paper/2019/file/2974788b53f73e7950e8aa49f3a306db-Paper.pdf)] [[Code](https://github.com/biomedia-mira/masf)]
- Adversarial Invariant Feature Learning with Accuracy Constraint for Domain Generalization [[ECMLPKDD 2019](https://arxiv.53yu.com/pdf/1904.12543)] [[Code](https://github.com/akuzeee/AFLAC)]
- Feature Alignment and Restoration for Domain Generalization and Adaptation [[arXiv 2020](https://arxiv.org/pdf/2006.12009)]
- Representation via Representations: Domain Generalization via Adversarially Learned Invariant Representations [[arXiv 2020](https://arxiv.53yu.com/pdf/2006.11478)]
- Feature alignment and restoration for domain generalization and adaptation [[arXiv 2020](https://arxiv.53yu.com/pdf/2006.12009)]
- Correlation-aware Adversarial Domain Adaptation and Generalization [[PR 2020](https://arxiv.53yu.com/pdf/1911.12983)] [[Code](https://github.com/mahfujur1/CA-DA-DG)]
- Domain Generalization Using a Mixture of Multiple Latent Domains [[AAAI 2020](https://ojs.aaai.org/index.php/AAAI/article/view/6846/6700)] [[Code](https://github.com/mil-tokyo/dg_mmld)]
- Single-Side Domain Generalization for Face Anti-Spoofing [[CVPR 2020](http://openaccess.thecvf.com/content_CVPR_2020/papers/Jia_Single-Side_Domain_Generalization_for_Face_Anti-Spoofing_CVPR_2020_paper.pdf)] [[Code](https://github.com/taylover-pei/SSDG-CVPR2020)]
- Scanner Invariant Multiple Sclerosis Lesion Segmentation from MRI [[ISBI 2020](https://arxiv.53yu.com/pdf/1910.10035)]
- Respecting Domain Relations: Hypothesis Invariance for Domain Generalization [[ICPR 2020](https://arxiv.53yu.com/pdf/2010.07591)]
- Domain Generalization via Multidomain Discriminant Analysis [[UAI 2020](http://proceedings.mlr.press/v115/hu20a/hu20a.pdf)] [[Code](https://github.com/amber0309/Multidomain-Discriminant-Analysis)]
- Domain Generalization for Medical Imaging Classification with Linear-Dependency Regularization [[NeurIPS 2020](https://proceedings.neurips.cc/paper/2020/file/201d7288b4c18a679e48b31c72c30ded-Paper.pdf)] [[Code](https://github.com/wyf0912/LDDG)]
- Domain Generalization via Entropy Regularization [[NeurIPS 2020](https://proceedings.neurips.cc/paper/2020/file/b98249b38337c5088bbc660d8f872d6a-Paper.pdf)] [[Code](https://github.com/sshan-zhao/DG_via_ER)]
- Iterative Feature Matching: Toward Provable Domain Generalization with Logarithmic Environments [[arXiv 2021](https://arxiv.org/pdf/2106.09913)]
- Semi-Supervised Domain Generalization in RealWorld: New Benchmark and Strong Baseline [[arXiv 2021](https://arxiv.org/pdf/2111.10221)]
- Collaborative Semantic Aggregation and Calibration for Separated Domain Generalization [[arXiv 2021](https://arxiv.org/pdf/2110.06736)] [[Code](https://github.com/junkunyuan/CSAC)]
- Multi-Domain Adversarial Feature Generalization for Person Re-Identification [[TIP 2021](https://ieeexplore.ieee.org/iel7/83/9263394/09311771.pdf)]
- Scale Invariant Domain Generalization Image Recapture Detection [[ICONIP 2021](https://arxiv.org/pdf/2110.03496)]
- Domain Generalization under Conditional and Label Shifts via Variational Bayesian Inference [[IJCAI 2021](https://arxiv.org/pdf/2107.10931)]
- Domain Generalization using Causal Matching [[ICML 2021](http://proceedings.mlr.press/v139/mahajan21b/mahajan21b.pdf)] [[Code](https://github.com/microsoft/robustdg)]
- Generalization on Unseen Domains via Inference-Time Label-Preserving Target Projections [[CVPR 2021](http://openaccess.thecvf.com/content/CVPR2021/papers/Pandey_Generalization_on_Unseen_Domains_via_Inference-Time_Label-Preserving_Target_Projections_CVPR_2021_paper.pdf)] [[Code](https://github.com/yys-Polaris/InferenceTimeDG)]
- Progressive Domain Expansion Network for Single Domain Generalization [[CVPR 2021](http://openaccess.thecvf.com/content/CVPR2021/papers/Li_Progressive_Domain_Expansion_Network_for_Single_Domain_Generalization_CVPR_2021_paper.pdf)] [[Code](https://github.com/lileicv/PDEN)]
- Confidence Calibration for Domain Generalization Under Covariate Shift [[ICCV 2021](https://openaccess.thecvf.com/content/ICCV2021/papers/Gong_Confidence_Calibration_for_Domain_Generalization_Under_Covariate_Shift_ICCV_2021_paper.pdf)]
- On Calibration and Out-of-domain Generalization [[NeurIPS 2021](https://proceedings.neurips.cc/paper/2021/file/118bd558033a1016fcc82560c65cca5f-Paper.pdf)]

### Data Augmentation-Based Methods
> Data augmentation-based methods augment original data and train the model on the generated data to improve model robustness.

- Certifying Some Distributional Robustness with Principled Adversarial Training [[arXiv 2017](https://arxiv.53yu.com/pdf/1710.10571.pdf])] [[Code](https://github.com/duchi-lab/certifiable-distributional-robustness)]
- Generalizing across Domains via Cross-Gradient Training [[ICLR 2018](https://arxiv.53yu.com/pdf/1804.10745)] [[Code](https://github.com/vihari/crossgrad)]
- Generalizing to Unseen Domains via Adversarial Data Augmentation [[NeurIPS 2018](https://proceedings.neurips.cc/paper/2018/file/1d94108e907bb8311d8802b48fd54b4a-Paper.pdf)] [[Code](https://github.com/ricvolpi/generalize-unseen-domains)]
- Staining Invariant Features for Improving Generalization of Deep Convolutional Neural Networks in Computational Pathology [[Frontiers in Bioengineering and Biotechnology 2019](https://www.frontiersin.org/articles/10.3389/fbioe.2019.00198/full?&utm_source=Email_to_authors_&utm_medium=Email&utm_content=T1_11.5e1_author&utm_campaign=Email_publication&field=&journalName=Frontiers_in_Bioengineering_and_Biotechnology&id=474781)]
- Multi-component Image Translation for Deep Domain Generalization [[WACV 2019](https://arxiv.53yu.com/pdf/1812.08974)] [[Code](https://github.com/mahfujur1/mit-DG)]
- Domain Generalization by Solving Jigsaw Puzzles [[CVPR 2019](https://openaccess.thecvf.com/content_CVPR_2019/papers/Carlucci_Domain_Generalization_by_Solving_Jigsaw_Puzzles_CVPR_2019_paper.pdf)] [[Code](https://github.com/fmcarlucci/JigenDG)]
- Addressing Model Vulnerability to Distributional Shifts Over Image Transformation Sets [[ICCV 2019](https://openaccess.thecvf.com/content_ICCV_2019/papers/Volpi_Addressing_Model_Vulnerability_to_Distributional_Shifts_Over_Image_Transformation_Sets_ICCV_2019_paper.pdf)] [[Code](https://github.com/ricvolpi/domain-shift-robustness)]
- Domain Randomization and Pyramid Consistency: Simulation-to-Real Generalization Without Accessing Target Domain Data [[ICCV 2019](http://openaccess.thecvf.com/content_ICCV_2019/papers/Yue_Domain_Randomization_and_Pyramid_Consistency_Simulation-to-Real_Generalization_Without_Accessing_Target_ICCV_2019_paper.pdf)] [[Code](https://github.com/xyyue/DRPC)]
- Hallucinating Agnostic Images to Generalize Across Domains [[ICCV workshop 2019](https://arxiv.53yu.com/pdf/1808.01102)]
- Improving the Generalizability of Convolutional Neural Network-Based Segmentation on CMR Images [[Frontiers in Cardiovascular Medicine 2020](https://www.frontiersin.org/articles/10.3389/fcvm.2020.00105/full)]
- Generalizing Deep Learning for Medical Image Segmentation to Unseen Domains via Deep Stacked Transformation [[TMI 2020](https://www.ncbi.nlm.nih.gov/pmc/articles/pmc7393676/)]
- Deep Domain-Adversarial Image Generation for Domain Generalisation [[AAAI 2020](https://ojs.aaai.org/index.php/AAAI/article/download/7003/6857)] [[Code](https://github.com/KaiyangZhou/Dassl.pytorch)]
- Towards Universal Representation Learning for Deep Face Recognition [[CVPR 2020](http://openaccess.thecvf.com/content_CVPR_2020/papers/Shi_Towards_Universal_Representation_Learning_for_Deep_Face_Recognition_CVPR_2020_paper.pdf)] [[Code](https://github.com/MatyushinMA/uni_rep_deep_faces)]
- Heterogeneous Domain Generalization via Domain Mixup [[ICASSP 2020](https://arxiv.org/pdf/2009.05448)] [[Code](https://github.com/wyf0912/MIXALL)]
- Learning to Generate Novel Domains for Domain Generalization [[ECCV 2020](https://arxiv.org/pdf/2007.03304)] [[Code](https://github.com/mousecpn/L2A-OT)]
- Learning from Extrinsic and Intrinsic Supervisions for Domain Generalization [[ECCV 2020](https://arxiv.53yu.com/pdf/2007.09316)] [[Code](https://github.com/emma-sjwang/EISNet)]
- Towards Recognizing Unseen Categories in Unseen Domains [[ECCV 2020](https://arxiv.53yu.com/pdf/2007.12256.pdf?ref=https://githubhelp.com)] [[Code](https://github.com/mancinimassimiliano/CuMix)]
- Rethinking Domain Generalization Baselines [[ICPR 2020](https://arxiv.53yu.com/pdf/2101.09060)]
- More is Better: A Novel Multi-view Framework for Domain Generalization [[arXiv 2021](https://arxiv.org/pdf/2112.12329)]
- Semi-Supervised Domain Generalization with Stochastic StyleMatch [[arXiv 2021](https://arxiv.53yu.com/pdf/2106.00592)] [[Code](https://github.com/KaiyangZhou/ssdg-benchmark)]
- Better Pseudo-label Joint Domain-aware Label and Dual-classifier for Semi-supervised Domain Generalization [[arXiv 2021](https://arxiv.53yu.com/pdf/2110.04820)]
- Out-of-domain Generalization from a Single Source: A Uncertainty Quantification Approach [[arXiv 2021](https://arxiv.53yu.com/pdf/2108.02888)]
- Towards Principled Disentanglement for Domain Generalization [[arXiv 2021](https://arxiv.org/pdf/2111.13839)] [[Code](https://github.com/hlzhang109/DDG)]
- MixStyle Neural Networks for Domain Generalization and Adaptation [[arXiv 2021](https://arxiv.53yu.com/pdf/2107.02053)] [[Code](https://github.com/KaiyangZhou/mixstyle-release)]
- VideoDG: Generalizing Temporal Relations in Videos to Novel Domains [[TPAMI 2021](https://arxiv.org/pdf/1912.03716)] [[Code](https://github.com/thuml/VideoDG)]
- Domain Generalization by Marginal Transfer Learning [[JMLR 2021](https://www.jmlr.org/papers/volume22/17-679/17-679.pdf)] [[Code](https://github.com/aniketde/DomainGeneralizationMarginal)]
- Domain Generalisation with Domain Augmented Supervised Contrastive Learning [[AAAI Student Abstract 2021](https://www.aaai.org/AAAI21Papers/SA-197.LeHS.pdf)]
- DecAug: Out-of-Distribution Generalization via Decomposed Feature Representation and Semantic Augmentation [[AAAI 2021](https://arxiv.org/pdf/2012.09382)] [[Code](https://github.com/HaoyueBaiZJU/DecAug)]
- Domain Generalization with Mixstyle [[ICLR 2021](https://arxiv.53yu.com/pdf/2104.02008)] [[Code](https://github.com/KaiyangZhou/mixstyle-release)]
- Robust and Generalizable Visual Representation Learning via Random Convolutions [[ICLR 2021](https://arxiv.53yu.com/pdf/2007.13003)] [[Code](https://github.com/wildphoton/RandConv)]
- Learning to Learn Single Domain Generalization [[CVPR 2020](https://openaccess.thecvf.com/content_CVPR_2020/papers/Qiao_Learning_to_Learn_Single_Domain_Generalization_CVPR_2020_paper.pdf)] [[Code](https://github.com/joffery/M-ADA)]
- FSDR: Frequency Space Domain Randomization for Domain Generalization [[CVPR 2021](http://openaccess.thecvf.com/content/CVPR2021/papers/Huang_FSDR_Frequency_Space_Domain_Randomization_for_Domain_Generalization_CVPR_2021_paper.pdf)] [[Code](https://github.com/jxhuang0508/FSDR)]
- FedDG: Federated Domain Generalization on Medical Image Segmentation via Episodic Learning in Continuous Frequency Space [[CVPR 2021](http://openaccess.thecvf.com/content/CVPR2021/papers/Liu_FedDG_Federated_Domain_Generalization_on_Medical_Image_Segmentation_via_Episodic_CVPR_2021_paper.pdf)] [[Code](https://github.com/liuquande/FedDG-ELCFS)]
- Uncertainty-guided Model Generalization to Unseen Domains [[CVPR 2021](http://openaccess.thecvf.com/content/CVPR2021/papers/Qiao_Uncertainty-Guided_Model_Generalization_to_Unseen_Domains_CVPR_2021_paper.pdf)] [[Code](https://github.com/joffery/UMGUD)]
- Continual Adaptation of Visual Representations via Domain Randomization and Meta-learning [[CVPR 2021](http://openaccess.thecvf.com/content/CVPR2021/papers/Volpi_Continual_Adaptation_of_Visual_Representations_via_Domain_Randomization_and_Meta-Learning_CVPR_2021_paper.pdf)]
- A Fourier-Based Framework for Domain Generalization [[CVPR 2021](https://openaccess.thecvf.com/content/CVPR2021/papers/Xu_A_Fourier-Based_Framework_for_Domain_Generalization_CVPR_2021_paper.pdf)] [[Code](https://github.com/MediaBrain-SJTU/FACT)]
- Open Domain Generalization with Domain-Augmented Meta-Learning [[CVPR 2021](http://openaccess.thecvf.com/content/CVPR2021/papers/Shu_Open_Domain_Generalization_with_Domain-Augmented_Meta-Learning_CVPR_2021_paper.pdf)] [[Code](https://github.com/thuml/OpenDG-DAML)]
- A Simple Feature Augmentation for Domain Generalization [[ICCV 2021](https://openaccess.thecvf.com/content/ICCV2021/papers/Li_A_Simple_Feature_Augmentation_for_Domain_Generalization_ICCV_2021_paper.pdf)]
- Universal Cross-Domain Retrieval Generalizing Across Classes and Domains [[ICCV 2021](https://openaccess.thecvf.com/content/ICCV2021/papers/Paul_Universal_Cross-Domain_Retrieval_Generalizing_Across_Classes_and_Domains_ICCV_2021_paper.pdf)] [[Code](https://github.com/mvp18/UCDR)]
- Feature Stylization and Domain-aware Contrastive Learning for Domain Generalization [[MM 2021](https://dl.acm.org/doi/pdf/10.1145/3474085.3475271)]
- Adversarial Teacher-Student Representation Learning for Domain Generalization [[NeurIPS 2021](https://proceedings.neurips.cc/paper/2021/file/a2137a2ae8e39b5002a3f8909ecb88fe-Paper.pdf)]
- Model-Based Domain Generalization [[NeurIPS 2021](https://proceedings.neurips.cc/paper/2021/file/a8f12d9486cbcc2fe0cfc5352011ad35-Paper.pdf)] [[Code](https://github.com/arobey1/mbdg)]

### Meta-Learning-Based Methods
> Meta-learning-based methods train the model on a meta-train set and improve its performance on a meta-test set for boosting out-of-domain generalization ability.  

- Learning to Generalize: Meta-Learning for Domain Generalization [[AAAI 2018](https://www.aaai.org/ocs/index.php/AAAI/AAAI18/paper/viewFile/16067/16547)] [[Code](https://github.com/HAHA-DL/MLDG)]
- MetaReg: Towards Domain Generalization using Meta-Regularization [[NeurIPS 2018](https://proceedings.neurips.cc/paper/2018/file/647bba344396e7c8170902bcf2e15551-Paper.pdf)] [[Code*](https://github.com/elliotbeck/MetaReg_PyTorch)]
- Feature-Critic Networks for Heterogeneous Domain Generalisation [[ICML 2019](http://proceedings.mlr.press/v97/li19l/li19l.pdf)] [[Code](https://github.com/liyiying/Feature_Critic)]
- Episodic Training for Domain Generalization [[ICCV 2019](https://openaccess.thecvf.com/content_ICCV_2019/papers/Li_Episodic_Training_for_Domain_Generalization_ICCV_2019_paper.pdf)] [[Code](https://github.com/HAHA-DL/Episodic-DG)]
- Domain Generalization via Model-Agnostic Learning of Semantic Features [[NeurIPS 2019](https://proceedings.neurips.cc/paper/2019/file/2974788b53f73e7950e8aa49f3a306db-Paper.pdf)] [[Code](https://github.com/biomedia-mira/masf)]
- Domain Generalization via Semi-supervised Meta Learning [[arXiv 2020](https://arxiv.org/pdf/2009.12658)] [[Code](https://github.com/hosseinshn/DGSML)]
- Frustratingly Simple Domain Generalization via Image Stylization [[arXiv 2020](https://arxiv.53yu.com/pdf/2006.11207)] [[Code](https://github.com/GT-RIPL/DomainGeneralization-Stylization)]
- Domain Generalization for Named Entity Boundary Detection via Metalearning [[TNNLS 2020](https://ieeexplore.ieee.org/abstract/document/9174763/)]
- Learning to Learn Single Domain Generalization [[CVPR 2020](https://openaccess.thecvf.com/content_CVPR_2020/papers/Qiao_Learning_to_Learn_Single_Domain_Generalization_CVPR_2020_paper.pdf)] [[Code](https://github.com/joffery/M-ADA)]
- Learning to Learn with Variational Information Bottleneck for Domain Generalization [[ECCV 2020](https://arxiv.org/pdf/2007.07645)]
- Sequential Learning for Domain Generalization [[ECCV workshop 2020](https://arxiv.org/pdf/2004.01377)]
- Shape-Aware Meta-Learning for Generalizing Prostate MRI Segmentation to Unseen Domains [[MICCAI 2020](https://arxiv.org/pdf/2007.02035)] [[Code](https://github.com/liuquande/SAML)]
- More is Better: A Novel Multi-view Framework for Domain Generalization [[arXiv 2021](https://arxiv.org/pdf/2112.12329)]
- Few-Shot Classification in Unseen Domains by Episodic Meta-Learning Across Visual Domains [[ICIP 2021](https://arxiv.org/pdf/2112.13539)]
- Meta-Learned Feature Critics for Domain Generalized Semantic Segmentation [[ICIP 2021](https://arxiv.org/pdf/2112.13538)]
- MetaNorm: Learning to Normalize Few-Shot Batches Across Domains [[ICLR 2021](https://openreview.net/pdf?id=9z_dNsC4B5t)] [[Code](https://github.com/YDU-AI/MetaNorm)]
- Learning to Generalize Unseen Domains via Memory-based Multi-Source Meta-Learning for Person Re-Identification [[CVPR 2021](https://openaccess.thecvf.com/content/CVPR2021/papers/Zhao_Learning_to_Generalize_Unseen_Domains_via_Memory-based_Multi-Source_Meta-Learning_for_CVPR_2021_paper.pdf)] [[Code](https://github.com/HeliosZhao/M3L)]
- Uncertainty-guided Model Generalization to Unseen Domains [[CVPR 2021](http://openaccess.thecvf.com/content/CVPR2021/papers/Qiao_Uncertainty-Guided_Model_Generalization_to_Unseen_Domains_CVPR_2021_paper.pdf)] [[Code](https://github.com/joffery/UMGUD)]
- Continual Adaptation of Visual Representations via Domain Randomization and Meta-learning [[CVPR 2021](http://openaccess.thecvf.com/content/CVPR2021/papers/Volpi_Continual_Adaptation_of_Visual_Representations_via_Domain_Randomization_and_Meta-Learning_CVPR_2021_paper.pdf)]
- Meta Batch-Instance Normalization for Generalizable Person Re-Identification [[CVPR 2021](https://openaccess.thecvf.com/content/CVPR2021/papers/Choi_Meta_Batch-Instance_Normalization_for_Generalizable_Person_Re-Identification_CVPR_2021_paper.pdf)] [[Code](https://github.com/bismex/MetaBIN)]
- Open Domain Generalization with Domain-Augmented Meta-Learning [[CVPR 2021](http://openaccess.thecvf.com/content/CVPR2021/papers/Shu_Open_Domain_Generalization_with_Domain-Augmented_Meta-Learning_CVPR_2021_paper.pdf)] [[Code](https://github.com/thuml/OpenDG-DAML)]
- Exploiting Domain-Specific Features to Enhance Domain Generalization [[NeurIPS 2021](https://proceedings.neurips.cc/paper/2021/file/b0f2ad44d26e1a6f244201fe0fd864d1-Paper.pdf)] [[Code](https://github.com/manhhabui/mDSDI)]

### Ensemble Learning-Based Methods
> Ensemble learning-based methods mainly train a domain-specific model on each source domain, and then draw on collective wisdom to make accurate prediction. 

- Exploiting Low-Rank Structure from Latent Domains for Domain Generalization [[ECCV 2014](https://linkspringer.53yu.com/content/pdf/10.1007/978-3-319-10578-9_41.pdf)]
- Visual recognition by learning from web data: A weakly supervised domain generalization approach [[CVPR 2015](https://openaccess.thecvf.com/content_cvpr_2015/papers/Niu_Visual_Recognition_by_2015_CVPR_paper.pdf)]
- Multi-View Domain Generalization for Visual Recognition [[ICCV 2015](http://openaccess.thecvf.com/content_iccv_2015/papers/Niu_Multi-View_Domain_Generalization_ICCV_2015_paper.pdf)]
- Deep Domain Generalization With Structured Low-Rank Constraint [[TIP 2017](https://par.nsf.gov/servlets/purl/10065328)]
- Visual Recognition by Learning From Web Data via Weakly Supervised Domain Generalization [[TNNLS 2017](https://bcmi.sjtu.edu.cn/home/niuli/paper/Visual%20Recognition%20by%20Learning%20From%20Web%20Data%20via%20Weakly%20Supervised%20Domain%20Generalization.pdf)]
- Robust Place Categorization with Deep Domain Generalization [[IEEE Robotics and Automation Letters 2018](https://arxiv.53yu.com/pdf/1805.12048)] [[Code](https://github.com/mancinimassimiliano/caffe)] 
- Multi-View Domain Generalization Framework for Visual Recognition [[TNNLS 2018](http://openaccess.thecvf.com/content_iccv_2015/papers/Niu_Multi-View_Domain_Generalization_ICCV_2015_paper.pdf)]
- Domain Generalization with Domain-Specific Aggregation Modules [[GCPR 2018](https://arxiv.53yu.com/pdf/1809.10966)]
- Best Sources Forward: Domain Generalization through Source-Specific Nets [[ICIP 2018](https://arxiv.53yu.com/pdf/1806.05810)]
- Batch Normalization Embeddings for Deep Domain Generalization [[arXiv 2020](https://arxiv.53yu.com/pdf/2011.12672)]
- DoFE: Domain-oriented Feature Embedding for Generalizable Fundus Image Segmentation on Unseen Datasets [[TMI 2020](https://arxiv.53yu.com/pdf/2010.06208)]
- MS-Net: Multi-Site Network for Improving Prostate Segmentation with Heterogeneous MRI Data [[TMI 2020](https://arxiv.53yu.com/pdf/2002.03366)] [[Code](https://github.com/liuquande/MS-Net)]
- Generalized Convolutional Forest Networks for Domain Generalization and Visual Recognition [[ICLR 2020](https://openreview.net/pdf?id=H1lxVyStPH)]
- Learning to Optimize Domain Specific Normalization for Domain Generalization [[ECCV 2020](https://arxiv.53yu.com/pdf/1907.04275)]
- Class-conditioned Domain Generalization via Wasserstein Distributional Robust Optimization [[ICLR workshop 2021](https://arxiv.org/pdf/2109.03676)]
- Domain and Content Adaptive Convolution for Domain Generalization in Medical Image Segmentation [[arXiv 2021](https://arxiv.org/pdf/2109.05676)]
- Dynamically Decoding Source Domain Knowledge for Unseen Domain Generalization [[arXiv 2021](https://www.researchgate.net/profile/Karthik-Nandakumar-3/publication/355142270_Dynamically_Decoding_Source_Domain_Knowledge_For_Unseen_Domain_Generalization/links/61debe18034dda1b9ef16fc6/Dynamically-Decoding-Source-Domain-Knowledge-For-Unseen-Domain-Generalization.pdf)]
- Domain Adaptive Ensemble Learning [[TIP 2021](https://arxiv.53yu.com/pdf/2003.07325)] [[Code](https://github.com/KaiyangZhou/Dassl.pytorch)]
- Generalizable Person Re-identification with Relevance-aware Mixture of Experts [[CVPR 2021](https://openaccess.thecvf.com/content/CVPR2021/papers/Dai_Generalizable_Person_Re-Identification_With_Relevance-Aware_Mixture_of_Experts_CVPR_2021_paper.pdf)]
- Learning Transferrable and Interpretable Representations for Domain Generalization [[MM 2021](https://dl.acm.org/doi/pdf/10.1145/3474085.3475488)]
- Embracing the Dark Knowledge: Domain Generalization Using Regularized Knowledge Distillation [[MM 2021](https://arxiv.53yu.com/pdf/2110.04820)]
- TransMatcher: Deep Image Matching Through Transformers for Generalizable Person Re-identification [[NeurIPS 2021](https://proceedings.neurips.cc/paper/2021/file/0f49c89d1e7298bb9930789c8ed59d48-Paper.pdf)] [[Code](https://github.com/ShengcaiLiao/QAConv)]

### Self-Supervised Learning-Based Methods
> Self-supervised learning-based methods improve model generalization by solving some pretext tasks with data itself.

- Domain Generalization for Object Recognition with Multi-Task Autoencoders [[ICCV 2015](http://openaccess.thecvf.com/content_iccv_2015/papers/Ghifary_Domain_Generalization_for_ICCV_2015_paper.pdf)] [[Code](https://github.com/Emma0118/mate)]
- Domain Generalization by Solving Jigsaw Puzzles [[CVPR 2019](https://openaccess.thecvf.com/content_CVPR_2019/papers/Carlucci_Domain_Generalization_by_Solving_Jigsaw_Puzzles_CVPR_2019_paper.pdf)] [[Code](https://github.com/fmcarlucci/JigenDG)]
- Improving Out-Of-Distribution Generalization via Multi-Task Self-Supervised Pretraining [[arXiv 2020](https://arxiv.53yu.com/pdf/2003.13525)]
- Generalized Convolutional Forest Networks for Domain Generalization and Visual Recognition [[ICLR 2020](https://openreview.net/pdf?id=H1lxVyStPH)]
- Learning from Extrinsic and Intrinsic Supervisions for Domain Generalization [[ECCV 2020](https://arxiv.53yu.com/pdf/2007.09316)] [[Code](https://github.com/emma-sjwang/EISNet)]
- Zero Shot Domain Generalization [[BMVC 2020](https://arxiv.53yu.com/pdf/2008.07443)] [[Code](https://github.com/aniketde/ZeroShotDG)]
- Out-of-domain Generalization from a Single Source: A Uncertainty Quantification Approach [[arXiv 2021](https://arxiv.53yu.com/pdf/2108.02888)]
- Unsupervised Domain Generalization by Learning a Bridge Across Domains [[arXiv 2021](https://arxiv.org/pdf/2112.02300)]
- Self-Supervised Learning Across Domains [[TPAMI 2021](https://arxiv.53yu.com/pdf/2007.12368)] [[Code](https://github.com/silvia1993/Self-Supervised_Learning_Across_Domains)]
- Multi-Domain Adversarial Feature Generalization for Person Re-Identification [[TIP 2021](https://ieeexplore.ieee.org/iel7/83/9263394/09311771.pdf)]
- Scale Invariant Domain Generalization Image Recapture Detection [[ICONIP 2021](https://arxiv.org/pdf/2110.03496)]
- Domain Generalisation with Domain Augmented Supervised Contrastive Learning [[AAAI Student Abstract 2021](https://www.aaai.org/AAAI21Papers/SA-197.LeHS.pdf)]
- Progressive Domain Expansion Network for Single Domain Generalization [[CVPR 2021](http://openaccess.thecvf.com/content/CVPR2021/papers/Li_Progressive_Domain_Expansion_Network_for_Single_Domain_Generalization_CVPR_2021_paper.pdf)] [[Code](https://github.com/lileicv/PDEN)]
- FedDG: Federated Domain Generalization on Medical Image Segmentation via Episodic Learning in Continuous Frequency Space [[CVPR 2021](http://openaccess.thecvf.com/content/CVPR2021/papers/Liu_FedDG_Federated_Domain_Generalization_on_Medical_Image_Segmentation_via_Episodic_CVPR_2021_paper.pdf)] [[Code](https://github.com/liuquande/FedDG-ELCFS)]
- Boosting the Generalization Capability in Cross-Domain Few-shot Learning via Noise-enhanced Supervised Autoencoder [[ICCV 2021](https://openaccess.thecvf.com/content/ICCV2021/papers/Liang_Boosting_the_Generalization_Capability_in_Cross-Domain_Few-Shot_Learning_via_Noise-Enhanced_ICCV_2021_paper.pdf)]
- A Style and Semantic Memory Mechanism for Domain Generalization [[ICCV 2021](http://openaccess.thecvf.com/content/ICCV2021/papers/Chen_A_Style_and_Semantic_Memory_Mechanism_for_Domain_Generalization_ICCV_2021_paper.pdf)]
- SelfReg: Self-Supervised Contrastive Regularization for Domain Generalization [[ICCV 2021](http://openaccess.thecvf.com/content/ICCV2021/papers/Kim_SelfReg_Self-Supervised_Contrastive_Regularization_for_Domain_Generalization_ICCV_2021_paper.pdf)]
- Domain Generalization for Mammography Detection via Multi-style and Multi-view Contrastive Learning [[MICCAI 2021](https://arxiv.org/pdf/2111.10827)] [[Code](https://github.com/lizheren/MSVCL_MICCAI2021)]
- Feature Stylization and Domain-aware Contrastive Learning for Domain Generalization [[MM 2021](https://dl.acm.org/doi/pdf/10.1145/3474085.3475271)]
- Adversarial Teacher-Student Representation Learning for Domain Generalization [[NeurIPS 2021](https://proceedings.neurips.cc/paper/2021/file/a2137a2ae8e39b5002a3f8909ecb88fe-Paper.pdf)]

### Disentangled Representation Learning-Based Methods
> Disentangled representation learning-based methods aim to disentangle domain-specific and domain-invariant parts from source data, and then adopt the domain-invariant one for inference on the target domains. 

- Undoing the Damage of Dataset Bias [[ECCV 2012](https://linkspringer.53yu.com/content/pdf/10.1007/978-3-642-33718-5_12.pdf)] [[Code](https://github.com/adikhosla/undoing-bias)] 
- Deeper, Broader and Artier Domain Generalization [[ICCV 2017](https://openaccess.thecvf.com/content_ICCV_2017/papers/Li_Deeper_Broader_and_ICCV_2017_paper.pdf)] [[Code](https://dali-dl.github.io/project_iccv2017.html)]
- DIVA: Domain Invariant Variational Autoencoders [[ICML workshop 2019](http://proceedings.mlr.press/v121/ilse20a/ilse20a.pdf)] [[Code](https://github.com/AMLab-Amsterdam/DIVA)]
- Efficient Domain Generalization via Common-Specific Low-Rank Decomposition [[ICML 2020](http://proceedings.mlr.press/v119/piratla20a/piratla20a.pdf)] [[Code](https://github.com/vihari/CSD)]
- Cross-Domain Face Presentation Attack Detection via Multi-Domain Disentangled Representation Learning [[CVPR 2020](http://openaccess.thecvf.com/content_CVPR_2020/papers/Wang_Cross-Domain_Face_Presentation_Attack_Detection_via_Multi-Domain_Disentangled_Representation_Learning_CVPR_2020_paper.pdf)]
- Learning to Balance Specificity and Invariance for In and Out of Domain Generalization [[ECCV 2020](https://arxiv.53yu.com/pdf/2008.12839)] [[Code](https://github.com/prithv1/DMG)]
- Towards Principled Disentanglement for Domain Generalization [[arXiv 2021](https://arxiv.org/pdf/2111.13839)] [[Code](https://github.com/hlzhang109/DDG)]
- Meta-Learned Feature Critics for Domain Generalized Semantic Segmentation [[ICIP 2021](https://arxiv.org/pdf/2112.13538)]
- DecAug: Out-of-Distribution Generalization via Decomposed Feature Representation and Semantic Augmentation [[AAAI 2021](https://arxiv.org/pdf/2012.09382)] [[Code](https://github.com/HaoyueBaiZJU/DecAug)]
- Robustnet: Improving Domain Generalization in Urban-Scene Segmentation via Instance Selective Whitening [[CVPR 2021](https://openaccess.thecvf.com/content/CVPR2021/papers/Choi_RobustNet_Improving_Domain_Generalization_in_Urban-Scene_Segmentation_via_Instance_Selective_CVPR_2021_paper.pdf)] [[Code](https://github.com/shachoi/RobustNet)]
- Shape-Biased Domain Generalization via Shock Graph Embeddings [[ICCV 2021](https://openaccess.thecvf.com/content/ICCV2021/papers/Narayanan_Shape-Biased_Domain_Generalization_via_Shock_Graph_Embeddings_ICCV_2021_paper.pdf)]
- Domain-Invariant Disentangled Network for Generalizable Object Detection [[ICCV 2021](https://openaccess.thecvf.com/content/ICCV2021/papers/Lin_Domain-Invariant_Disentangled_Network_for_Generalizable_Object_Detection_ICCV_2021_paper.pdf)] 
- Domain Generalization via Feature Variation Decorrelation [[MM 2021](https://dl.acm.org/doi/pdf/10.1145/3474085.3475311)]
- Exploiting Domain-Specific Features to Enhance Domain Generalization [[NeurIPS 2021](https://proceedings.neurips.cc/paper/2021/file/b0f2ad44d26e1a6f244201fe0fd864d1-Paper.pdf)] [[Code](https://github.com/manhhabui/mDSDI)]

### Regularization-Based Methods
> Regularization-based methods leverage regularization terms to prevent the overfitting, or design optimization strategies to guide the training.

- Generalizing from Several Related Classification Tasks to a New Unlabeled Sample [[NeurIPS 2011](https://proceedings.neurips.cc/paper/2011/file/b571ecea16a9824023ee1af16897a582-Paper.pdf)]
- MetaReg: Towards Domain Generalization using Meta-Regularization [[NeurIPS 2018](https://proceedings.neurips.cc/paper/2018/file/647bba344396e7c8170902bcf2e15551-Paper.pdf)] [[Code*](https://github.com/elliotbeck/MetaReg_PyTorch)]
- Invariant Risk Minimization [[arXiv 2019](https://arxiv.53yu.com/pdf/1907.02893.pdf;)] [[Code](https://github.com/facebookresearch/InvariantRiskMinimization)]
- Learning Robust Representations by Projecting Superficial Statistics Out [[ICLR 2019](https://arxiv.53yu.com/pdf/1903.06256)] [[Code](https://github.com/HaohanWang/HEX)]
- Self-challenging Improves Cross-Domain Generalization [[ECCV 2020](https://arxiv.53yu.com/pdf/2007.02454)] [[Code](https://github.com/DeLightCMU/RSC)]
- Energy-based Out-of-distribution Detection [[NeurIPS 2020](https://proceedings.neurips.cc/paper/2020/file/f5496252609c43eb8a3d147ab9b9c006-Paper.pdf)] [[Code](https://github.com/xieshuqin/Energy-OOD)]
- Fishr: Invariant Gradient Variances for Our-of-distribution Generalization [[arXiv 2021](https://arxiv.org/pdf/2109.02934)] [[Code](https://github.com/alexrame/fishr)]
- Out-of-Distribution Generalization via Risk Extrapolation [[ICML 2021](http://proceedings.mlr.press/v139/krueger21a/krueger21a.pdf)]
- A Fourier-Based Framework for Domain Generalization [[CVPR 2021](https://openaccess.thecvf.com/content/CVPR2021/papers/Xu_A_Fourier-Based_Framework_for_Domain_Generalization_CVPR_2021_paper.pdf)] [[Code](https://github.com/MediaBrain-SJTU/FACT)]
- Domain Generalization via Gradient Surgery [[ICCV 2021](https://openaccess.thecvf.com/content/ICCV2021/papers/Mansilla_Domain_Generalization_via_Gradient_Surgery_ICCV_2021_paper.pdf)] [[Code](https://github.com/lucasmansilla/DGvGS)]
- SelfReg: Self-Supervised Contrastive Regularization for Domain Generalization [[ICCV 2021](http://openaccess.thecvf.com/content/ICCV2021/papers/Kim_SelfReg_Self-Supervised_Contrastive_Regularization_for_Domain_Generalization_ICCV_2021_paper.pdf)]
- Embracing the Dark Knowledge: Domain Generalization Using Regularized Knowledge Distillation [[MM 2021](https://arxiv.53yu.com/pdf/2110.04820)]
- Model-Based Domain Generalization [[NeurIPS 2021](https://proceedings.neurips.cc/paper/2021/file/a8f12d9486cbcc2fe0cfc5352011ad35-Paper.pdf)] [[Code](https://github.com/arobey1/mbdg)]
- Swad: Domain Generalization by Seeking Flat Minima [[NeurIPS 2021](https://proceedings.neurips.cc/paper/2021/file/bcb41ccdc4363c6848a1d760f26c28a0-Paper.pdf)] [[Code](https://github.com/khanrc/swad)]
- Training for the Future: A Simple Gradient Interpolation Loss to Generalize Along Time [[NeurIPS 2021](https://proceedings.neurips.cc/paper/2021/file/a02ef8389f6d40f84b50504613117f88-Paper.pdf)] [[Code](https://github.com/anshuln/Training-for-the-Future)]
- Quantifying and Improving Transferability in Domain Generalization [[NeurIPS 2021](https://proceedings.neurips.cc/paper/2021/file/5adaacd4531b78ff8b5cedfe3f4d5212-Paper.pdf)] [[Code](https://github.com/Gordon-Guojun-Zhang/Transferability-NeurIPS2021)]

### Normalization-Based Methods
> Normalization-based methods calibrate data from different domains by normalizing them with their statistic.

- Batch Normalization Embeddings for Deep Domain Generalization [[arXiv 2020](https://arxiv.53yu.com/pdf/2011.12672)]
- Learning to Optimize Domain Specific Normalization for Domain Generalization [[ECCV 2020](https://arxiv.53yu.com/pdf/1907.04275)]
- MetaNorm: Learning to Normalize Few-Shot Batches Across Domains [[ICLR 2021](https://openreview.net/pdf?id=9z_dNsC4B5t)] [[Code](https://github.com/YDU-AI/MetaNorm)]
- Meta Batch-Instance Normalization for Generalizable Person Re-Identification [[CVPR 2021](https://openaccess.thecvf.com/content/CVPR2021/papers/Choi_Meta_Batch-Instance_Normalization_for_Generalizable_Person_Re-Identification_CVPR_2021_paper.pdf)] [[Code](https://github.com/bismex/MetaBIN)]
- Learning to Generalize Unseen Domains via Memory-based Multi-Source Meta-Learning for Person Re-Identification [[CVPR 2021](https://openaccess.thecvf.com/content/CVPR2021/papers/Zhao_Learning_to_Generalize_Unseen_Domains_via_Memory-based_Multi-Source_Meta-Learning_for_CVPR_2021_paper.pdf)] [[Code](https://github.com/HeliosZhao/M3L)]
- Adversarially Adaptive Normalization for Single Domain Generalization [[CVPR 2021](https://openaccess.thecvf.com/content/CVPR2021/papers/Fan_Adversarially_Adaptive_Normalization_for_Single_Domain_Generalization_CVPR_2021_paper.pdf)]
- Collaborative Optimization and Aggregation for Decentralized Domain Generalization and Adaptation [[ICCV 2021](https://openaccess.thecvf.com/content/ICCV2021/papers/Wu_Collaborative_Optimization_and_Aggregation_for_Decentralized_Domain_Generalization_and_Adaptation_ICCV_2021_paper.pdf)]
- Domain Generalization through Audio-Visual Relative Norm Alignment in First Person Action Recognition [[WACV 2022](https://openaccess.thecvf.com/content/WACV2022/papers/Planamente_Domain_Generalization_Through_Audio-Visual_Relative_Norm_Alignment_in_First_Person_WACV_2022_paper.pdf)]

### Information-Based Methods
> Information-based methods utilize techniques of information theory to realize domain generalization. 

- Learning to Learn with Variational Information Bottleneck for Domain Generalization [[ECCV 2020](https://arxiv.org/pdf/2007.07645)]
- Progressive Domain Expansion Network for Single Domain Generalization [[CVPR 2021](http://openaccess.thecvf.com/content/CVPR2021/papers/Li_Progressive_Domain_Expansion_Network_for_Single_Domain_Generalization_CVPR_2021_paper.pdf)] [[Code](https://github.com/lileicv/PDEN)]
- Learning To Diversify for Single Domain Generalization [[ICCV 2021](https://openaccess.thecvf.com/content/ICCV2021/papers/Wang_Learning_To_Diversify_for_Single_Domain_Generalization_ICCV_2021_paper.pdf)] [[Code](https://github.com/BUserName/Learning)]
- Invariance Principle Meets Information Bottleneck for Out-Of-Distribution Generalization [[NeurIPS 2021](https://proceedings.neurips.cc/paper/2021/file/1c336b8080f82bcc2cd2499b4c57261d-Paper.pdf)] [[Code](https://github.com/ahujak/IB-IRM)]
- Exploiting Domain-Specific Features to Enhance Domain Generalization [[NeurIPS 2021](https://proceedings.neurips.cc/paper/2021/file/b0f2ad44d26e1a6f244201fe0fd864d1-Paper.pdf)] [[Code](https://github.com/manhhabui/mDSDI)]
- Invariant Information Bottleneck for Domain Generalization [[AAAI 2022](https://arxiv.org/pdf/2106.06333)] [[Code](https://github.com/Luodian/IIB/tree/IIB)]

### Causality-Based Methods
> Causality-based methods analyze and address the domain generalization problem from a causal perspective.

- Invariant Risk Minimization [[arXiv 2019](https://arxiv.53yu.com/pdf/1907.02893.pdf;)] [[Code](https://github.com/facebookresearch/InvariantRiskMinimization)]
- Learning Domain-Invariant Relationship with Instrumental Variable for Domain Generalization [[arXiv 2021](https://arxiv.org/pdf/2110.01438)]
- A Causal Framework for Distribution Generalization [[TPAMI 2021](https://arxiv.org/pdf/2006.07433)] [[Code](https://runesen.github.io/NILE/)]
- Domain Generalization using Causal Matching [[ICML 2021](http://proceedings.mlr.press/v139/mahajan21b/mahajan21b.pdf)] [[Code](https://github.com/microsoft/robustdg)]
- Deep Stable Learning for Out-of-Distribution Generalization [[CVPR 2021](http://openaccess.thecvf.com/content/CVPR2021/papers/Zhang_Deep_Stable_Learning_for_Out-of-Distribution_Generalization_CVPR_2021_paper.pdf)] [[Code](https://github.com/xxgege/StableNet)]
- A Style and Semantic Memory Mechanism for Domain Generalization [[ICCV 2021](http://openaccess.thecvf.com/content/ICCV2021/papers/Chen_A_Style_and_Semantic_Memory_Mechanism_for_Domain_Generalization_ICCV_2021_paper.pdf)]
- Learning Causal Semantic Representation for Out-of-Distribution Prediction [[NeurIPS 2021](https://proceedings.neurips.cc/paper/2021/file/310614fca8fb8e5491295336298c340f-Paper.pdf)] [[Code](https://github.com/changliu00/causal-semantic-generative-model)]
- Recovering Latent Causal Factor for Generalization to Distributional Shifts [[NeurIPS 2021](https://proceedings.neurips.cc/paper/2021/file/8c6744c9d42ec2cb9e8885b54ff744d0-Paper.pdf)] [[Code](https://github.com/wubotong/LaCIM)]
- On Calibration and Out-of-domain Generalization [[NeurIPS 2021](https://proceedings.neurips.cc/paper/2021/file/118bd558033a1016fcc82560c65cca5f-Paper.pdf)]
- Invariance Principle Meets Information Bottleneck for Out-Of-Distribution Generalization [[NeurIPS 2021](https://proceedings.neurips.cc/paper/2021/file/1c336b8080f82bcc2cd2499b4c57261d-Paper.pdf)] [[Code](https://github.com/ahujak/IB-IRM)]
- Invariant Information Bottleneck for Domain Generalization [[AAAI 2022](https://arxiv.org/pdf/2106.06333)] [[Code](https://github.com/Luodian/IIB/tree/IIB)]

### Inference-Time-Based Methods
> Inference-time-based methods leverage the unlabeled target data, which is available at inference-time, to improve generalization performance without further model training. 

- Generalization on Unseen Domains via Inference-Time Label-Preserving Target Projections [[CVPR 2021](http://openaccess.thecvf.com/content/CVPR2021/papers/Pandey_Generalization_on_Unseen_Domains_via_Inference-Time_Label-Preserving_Target_Projections_CVPR_2021_paper.pdf)] [[Code](https://github.com/yys-Polaris/InferenceTimeDG)]
- Adaptive Methods for Real-World Domain Generalization [[CVPR 2021](http://openaccess.thecvf.com/content/CVPR2021/papers/Dubey_Adaptive_Methods_for_Real-World_Domain_Generalization_CVPR_2021_paper.pdf)] [[Code](https://github.com/abhimanyudubey/GeoYFCC)]
- Test-Time Classifier Adjustment Module for Model-Agnostic Domain Generalization [[NeurIPS 2021](https://proceedings.neurips.cc/paper/2021/file/1415fe9fea0fa1e45dddcff5682239a0-Paper.pdf)] [[Code](https://github.com/matsuolab/T3A)]

### Neural Architecture Search-based Methods
> Neural architecture search-based methods aim to dynamically tune the network architecture to improve out-of-domain generalization.  

- NAS-OoD Neural Architecture Search for Out-of-Distribution Generalization [[ICCV 2021](https://openaccess.thecvf.com/content/ICCV2021/papers/Bai_NAS-OoD_Neural_Architecture_Search_for_Out-of-Distribution_Generalization_ICCV_2021_paper.pdf)]


## Single Domain Generalization
> The goal of single domain generalization task is to improve model performance on unknown target domains by using data from only one source domain. 

- Learning to Learn Single Domain Generalization [[CVPR 2020](https://openaccess.thecvf.com/content_CVPR_2020/papers/Qiao_Learning_to_Learn_Single_Domain_Generalization_CVPR_2020_paper.pdf)] [[Code](https://github.com/joffery/M-ADA)]
- Out-of-domain Generalization from a Single Source: A Uncertainty Quantification Approach [[arXiv 2021](https://arxiv.53yu.com/pdf/2108.02888)]
- Uncertainty-guided Model Generalization to Unseen Domains [[CVPR 2021](http://openaccess.thecvf.com/content/CVPR2021/papers/Qiao_Uncertainty-Guided_Model_Generalization_to_Unseen_Domains_CVPR_2021_paper.pdf)] [[Code](https://github.com/joffery/UMGUD)]
- Adversarially Adaptive Normalization for Single Domain Generalization [[CVPR 2021](https://openaccess.thecvf.com/content/CVPR2021/papers/Fan_Adversarially_Adaptive_Normalization_for_Single_Domain_Generalization_CVPR_2021_paper.pdf)]
- Progressive Domain Expansion Network for Single Domain Generalization [[CVPR 2021](http://openaccess.thecvf.com/content/CVPR2021/papers/Li_Progressive_Domain_Expansion_Network_for_Single_Domain_Generalization_CVPR_2021_paper.pdf)] [[Code](https://github.com/lileicv/PDEN)]
- Learning To Diversify for Single Domain Generalization [[ICCV 2021](https://openaccess.thecvf.com/content/ICCV2021/papers/Wang_Learning_To_Diversify_for_Single_Domain_Generalization_ICCV_2021_paper.pdf)] [[Code](https://github.com/BUserName/Learning)]

## Semi/Weak/Un-Supervised Domain Generalization
> Semi/weak-supervised domain generalization assumes that a part of the source data is unlabeled, while unsupervised domain generalization assumes no training supervision.

- Visual recognition by learning from web data: A weakly supervised domain generalization approach [[CVPR 2015](https://openaccess.thecvf.com/content_cvpr_2015/papers/Niu_Visual_Recognition_by_2015_CVPR_paper.pdf)]
- Visual Recognition by Learning From Web Data via Weakly Supervised Domain Generalization [[TNNLS 2017](https://bcmi.sjtu.edu.cn/home/niuli/paper/Visual%20Recognition%20by%20Learning%20From%20Web%20Data%20via%20Weakly%20Supervised%20Domain%20Generalization.pdf)]
- Domain Generalization via Semi-supervised Meta Learning [[arXiv 2020](https://arxiv.org/pdf/2009.12658)] [[Code](https://github.com/hosseinshn/DGSML)]
- Deep Semi-supervised Domain Generalization Network for Rotary Machinery Fault Diagnosis under Variable Speed [[IEEE Transactions on Instrumentation and Measurement 2020](https://www.researchgate.net/profile/Yixiao-Liao/publication/341199775_Deep_Semisupervised_Domain_Generalization_Network_for_Rotary_Machinery_Fault_Diagnosis_Under_Variable_Speed/links/613f088201846e45ef450a0a/Deep-Semisupervised-Domain-Generalization-Network-for-Rotary-Machinery-Fault-Diagnosis-Under-Variable-Speed.pdf)]
- Semi-Supervised Domain Generalization with Stochastic StyleMatch [[arXiv 2021](https://arxiv.53yu.com/pdf/2106.00592)] [[Code](https://github.com/KaiyangZhou/ssdg-benchmark)]
- Better Pseudo-label Joint Domain-aware Label and Dual-classifier for Semi-supervised Domain Generalization [[arXiv 2021](https://arxiv.53yu.com/pdf/2110.04820)]
- Semi-Supervised Domain Generalization in RealWorld: New Benchmark and Strong Baseline [[arXiv 2021](https://arxiv.org/pdf/2111.10221)]
- Unsupervised Domain Generalization by Learning a Bridge Across Domains [[arXiv 2021](https://arxiv.org/pdf/2112.02300)]
- Domain-Specific Bias Filtering for Single Labeled Domain Generalization [[arXiv 2021](https://arxiv.org/pdf/2110.00726)] [[Code](https://github.com/junkunyuan/DSBF)]


## Open/Heterogeneous Domain Generalization
> Open/heterogeneous domain generalization assumes the label space of one domain is different from that of another domain.

- Feature-Critic Networks for Heterogeneous Domain Generalisation [[ICML 2019](http://proceedings.mlr.press/v97/li19l/li19l.pdf)] [[Code](https://github.com/liyiying/Feature_Critic)]
- Episodic Training for Domain Generalization [[ICCV 2019](https://openaccess.thecvf.com/content_ICCV_2019/papers/Li_Episodic_Training_for_Domain_Generalization_ICCV_2019_paper.pdf)] [[Code](https://github.com/HAHA-DL/Episodic-DG)]
- Towards Recognizing Unseen Categories in Unseen Domains [[ECCV 2020](https://arxiv.53yu.com/pdf/2007.12256.pdf?ref=https://githubhelp.com)] [[Code](https://github.com/mancinimassimiliano/CuMix)]
- Heterogeneous Domain Generalization via Domain Mixup [[ICASSP 2020](https://arxiv.org/pdf/2009.05448)] [[Code](https://github.com/wyf0912/MIXALL)]
- Open Domain Generalization with Domain-Augmented Meta-Learning [[CVPR 2021](http://openaccess.thecvf.com/content/CVPR2021/papers/Shu_Open_Domain_Generalization_with_Domain-Augmented_Meta-Learning_CVPR_2021_paper.pdf)] [[Code](https://github.com/thuml/OpenDG-DAML)]
- Universal Cross-Domain Retrieval Generalizing Across Classes and Domains [[ICCV 2021](https://openaccess.thecvf.com/content/ICCV2021/papers/Paul_Universal_Cross-Domain_Retrieval_Generalizing_Across_Classes_and_Domains_ICCV_2021_paper.pdf)] [[Code](https://github.com/mvp18/UCDR)]


## Federated Domain Generalization
> Federated domain generalization assumes that source data is distributed and can not be fused for data privacy protection. 

- Collaborative Semantic Aggregation and Calibration for Separated Domain Generalization [[arXiv 2021](https://arxiv.org/pdf/2110.06736)] [[Code](https://github.com/junkunyuan/CSAC)]
- FedDG: Federated Domain Generalization on Medical Image Segmentation via Episodic Learning in Continuous Frequency Space [[CVPR 2021](http://openaccess.thecvf.com/content/CVPR2021/papers/Liu_FedDG_Federated_Domain_Generalization_on_Medical_Image_Segmentation_via_Episodic_CVPR_2021_paper.pdf)] [[Code](https://github.com/liuquande/FedDG-ELCFS)]
- Collaborative Optimization and Aggregation for Decentralized Domain Generalization and Adaptation [[ICCV 2021](https://openaccess.thecvf.com/content/ICCV2021/papers/Wu_Collaborative_Optimization_and_Aggregation_for_Decentralized_Domain_Generalization_and_Adaptation_ICCV_2021_paper.pdf)]

## Applications
### Person Re-Identification
- Deep Domain-Adversarial Image Generation for Domain Generalisation [[AAAI 2020](https://ojs.aaai.org/index.php/AAAI/article/download/7003/6857)] [[Code](https://github.com/KaiyangZhou/Dassl.pytorch)]
- Learning to Generate Novel Domains for Domain Generalization [[ECCV 2020](https://arxiv.org/pdf/2007.03304)] [[Code](https://github.com/mousecpn/L2A-OT)]
- Learning Generalisable Omni-Scale Representations for Person Re-Identification [[TPAMI 2021](https://arxiv.org/pdf/1910.06827)] [[Code](https://github.com/KaiyangZhou/deep-person-reid)]
- Multi-Domain Adversarial Feature Generalization for Person Re-Identification [[TIP 2021](https://ieeexplore.ieee.org/iel7/83/9263394/09311771.pdf)]
- Domain Generalization with Mixstyle [[ICLR 2021](https://arxiv.53yu.com/pdf/2104.02008)] [[Code](https://github.com/KaiyangZhou/mixstyle-release)]
- Learning to Generalize Unseen Domains via Memory-based Multi-Source Meta-Learning for Person Re-Identification [[CVPR 2021](https://openaccess.thecvf.com/content/CVPR2021/papers/Zhao_Learning_to_Generalize_Unseen_Domains_via_Memory-based_Multi-Source_Meta-Learning_for_CVPR_2021_paper.pdf)] [[Code](https://github.com/HeliosZhao/M3L)]
- Meta Batch-Instance Normalization for Generalizable Person Re-Identification [[CVPR 2021](https://openaccess.thecvf.com/content/CVPR2021/papers/Choi_Meta_Batch-Instance_Normalization_for_Generalizable_Person_Re-Identification_CVPR_2021_paper.pdf)] [[Code](https://github.com/bismex/MetaBIN)]
- Generalizable Person Re-identification with Relevance-aware Mixture of Experts [[CVPR 2021](https://openaccess.thecvf.com/content/CVPR2021/papers/Dai_Generalizable_Person_Re-Identification_With_Relevance-Aware_Mixture_of_Experts_CVPR_2021_paper.pdf)]
- TransMatcher: Deep Image Matching Through Transformers for Generalizable Person Re-identification [[NeurIPS 2021](https://proceedings.neurips.cc/paper/2021/file/0f49c89d1e7298bb9930789c8ed59d48-Paper.pdf)] [[Code](https://github.com/ShengcaiLiao/QAConv)]

### Face Recognition & Anti-Spoofing
- Multi-Adversarial Discriminative Deep Domain Generalization for Face Presentation Attack Detection [[CVPR 2019](http://openaccess.thecvf.com/content_CVPR_2019/papers/Shao_Multi-Adversarial_Discriminative_Deep_Domain_Generalization_for_Face_Presentation_Attack_Detection_CVPR_2019_paper.pdf)] [[Code](https://github.com/rshaojimmy/CVPR2019-MADDoG)]
- Towards Universal Representation Learning for Deep Face Recognition [[CVPR 2020](http://openaccess.thecvf.com/content_CVPR_2020/papers/Shi_Towards_Universal_Representation_Learning_for_Deep_Face_Recognition_CVPR_2020_paper.pdf)] [[Code](https://github.com/MatyushinMA/uni_rep_deep_faces)]
- Cross-Domain Face Presentation Attack Detection via Multi-Domain Disentangled Representation Learning [[CVPR 2020](http://openaccess.thecvf.com/content_CVPR_2020/papers/Wang_Cross-Domain_Face_Presentation_Attack_Detection_via_Multi-Domain_Disentangled_Representation_Learning_CVPR_2020_paper.pdf)]
- Single-Side Domain Generalization for Face Anti-Spoofing [[CVPR 2020](http://openaccess.thecvf.com/content_CVPR_2020/papers/Jia_Single-Side_Domain_Generalization_for_Face_Anti-Spoofing_CVPR_2020_paper.pdf)] [[Code](https://github.com/taylover-pei/SSDG-CVPR2020)]


## Related Topics
### Life-Long Learning
- Sequential Learning for Domain Generalization [[ECCV workshop 2020](https://arxiv.org/pdf/2004.01377)]
- Continual Adaptation of Visual Representations via Domain Randomization and Meta-learning [[CVPR 2021](http://openaccess.thecvf.com/content/CVPR2021/papers/Volpi_Continual_Adaptation_of_Visual_Representations_via_Domain_Randomization_and_Meta-Learning_CVPR_2021_paper.pdf)]

# Datasets
> Evaluations on the following datasets often follow leave-one-domain-out protocol: randomly choose one domain to hold out as the target domain, while the others are used as the  source domain(s).

| Datasets (download link) | Description | Related papers in [Paper Index](#paper-index) |
| :---- | :----: | :----: | :----: |
| **Colored MNIST** [165] | Handwritten digit recognition; 3 domains: {0.1, 0.3, 0.9}; 70,000 samples of dimension (2, 28, 28); 2 classes | [82], [138], [140], [149], [152], [154], [165], [171], [173], [190], [200], [202] |
| **Rotated MNIST** [6] ([original](https://github.com/Emma0118/mate)) | Handwritten digit recognition; 6 domains with rotated degree: {0, 15, 30, 45, 60, 75}; 7,000 samples of dimension (1, 28, 28); 10 classes | [5], [6], [15], [35], [53], [55], [63], [71], [73], [74], [76], [77], [86], [90], [105], [107], [138], [140], [170], [173], [202], [204], [206] |
| **Digits-DG** [28] | Handwritten digit recognition; 4 domains: {MNIST [29], MNIST-M [30], SVHN [31], SYN [30]}; 24,000 samples; 10 classes | [21], [25], [27], [28], [34], [35], [55], [59], [63], [69], [94], [98], [116], [118], [130], [141], [142], [146], [151], [153], [157], [158], [159], [160], [166], [168], [179], [189], [203] |
| **VLCS** [16] ([1](https://drive.google.com/uc?id=1skwblH1_okBwxWxmRsp9_qi15hyPpxg8); or [original](https://www.mediafire.com/file/7yv132lgn1v267r/vlcs.tar.gz/file)) | Object recognition; 4 domains: {Caltech [8], LabelMe [9], PASCAL [10], SUN [11]}; 10,729 samples of dimension (3, 224, 224); 5 classes; about 3.6 GB | [2], [6], [7], [14], [15], [18], [60], [61], [64], [67], [68], [70], [71], [74], [76], [77], [81], [83], [86], [91], [98], [99], [101], [102], [103], [117], [118], [126], [127], [131], [132], [136], [138], [140], [142], [145], [146], [148], [149], [161], [170], [173], [174], [184], [190], [195], [199], [201], [202], [203] |
| **Office31+Caltech** [32] ([1](https://drive.google.com/file/d/14OIlzWFmi5455AjeBZLak2Ku-cFUrfEo/view)) | Object recognition; 4 domains: {Amazon, Webcam, DSLR, Caltech}; 4,652 samples in 31 classes (office31) or 2,533 samples in 10 classes (office31+caltech); 51 MB | [6], [35], [67], [68], [70], [71], [80], [91], [96], [119], [131], [167] | 
| **OfficeHome** [20] ([1](https://drive.google.com/uc?id=1uY0pj7oFsjMxRwaD3Sxy0jgel0fsYXLC); or [original](https://drive.google.com/file/d/0B81rNlvomiwed0V1YUxQdC1uOTg/view?resourcekey=0-2SNWq0CDAuWOBRRBL7ZZsw)) | Object recognition; 4 domains: {Art, Clipart, Product, Real World}; 15,588 samples of dimension (3, 224, 224); 65 classes; 1.1 GB | [19], [54], [28], [34], [55], [58], [60], [61], [64], [69], [80], [92], [94], [98], [101], [118], [126], [130], [131], [132], [133], [137], [138], [140], [146], [148], [156], [159], [160], [162], [163], [167], [173], [174], [178], [179], [184], [189], [190], [199], [201], [202], [203], [206] |
| **PACS** [2] ([1](https://drive.google.com/uc?id=1JFr8f805nMUelQWWmfnJR3y4_SYoN5Pd); or [original](https://drive.google.com/drive/folders/0B6x7gtvErXgfUU1WcGY5SzdwZVk?resourcekey=0-2fvpQY_QSyJf2uIECzqPuQ)) | Object recognition; 4 domains: {photo, art_painting, cartoon, sketch}; 9,991 samples of dimension (3, 224, 224); 7 classes; 174 MB | [1], [2], [4], [5], [14], [15], [18], [19], [34], [54], [28], [35], [55], [56], [57], [58], [59], [60], [61], [64], [69], [73], [77], [80], [81], [82], [83], [84], [86], [90], [92], [94], [96], [98], [99], [101], [102], [104], [105], [116], [117], [118], [127], [129], [130], [131], [132], [136], [137], [138], [139], [140], [142], [145], [146], [148], [149], [153], [156], [157], [158], [159], [160], [161], [162], [163], [167], [170], [171], [173], [174], [178], [179], [180], [184], [189], [190], [195], [199], [200], [201], [202], [203], [206] |
| **DomainNet** [33] ([clipart](http://csr.bu.edu/ftp/visda/2019/multi-source/groundtruth/clipart.zip), [infograph](http://csr.bu.edu/ftp/visda/2019/multi-source/infograph.zip), [painting](http://csr.bu.edu/ftp/visda/2019/multi-source/groundtruth/painting.zip), [quick-draw](http://csr.bu.edu/ftp/visda/2019/multi-source/quickdraw.zip), [real](http://csr.bu.edu/ftp/visda/2019/multi-source/real.zip), and [sketch](http://csr.bu.edu/ftp/visda/2019/multi-source/sketch.zip); or [original](http://ai.bu.edu/M3SDA/)) | Object recognition; 6 domains: {clipart, infograph, painting, quick-draw, real, sketch}; 586,575 samples of dimension (3, 224, 224); 345 classes; 1.2 GB + 4.0 GB + 3.4 GB + 439 MB + 5.6 GB + 2.5 GB | [34], [57], [104], [119], [130], [131], [132], [133], [138], [140], [150], [173], [178], [189], [201], [202], [203] |
| **mini-DomainNet** [34] | Object recognition; a smaller and less noisy version of DomainNet; 4 domains: {clipart, painting, real, sketch}; 140,006 samples | [34], [69], [130], [156], [157] |
**ImageNet-Sketch** [35] | Object recognition; 2 domains: {real, sketch}; 50,000 samples | [64] |
**VisDA-17** [36] | Object recognition; 3 domains of synthetic-to-real generalization; 280,157 samples | [119], [178] |
**CIFAR-10-C** / **CIFAR-100-C** / **ImageNet-C** [37] ([original](https://github.com/hendrycks/robustness/)) | Object recognition; the test data are damaged by 15 corruptions (each with 5 intensity levels) drawn from 4 categories (noise, blur, weather, and digital); 60,000/60,000/1.3M samples | [27], [74], [116], [141], [151], [168] |
| **Visual Decathlon (VD)** [38] | Object/action/handwritten/digit recognition; 10 domains from the combination of 10 datasets; 1,659,142 samples | [5], [7], [128] |
**IXMAS** [39] | Action recognition; 5 domains with 5 camera views, 10 subjects, and 5 actions; 1,650 samples | [7], [14], [67], [76] |
**SYNTHIA** [42] | Semantic segmentation; 15 domains with 4 locations and 5 weather conditions; 2,700 samples | [27], [62], [115], [141], [151], [185], [193] | 
**GTA5-Cityscapes** [43], [44] | Semantic segmentation; 2 domains of synthetic-to-real generalization; 29,966 samples | [62], [115], [185], [193] |
**Terra Incognita (TerraInc)** [45] ([1](https://lilablobssc.blob.core.windows.net/caltechcameratraps/eccv_18_all_images_sm.tar.gz) and [2](https://lilablobssc.blob.core.windows.net/caltechcameratraps/labels/caltech_camera_traps.json.zip); or [original](https://lila.science/datasets/caltech-camera-traps)) | Animal classification | 4 domains captured at different geographical locations: {L100, L38, L43, L46}; 24,788 samples of dimension (3, 224, 224); 10 classes; 6.0 GB + 8.6 MB | [132], [136], [138], [140], [173], [201], [202], [207] |
**Market-Duke** [46], [47] | Person re-idetification; cross-dataset re-ID; heterogeneous DG with 2 domains; 69,079 samples | [12], [13], [28], [55], [56], [58], [114], [144], [187], [208] |
<!-- **UCF-HMDB** [40], [41] | Action recognition | 2 domains with 12 overlapping actions; 3809 samples |  | -->
<!-- **Face** [22] | >5M | 9 | Face recognition | Combination of 9 face datasets |  |
**COMI** [48], [49], [50], [51] | 8500 | 4 | Face anti-spoofing | Combination of 4 face anti-spoofing datasets |  | -->



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
- Adaptation and Generalization Across Domains in Visual Recognition with Deep Neural Networks [[PhD 2020, Kaiyang Zhou (University of Surrey)](https://openresearch.surrey.ac.uk/esploro/outputs/doctoral/Adaptation-and-Generalization-Across-Domains-in/99513024202346)]



# Paper Index
> We list all the papers for quick check, including method abbreviation, keywords, etc.

[1] Learning to Generalize: Meta-Learning for Domain Generalization [[AAAI 2018](https://www.aaai.org/ocs/index.php/AAAI/AAAI18/paper/viewFile/16067/16547)] [[Code](https://github.com/HAHA-DL/MLDG)] *(MLDG, [meta-learning](#meta-learning-based-methods))*

[2] Deeper, Broader and Artier Domain Generalization [[ICCV 2017](https://openaccess.thecvf.com/content_ICCV_2017/papers/Li_Deeper_Broader_and_ICCV_2017_paper.pdf)] [[Code](https://dali-dl.github.io/project_iccv2017.html)] *([disentangled representation learning](#disentangled-representation-learning-based-methods), [PACS dataset](#datasets))*

[3] Domain Generalization in Vision: A Survey [[arXiv 2021](https://arxiv.org/abs/2103.02503)] *([survey](#survey))*

[4] MetaReg: Towards Domain Generalization using Meta-Regularization [[NeurIPS 2018](https://proceedings.neurips.cc/paper/2018/file/647bba344396e7c8170902bcf2e15551-Paper.pdf)] [[Code*](https://github.com/elliotbeck/MetaReg_PyTorch)] *(MetaReg, [meta-learning](#meta-learning-based-methods), [regularization](#regularization-based-methods))*

[5] Feature-Critic Networks for Heterogeneous Domain Generalisation [[ICML 2019](http://proceedings.mlr.press/v97/li19l/li19l.pdf)] [[Code](https://github.com/liyiying/Feature_Critic)] *(Feature-Critic, [meta-learning](#meta-learning-based-methods), [open/heterogeneous domain generalization](#openheterogeneous-domain-generalization))*

[6] Domain Generalization for Object Recognition with Multi-Task Autoencoders [[ICCV 2015](http://openaccess.thecvf.com/content_iccv_2015/papers/Ghifary_Domain_Generalization_for_ICCV_2015_paper.pdf)] [[Code](https://github.com/Emma0118/mate)] *(MTAE, [self-supervised learning](#self-supervised-learning-based-methods), [Rotated MNIST dataset](#datasets))*

[7] Episodic Training for Domain Generalization [[ICCV 2019](https://openaccess.thecvf.com/content_ICCV_2019/papers/Li_Episodic_Training_for_Domain_Generalization_ICCV_2019_paper.pdf)] [[Code](https://github.com/HAHA-DL/Episodic-DG)] *(Epi-FCR, [meta-learning](#meta-learning-based-methods), [open/heterogeneous domain generalization](#openheterogeneous-domain-generalization))*

[8] Learning Generative Visual Models from Few Training Examples: An Incremental Bayesian Approach Tested on 101 Object Categories [[CVPR workshop 2004](http://www.vision.caltech.edu/publications/Fei-FeiCompVIsImageU2007.pdf)] *([Caltech dataset](#datasets))*

[9] Labelme: A Database and Web-Based Tool for Image Annotation [[IJCV 2008](https://idp.springer.com/authorize/casa?redirect_uri=https://link.springer.com/content/pdf/10.1007/s11263-007-0090-8.pdf&casa_token=n3w4Sen-huAAAAAA:sJY2dHreDGe2V4KE9jDehftM1W-Sn1z8bqeF_WK8Q9t4B0dFk5OXEAlIP7VYnr8UfiWLAOPG7dK0ZveYWs8)] *([LabelMe dataset](#datasets))*

[10] The pascal visual object classes (voc) challenge [[IJCV 2010](https://idp.springer.com/authorize/casa?redirect_uri=https://link.springer.com/content/pdf/10.1007/s11263-009-0275-4.pdf&casa_token=Zb6LfMuhy_sAAAAA:Sqk_aoTWdXx37FQjUFaZN9ZMQxrUhqO2S_HbOO2a9BKtejW7CMekg-3PDVw6Yjw7BZqihyjP0D_Y6H2msBo)] *([PASCAL dataset](#datasets))*

[11] Sun Database: Large-Scale Scene Recognition from Abbey to Zoo [[CVPR 2010](https://dspace.mit.edu/bitstream/handle/1721.1/60690/Oliva_SUN%20database.pdf?sequence=1&isAllowed=y)] *([Sun dataset](#datasets))*

[12] Learning to Generalize Unseen Domains via Memory-based Multi-Source Meta-Learning for Person Re-Identification [[CVPR 2021](https://openaccess.thecvf.com/content/CVPR2021/papers/Zhao_Learning_to_Generalize_Unseen_Domains_via_Memory-based_Multi-Source_Meta-Learning_for_CVPR_2021_paper.pdf)] [[Code](https://github.com/HeliosZhao/M3L)] *(M3L, [meta-learning](#meta-learning-based-methods), [normalization](#normalization-based-methods), [person re-identification](#person-re-identification))*

[13] Meta Batch-Instance Normalization for Generalizable Person Re-Identification [[CVPR 2021](https://openaccess.thecvf.com/content/CVPR2021/papers/Choi_Meta_Batch-Instance_Normalization_for_Generalizable_Person_Re-Identification_CVPR_2021_paper.pdf)] [[Code](https://github.com/bismex/MetaBIN)] *(MetaBIN, [meta-learning](#meta-learning-based-methods), [normalization](#normalization-based-methods), [person re-identification](#person-re-identification))*

[14] Sequential Learning for Domain Generalization [[ECCV workshop 2020](https://arxiv.org/pdf/2004.01377)] *(S-MLDG, [meta-learning](#meta-learning-based-methods), [life-long learning](#life-long-learning))*

[15] Learning to Learn with Variational Information Bottleneck for Domain Generalization [[ECCV 2020](https://arxiv.org/pdf/2007.07645)] *(MetaVIB, [meta-learning](#meta-learning-based-methods), [information](#information-based-methods))*

[16] Unbiased Metric Learning: On the Utilization of Multiple Datasets and Web Images for Softening Bias [[ICCV 2013](http://openaccess.thecvf.com/content_iccv_2013/papers/Fang_Unbiased_Metric_Learning_2013_ICCV_paper.pdf)] *([VLCS dataset](#datasets))*

[17] Shape-Aware Meta-Learning for Generalizing Prostate MRI Segmentation to Unseen Domains [[MICCAI 2020](https://arxiv.org/pdf/2007.02035)] [[Code](https://github.com/liuquande/SAML)] *(SAML, [meta-learning](#meta-learning-based-methods))*

[18] Domain Generalization via Model-Agnostic Learning of Semantic Features [[NeurIPS 2019](https://proceedings.neurips.cc/paper/2019/file/2974788b53f73e7950e8aa49f3a306db-Paper.pdf)] [[Code](https://github.com/biomedia-mira/masf)] *(MASF, [domain alignment](#domain-alignment-based-methods), [meta-learning](#meta-learning-based-methods))*

[19] MetaNorm: Learning to Normalize Few-Shot Batches Across Domains [[ICLR 2021](https://openreview.net/pdf?id=9z_dNsC4B5t)] [[Code](https://github.com/YDU-AI/MetaNorm)] *(MetaNorm, [meta-learning](#meta-learning-based-methods), [normalization](#normalization-based-methods))*

[20] Deep Hashing Network for Unsupervised Domain Adaptation [[CVPR 2017](http://openaccess.thecvf.com/content_cvpr_2017/papers/Venkateswara_Deep_Hashing_Network_CVPR_2017_paper.pdf)] [[Code](https://github.com/hemanthdv/da-hash)] *([OfficeHome dataset](#datasets))*

[21] Addressing Model Vulnerability to Distributional Shifts Over Image Transformation Sets [[ICCV 2019](https://openaccess.thecvf.com/content_ICCV_2019/papers/Volpi_Addressing_Model_Vulnerability_to_Distributional_Shifts_Over_Image_Transformation_Sets_ICCV_2019_paper.pdf)] [[Code](https://github.com/ricvolpi/domain-shift-robustness)] *([data augmentation](#data-augmentation-based-methods))*

[22] Towards Universal Representation Learning for Deep Face Recognition [[CVPR 2020](http://openaccess.thecvf.com/content_CVPR_2020/papers/Shi_Towards_Universal_Representation_Learning_for_Deep_Face_Recognition_CVPR_2020_paper.pdf)] [[Code](https://github.com/MatyushinMA/uni_rep_deep_faces)] *([data augmentation](#data-augmentation-based-methods), [face recognition & anti-spoofing](#face-recognition--anti-spoofing))*

[23] Generalizing Deep Learning for Medical Image Segmentation to Unseen Domains via Deep Stacked Transformation [[TMI 2020](https://www.ncbi.nlm.nih.gov/pmc/articles/pmc7393676/)] *(BigAug, [data augmentation](#data-augmentation-based-methods))*

[24] Improving the Generalizability of Convolutional Neural Network-Based Segmentation on CMR Images [[Frontiers in Cardiovascular Medicine 2020](https://www.frontiersin.org/articles/10.3389/fcvm.2020.00105/full)] *([data augmentation](#data-augmentation-based-methods))*

[25] Generalizing to Unseen Domains via Adversarial Data Augmentation [[NeurIPS 2018](https://proceedings.neurips.cc/paper/2018/file/1d94108e907bb8311d8802b48fd54b4a-Paper.pdf)] [[Code](https://github.com/ricvolpi/generalize-unseen-domains)] *([data augmentation](#data-augmentation-based-methods))*

[26] Staining Invariant Features for Improving Generalization of Deep Convolutional Neural Networks in Computational Pathology [[Frontiers in Bioengineering and Biotechnology 2019](https://www.frontiersin.org/articles/10.3389/fbioe.2019.00198/full?&utm_source=Email_to_authors_&utm_medium=Email&utm_content=T1_11.5e1_author&utm_campaign=Email_publication&field=&journalName=Frontiers_in_Bioengineering_and_Biotechnology&id=474781)] *([data augmentation](#data-augmentation-based-methods))*

[27] Learning to Learn Single Domain Generalization [[CVPR 2020](https://openaccess.thecvf.com/content_CVPR_2020/papers/Qiao_Learning_to_Learn_Single_Domain_Generalization_CVPR_2020_paper.pdf)] [[Code](https://github.com/joffery/M-ADA)] (M-ADA, [data augmentation](#data-augmentation-based-methods), [meta-learning](#meta-learning-based-methods), [single domain generalization](#single-domain-generalization))

[28] Learning to Generate Novel Domains for Domain Generalization [[ECCV 2020](https://arxiv.org/pdf/2007.03304)] [[Code](https://github.com/mousecpn/L2A-OT)] *(L2A-OT, [data augmentation](#data-augmentation-based-methods), [person re-identification](#person-re-identification), [Digits-DG dataset](#datasets))*

[29] Gradient-Based Learning Applied to Document Recognition [[IEEE 1998](http://lushuangning.oss-cn-beijing.aliyuncs.com/CNN%E5%AD%A6%E4%B9%A0%E7%B3%BB%E5%88%97/Gradient-Based_Learning_Applied_to_Document_Recognition.pdf)] *([MNIST dataset](#datasets))*

[30] Unsupervised Domain Adaptation by Backpropagation [[ICML 2015](http://proceedings.mlr.press/v37/ganin15.pdf)] *([MNIST-M dataset](#datasets))*

[31] Reading Digits in Natural Images with Unsupervised Feature Learning [[NeurIPS workshop 2011](https://research.google/pubs/pub37648.pdf)] *([SVHN dataset](#datasets))*

[32] Adapting Visual Category Models to New Domains [[ECCV 2010](https://linkspringer.53yu.com/content/pdf/10.1007/978-3-642-15561-1_16.pdf)] *([Office31 dataset](#datasets))*

[33] Moment Matching for Multi-Source Domain Adaptation [[ICCV 2019](https://openaccess.thecvf.com/content_ICCV_2019/papers/Peng_Moment_Matching_for_Multi-Source_Domain_Adaptation_ICCV_2019_paper.pdf)] *([DomainNet dataset](#datasets))*

[34] Domain Adaptive Ensemble Learning [[TIP 2021](https://arxiv.53yu.com/pdf/2003.07325)] [[Code](https://github.com/KaiyangZhou/Dassl.pytorch)] *([ensemble learning](#ensemble-learning-based-methods), [mini-DomainNet dataset](#datasets))*

[35] Learning Robust Representations by Projecting Superficial Statistics Out [[ICLR 2019](https://arxiv.53yu.com/pdf/1903.06256)] [[Code](https://github.com/HaohanWang/HEX)] *(HEX, [ImageNet-Sketch dataset](#datasets), [regularization](#regularization-based-methods))*

[36] Visda: The visual domain adaptation challenge [[arXiv 2017](https://arxiv.53yu.com/pdf/1710.06924)] *([Visda dataset](#datasets))*

[37] Benchmarking Neural Network Robustness to Common Corruptions and Perturbations [[ICLR 2019](https://arxiv.53yu.com/pdf/1903.12261.pdf?ref=https://githubhelp.com)] *([CIFAR-10-C, CIFAR-100-C, ImageNet-C datasets](#datasets))*

[38] Learning Multiple Visual Domains with Residual Adapters [[NeurIPS 2017](https://proceedings.neurips.cc/paper/2017/file/e7b24b112a44fdd9ee93bdf998c6ca0e-Paper.pdf)] *([Visual Decathlon dataset](#datasets))*

[39] Free Viewpoint Action Recognition Using Mmotion History Volumes [[CVIU 2006](https://hal.inria.fr/docs/00/54/46/29/PDF/cviu_motion_history_volumes.pdf)] *([IXMAS dataset](#datasets))*

[40] Ucf101: A dataset of 101 Human Actions Classes from Videos in the Wild [[arXiv 2012](https://arxiv.53yu.com/pdf/1212.0402.pdf?ref=https://githubhelp.com)] *([UCF-HMDB dataset](#datasets))*

[41] Hmdb: Large Video Database for Human Motion Recognition [[ICCV 2011](https://dspace.mit.edu/bitstream/handle/1721.1/69981/Poggio-HMDB.pdf?sequence=1&isAllowed=y)] *([UCF-HMDB dataset](#datasets))*

[42] The Synthia Dataset: A Large Collection of Synthetic Images for Semantic Segmentation of Urban Scenes [[CVPR 2016](https://www.cv-foundation.org/openaccess/content_cvpr_2016/papers/Ros_The_SYNTHIA_Dataset_CVPR_2016_paper.pdf)] *([SYNTHIA dataset](#datasets))*

[43] Playing for Data: Ground Truth from Computer Games [[ECCV 2016](https://linkspringer.53yu.com/chapter/10.1007/978-3-319-46475-6_7)] *([GTA5-Cityscapes dataset](#datasets))*

[44] The Cityscapes Dataset for Semantic Urban Scene Understanding [[CVPR 2016](http://openaccess.thecvf.com/content_cvpr_2016/papers/Cordts_The_Cityscapes_Dataset_CVPR_2016_paper.pdf)] *([GTA5-Cityscapes dataset](#datasets))*

[45] Recognition in Terra Incognita [[ECCV 2018](http://openaccess.thecvf.com/content_ECCV_2018/papers/Beery_Recognition_in_Terra_ECCV_2018_paper.pdf)] *([TerraInc dataset](#datasets))*

[46] Scalable Person Re-Identification: A Benchmark [[ICCV 2015](https://www.cv-foundation.org/openaccess/content_iccv_2015/papers/Zheng_Scalable_Person_Re-Identification_ICCV_2015_paper.pdf)] *([Market-Duke dataset](#datasets))*

[47] Performance Measures and a Data Set for Multi-target, Multi-Camera Tracking [[ECCV 2016](https://linkspringer.53yu.com/chapter/10.1007/978-3-319-48881-3_2)] *([Market-Duke dataset](#datasets))*

[48] A Face Antispoofing Database with Diverse Attacks [[ICB 2012](http://www.cbsr.ia.ac.cn/users/jjyan/zhang-icb2012.pdf)] *([COMI dataset](#datasets))*

[49] Oulu-npu: A Mobile Face Presentation Attack Database with Realworld Variations [FG 2017] *([COMI dataset](#datasets))*

[50] Face Spoof Detection with Image Distortion Analysis [TIFS 2015] *([COMI dataset](#datasets))*

[51] On the Effectiveness of Local Binary Patterns in Face Anti-Spoofing [[BIOSIG 2012](https://dl.gi.de/bitstream/handle/20.500.12116/18295/183.pdf?sequence=1)] *([COMI dataset](#datasets))*

[52] Certifying Some Distributional Robustness with Principled Adversarial Training [[arXiv 2017](https://arxiv.53yu.com/pdf/1710.10571.pdf])] [[Code](https://github.com/duchi-lab/certifiable-distributional-robustness)] *([data augmentation](#data-augmentation-based-methods))*

[53] Generalizing across Domains via Cross-Gradient Training [[ICLR 2018](https://arxiv.53yu.com/pdf/1804.10745)] [[Code](https://github.com/vihari/crossgrad)] *(CrossGrad, [data augmentation](#data-augmentation-based-methods))*

[54] Semi-Supervised Domain Generalization with Stochastic StyleMatch [[arXiv 2021](https://arxiv.53yu.com/pdf/2106.00592)] [[Code](https://github.com/KaiyangZhou/ssdg-benchmark)] *(StyleMatch, [data augmentation](#data-augmentation-based-methods), [semi/weak/un-supervised domain generalization](#semiweakun-supervised-domain-generalization))*

[55] Deep Domain-Adversarial Image Generation for Domain Generalisation [[AAAI 2020](https://ojs.aaai.org/index.php/AAAI/article/download/7003/6857)] [[Code](https://github.com/KaiyangZhou/Dassl.pytorch)] *(DDAIG, [data augmentation](#data-augmentation-based-methods), [person re-identification](#person-re-identification))*

[56] Domain Generalization with Mixstyle [[ICLR 2021](https://arxiv.53yu.com/pdf/2104.02008)] [[Code](https://github.com/KaiyangZhou/mixstyle-release)] *(MixStyle, [data augmentation](#data-augmentation-based-methods), [person re-identification](#person-re-identification))*

[57] Towards Recognizing Unseen Categories in Unseen Domains [[ECCV 2020](https://arxiv.53yu.com/pdf/2007.12256.pdf?ref=https://githubhelp.com)] [[Code](https://github.com/mancinimassimiliano/CuMix)] *(CuMix, [data augmentation](#data-augmentation-based-methods), [open/heterogeneous domain generalization](#openheterogeneous-domain-generalization))*

[58] MixStyle Neural Networks for Domain Generalization and Adaptation [[arXiv 2021](https://arxiv.53yu.com/pdf/2107.02053)] [[Code](https://github.com/KaiyangZhou/mixstyle-release)] *(MixStyle, [data augmentation](#data-augmentation-based-methods))*

[59] Robust and Generalizable Visual Representation Learning via Random Convolutions [[ICLR 2021](https://arxiv.53yu.com/pdf/2007.13003)] [[Code](https://github.com/wildphoton/RandConv)] *(RC, [data augmentation](#data-augmentation-based-methods))*

[60] Frustratingly Simple Domain Generalization via Image Stylization [[arXiv 2020](https://arxiv.53yu.com/pdf/2006.11207)] [[Code](https://github.com/GT-RIPL/DomainGeneralization-Stylization)] *([data augmentation](#data-augmentation-based-methods))*

[61] Rethinking Domain Generalization Baselines [[ICPR 2020](https://arxiv.53yu.com/pdf/2101.09060)] *([data augmentation](#data-augmentation-based-methods))*

[62] Domain Randomization and Pyramid Consistency: Simulation-to-Real Generalization Without Accessing Target Domain Data [[ICCV 2019](http://openaccess.thecvf.com/content_ICCV_2019/papers/Yue_Domain_Randomization_and_Pyramid_Consistency_Simulation-to-Real_Generalization_Without_Accessing_Target_ICCV_2019_paper.pdf)] [[Code](https://github.com/xyyue/DRPC)] *([data augmentation](#data-augmentation-based-methods))*

[63] Hallucinating Agnostic Images to Generalize Across Domains [[ICCV workshop 2019](https://arxiv.53yu.com/pdf/1808.01102)] [[Code](https://github.com/fmcarlucci/ADAGE)] *([data augmentation](#data-augmentation-based-methods))*

[64] Self-challenging Improves Cross-Domain Generalization [[ECCV 2020](https://arxiv.53yu.com/pdf/2007.02454)] [[Code](https://github.com/DeLightCMU/RSC)] *(RSC, [regularization](#regularization-based-methods))*

[65] Domain Generalization via Invariant Feature Representation [[ICML 2013](http://proceedings.mlr.press/v28/muandet13.pdf)] [[Code](https://github.com/krikamol/dg-dica)] *(DICA, [domain alignment](#domain-alignment-based-methods))*

[66] Robust Domain Generalisation by Enforcing Distribution Invariance [[IJCAI 2016](https://eprints.qut.edu.au/115382/15/Erfani2016IJCAI.pdf)] *(ESRand, [domain alignment](#domain-alignment-based-methods))*

[67] Scatter Component Analysis A Unified Framework for Domain Adaptation and Domain Generalization [[TPAMI 2017](https://arxiv.53yu.com/pdf/1510.04373)] *(SCA, [domain alignment](#domain-alignment-based-methods))*

[68] Domain Generalization via Conditional Invariant Representation [[AAAI 2018](https://ojs.aaai.org/index.php/AAAI/article/view/11682/11541)] *(CIDG, [domain alignment](#domain-alignment-based-methods))*

[69] Feature alignment and restoration for domain generalization and adaptation [[arXiv 2020](https://arxiv.53yu.com/pdf/2006.12009)] *(FAR, [domain alignment](#domain-alignment-based-methods))*

[70] Domain Generalization via Multidomain Discriminant Analysis [[UAI 2020](http://proceedings.mlr.press/v115/hu20a/hu20a.pdf)] [[Code](https://github.com/amber0309/Multidomain-Discriminant-Analysis)] *(MDA, [domain alignment](#domain-alignment-based-methods))*

[71] Unified Deep Supervised Domain Adaptation and Generalization [[ICCV 2017](https://openaccess.thecvf.com/content_ICCV_2017/papers/Motiian_Unified_Deep_Supervised_ICCV_2017_paper.pdf)] [[Code](https://github.com/samotiian/CCSA)] *(CCSA, [domain alignment](#domain-alignment-based-methods))*

[72] Generalizable Feature Learning in the Presence of Data Bias and Domain Class Imbalance with Application to Skin Lesion Classification [[MICCAI 2019](https://www.cs.sfu.ca/~hamarneh/ecopy/miccai2019d.pdf)] *([domain alignment](#domain-alignment-based-methods))*

[73] Domain Generalization using Causal Matching [[ICML 2021](http://proceedings.mlr.press/v139/mahajan21b/mahajan21b.pdf)] [[Code](https://github.com/microsoft/robustdg)] *(MatchDG, [domain alignment](#domain-alignment-based-methods), [causality](#causality-based-methods))*

[74] Respecting Domain Relations: Hypothesis Invariance for Domain Generalization [[ICPR 2020](https://arxiv.53yu.com/pdf/2010.07591)] *(HIR, [domain alignment](#domain-alignment-based-methods))*

[75] Domain Generalization for Medical Imaging Classification with Linear-Dependency Regularization [[NeurIPS 2020](https://proceedings.neurips.cc/paper/2020/file/201d7288b4c18a679e48b31c72c30ded-Paper.pdf)] [[Code](https://github.com/wyf0912/LDDG)] *(LDDG, [domain alignment](#domain-alignment-based-methods))*

[76] Domain Generalization with Adversarial Feature Learning [[CVPR 2018](https://openaccess.thecvf.com/content_cvpr_2018/papers/Li_Domain_Generalization_With_CVPR_2018_paper.pdf)] [[Code](https://github.com/YuqiCui/MMD_AAE)] *(MMD-AAE, [domain alignment](#domain-alignment-based-methods))*

[77] Deep Domain Generalization via Conditional Invariant Adversarial Networks [[ECCV 2018](http://openaccess.thecvf.com/content_ECCV_2018/papers/Ya_Li_Deep_Domain_Generalization_ECCV_2018_paper.pdf)] *(CIDDG, [domain alignment](#domain-alignment-based-methods))*

[78] Multi-Adversarial Discriminative Deep Domain Generalization for Face Presentation Attack Detection [[CVPR 2019](http://openaccess.thecvf.com/content_CVPR_2019/papers/Shao_Multi-Adversarial_Discriminative_Deep_Domain_Generalization_for_Face_Presentation_Attack_Detection_CVPR_2019_paper.pdf)] [[Code](https://github.com/rshaojimmy/CVPR2019-MADDoG)] *(MADDG, [domain alignment](#domain-alignment-based-methods), [face recognition & anti-spoofing](#face-recognition--anti-spoofing))*

[79] Single-Side Domain Generalization for Face Anti-Spoofing [[CVPR 2020](http://openaccess.thecvf.com/content_CVPR_2020/papers/Jia_Single-Side_Domain_Generalization_for_Face_Anti-Spoofing_CVPR_2020_paper.pdf)] [[Code](https://github.com/taylover-pei/SSDG-CVPR2020)] *(SSDG, [domain alignment](#domain-alignment-based-methods), [face recognition & anti-spoofing](#face-recognition--anti-spoofing))*

[80] Correlation-aware Adversarial Domain Adaptation and Generalization [[PR 2020](https://arxiv.53yu.com/pdf/1911.12983)] [[Code](https://github.com/mahfujur1/CA-DA-DG)] *(CAADA, [domain alignment](#domain-alignment-based-methods))*

[81] Generalizing to Unseen Domains via Distribution Matching [[arXiv 2019](https://arxiv.53yu.com/pdf/1911.00804)] [[Code](https://github.com/belaalb/G2DM)] *(G2DM, [domain alignment](#domain-alignment-based-methods))*

[82] Representation via Representations: Domain Generalization via Adversarially Learned Invariant Representations [[arXiv 2020](https://arxiv.53yu.com/pdf/2006.11478)] *(RVR, [domain alignment](#domain-alignment-based-methods))*

[83] Domain Generalization Using a Mixture of Multiple Latent Domains [[AAAI 2020](https://ojs.aaai.org/index.php/AAAI/article/view/6846/6700)] [[Code](https://github.com/mil-tokyo/dg_mmld)] *([domain alignment](#domain-alignment-based-methods))*

[84] Adversarial Invariant Feature Learning with Accuracy Constraint for Domain Generalization [[ECMLPKDD 2019](https://arxiv.53yu.com/pdf/1904.12543)] [[Code](https://github.com/akuzeee/AFLAC)] *(AFLAC, [domain alignment](#domain-alignment-based-methods))*

[85] Scanner Invariant Multiple Sclerosis Lesion Segmentation from MRI [[ISBI 2020](https://arxiv.53yu.com/pdf/1910.10035)] *([domain alignment](#domain-alignment-based-methods))*

[86] Domain Generalization via Entropy Regularization [[NeurIPS 2020](https://proceedings.neurips.cc/paper/2020/file/b98249b38337c5088bbc660d8f872d6a-Paper.pdf)] [[Code](https://github.com/sshan-zhao/DG_via_ER)] *([domain alignment](#domain-alignment-based-methods))*

[87] Exploiting Low-Rank Structure from Latent Domains for Domain Generalization [[ECCV 2014](https://linkspringer.53yu.com/content/pdf/10.1007/978-3-319-10578-9_41.pdf)] *([ensemble learning](#ensemble-learning-based-methods))*

[88] Multi-View Domain Generalization for Visual Recognition [[ICCV 2015](http://openaccess.thecvf.com/content_iccv_2015/papers/Niu_Multi-View_Domain_Generalization_ICCV_2015_paper.pdf)] *(MVDG, [ensemble learning](#ensemble-learning-based-methods))*

[89] Visual recognition by learning from web data: A weakly supervised domain generalization approach [[CVPR 2015](https://openaccess.thecvf.com/content_cvpr_2015/papers/Niu_Visual_Recognition_by_2015_CVPR_paper.pdf)] *([ensemble learning](#ensemble-learning-based-methods), [semi/weak/un-supervised domain generalization](#semiweakun-supervised-domain-generalization))*

[90] Best Sources Forward: Domain Generalization through Source-Specific Nets [[ICIP 2018](https://arxiv.53yu.com/pdf/1806.05810)] *([ensemble learning](#ensemble-learning-based-methods))*

[91] Deep Domain Generalization With Structured Low-Rank Constraint [[TIP 2017](https://par.nsf.gov/servlets/purl/10065328)] *([ensemble learning](#ensemble-learning-based-methods))*

[92] Domain Generalization with Domain-Specific Aggregation Modules [[GCPR 2018](https://arxiv.53yu.com/pdf/1809.10966)] *(D-SAMs, [ensemble learning](#ensemble-learning-based-methods))*

[93] DoFE: Domain-oriented Feature Embedding for Generalizable Fundus Image Segmentation on Unseen Datasets [[TMI 2020](https://arxiv.53yu.com/pdf/2010.06208)] *(DoFE, [ensemble learning](#ensemble-learning-based-methods))*

[94] Learning to Optimize Domain Specific Normalization for Domain Generalization [[ECCV 2020](https://arxiv.53yu.com/pdf/1907.04275)] *(DSON, [ensemble learning](#ensemble-learning-based-methods), [normalization](#normalization-based-methods))*

[95] MS-Net: Multi-Site Network for Improving Prostate Segmentation with Heterogeneous MRI Data [[TMI 2020](https://arxiv.53yu.com/pdf/2002.03366)] [[Code](https://github.com/liuquande/MS-Net)] *(MS-Net, [ensemble learning](#ensemble-learning-based-methods))*

[96] Batch Normalization Embeddings for Deep Domain Generalization [[arXiv 2020](https://arxiv.53yu.com/pdf/2011.12672)] *(BNE, [ensemble learning](#ensemble-learning-based-methods), [normalization](#normalization-based-methods))*

[97] Robust Place Categorization with Deep Domain Generalization [[IEEE Robotics and Automation Letters 2018](https://arxiv.53yu.com/pdf/1805.12048)] [[Code](https://github.com/mancinimassimiliano/caffe)] *(COLD, [ensemble learning](#ensemble-learning-based-methods))*

[98] Domain Generalization by Solving Jigsaw Puzzles [[CVPR 2019](https://openaccess.thecvf.com/content_CVPR_2019/papers/Carlucci_Domain_Generalization_by_Solving_Jigsaw_Puzzles_CVPR_2019_paper.pdf)] [[Code](https://github.com/fmcarlucci/JigenDG)] *(JiGen, [data augmentation](#data-augmentation-based-methods), [self-supervised learning](#self-supervised-learning-based-methods))*

[99] Learning from Extrinsic and Intrinsic Supervisions for Domain Generalization [[ECCV 2020](https://arxiv.53yu.com/pdf/2007.09316)] [[Code](https://github.com/emma-sjwang/EISNet)] *(EISNet, [data augmentation](#data-augmentation-based-methods), [self-supervised learning](#self-supervised-learning-based-methods))*

[100] Zero Shot Domain Generalization [[BMVC 2020](https://arxiv.53yu.com/pdf/2008.07443)] [[Code](https://github.com/aniketde/ZeroShotDG)] *([self-supervised learning](#self-supervised-learning-based-methods))*

[101] Self-Supervised Learning Across Domains [[TPAMI 2021](https://arxiv.53yu.com/pdf/2007.12368)] [[Code](https://github.com/silvia1993/Self-Supervised_Learning_Across_Domains)] *([self-supervised learning](#self-supervised-learning-based-methods))*

[102] Improving Out-Of-Distribution Generalization via Multi-Task Self-Supervised Pretraining [[arXiv 2020](https://arxiv.53yu.com/pdf/2003.13525)] *([self-supervised learning](#self-supervised-learning-based-methods))*

[103] Undoing the Damage of Dataset Bias [[ECCV 2012](https://linkspringer.53yu.com/content/pdf/10.1007/978-3-642-33718-5_12.pdf)] [[Code](https://github.com/adikhosla/undoing-bias)] *([disentangled representation learning](#disentangled-representation-learning-based-methods))*

[104] Learning to Balance Specificity and Invariance for In and Out of Domain Generalization [[ECCV 2020](https://arxiv.53yu.com/pdf/2008.12839)] [[Code](https://github.com/prithv1/DMG)] *(DMG, [disentangled representation learning](#disentangled-representation-learning-based-methods))*

[105] Efficient Domain Generalization via Common-Specific Low-Rank Decomposition [[ICML 2020](http://proceedings.mlr.press/v119/piratla20a/piratla20a.pdf)] [[Code](https://github.com/vihari/CSD)] *(CSD, [disentangled representation learning](#disentangled-representation-learning-based-methods))*

[106] Cross-Domain Face Presentation Attack Detection via Multi-Domain Disentangled Representation Learning [[CVPR 2020](http://openaccess.thecvf.com/content_CVPR_2020/papers/Wang_Cross-Domain_Face_Presentation_Attack_Detection_via_Multi-Domain_Disentangled_Representation_Learning_CVPR_2020_paper.pdf)] *([disentangled representation learning](#disentangled-representation-learning-based-methods), [face recognition/antispoofing](#face-recognition-anti-spoofing))*

[107] DIVA: Domain Invariant Variational Autoencoders [[ICML workshop 2019](http://proceedings.mlr.press/v121/ilse20a/ilse20a.pdf)] [[Code](https://github.com/AMLab-Amsterdam/DIVA)] *(DIVA, [disentangled representation learning](#disentangled-representation-learning-based-methods))*

[108] Dataset Shift in Machine Learning [MIT 2019] *([dataset shift](#domain-generalization))*

[109] A Unifying View on Dataset Shift in Classification [[PR 2012](https://www.sciencedirect.com/science/article/pii/S0031320311002901?casa_token=qIu5tyPmlgQAAAAA:IDLcYED3jzUGsissKY_EuDLQTMCkGQrEWoAq542Cbcd4FKQinvp78Wgb6jhRiSLqGdQCvcifwprz)] *([dataset shift](#domain-generalization))*

[110] Do Imagenet Classifiers Generalize to Imagenet? [[ICML 2019](http://proceedings.mlr.press/v97/recht19a/recht19a.pdf)] *([dataset shift](#domain-generalization))*

[111] A Theory of Learning from Different Domains [[ML 2010](https://link.springer.com/content/pdf/10.1007/s10994-009-5152-4.pdf)] *([dataset shift](#domain-generalization))*

[112] Measuring Robustness to Natural Distribution Shifts in Image Classification [[NeurIPS 2020](https://proceedings.neurips.cc/paper/2020/file/d8330f857a17c53d217014ee776bfd50-Paper.pdf)] [[Code](https://github.com/modestyachts/imagenet-testbed)] *([dataset shift](#domain-generalization))*

[113] Generalizing from Several Related Classification Tasks to a New Unlabeled Sample [[NeurIPS 2011](https://proceedings.neurips.cc/paper/2011/file/b571ecea16a9824023ee1af16897a582-Paper.pdf)] *([domain generalization](#domain-generalization))*

[114] Learning Generalisable Omni-Scale Representations for Person Re-Identification [[TPAMI 2021](https://arxiv.org/pdf/1910.06827)] [[Code](https://github.com/KaiyangZhou/deep-person-reid)] *([person re-identification](#person-re-identification))*

[115] FSDR: Frequency Space Domain Randomization for Domain Generalization [[CVPR 2021](http://openaccess.thecvf.com/content/CVPR2021/papers/Huang_FSDR_Frequency_Space_Domain_Randomization_for_Domain_Generalization_CVPR_2021_paper.pdf)] [[Code](https://github.com/jxhuang0508/FSDR)] *(FSDR, [data augmentation](#data-augmentation-based-methods))*

[116] Adversarially Adaptive Normalization for Single Domain Generalization [[CVPR 2021](https://openaccess.thecvf.com/content/CVPR2021/papers/Fan_Adversarially_Adaptive_Normalization_for_Single_Domain_Generalization_CVPR_2021_paper.pdf)]  *(ASR, [normalization](#normalization-based-methods), [single domain generalization](#single-domain-generalization))*

[117] Deep Stable Learning for Out-of-Distribution Generalization [[CVPR 2021](http://openaccess.thecvf.com/content/CVPR2021/papers/Zhang_Deep_Stable_Learning_for_Out-of-Distribution_Generalization_CVPR_2021_paper.pdf)] [[Code](https://github.com/xxgege/StableNet)] *(StableNet, [causality](#causality-based-methods))*

[118] Generalization on Unseen Domains via Inference-Time Label-Preserving Target Projections [[CVPR 2021](http://openaccess.thecvf.com/content/CVPR2021/papers/Pandey_Generalization_on_Unseen_Domains_via_Inference-Time_Label-Preserving_Target_Projections_CVPR_2021_paper.pdf)] [[Code](https://github.com/yys-Polaris/InferenceTimeDG)] *([domain alignment](#domain-alignment-based-methods), [inference-time](#inference-time-based-methods))*

[119] Open Domain Generalization with Domain-Augmented Meta-Learning [[CVPR 2021](http://openaccess.thecvf.com/content/CVPR2021/papers/Shu_Open_Domain_Generalization_with_Domain-Augmented_Meta-Learning_CVPR_2021_paper.pdf)] [[Code](https://github.com/thuml/OpenDG-DAML)] *(DAML, [data augmentation](#data-augmentation-based-methods), [meta-learning](#meta-learning-based-methods), [open/heterogeneous domain generalization](#openheterogeneous-domain-generalization))*

[120] Learning Attributes Equals Multi-Source Domain Generalization [[CVPR 2016](https://www.cv-foundation.org/openaccess/content_cvpr_2016/papers/Gan_Learning_Attributes_Equals_CVPR_2016_paper.pdf)] *(UDICA, [domain alignment](#domain-alignment-based-methods))*

[121] Visual Recognition by Learning From Web Data via Weakly Supervised Domain Generalization [[TNNLS 2017](https://bcmi.sjtu.edu.cn/home/niuli/paper/Visual%20Recognition%20by%20Learning%20From%20Web%20Data%20via%20Weakly%20Supervised%20Domain%20Generalization.pdf)] *([ensemble learning](#ensemble-learning-based-methods), [semi/weak/un-supervised domain generalization](#semiweakun-supervised-domain-generalization))*

[122] Multi-View Domain Generalization Framework for Visual Recognition [[TNNLS 2018](http://openaccess.thecvf.com/content_iccv_2015/papers/Niu_Multi-View_Domain_Generalization_ICCV_2015_paper.pdf)] *([ensemble learning](#ensemble-learning-based-methods))*

[123] A Generalization Error Bound for Multi-Class Domain Generalization [[arXiv 2019](https://arxiv.org/pdf/1905.10392)] *([theory & analysis](#theory--analysis))*

[124] Domain Generalization for Named Entity Boundary Detection via Metalearning [[TNNLS 2020](https://ieeexplore.ieee.org/abstract/document/9174763/)] *(METABDRY, [meta-learning](#meta-learning-based-methods))*

[125] Deep Semi-supervised Domain Generalization Network for Rotary Machinery Fault Diagnosis under Variable Speed [[IEEE Transactions on Instrumentation and Measurement 2020](https://www.researchgate.net/profile/Yixiao-Liao/publication/341199775_Deep_Semisupervised_Domain_Generalization_Network_for_Rotary_Machinery_Fault_Diagnosis_Under_Variable_Speed/links/613f088201846e45ef450a0a/Deep-Semisupervised-Domain-Generalization-Network-for-Rotary-Machinery-Fault-Diagnosis-Under-Variable-Speed.pdf)] *(DSDGN, [semi/weak/un-supervised domain generalization](#semiweakun-supervised-domain-generalization))*

[126] Generalized Convolutional Forest Networks for Domain Generalization and Visual Recognition [[ICLR 2020](https://openreview.net/pdf?id=H1lxVyStPH)] *(GCFN, [ensemble learning](#ensemble-learning-based-methods), [self-supervised learning](#self-supervised-learning-based-methods))*

[127] Domain Generalization via Semi-supervised Meta Learning [[arXiv 2020](https://arxiv.org/pdf/2009.12658)] [[Code](https://github.com/hosseinshn/DGSML)] *(DGSML, [meta-learning](#meta-learning-based-methods), [semi/weak/un-supervised domain generalization](#semiweakun-supervised-domain-generalization))*

[128] Heterogeneous Domain Generalization via Domain Mixup [[ICASSP 2020](https://arxiv.org/pdf/2009.05448)] [[Code](https://github.com/wyf0912/MIXALL)] *([data augmentation](#data-augmentation-based-methods), [open/heterogeneous domain generalization](#openheterogeneous-domain-generalization))*

[129] NAS-OoD Neural Architecture Search for Out-of-Distribution Generalization [[ICCV 2021](https://openaccess.thecvf.com/content/ICCV2021/papers/Bai_NAS-OoD_Neural_Architecture_Search_for_Out-of-Distribution_Generalization_ICCV_2021_paper.pdf)] *(NAS-OoD, [neural architecture search](#neural-architecture-search-based-methods))*

[130] A Style and Semantic Memory Mechanism for Domain Generalization [[ICCV 2021](http://openaccess.thecvf.com/content/ICCV2021/papers/Chen_A_Style_and_Semantic_Memory_Mechanism_for_Domain_Generalization_ICCV_2021_paper.pdf)] *(STEAM, [self-supervised learning](#self-supervised-learning-based-methods), [causality](#causality-based-methods))*

[131] Learning Transferrable and Interpretable Representations for Domain Generalization [[MM 2021](https://dl.acm.org/doi/pdf/10.1145/3474085.3475488)] *(DTN, [ensemble learning](#ensemble-learning-based-methods))*

[132] Adaptive Methods for Real-World Domain Generalization [[CVPR 2021](http://openaccess.thecvf.com/content/CVPR2021/papers/Dubey_Adaptive_Methods_for_Real-World_Domain_Generalization_CVPR_2021_paper.pdf)] [[Code](https://github.com/abhimanyudubey/GeoYFCC)] *(DA-ERM, [inference-time](#inference-time-based-methods))*

[133] Confidence Calibration for Domain Generalization Under Covariate Shift [[ICCV 2021](https://openaccess.thecvf.com/content/ICCV2021/papers/Gong_Confidence_Calibration_for_Domain_Generalization_Under_Covariate_Shift_ICCV_2021_paper.pdf)] *([domain alignment](#domain-alignment-based-methods))*

[134] In Search of Lost Domain Generalization [[ICLR 2021](https://arxiv.org/pdf/2007.01434.pdf?fbclid=IwAR1YkUXkIhC6fhr6eI687zBXo_W2tTjjTAFnyjEWvmq4gQKon_4pIDbTnQ4)] *([theory & analysis](#theory--analysis))*

[135] The Many Faces of Robustness: A Critical Analysis of Out-of-Distribution Generalization [[ICCV 2021](https://openaccess.thecvf.com/content/ICCV2021/papers/Hendrycks_The_Many_Faces_of_Robustness_A_Critical_Analysis_of_Out-of-Distribution_ICCV_2021_paper.pdf)] [[Code](https://github.com/hendrycks/imagenet-r)] *([theory & analysis](#theory--analysis))*

[136] Test-Time Classifier Adjustment Module for Model-Agnostic Domain Generalization [[NeurIPS 2021](https://proceedings.neurips.cc/paper/2021/file/1415fe9fea0fa1e45dddcff5682239a0-Paper.pdf)] [[Code](https://github.com/matsuolab/T3A)] *(T3A, [inference-time](#inference-time-based-methods))*

[137] Feature Stylization and Domain-aware Contrastive Learning for Domain Generalization [[MM 2021](https://dl.acm.org/doi/pdf/10.1145/3474085.3475271)] *([data augmentation](#data-augmentation-based-methods), [self-supervised learning](#self-supervised-learning-based-methods))*

[138] SelfReg: Self-Supervised Contrastive Regularization for Domain Generalization [[ICCV 2021](http://openaccess.thecvf.com/content/ICCV2021/papers/Kim_SelfReg_Self-Supervised_Contrastive_Regularization_for_Domain_Generalization_ICCV_2021_paper.pdf)] *(SelfReg, [self-supervised learning](#self-supervised-learning-based-methods), [regularization](#regularization-based-methods))*

[139] Domain Generalisation with Domain Augmented Supervised Contrastive Learning [[AAAI Student Abstract 2021](https://www.aaai.org/AAAI21Papers/SA-197.LeHS.pdf)] *(DASCL, [data augmentation](#data-augmentation-based-methods), [self-supervised learning](#self-supervised-learning-based-methods))*

[140] Invariant Information Bottleneck for Domain Generalization [[AAAI 2022](https://arxiv.org/pdf/2106.06333)] [[Code](https://github.com/Luodian/IIB/tree/IIB)] *(IIB, [information](#information-based-methods), [causality](#causality-based-methods))*

[141] Progressive Domain Expansion Network for Single Domain Generalization [[CVPR 2021](http://openaccess.thecvf.com/content/CVPR2021/papers/Li_Progressive_Domain_Expansion_Network_for_Single_Domain_Generalization_CVPR_2021_paper.pdf)] [[Code](https://github.com/lileicv/PDEN)] *(PDEN, [domain alignment](#domain-alignment-based-methods), [self-supervised learning](#self-supervised-learning-based-methods), [information](#information-based-methods), [single domain generalization](#single-domain-generalization))*

[142] A Simple Feature Augmentation for Domain Generalization [[ICCV 2021](https://openaccess.thecvf.com/content/ICCV2021/papers/Li_A_Simple_Feature_Augmentation_for_Domain_Generalization_ICCV_2021_paper.pdf)] *(SFA, [data augmentation](#data-augmentation-based-methods))*

[143] Domain-Invariant Disentangled Network for Generalizable Object Detection [[ICCV 2021](https://openaccess.thecvf.com/content/ICCV2021/papers/Lin_Domain-Invariant_Disentangled_Network_for_Generalizable_Object_Detection_ICCV_2021_paper.pdf)] *([disentangled representation learning](#disentangled-representation-learning-based-methods))*

[144] Multi-Domain Adversarial Feature Generalization for Person Re-Identification [[TIP 2021](https://ieeexplore.ieee.org/iel7/83/9263394/09311771.pdf)] *(MMFA-AAE, [domain alignment](#domain-alignment-based-methods), [self-supervised learning](#self-supervised-learning-based-methods), [person re-identification](#person-re-identification))*

[145] Learning Causal Semantic Representation for Out-of-Distribution Prediction [[NeurIPS 2021](https://proceedings.neurips.cc/paper/2021/file/310614fca8fb8e5491295336298c340f-Paper.pdf)] [[Code](https://github.com/changliu00/causal-semantic-generative-model)] *(CSG-ind, [causality](#causality-based-methods))*

[146] Domain Generalization via Feature Variation Decorrelation [[MM 2021](https://dl.acm.org/doi/pdf/10.1145/3474085.3475311)] *([disentangled representation learning](#disentangled-representation-learning-based-methods))*

[147] FedDG: Federated Domain Generalization on Medical Image Segmentation via Episodic Learning in Continuous Frequency Space [[CVPR 2021](http://openaccess.thecvf.com/content/CVPR2021/papers/Liu_FedDG_Federated_Domain_Generalization_on_Medical_Image_Segmentation_via_Episodic_CVPR_2021_paper.pdf)] [[Code](https://github.com/liuquande/FedDG-ELCFS)] *(FedDG, [data augmentation](#data-augmentation-based-methods), [self-supervised learning](#self-supervised-learning-based-methods), [federated domain generalization](#federated-domain-generalization))*

[148] Domain Generalization via Gradient Surgery [[ICCV 2021](https://openaccess.thecvf.com/content/ICCV2021/papers/Mansilla_Domain_Generalization_via_Gradient_Surgery_ICCV_2021_paper.pdf)] [[Code](https://github.com/lucasmansilla/DGvGS)] *(Agr, [regularization](#regularization-based-methods))*

[149] Shape-Biased Domain Generalization via Shock Graph Embeddings [[ICCV 2021](https://openaccess.thecvf.com/content/ICCV2021/papers/Narayanan_Shape-Biased_Domain_Generalization_via_Shock_Graph_Embeddings_ICCV_2021_paper.pdf)] *([disentangled representation learning](#disentangled-representation-learning-based-methods))*

[150] Universal Cross-Domain Retrieval Generalizing Across Classes and Domains [[ICCV 2021](https://openaccess.thecvf.com/content/ICCV2021/papers/Paul_Universal_Cross-Domain_Retrieval_Generalizing_Across_Classes_and_Domains_ICCV_2021_paper.pdf)] [[Code](https://github.com/mvp18/UCDR)] *(SnMpNet, [data augmentation](#data-augmentation-based-methods), [open/heterogeneous domain generalization](#openheterogeneous-domain-generalization))*

[151] Out-of-domain Generalization from a Single Source: A Uncertainty Quantification Approach [[arXiv 2021](https://arxiv.53yu.com/pdf/2108.02888)] *([data augmentation](#data-augmentation-based-methods), [self-supervised learning](#self-supervised-learning-based-methods), [single domain generalization](#single-domain-generalization))*

[152] Recovering Latent Causal Factor for Generalization to Distributional Shifts [[NeurIPS 2021](https://proceedings.neurips.cc/paper/2021/file/8c6744c9d42ec2cb9e8885b54ff744d0-Paper.pdf)] [[Code](https://github.com/wubotong/LaCIM)] *(LaCIM, [causality](#causality-based-methods))*

[153] Continual Adaptation of Visual Representations via Domain Randomization and Meta-learning [[CVPR 2021](http://openaccess.thecvf.com/content/CVPR2021/papers/Volpi_Continual_Adaptation_of_Visual_Representations_via_Domain_Randomization_and_Meta-Learning_CVPR_2021_paper.pdf)] *(Meta-DR, [data augmentation](#data-augmentation-based-methods), [meta-learning](#meta-learning-based-methods), [life-long learning](#life-long-learning))*

[154] On Calibration and Out-of-domain Generalization [[NeurIPS 2021](https://proceedings.neurips.cc/paper/2021/file/118bd558033a1016fcc82560c65cca5f-Paper.pdf)] *([domain alignment](#domain-alignment-based-methods), [causality](#causality-based-methods))*

[155] Generalizing to Unseen Domains: A Survey on Domain Generalization [[IJCAI 2021](https://arxiv.53yu.com/pdf/2103.03097)] [[Slides](http://jd92.wang/assets/files/DGSurvey-ppt.pdf)] *([survey](#survey))*

[156] Better Pseudo-label Joint Domain-aware Label and Dual-classifier for Semi-supervised Domain Generalization [[arXiv 2021](https://arxiv.53yu.com/pdf/2110.04820)] *([data augmentation](#data-augmentation-based-methods),  [semi/weak/un-supervised domain generalization](#semiweakun-supervised-domain-generalization))*

[157] Embracing the Dark Knowledge: Domain Generalization Using Regularized Knowledge Distillation [[MM 2021](https://arxiv.53yu.com/pdf/2110.04820)] *(KDDG, [ensemble learning](#ensemble-learning-based-methods), [regularization](#regularization-based-methods))*

[158] Learning To Diversify for Single Domain Generalization [[ICCV 2021](https://openaccess.thecvf.com/content/ICCV2021/papers/Wang_Learning_To_Diversify_for_Single_Domain_Generalization_ICCV_2021_paper.pdf)] [[Code](https://github.com/BUserName/Learning)] *([information](#information-based-methods), [single domain generalization](#single-domain-generalization))*

[159] Collaborative Optimization and Aggregation for Decentralized Domain Generalization and Adaptation [[ICCV 2021](https://openaccess.thecvf.com/content/ICCV2021/papers/Wu_Collaborative_Optimization_and_Aggregation_for_Decentralized_Domain_Generalization_and_Adaptation_ICCV_2021_paper.pdf)] *(COPDA, [normalization](#normalization-based-methods), [federated domain generalization](#federated-domain-generalization))*

[160] A Fourier-Based Framework for Domain Generalization [[CVPR 2021](https://openaccess.thecvf.com/content/CVPR2021/papers/Xu_A_Fourier-Based_Framework_for_Domain_Generalization_CVPR_2021_paper.pdf)] [[Code](https://github.com/MediaBrain-SJTU/FACT)] *(FACT, [data augmentation](#data-augmentation-based-methods), [regularization](#regularization-based-methods))*

[161] Collaborative Semantic Aggregation and Calibration for Separated Domain Generalization [[arXiv 2021](https://arxiv.org/pdf/2110.06736)] [[Code](https://github.com/junkunyuan/CSAC)] *(CSAC, [domain alignment](#domain-alignment-based-methods), [federated domain generalization](#federated-domain-generalization))*

[162] Domain-Specific Bias Filtering for Single Labeled Domain Generalization [[arXiv 2021](https://arxiv.org/pdf/2110.00726)] [[Code](https://github.com/junkunyuan/DSBF)] *(DSBF, [semi/weak/un-supervised domain generalization](#semiweakun-supervised-domain-generalization))*

[163] Learning Domain-Invariant Relationship with Instrumental Variable for Domain Generalization [[arXiv 2021](https://arxiv.org/pdf/2110.01438)] *(DRIVE, [causality](#causality-based-methods))* 

[164] Adaptation and Generalization Across Domains in Visual Recognition with Deep Neural Networks [[PhD 2020](https://openresearch.surrey.ac.uk/esploro/outputs/doctoral/Adaptation-and-Generalization-Across-Domains-in/99513024202346)] *([other resources](#other-resources))*

[165] Invariant Risk Minimization [[arXiv 2019](https://arxiv.53yu.com/pdf/1907.02893.pdf;)] [[Code](https://github.com/facebookresearch/InvariantRiskMinimization)] *(IRM, [regularization](#regularization-based-methods), [causality](#causality-based-methods))*

[166] Beyond Domain Adaptation: Unseen Domain Encapsulation via Universal Non-volume Preserving Models [[arXiv 2018](https://arxiv.53yu.com/pdf/1812.03407)] *(UNVP, [domain alignment](#domain-alignment-based-methods))*

[167] Multi-component Image Translation for Deep Domain Generalization [[WACV 2019](https://arxiv.53yu.com/pdf/1812.08974)] [[Code](https://github.com/mahfujur1/mit-DG)] *([data augmentation](#data-augmentation-based-methods))*

[168] Uncertainty-guided Model Generalization to Unseen Domains [[CVPR 2021](http://openaccess.thecvf.com/content/CVPR2021/papers/Qiao_Uncertainty-Guided_Model_Generalization_to_Unseen_Domains_CVPR_2021_paper.pdf)] [[Code](https://github.com/joffery/UMGUD)] *([data augmentation](#data-augmentation-based-methods), [meta-learning](#meta-learning-based-methods), [single domain generalization](#single-domain-generalization))*

[169] Image Alignment in Unseen Domains via Domain Deep Generalization [[arXiv 2019](https://arxiv.org/pdf/1905.12028)] *(DeGIA, [domain alignment](#domain-alignment-based-methods))*

[170] Towards Principled Disentanglement for Domain Generalization [[arXiv 2021](https://arxiv.org/pdf/2111.13839)] [[Code](https://github.com/hlzhang109/DDG)] *(DDG, [data augmentation](#data-augmentation-based-methods), [disentangled representation learning](#disentangled-representation-learning-based-methods))*

[171] DecAug: Out-of-Distribution Generalization via Decomposed Feature Representation and Semantic Augmentation [[AAAI 2021](https://arxiv.org/pdf/2012.09382)] [[Code](https://github.com/HaoyueBaiZJU/DecAug)] *(DecAug, [data augmentation](#data-augmentation-based-methods), [disentangled representation learning](#disentangled-representation-learning-based-methods))*

[172] Domain Generalization for Mammography Detection via Multi-style and Multi-view Contrastive Learning [[MICCAI 2021](https://arxiv.org/pdf/2111.10827)] [[Code](https://github.com/lizheren/MSVCL_MICCAI2021)] *(MSVCL, [self-supervised learning](#self-supervised-learning-based-methods))*

[173] Fishr: Invariant Gradient Variances for Our-of-distribution Generalization [[arXiv 2021](https://arxiv.org/pdf/2109.02934)] [[Code](https://github.com/alexrame/fishr)] *(Fishr, [regularization](#regularization-based-methods))*

[174] Dynamically Decoding Source Domain Knowledge for Unseen Domain Generalization [[arXiv 2021](https://www.researchgate.net/profile/Karthik-Nandakumar-3/publication/355142270_Dynamically_Decoding_Source_Domain_Knowledge_For_Unseen_Domain_Generalization/links/61debe18034dda1b9ef16fc6/Dynamically-Decoding-Source-Domain-Knowledge-For-Unseen-Domain-Generalization.pdf)] *(D2SDK, [ensemble learning](#ensemble-learning-based-methods))*

[175] Class-conditioned Domain Generalization via Wasserstein Distributional Robust Optimization [[ICLR workshop 2021](https://arxiv.org/pdf/2109.03676)] *([ensemble learning](#ensemble-learning-based-methods))*

[176] Domain and Content Adaptive Convolution for Domain Generalization in Medical Image Segmentation [[arXiv 2021](https://arxiv.org/pdf/2109.05676)] *(DCAC, [ensemble learning](#ensemble-learning-based-methods))*

[177] Scale Invariant Domain Generalization Image Recapture Detection [[ICONIP 2021](https://arxiv.org/pdf/2110.03496)] *(SADG, [domain alignment](#domain-alignment-based-methods), [self-supervised learning](#self-supervised-learning-based-methods))*

[178] Unsupervised Domain Generalization by Learning a Bridge Across Domains [[arXiv 2021](https://arxiv.org/pdf/2112.02300)] ([self-supervised learning](#self-supervised-learning-based-methods), [semi/weak/un-supervised domain generalization](#semiweakun-supervised-domain-generalization))

[179] Semi-Supervised Domain Generalization in RealWorld: New Benchmark and Strong Baseline [[arXiv 2021](https://arxiv.org/pdf/2111.10221)] *([domain alignment](#domain-alignment-based-methods), [semi/weak/un-supervised domain generalization](#semiweakun-supervised-domain-generalization))*

[180] Few-Shot Classification in Unseen Domains by Episodic Meta-Learning Across Visual Domains [[ICIP 2021](https://arxiv.org/pdf/2112.13539)] *(x-EML, [meta-learning](#meta-learning-based-methods))*

[181] Energy-based Out-of-distribution Detection [[NeurIPS 2020](https://proceedings.neurips.cc/paper/2020/file/f5496252609c43eb8a3d147ab9b9c006-Paper.pdf)] [[Code](https://github.com/xieshuqin/Energy-OOD)] ([regularization](#regularization-based-methods))

[182] ROBIN : A Benchmark for Robustness to Individual Nuisances in Real-World Out-of-Distribution Shifts [[arXiv 2021](https://arxiv.org/pdf/2111.14341)] *([ROBIN dataset](#datasets))* [[Code](https://bzhao.me/ROBIN/)]

[183] Towards Non-I.I.D. Image Classification: A Dataset and Baselines [[PR 2021](https://www.sciencedirect.com/science/article/pii/S0031320320301862?casa_token=y_fKwAxq7egAAAAA:BGofuCXu-RA4XWbYcbDjwWXkPvXpj983lsle7WO5fOtoRFrDlI6GG_qELpPt_zApzM_yOXoFlUX0)] *([NICO dataset](#datasets))*

[184] More is Better: A Novel Multi-view Framework for Domain Generalization [[arXiv 2021](https://arxiv.org/pdf/2112.12329)] *([data augmentation](#data-augmentation-based-methods), [meta-learning](#meta-learning-based-methods))*

[185] Meta-Learned Feature Critics for Domain Generalized Semantic Segmentation [[ICIP 2021](https://arxiv.org/pdf/2112.13538)] *([meta-learning](#meta-learning-based-methods), [disentangled representation learning](#disentangled-representation-learning-based-methods))*

[186] Domain Generalization through Audio-Visual Relative Norm Alignment in First Person Action Recognition [[WACV 2022](https://openaccess.thecvf.com/content/WACV2022/papers/Planamente_Domain_Generalization_Through_Audio-Visual_Relative_Norm_Alignment_in_First_Person_WACV_2022_paper.pdf)] *(RNA-Net, [normalization](#normalization-based-methods))*


[187] Generalizable Person Re-identification with Relevance-aware Mixture of Experts [[CVPR 2021](https://openaccess.thecvf.com/content/CVPR2021/papers/Dai_Generalizable_Person_Re-Identification_With_Relevance-Aware_Mixture_of_Experts_CVPR_2021_paper.pdf)] *(RaMoE, [ensemble learning](#ensemble-learning-based-methods), [person re-identification](#person-re-identification))*

[188] Domain Generalization by Marginal Transfer Learning [[JMLR 2021](https://www.jmlr.org/papers/volume22/17-679/17-679.pdf)] [[Code](https://github.com/aniketde/DomainGeneralizationMarginal)] *([theory & analysis](#theory--analysis), [data augmentation](#data-augmentation-based-methods))*

[189] Feature Alignment and Restoration for Domain Generalization and Adaptation [[arXiv 2020](https://arxiv.org/pdf/2006.12009)] *(FAR, [domain alignment](#domain-alignment-based-methods))*

[190] Out-of-Distribution Generalization via Risk Extrapolation [[ICML 2021](http://proceedings.mlr.press/v139/krueger21a/krueger21a.pdf)] *(REx, [regularization](#regularization-based-methods))*

[191] A Causal Framework for Distribution Generalization [[TPAMI 2021](https://arxiv.org/pdf/2006.07433)] [[Code](https://runesen.github.io/NILE/)] *(NILE, [causality](#causality-based-methods))*

[192] Iterative Feature Matching: Toward Provable Domain Generalization with Logarithmic Environments [[arXiv 2021](https://arxiv.org/pdf/2106.09913)] *([domain alignment](#domain-alignment-based-methods))*

[193] Robustnet: Improving Domain Generalization in Urban-Scene Segmentation via Instance Selective Whitening [[CVPR 2021](https://openaccess.thecvf.com/content/CVPR2021/papers/Choi_RobustNet_Improving_Domain_Generalization_in_Urban-Scene_Segmentation_via_Instance_Selective_CVPR_2021_paper.pdf)] [[Code](https://github.com/shachoi/RobustNet)] *(RobustNet, [disentangled representation learning](#disentangled-representation-learning-based-methods))*

[194] Boosting the Generalization Capability in Cross-Domain Few-shot Learning via Noise-enhanced Supervised Autoencoder [[ICCV 2021](https://openaccess.thecvf.com/content/ICCV2021/papers/Liang_Boosting_the_Generalization_Capability_in_Cross-Domain_Few-Shot_Learning_via_Noise-Enhanced_ICCV_2021_paper.pdf)] *(NSAE, [self-supervised learning](#self-supervised-learning-based-methods))*

[195] Domain Generalization under Conditional and Label Shifts via Variational Bayesian Inference [[IJCAI 2021](https://arxiv.org/pdf/2107.10931)] *(VBCLS, [domain alignment](#domain-alignment-based-methods))*

[196] The Risks of Invariant Risk Minimization [[ICLR 2021](https://arxiv.org/pdf/2010.05761)] *([theory & analysis](#theory--analysis))*

[197] VideoDG: Generalizing Temporal Relations in Videos to Novel Domains [[TPAMI 2021](https://arxiv.org/pdf/1912.03716)] [[Code](https://github.com/thuml/VideoDG)] *(APN, [data augmentation](#data-augmentation-based-methods))*

[198] An Empirical Investigation of Domain Generalization with Empirical Risk Minimizers [[NeurIPS 2021](https://proceedings.neurips.cc/paper/2021/file/ecf9902e0f61677c8de25ae60b654669-Paper.pdf)] [[Code](https://github.com/facebookresearch/domainbed_measures)] *([theory & analysis](#theory--analysis))*

[199] Towards a Theoretical Framework of Out-Of-Distribution Generalization [[NeurIPS 2021](https://proceedings.neurips.cc/paper/2021/file/c5c1cb0bebd56ae38817b251ad72bedb-Paper.pdf)] *([theory & analysis](#theory--analysis))*

[200] Model-Based Domain Generalization [[NeurIPS 2021](https://proceedings.neurips.cc/paper/2021/file/a8f12d9486cbcc2fe0cfc5352011ad35-Paper.pdf)] [[Code](https://github.com/arobey1/mbdg)] *(MBDG, [data augmentation](#data-augmentation-based-methods), [regularization](#regularization-based-methods))*

[201] Swad: Domain Generalization by Seeking Flat Minima [[NeurIPS 2021](https://proceedings.neurips.cc/paper/2021/file/bcb41ccdc4363c6848a1d760f26c28a0-Paper.pdf)] [[Code](https://github.com/khanrc/swad)] *(SWAD, [regularization](#regularization-based-methods))*

[202] Exploiting Domain-Specific Features to Enhance Domain Generalization [[NeurIPS 2021](https://proceedings.neurips.cc/paper/2021/file/b0f2ad44d26e1a6f244201fe0fd864d1-Paper.pdf)] [[Code](https://github.com/manhhabui/mDSDI)] *(mDSDI, [meta-learning](#meta-learning-based-methods), [disentangled representation learning](#disentangled-representation-learning-based-methods), [information](#information-based-methods))*

[203] Adversarial Teacher-Student Representation Learning for Domain Generalization [[NeurIPS 2021](https://proceedings.neurips.cc/paper/2021/file/a2137a2ae8e39b5002a3f8909ecb88fe-Paper.pdf)] *([data augmentation](#data-augmentation-based-methods), [self-supervised learning](#self-supervised-learning-based-methods))*

[204] Training for the Future: A Simple Gradient Interpolation Loss to Generalize Along Time [[NeurIPS 2021](https://proceedings.neurips.cc/paper/2021/file/a02ef8389f6d40f84b50504613117f88-Paper.pdf)] [[Code](https://github.com/anshuln/Training-for-the-Future)] *(GI, [regularization](#regularization-based-methods))*

[205] Out-of-Distribution Generalization in Kernel Regression [[NeurIPS 2021](https://proceedings.neurips.cc/paper/2021/file/691dcb1d65f31967a874d18383b9da75-Paper.pdf)] *([theory & analysis](#theory--analysis))*

[206] Quantifying and Improving Transferability in Domain Generalization [[NeurIPS 2021](https://proceedings.neurips.cc/paper/2021/file/5adaacd4531b78ff8b5cedfe3f4d5212-Paper.pdf)] [[Code](https://github.com/Gordon-Guojun-Zhang/Transferability-NeurIPS2021)] *([theory & analysis](#theory--analysis), [regularization](#regularization-based-methods))*

[207] Invariance Principle Meets Information Bottleneck for Out-Of-Distribution Generalization [[NeurIPS 2021](https://proceedings.neurips.cc/paper/2021/file/1c336b8080f82bcc2cd2499b4c57261d-Paper.pdf)] [[Code](https://github.com/ahujak/IB-IRM)] *(IB-IRM, [information](#information-based-methods), [causality](#causality-based-methods))*

[208] TransMatcher: Deep Image Matching Through Transformers for Generalizable Person Re-identification [[NeurIPS 2021](https://proceedings.neurips.cc/paper/2021/file/0f49c89d1e7298bb9930789c8ed59d48-Paper.pdf)] [[Code](https://github.com/ShengcaiLiao/QAConv)] *(TransMatcher, [ensemble learning](#ensemble-learning-based-methods), [person re-identification](#person-re-identification))*


# Contributing & Contact
Feel free to contribute to our repository.

- If you woulk like to **correct mistakes**, please do it directly;
- If you would like to **add/update papers**, please finish the following tasks (if necessary):
    1. Update [Paper Index](#paper-index).
    2. Update [Papers](#papers). 
    3. Update [Datasets](#datasets) with reference of [Paper Index](#paper-index).
- If you have any **questions or advice**, please contact us by email (yuanjk@zju.edu.cn) or GitHub issues.

Thank you for your cooperation and contributions!



# Acknowledgements
- We refer to [awesome-domain-adaptation](https://github.com/zhaoxin94/awesome-domain-adaptation#unsupervised-da) to design the hierarchy of the [Contents](#contents).
- We refer to [3] to design the [Contents](#contents) and the table of [Datasets](#datasets).