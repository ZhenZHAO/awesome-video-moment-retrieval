# VMR With Codes

> - Temporal Sentence Grounding in Videos (**TSGV**) 
> - Natural Language Video Localization (**NLVL**) 
> - Video Moment Retrieval (**VMR**) 
>
> Keywords: temporal grounding, moment retrieval, Language Video Localization
>
> - https://github.com/IsaacChanghau/DL-NLP-Readings/blob/master/readme/grounding/video/video_grounding.md



CV比赛，数据集 

- 不完全相同，可以参考

- [ego4d-data](https://ego4d-data.org/docs/challenge/)
  - [ECCV2022-workshop](https://ego4d-data.org/workshops/eccv22/)
  - [CVPR22-workshop](https://ego4d-data.org/workshops/cvpr22/)
  - https://www.youtube.com/watch?v=KH9VP9tDbdI

- [PIC](http://www.picdataset.com/)
  - http://www.picdataset.com/challenge/leaderboard/hcvg2022





最初的文章, 提的任务和数据集

- https://github.com/LisaAnne/LocalizingMoments
  - Localizing Moments in Video with Natural Language, ICCV 2017
- https://github.com/LisaAnne/TemporalLanguageRelease
  - Localizing Moments in Video with Temporal Language, EMNLP (2018)
- https://github.com/jiyanggao/TALL
  - TALL: Temporal Activity Localization via Language Query, ICCV 2017 paper



### Papers by Issues

Transformers-based

- https://github.com/jayleicn/moment_detr
  - QVHighlights: Detecting Moments and Highlights in Videos via Natural Language Queries, NeurIPS 2021, Moment_DETR
  - 各种video-laug的任务，https://jayleicn.github.io/
- https://github.com/TencentARC/UMT
  
  - UMT: Unified Multi-modal Transformers for Joint Video Moment Retrieval and Highlight Detection, CVPR 2022
- Multi-stage aggregated transformer network for temporal language localization in videos, CVPR, 2021

  - no code
- On Pursuit of Designing Multi-modal Transformer for Video Grounding, EMNLP, 2021

  - no code
  - from raw videos
  - https://mengcaopku.github.io/
  - https://zjuchenlong.github.io/
- LocVTP: Video-Text Pre-training for Temporal Localization， ECCV 22
    - https://github.com/mengcaopku/LocVTP
- Explore-And-Match: Bridging Proposal-Based and Proposal-Free With Transformer for Sentence Grounding in Videos 
  
  - https://github.com/sangminwoo/Explore-And-Match
- [Hierarchical local- global transformer for temporal sentence grounding](https://arxiv.org/pdf/2208.14882.pdf), arxiv,
- no code from Daizong.
- [Locformer: Enabling transformers to perform temporal moment localization on long untrimmed videos with a feature sampling approach](https://arxiv.org/pdf/2112.10066.pdf) , arxiv
- 其实相近的领域，比如activity location，基于transformer的文章会多很多
  
  - https://github.com/Alvin-Zeng/Awesome-Temporal-Action-Localization
  - https://arxiv.org/abs/2204.01680
  - https://arxiv.org/abs/2202.07925
- 没有开源的文章:
  - **PPT**: Point Prompt Tuning for Temporally Language Grounding, 使用clip, SIGGAR 2022
  - Zero-shot Video Moment Retrieval With Off-the-Shelf Models 
- [ego4d-data](https://ego4d-data.org/docs/challenge/) 的比赛
  - 其中各种使用clip和transformer的
  - 衍生出相关的文章
  - Egocentric Video-Language Pretraining， NIPS2022， https://arxiv.org/pdf/2206.01670.pdf
  - https://ego4d-data.org/workshops/cvpr22/
  - https://ego4d-data.org/workshops/eccv22/
- 相似的领域也是有完全基于transformer的
  - TubeDETR: Spatio-Temporal Video Grounding with Transformers, CVPR 22
    - https://github.com/antoyang/TubeDETR
  - Embracing Consistency: A One-Stage Approach for Spatio-Temporal Video Grounding, NIPS 22
    - https://github.com/jy0205/STCAT?utm_source=catalyzex.com





Metric-learning 角度看问题

- https://github.com/MCG-NJU/MMN
  - Negative Sample Matters: A Renaissance of Metric Learning for Temporal Grounding, AAAI 2022
  - https://zhuanlan.zhihu.com/p/446203594
  - 从metric learning角度
- https://github.com/asrafulashiq/wsad
  - Weakly Supervised Temporal Action Localization Using Deep Metric Learning, WACV 2020
  - 从metric learning角度



**Temporal biases**


- Uncovering Hidden Challenges in Query-Based Video Moment Retrieval, BMVC 2020
  - https://github.com/mayu-ot/hidden-challenges-MR
- A Closer Look at Debiased Temporal Sentence Grounding in Videos: Dataset, Metric, and Approach 

  - https://arxiv.org/abs/2203.05243
- A Closer Look at Temporal Sentence Grounding in Videos: Dataset and Metric 

  - https://arxiv.org/abs/2101.09028
- Can Shuffling Video Benefit Temporal Bias Problem: A Novel Training Framework for Temporal Grounding, ECCV 2022
  - https://github.com/haojc/ShufflingVideosForTSG
- Interventional Video Grounding with Dual Contrastive Learning， CVPR21

  - https://github.com/nanguoshun/IVG
- Deconfounded Video Moment Retrieval with Causal Intervention， SIGGAR21

  - https://github.com/Xun-Yang/Causal_Video_Moment_Retrieval
- Learning Sample Importance for Cross-Scenario Video Temporal Grounding，ICMR2022

  - https://arxiv.org/abs/2201.02848
  - Hybrid-Learning Video Moment Retrieval across Multi-Domain Labels， BMVC 22



### Papers by People



ZHANG HAO - NTU

- https://github.com/IsaacChanghau/VSLNet
  - Span-based Localizing Network for Natural Language Video Localization (ACL 2020)
- https://github.com/IsaacChanghau/SeqPAN
  - Parallel Attention Network with Sequence Matching for Video Grounding, Findings of ACL 2021
- https://github.com/IsaacChanghau/ReLoCLNet
  - Video Corpus Moment Retrieval with Contrastive Learning, SIGIR 2021
- review: https://arxiv.org/abs/2201.08071
  - PhD thesis



Songyang Zhang - Rochester


- https://github.com/Sy-Zhang/TCMN-Release
  - Exploiting Temporal Relationships in Video Moment Localization with Natural Language, In ACM Multimedia 2019
- https://github.com/microsoft/VideoX/tree/master/2D-TAN
  - Learning 2D Temporal Adjacent Networks for Moment Localization with Natural Language, AAAI 2020
- https://github.com/ChenJoya/2dtan  (improved verions)

  - An optimized re-implementation for 2D-TAN: Learning 2D Temporal Localization Networks for Moment Localization with Natural Language, AAAI 2020.
- https://github.com/microsoft/VideoX/tree/master/MS-2D-TAN
  - Multi-Scale 2D Temporal Adjacent Networks for Moment Localization with Natural Language, TPAMI21



Yawenzeng - hunan univ.


- https://github.com/nakaizura/AVMR
- Adversarial Video Moment Retrieval by Jointly Modeling Ranking and Localization (AVMR), MM 2020
- https://github.com/nakaizura/STRONG
  - STRONG: Spatio-Temporal Reinforcement Learning for Cross-Modal Video Moment Localization
  
- 关于VMR的review
  
  - https://blog.csdn.net/qq_39388410/article/details/107316185
  - https://github.com/nakaizura/Awesome-Cross-Modal-Video-Moment-Retrieval
- many papers, can hardly find source codes for recent papers.





Zhijie Lin - Sea AI Lab


- https://github.com/ikuinen/CMIN_moment_retrieval

  - Cross-Modal Interaction Networks for Query-Based Moment Retrieval in Videos, SIGIR 2019
- https://github.com/ikuinen/regularized_two-branch_proposal_network

  - Regularized Two-Branch Proposal Networks for Weakly-Supervised Moment Retrieval in Videos, MM 2020
- https://github.com/ikuinen/semantic_completion_network

  - Weakly-Supervised Video Moment Retrieval via Semantic Completion Network, AAAI 2020

- https://scholar.google.com/citations?user=xXMj6_EAAAAJ&hl=zh-CN

  - more papers... actually.





Xin Sun -上交

- https://github.com/Huntersxsx/RaNet
  - Relation-aware Video Reading Comprehension for Temporal Language Grounding, EMNLP 2021.
- https://github.com/Huntersxsx/MGPN
  - You Need to Read Again: Multi-granularity Perception Network for Moment Retrieval in Videos, SIGIR 2022.
- 关于VMR/TSGV 的review: 
  - https://github.com/Huntersxsx/TSGV-Learning-List
  - https://huntersxsx.github.io/





Liu daizong - 北大 

- Progressively Guide to Attend: An Iterative Alignment Framework for Temporal Sentence Grounding, EMNLP 2021
  - https://github.com/liudaizong/IA-Net --- cannot reproduce
- Context-aware Biaffine Localizing Network for Temporal Sentence Grounding, CVPR 2021
  - https://github.com/liudaizong/CBLN --- no code
- Jointly Cross- and Self-Modal Graph Attention Network for Query-Based Moment Localization, MM2020
  - https://github.com/liudaizong/CSMGAN 
- [Hierarchical local- global transformer for temporal sentence grounding](https://arxiv.org/pdf/2208.14882.pdf)
- [Memory-guided semantic learning network for temporal sentence grounding](https://www.aaai.org/AAAI22Papers/AAAI-111.LiuD.pdf), AAAI 2022
- [exploring motion and appearance information for temporal sentence grounding](https://www.aaai.org/AAAI22Papers/AAAI-112.LiuD.pdf), AAAI 2022
- Unsupervised temporal video grounding with deep semantic clustering, AAAI 2022
- Reducing the vision and language bias for temporal sentence grounding, MM 2022
- [Skimming, Locating, then Perusing: A Human-Like Framework for Natural Language Video Localization](https://arxiv.org/abs/2207.13450), MM 2022
- A Hybird Alignment Loss for Temporal Moment Localization with Natural Language, ICME 2022
- [Learning to Focus on the Foreground for Temporal Sentence Grounding](https://aclanthology.org/2022.coling-1.490.pdf), Coling 2022
- Reasoning step-by-step: Temporal sentence localization in videos via deep rectification-modulation network, Coling 2020
- Adaptive Proposal Generation Network for Temporal Sentence Localization in Videos， EMNLP 2021
- Few-Shot Temporal Sentence Grounding via Memory-Guided Semantic Learning, TCSVT, 2022
- **MA3SRN**: [Exploring Optical-Flow-Guided Motion and Detection-Based Appearance for Temporal Sentence Grounding](https://arxiv.org/pdf/2203.02966.pdf)
  - SOTA, high.. no code
- https://scholar.google.com/citations?user=lUw7tVIAAAAJ&hl=en, 
  - many many many papers on VMR, but where are useful source codes?



Yitian Yuan - meituan 

- https://github.com/yytzsy/SCDM

  - Semantic Conditioned Dynamic Modulation for Temporal Sentence Grounding in Videos, NIPS 2019
- https://github.com/yytzsy/SCDM-TPAMI

  - Semantic Conditioned Dynamic Modulation for Temporal Sentence Grounding in Videos, PAMI 2021
- https://github.com/yytzsy/ABLR_code
  - To Find Where You Talk: Temporal Sentence Localization in Video with Attention Based Location Regression, AAAI 2019
- https://scholar.google.com.hk/citations?user=sCHGSHIAAAAJ&hl=en&oi=sra





一人多篇工作


- https://github.com/crodriguezo/TMLGA

  - Proposal-free Temporal Moment Localization of a Natural-Language Query in Video using Guided Attention, WACV 2020
- https://github.com/crodriguezo/DORi

  - DORi: Discovering Object Relationships for Moment Localization of a Natural, WACV 2021

- https://github.com/JaywongWang/CBP (tf)
  - Temporally Grounding Language Queries in Videos by Contextual Boundary-aware Prediction, AAAI 2020
  - https://github.com/yyuanad/Pytorch_C3D_Feature_Extractor, load feats
- https://github.com/JaywongWang/TGN （tf）
  - Temporally Grounding Natural Sentence in Video, EMNLP 2018






- https://github.com/runzhouge/MAC

  - MAC: Mining Activity Concepts for Language-based Temporal Localization, WACV 2019.
- https://github.com/BonnieHuangxin/SLTA

  - Cross-Modal Video Moment Retrieval with Spatial and Language-Temporal Attention, ICMR 2019
- https://github.com/Alvin-Zeng/DRN
  - Dense Regression Network for Video Grounding, CVPR2020
  - https://github.com/Alvin-Zeng/Awesome-Temporal-Action-Localization
- https://github.com/JonghwanMun/LGI4temporalgrounding

  - Local-Global Video-Text Interactions for Temporal Grounding, cvpr 2020
- https://github.com/forwchen/HVTG

  - Hierarchical Visual-Textual Graph for Temporal Activity Localization via Language, ECCV 2020

- https://github.com/WuJie1010/TSP-PRL
  - Tree-Structured Policy based Progressive Reinforcement Learning for Temporally Language Grounding in Video, AAAI2020
  - https://github.com/WuJie1010/Temporally-language-grounding， code base





Others


- https://github.com/r-cui/ViGA， 玩setting

  - Video Moment Retrieval from Text Queries via Single Frame Annotation, SIGIR 2022.

- https://github.com/gistvision/PSVL， zero-shot
  - Zero-shot Natural Language Video Localization. (ICCV 2021, Oral)
- 不错的codebase
  - https://github.com/Sense-X/X-Temporal 商汤video理解的base
  - https://github.com/WuJie1010/Temporally-language-grounding， 字节