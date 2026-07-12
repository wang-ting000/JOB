# CKM论文进展与GAP调研：过程与来源说明

## 报告任务

- 用户问题：基于《课题四问题梳理与解决计划》的11项问题，核验截至2026-07-12的CKM及相邻研究进展，判断是否仍有可发表的研究GAP。
- 受众：技术研究人员。
- 决策：哪些原始想法需要放弃“首创”表述，哪些可通过收缩问题边界形成主线论文。
- 时间范围：以2021年CKM奠基论文为起点，重点核验2024-2026年进展；检索截止2026-07-12。
- 比较基线：不是按“是否出现过关键词”判断，而是按概念、算法、系统仿真、真实测量/原型、跨场景验证和标准化六级证据判断。
- 成功标准：逐项给出已解决程度、最接近论文、尚缺的证据，以及可执行的研究重构建议。

## 证据边界与方法

1. 优先使用论文原文、作者项目页、开放论文仓库、IEEE/ACM/期刊页面、3GPP和ITU官方页面。
2. 代表性论文按“与11项问题直接相关、提供可核验实验或理论结论、能够改变GAP判断”的原则纳入；本报告不是系统综述或文献计量分析。
3. 2025-2026年预印本单独视为早期证据，不能与已同行评审论文等价。
4. “论文提出CKM可以降低导频”不等于已证明百流系统可运行；“构造误差更低”也不等于端到端吞吐、能耗和时延一定更优。
5. 未使用趋势图：论文成熟度是有序类别而不是连续定量时间序列，强行作图会制造虚假精度；报告采用逐项证据表和优先级表。

## 代表性一手来源清单

### 奠基、教程与近期总览

- Yong Zeng, Xiaoli Xu, “Toward Environment-Aware 6G Communications via Channel Knowledge Map,” IEEE Wireless Communications, 2021. DOI: https://doi.org/10.1109/MWC.001.2000327
- Yong Zeng et al., “A Tutorial on Environment-Aware Communications via Channel Knowledge Map for 6G,” IEEE Communications Surveys & Tutorials, 2024. https://arxiv.org/abs/2309.07460
- Zixiang Ren et al., “Channel Knowledge Map Construction: Recent Advances and Open Challenges,” 2025 preprint. https://arxiv.org/abs/2511.04944
- Shen Fu et al., “CKMDiff: a generative diffusion model for CKM construction via inverse problems with learned priors,” npj Wireless Technology, published 2026-07-03. https://www.nature.com/articles/s44459-026-00042-1

### 构图、跨场景与数据

- Ju-Hyung Lee, Andreas F. Molisch, “A Scalable and Generalizable Pathloss Map Prediction,” IEEE TWC, 2024. https://arxiv.org/abs/2312.03950
- Çağkan Yapar et al., “The First Pathloss Radio Map Prediction Challenge,” 2023/2024. https://arxiv.org/abs/2310.07658
- Stefanos Bakirtzis et al., “The First Indoor Pathloss Radio Map Prediction Challenge,” 2025. https://arxiv.org/abs/2501.13698
- Xiaoli Xu, Yong Zeng, “How Much Data is Needed for Channel Knowledge Map Construction?” IEEE TWC, 2024. https://arxiv.org/abs/2312.06966
- Di Wu et al., “CKMImageNet,” 2025 dataset paper. https://arxiv.org/abs/2504.09849
- RadioMapSeer dataset and challenge. https://radiomapseer.github.io/ and https://radiomapchallenge.github.io/results.html
- Kang Yang et al., “Generalizable Radio-Frequency Radiance Fields for Spatial Spectrum Synthesis,” CVPR 2026. https://kangyangg.com/projects/graf/
- Stefanos Bakirtzis et al., “Radio Propagation Modelling: To Differentiate or To Deep Learn, That Is The Question,” 2025 preprint; 13 cities and 10,000+ antennas. https://arxiv.org/abs/2509.19337
- Ali Saeizadeh et al., “AIRMap: AI-Generated Radio Maps for Wireless Digital Twins,” 2025/2026 preprint. https://arxiv.org/abs/2511.05522
- Prasenjit Dhara, Daniel Romero, “Learning the Channel Gain from Anywhere to Anywhere via Cross-environment Transformer Estimators,” 2026 preprint. https://arxiv.org/abs/2605.08211
- Kequan Zhou et al., “F4-CKM,” IEEE TCOM, 2026. https://arxiv.org/abs/2601.03601
- Jun Jiang et al., “CSI-CLIP++,” 2026 preprint. https://arxiv.org/abs/2606.25714
- Sheng Wang et al., “XFreq-GS: Cross-Frequency Wireless Radiation Field Reconstruction with 3D Gaussian Splatting,” 2026 preprint. https://arxiv.org/abs/2605.11432
- Zhonghao Jiu et al., “Anchor-CKM,” 2026-07 preprint. https://arxiv.org/abs/2607.01453

### 主动测量、参数反演与动态更新

- Jakob Hoydis et al., “Sionna RT: Differentiable Ray Tracing for Radio Propagation Modeling.” https://arxiv.org/abs/2303.11103
- Daniel Romero et al., “Spectrum Surveying: Active Radio Map Estimation with Autonomous UAVs,” 2022. https://arxiv.org/abs/2201.04125
- Polyzos et al., “Bayesian Active Learning for Sample Efficient 5G Radio Map Reconstruction,” IEEE TWC, 2024. https://doi.org/10.1109/TWC.2024.3483112
- Sun et al., “Flow Matching-Based Active Learning for Radio Map Construction with Low-Altitude UAVs,” ICASSP 2026 / 2025 preprint. https://arxiv.org/abs/2509.13822
- Jakob Hoydis et al., “Learning Radio Environments by Differentiable Ray Tracing,” IEEE TMLCN, 2024. https://arxiv.org/abs/2311.18558
- Jiang, Zhou, Zhong, “Trajectory Planning for UAV-Based Path Loss Data Collection to Enhance Permittivity Sensing in Ray-Tracing,” IEEE TWC, 2024. https://doi.org/10.1109/TWC.2024.3457016
- Wenlihan Lu et al., “mmDiff,” 2026 preprint. https://arxiv.org/abs/2605.26406
- An et al., “Taming Vision Priors for Data Efficient mmWave Channel Modeling,” 2026 preprint. https://arxiv.org/abs/2603.13383
- Fuhai Wang et al., “Radio-Frequency Inverse Rendering for Wireless Environment Modeling,” 2026 preprint. https://arxiv.org/abs/2604.07086
- Zhihan Zeng et al., “Sparse Gain Radio Map Reconstruction With Geometry Priors and Uncertainty-Guided Measurement Selection,” 2026 preprint. https://arxiv.org/abs/2604.05788
- Yunzhe Zhu et al., “Active Learning for CKM Construction via Bayesian Inference Diffusion Models,” 2026-06 preprint. https://arxiv.org/abs/2606.29862
- Wenlihan Lu et al., “Active Perception for Radio Map Reconstruction in Uncharted 3D Air-Ground Environments,” 2026-06 preprint. https://arxiv.org/abs/2606.12844
- Ting Wang et al., “Update Strategy for Channel Knowledge Map in Complex Environments,” 2025 preprint. https://arxiv.org/abs/2512.15154
- Jiang et al., “Dynamic Channel Knowledge Map Construction in MIMO-OFDM Systems,” 2025 preprint. https://arxiv.org/abs/2512.23470
- Qi et al., “A Novel 6G Dynamic Channel Map Based on a Hybrid Channel Model,” IEEE TVT, 2026 / preprint. https://arxiv.org/abs/2604.15083
- Kequan Zhou et al., “Location-Agnostic Channel Knowledge Map Construction for Dynamic Scenes,” 2026 preprint. https://arxiv.org/abs/2603.09273
- Chaoyue Zhang et al., “Prototyping and Experimental Results for ISAC-based Channel Knowledge Map,” IEEE TVT, 2025. https://arxiv.org/abs/2408.06164

### CKM辅助传输、复杂度与连续空间

- Di Wu et al., “Environment-Aware Hybrid Beamforming by Leveraging Channel Knowledge Map,” IEEE TWC, 2023. https://arxiv.org/abs/2206.08707
- Di Wu et al., “Prototyping and Experimental Results for Environment-Aware Millimeter Wave Beam Alignment via CKM,” IEEE TVT, 2024. https://arxiv.org/abs/2403.08200
- Xianling Wang et al., “Channel Knowledge Map-Aided Channel Prediction With Measurements-Based Evaluation,” IEEE TCOM, 2025. DOI: https://doi.org/10.1109/TCOMM.2024.3487310
- Wenjun Jiang et al., “Interference-Cancellation-Based CKM Construction and Its Applications to Channel Estimation,” IEEE TWC, 2025. https://arxiv.org/abs/2409.00461
- Shuaifei Chen et al., “Channel Map-Based Angle Domain Multiple Access for Cell-Free Massive MIMO Communications,” IEEE JSTSP, 2025. DOI: https://doi.org/10.1109/JSTSP.2025.3536289
- “Channel Knowledge Map Assisted Channel Estimation for an Extremely Large Antenna Array System,” IEEE Access, 2025. https://doi.org/10.1109/ACCESS.2025.3606947
- Juncong Zhou et al., “6D Channel Knowledge Map Construction via Bidirectional Wireless Gaussian Splatting,” 2025 preprint. https://arxiv.org/abs/2510.26166
- Le Zhao et al., “BeamCKMDiff,” 2026 preprint. https://arxiv.org/abs/2601.10207
- Chenghong Bian et al., “Eff-WRFGS,” 2026 preprint. https://arxiv.org/abs/2605.15324
- Zhaolin Wang et al., “Electromagnetic Signal and Information Theory: A Continuous-Aperture Array Perspective,” 2026 preprint. https://arxiv.org/abs/2605.12910
- Chongjun Ouyang et al., “Electromagnetic Degrees of Freedom for Continuous-Aperture Array Systems,” 2025 preprint. https://arxiv.org/abs/2502.14404
- Ahmed Hussain et al., “Spatial Degrees of Freedom in Near Field MIMO: Experimental Validation,” 2026 preprint. https://arxiv.org/abs/2602.21945

### 表示、标准与评测边界

- Jialin Wang et al., “Radio Environment Knowledge Pool for 6G Digital Twin Channel,” IEEE Communications Magazine, 2025. DOI: https://doi.org/10.1109/MCOM.003.2400168
- 3GPP TR 38.901 official archive (scenario-based channel model, not a CKM interchange standard): https://www.3gpp.org/ftp/Specs/archive/38_series/38.901/
- 3GPP TS 37.320 Release 18 (Minimization of Drive Tests data collection): referenced by Anchor-CKM; it standardizes measurement collection mechanisms, not CKM semantics.
- ITU-R M.2160 IMT-2030 framework: https://www.itu.int/rec/R-REC-M.2160-0-202311-I/en
- ICASSP pathloss radio-map challenge methodology: https://doi.org/10.1109/OJSP.2024.3419563

## 报告结构映射

- Technical summary → “技术结论”与“哪些想法已经不能再作为首创点”。
- Key findings → 11项问题逐项证据表、进展时间线表。
- Scope/data/definitions → “判断口径与证据等级”。
- Methodology → “如何判定GAP”。
- Limitations/robustness → “证据边界和不能下的结论”。
- Recommended next steps → 三条研究主线、首选主线及实验闭环。
- Further questions → 需要课题1数据补齐的可检验问题。
