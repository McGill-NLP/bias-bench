---
title: An Empirical Survey of the Effectiveness of Debiasing Techniques for Pre-trained Language Models
layout: default
---

# Overview
We investigate several different techniques for mitigating social bias in pre-trained language models in our paper [*An Empirical Survey of the Effectiveness of Debiasing Techniques for Pre-trained Language Models*](https://arxiv.org/abs/2110.08527) to be presented at ACL 2022. To track progress on the intrinsic bias benchmarks we investigate in our work, we introduce the leaderboard **Bias Bench**.

Bias Bench tracks the effectiveness of bias mitigation techniques across three intrinsic bias benchmarks: [StereoSet](https://aclanthology.org/2021.acl-long.416/), [CrowS-Pairs](https://aclanthology.org/2020.emnlp-main.154/), and the [Sentence Encoder Association Test (SEAT)](https://aclanthology.org/N19-1063/). Importantly, results on Bias Bench are ranked by StereoSet ICAT score, a metric which accounts for both the stereotype score and language modeling score of a model on StereoSet. We have included three of the strongest debiaising techniques we analyzed in our work as baselines: [Self-Debias](https://arxiv.org/abs/2103.00453), [CDA](https://arxiv.org/abs/2110.08527), and [Dropout](https://arxiv.org/abs/2010.06032).

We hope this leaderboard helps to better track progress in social bias mitigation research in NLP!

# Leaderboard
{% include leaderboard.html %}

# Submissions
To make a submission to Bias Bench, please contact nicholas.meade@mila.quebec.

# FAQ
**What metric is reported for SEAT in Bias Bench?** We report the average absolute effect size across the <i>gender</i>, <i>race</i>, and <i>religion</i> SEAT tests we evaluate in our work.

**What are the *ensemble* models reported in the leaderboard?** The baseline ensemble models (e.g., BERT + CDA) are aggregated results across three debiased models (one for each bias domain). The StereoSet language modeling scores and ICAT scores for these models are computed by aggregating results from each bias domain (e.g., gender bias) from each debiased model (e.g., gender debiased).

**How can I learn more about the intrinsic bias benchmarks used in Bias Bench?** To learn more about [StereoSet](https://aclanthology.org/2021.acl-long.416/), [CrowS-Pairs](https://aclanthology.org/2020.emnlp-main.154/), and the [Sentence Encoder Association Test (SEAT)](https://aclanthology.org/N19-1063/), we refer readers to their original respective works. Our [work](https://arxiv.org/abs/2110.08527) also summarizes each of these benchmarks.

**Why are there no SEAT results reported for the Self-Debias models?** There are no SEAT results for the Self-Debias models as Self-Debias is a *post-hoc* debiasing procedure which does not alter a model's internal representations. For more information on this, refer to [Section 3 of our work](https://arxiv.org/pdf/2110.08527.pdf#page=4).

**What do SS and LMS denote?** SS and LMS denote stereotype score and language modeling score, respectively.

# Ethical Considerations
In our [work](https://arxiv.org/pdf/2110.08527), we used a binary definition of gender while investigating gender bias in pre-trained language models.
While we fully recognize gender as non-binary, our survey closely follows the original methodology of the techniques explored in this work.
We believe it will be critical for future research in gender bias to use a more fluid definition of gender and we are encouraged by early work in this direction [(Manzini et al., 2019;](https://aclanthology.org/N19-1062/)[ Dinan et al., 2020b)](https://aclanthology.org/2020.emnlp-main.23/).
Similarly, our work makes use of a narrow definition of religious and racial bias.

We also note we do not investigate the *extrinsic* harm caused by any of the studied pre-trained language models, nor any potential *reduction* in harm by making use of any of our studied debiasing techniques.
In other words, we do not investigate how biases in pre-trained language models effect humans in real-world settings.

Finally, we highlight that all of the intrinsic bias benchmarks used in this work have only *positive* predictive power.
In other words, they can identify models as biased, but cannot verify a model as unbiased.
For example, a stereotype score of 50% on StereoSet or CrowS-Pairs is not indicative of an unbiased model.
Additionally, recent work demonstrated the potential unreliability of the bias benchmarks used in this work [(Blodgett et al., 2021)](https://aclanthology.org/2021.acl-long.81/).
Because of this, we caution readers from making definitive claims about bias in pre-trained language models based on these benchmarks alone.

# Citation
If you use our code, please cite the following paper:

    @inproceedings{meade_empirical_2022,
      address = {Online},
      title = {An Empirical Survey of the Effectiveness of Debiasing Techniques for Pre-trained Language Models},
      booktitle = {Proceedings of the 60th {Annual} {Meeting} of the {Association} for {Computational} {Linguistics},
      publisher = {Association for Computational Linguistics},
      author = {Meade, Nicholas and Poole-Dayan, Elinor and Reddy, Siva},
      month = may,
      year = {2022},
    }
