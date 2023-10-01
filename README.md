# weighted-smoothing
Mitigating the risks of Membership Inference Attacks (MIA) on neural network models is crucial for maintaining model privacy. The proposed Weighted Smoothing approach is a strategic answer to this challenge.

## Table of Contents
- [Project Summary](#project-summary)
- [Running the Code](#running-the-code)
- [Further Information](#further-information)
- [Contact](#contact)

## Project Summary

Membership Inference Attacks (MIA) threaten neural network models by compromising privacy. Despite the efforts of current differential privacy-based methods to alleviate this threat, a significant drop in accuracy is a common shortfall. This research is an in-depth exploration of the variation in MIA risk across different samples and a study into the potential reasons behind this variability. We introduce a tailored strategy, termed as "weighted smoothing". This technique selectively introduces noise to training samples, considering their class distribution, effectively mitigating MIA risks while preserving model accuracy. Our empirical analysis attests to the superior performance of our approach. It significantly undermines the effectiveness of MIA, bringing down the success rate of two advanced MIAs to near-random levels (i.e., a 0.5 success rate) with a negligible loss in accuracy.

## Running the Code

Execute the project using the `run.sh` file. Note that most datasets, such as Location30, Texas100, FACE, or HAM10000, are not available for download via official PyTorch channels. You will need to set up your data formats for successful execution.

## Further Information

Detailed introduction and code will be provided in the future. Stay tuned for more comprehensive resources and updates.

## Contact

Should you have any queries regarding the project, feel free to reach us via email. Your feedback and questions are always welcome.
wtd3gz@virginia.edu

