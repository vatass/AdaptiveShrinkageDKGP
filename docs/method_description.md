# Adaptive Shrinkage Deep Kernel Gaussian Process: Method Description

## Introduction

The Adaptive Shrinkage Deep Kernel Gaussian Process (AS-DKGP) is a novel approach for personalized prediction of biomarker trajectories. This method combines the strengths of population-level models with subject-specific adaptations through an adaptive shrinkage framework.

## Method Overview

### Problem Formulation

We model the biomarker trajectory prediction as a regression problem:

$$f: \mathcal{U} \rightarrow \mathcal{Y}$$

where:
- $\mathcal{U} \in \mathbb{R}^K$ is the input space
- $\mathcal{Y} \in \mathbb{R}$ is the output space (biomarker value)

The input space $\mathcal{U}$ consists of:
- $\mathbf{X}$: Imaging features (145 ROIs)
- $\mathbf{M}$: Clinical covariates
- $T$: Time from baseline visit

### Model Components

Our framework consists of three main components:

1. **Population-level DKGP Model**: A deep kernel Gaussian process trained on data from all subjects in the training set.
2. **Subject-Specific DKGP Model**: A personalized model trained on individual subject data.
3. **Adaptive Shrinkage Estimator**: A meta-model that learns to optimally combine predictions from the population and subject-specific models.

## Deep Kernel Gaussian Process

The DKGP model combines deep neural networks with Gaussian processes:

$$f(\mathbf{x}) \sim \mathcal{GP}(0, k(\phi(\mathbf{x}), \phi(\mathbf{x}')))$$

where:
- $\phi(\cdot)$ is a deep neural network feature extractor
- $k(\cdot, \cdot)$ is a kernel function (e.g., RBF kernel)

This allows the model to learn complex, non-linear relationships in the data while maintaining the uncertainty quantification benefits of Gaussian processes.

## Adaptive Shrinkage Framework

For a new subject with limited observations, we combine the population and subject-specific predictions:

$$\hat{y}_{\text{combined}} = \alpha \cdot \hat{y}_{\text{population}} + (1 - \alpha) \cdot \hat{y}_{\text{subject-specific}}$$

where $\alpha \in [0, 1]$ is the shrinkage parameter.

### Oracle Alpha

The optimal shrinkage parameter $\alpha^*$ (oracle alpha) minimizes the expected prediction error:

$$\alpha^* = \arg\min_{\alpha} \mathbb{E}[(y_{\text{true}} - \hat{y}_{\text{combined}})^2]$$

### Learning the Shrinkage Function

We train an XGBoost regression model to predict the optimal shrinkage parameter based on:
- Population model prediction ($\hat{y}_p$)
- Population model variance ($V_p$)
- Subject-specific model prediction ($\hat{y}_s$)
- Subject-specific model variance ($V_s$)
- Number of observations ($T_{obs}$)

$$\alpha = g(\hat{y}_p, V_p, \hat{y}_s, V_s, T_{obs})$$

## Training Pipeline

1. **Population Model Training**:
   - Train a DKGP model on all training subjects
   - Extract deep kernel parameters

2. **Oracle Dataset Creation**:
   - For each validation subject:
     - Train a subject-specific model
     - Calculate the oracle alpha
     - Store predictions, variances, and oracle alpha

3. **Adaptive Shrinkage Training**:
   - Train an XGBoost model to predict the oracle alpha
   - Optimize hyperparameters using cross-validation

4. **Personalization**:
   - For each test subject:
     - Generate population predictions
     - Train subject-specific model with available observations
     - Use adaptive shrinkage to combine predictions

## Advantages

1. **Optimal Knowledge Transfer**: Automatically determines how much to rely on population vs. subject-specific knowledge.
2. **Uncertainty-Aware**: Incorporates prediction uncertainties from both models.
3. **Data-Efficient**: Performs well even with limited subject-specific observations.
4. **Interpretable**: Provides insights into when personalization is beneficial.

## Applications

The AS-DKGP framework is particularly useful for:
- Predicting disease progression in neurodegenerative disorders
- Personalized treatment planning
- Biomarker trajectory forecasting
- Clinical trial enrichment

## Conclusion

The Adaptive Shrinkage Deep Kernel Gaussian Process provides a principled approach to personalized prediction by optimally combining population and subject-specific models. This method is especially valuable in healthcare applications where individual variations are significant but data per subject is limited. 