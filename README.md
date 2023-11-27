# Knee Osteoarthritis Pain Assessment Method

## Introduction

Knee Osteoarthritis (KOA) stands as the most prevalent musculoskeletal disorder. Presently, the diagnosis of KOA relies on the subjective evaluation of symptoms and patient scores, lacking a standardized and objective diagnostic process. In this study, we present a method for objectively assessing KOA pain grades using predominantly thermal imaging in a multi-modal dataset. We have developed a classification model that categorizes KOA pain grades into three distinct categories based on the Kellgren-Lawrence (KL) grading criteria.

## Methodology

Our proposed method includes feature weight optimization based on a Gaussian distribution function to enhance focus on different regions in knee images. Additionally, we employ the Synthetic Minority Over-sampling Technique (SMOTE) method to balance the dataset, ensuring the model is supported by sufficient samples across different pain grades. After completing the extraction of image features, we apply the gradient boosting tree algorithm for the pain grade classification task.

## Dataset and Evaluation

For the first time, we introduce the KOA dataset and evaluate it using the method proposed in this paper, achieving a classification accuracy of 89.29%, surpassing the best result of 85.71% from other models in comparative experiments. Furthermore, we use the new dataset for additional validation of the proposed method, achieving an accuracy of 70.83%, surpassing the best result of 62.5% for other models.

## Application Prospects

The application of this method is expected to significantly alleviate the laborious task of pain grade scoring for physicians. It provides crucial auxiliary information for the diagnosis of knee diseases, further enhancing efficiency and diagnostic accuracy in the field of KOA.