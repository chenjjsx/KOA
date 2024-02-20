# Objective Bi-Modal Assessment of Knee Osteoarthritis Severity Grades: Model and Mechanism
## Introduction
Knee Osteoarthritis (KOA) is a highly prevalent disease, and the Kellgren-Lawrence (KL) grading scoring method plays a crucial role in determining the severity and formulating treatment plans for knee osteoarthritis. However, existing assessment methods rely on experts' interpretation of X-ray images, which may be subject to image overlap or blurring and variations in the professional expertise of physicians, especially during long-term and repeated examinations, posing potential risks to health. In this study, we developed a model for predicting the severity of knee osteoarthritis using thermal image combined with health data. We have developed a classification model that categorizes KOA grades into three distinct categories based on the Kellgren-Lawrence (KL) grading criteria.

## Methodology

Our proposed method includes feature weight optimization based on a Gaussian distribution function to enhance focus on different regions in knee images. Additionally, we employ the Synthetic Minority Over-sampling Technique (SMOTE) method to balance the dataset, ensuring the model is supported by sufficient samples across different pain grades. After completing the extraction of image features, we apply the gradient boosting tree algorithm for the pain grade classification task.

## Dataset and Evaluation

For the first time, we introduce the KOA dataset and evaluate it using the method proposed in this paper, achieving a classification accuracy of 89.29%, surpassing the best result of 85.71% from other models in comparative experiments. Furthermore, we use the new dataset for additional validation of the proposed method, achieving an accuracy of 70.83%, surpassing the best result of 62.5% for other models.

## Application Prospects

The model provides a new method for predicting the severity of KOA and provides important auxiliary information for physicians' diagnosis.
