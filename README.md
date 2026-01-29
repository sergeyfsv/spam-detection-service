# ML-based Python SPAM Detection Service
## Enron dataset
https://www.kaggle.com/datasets/bayes2003/emails-for-spam-or-ham-classification-enron-2006
## SpamAssasin dataset
https://spamassassin.apache.org/old/publiccorpus/
## Ling-Spam dataset
https://www.kaggle.com/datasets/mandygu/lingspam-dataset
## spam-detection-service
Deployment-ready модель для HugginfFace
Endpoint: https://l0id-spam-detection-service.hf.space
## spam-detection-service-test (online)
Online тест модели https://l0id-spam-detection-service.hf.space с использованием ling-spam dataset
## spam-detection-service-test-offline (offline)
Offline тест модели подготовленной с spam-detection-train-workflow с использованием ling-spam dataset
## spam-detection-train-workflow
Код по нормализации данных Enron/SpamAssasin dataset и подготовки модели с кросс-валидацией и выбором порога классификации для минимизации False Positive
