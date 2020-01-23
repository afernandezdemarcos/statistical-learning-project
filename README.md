This project is an introduction to Computer Vision applications. The aim is to classify a set of garbage images into six categories: cardboard, glass, metal, paper, plastic and trash.

Due to the limited number of images directly available, transfer learning from MobileNet is applied to our CNN, which then will be trained on our specific dataset to improve the accuracy on the task of classifying garbage. Only last layers' weights will be updated.

Then it's time to extract the latent features that the CNN is capacble to "see". This features will serve as the input for a Support Vector Classifier.