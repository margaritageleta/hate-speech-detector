# A DistilBERT-Based Transfer Learning Approach for Hate Speech Detection 👍

### Motivation 💡
Identifying offensive contributions and hate speech would allow for the creation of safer online communities and social media spaces. 
This project aims to address the problem of identifying toxic comments on the web using English-only text. 

### Proposal 🛠
The latest methods perform sentiment analysis combining text features with neural network-based classifiers. Usually, text featurization models require large amounts of computational resources for training. We propose using a transfer learning approach based on the distilled version of BERT, DistilBERT, which is lighter in comparison to other state-of-the-art models, allowing deployment under low latency constraints. Different machine learning methods have been explored for classification: Random Forests, Support Vector Machines, or Multi-Layer Perceptrons. 

### Challenges 🚨
The main challenge has been the unbalancedness of the data with an inherent bias towards non-toxic comments. Moreover, the data contains inconsistencies in the labeling due to the manual tagging and the subjectivity of the task. Despite these caveats, our solution has been successful in obtaining high precision in the predictions with the additional relaxation of computational power requirements compared to previous methods.
