## Supervised Learning - Custom CNN (Saeed/ Kayal 🍁)

For the supervised learning experiment, a custom convolutional neural network (CNN) was designed to classify 96x96 histopathologic image patches into two classes: cancer present and cancer absent. The model was built from scratch to satisfy the project requirement of implementing a supervised deep learning solution rather than relying only on pretrained architectures.

The proposed CNN contains four convolutional blocks with filter sizes of 32, 64, 128, and 256. Each block uses two 3x3 convolution layers followed by ReLU activation, optional batch normalization, max pooling, and dropout. This design was selected to gradually learn low-level tissue textures in early layers and more discriminative cancer patterns in deeper layers. After feature extraction, global average pooling and a dense layer with dropout were used before the final sigmoid output layer for binary classification.

Several hyperparameters were prepared for experimentation, including the number of convolutional blocks, filter depth, dense layer size, dropout rate, batch size, and learning rate. The baseline configuration used dropout = 0.40, dense units = 256, Adam optimizer with learning rate = 0.001, batch size = 64, and 15 training epochs. Early stopping and learning rate reduction on plateau were applied to improve generalization and reduce overfitting.

Model performance was evaluated using validation and test accuracy, binary cross-entropy loss, precision, recall, and ROC-AUC. Accuracy and loss curves were saved for every run to support performance analysis. Training logs were exported as CSV files to make the experiment reproducible and easy to compare with the transfer learning and EfficientNet results developed by other group members.

This CNN serves as the baseline supervised learning model for Group 5. Its results will be compared against feature-extraction/transfer-learning approaches and a state-of-the-art model in order to analyze the trade-off between custom architecture design and stronger pretrained representations.
