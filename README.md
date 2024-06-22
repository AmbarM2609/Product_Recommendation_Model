# Fashion-Recommendation-Model

Dataset Link : https://www.kaggle.com/datasets/paramaggarwal/fashion-product-images-dataset

This project demonstrates how to build a fashion recommendation system using Convolutional Neural Networks (CNN) and transfer learning. The model is designed to recommend similar fashion items based on image inputs.

## Table of Contents
- [Overview](#overview)
- [Dataset](#dataset)
- [Model Architecture](#model-architecture)
- [Feature Extraction](#feature-extraction)
- [Similarity Computation](#similarity-computation)
- [Results](#results)
- [Installation](#installation)
- [Usage](#usage)
- [Future Work](#future-work)
- [Acknowledgements](#acknowledgements)

## Overview
This project leverages deep learning techniques to provide a fashion recommendation system. Given an image of a fashion item, the model retrieves and suggests similar items from a predefined dataset. The system utilizes a pre-trained DenseNet121 model for feature extraction, followed by similarity computation using cosine similarity.

## Dataset
The dataset consists of a collection of fashion item images. Each image is labeled with a unique identifier and relevant attributes (e.g., type, category). The images are preprocessed and resized to fit the input dimensions required by the DenseNet121 model.

## Model Architecture
The model architecture involves the following steps:
1. **Base Model**: DenseNet121 pre-trained on ImageNet is used as the base model without the top fully connected layers.
2. **Feature Extraction Layer**: A GlobalMaxPooling2D layer is added on top of the base model to extract meaningful features from the images.
3. **Similarity Computation**: The extracted features are compared using cosine similarity to find the most similar items.

### Code Snippet for Model Definition
```python
from keras.applications.densenet import DenseNet121
from keras.layers import GlobalMaxPooling2D
from keras.models import Model

base_model = DenseNet121(weights='imagenet', include_top=False)
x = GlobalMaxPooling2D()(base_model.output)
model = Model(inputs=base_model.input, outputs=x)
```

## Feature Extraction
The feature extraction process involves passing each image through the DenseNet121 model and retrieving the feature vector from the GlobalMaxPooling2D layer. These feature vectors are then stored for similarity computation.

### Code Snippet for Feature Extraction
```python
def extract_features(image_path, model):
    img = image.load_img(image_path, target_size=(224, 224))
    img_data = image.img_to_array(img)
    img_data = np.expand_dims(img_data, axis=0)
    img_data = preprocess_input(img_data)
    
    features = model.predict(img_data)
    return features
```

## Similarity Computation
To recommend similar items, the feature vectors of the query image and all images in the dataset are compared using cosine similarity. The top N most similar items are then recommended.

### Code Snippet for Similarity Computation
```python
from sklearn.metrics.pairwise import cosine_similarity

def get_recommendations(query_image_path, model, stored_features, top_n=5):
    query_features = extract_features(query_image_path, model)
    similarities = cosine_similarity(query_features, stored_features)
    similar_items_indices = similarities.argsort()[0][-top_n:][::-1]
    return similar_items_indices
```

## Results
The results are visualized by displaying the query image alongside the top N recommended images. This visual feedback helps evaluate the effectiveness of the recommendation system.

### Visualization Example
```python
import matplotlib.pyplot as plt

def visualize_recommendations(query_image_path, recommended_image_paths):
    query_img = mpimg.imread(query_image_path)
    plt.figure(figsize=(15, 5))
    
    plt.subplot(1, len(recommended_image_paths) + 1, 1)
    plt.imshow(query_img)
    plt.title('Query Image')
    plt.axis('off')
    
    for i, img_path in enumerate(recommended_image_paths, start=2):
        img = mpimg.imread(img_path)
        plt.subplot(1, len(recommended_image_paths) + 1, i)
        plt.imshow(img)
        plt.title(f'Recommendation {i-1}')
        plt.axis('off')
    
    plt.show()
```

## Installation
To get started with the project, clone the repository and install the required dependencies.

```bash
git clone https://github.com/AmbarM2609/Fashion-Recommendation-Model.git
cd Fashion-Recommendation-Model
pip install -r requirements.txt
```

## Usage
1. **Prepare Dataset**: Ensure your dataset is in the correct format and paths are specified.
2. **Extract Features**: Run the feature extraction script to generate feature vectors for your dataset images.
3. **Compute Recommendations**: Use the recommendation function to get similar items for a given query image.
4. **Visualize Results**: Use the visualization script to see the recommendations.

### Example Usage
```python
# Extract features
features = extract_features('path_to_image', model)

# Get recommendations
recommended_indices = get_recommendations('path_to_query_image', model, stored_features)

# Visualize results
visualize_recommendations('path_to_query_image', [dataset_paths[i] for i in recommended_indices])
```

## Future Work
- **Enhanced Preprocessing**: Improve image preprocessing techniques for better feature extraction.
- **Additional Models**: Experiment with other pre-trained models like ResNet, VGG, or EfficientNet.
- **Fine-Tuning**: Fine-tune the pre-trained models on the specific fashion dataset to improve accuracy.
- **User Interface**: Develop a web or mobile application for easy user interaction with the recommendation system.


