import shap
import lime.lime_tabular
import torch
import numpy as np

def generate_shap_explanations(model, background_data, test_data, feature_names):
    # SHAP Global
    model.eval()
    
    # We use DeepExplainer for PyTorch models
    # Background and test data should be tensors
    background_tensor = torch.FloatTensor(background_data)
    test_tensor = torch.FloatTensor(test_data)
    
    explainer = shap.DeepExplainer(model, background_tensor)
    shap_values = explainer.shap_values(test_tensor)
    
    return shap_values, explainer

def generate_lime_explanations(model, train_data, feature_names, instance_idx, test_data):
    # LIME Local
    model.eval()
    
    def predict_fn(x):
        x_tensor = torch.FloatTensor(x)
        if len(x_tensor.shape) == 2:
            # Add sequence dimension if needed
            x_tensor = x_tensor.unsqueeze(1)
        with torch.no_grad():
            preds = model(x_tensor)
            probs = torch.softmax(preds, dim=1)
        return probs.numpy()

    # Flatten train data for LIME
    train_flat = train_data.reshape(train_data.shape[0], -1)
    test_flat = test_data.reshape(test_data.shape[0], -1)
    
    explainer = lime.lime_tabular.LimeTabularExplainer(
        training_data=train_flat,
        feature_names=feature_names,
        class_names=["Benign", "Barrage", "Single-tone", "Successive-pulse", "Protocol-aware"],
        mode="classification"
    )
    
    instance = test_flat[instance_idx]
    exp = explainer.explain_instance(instance, predict_fn, num_features=len(feature_names))
    
    return exp
