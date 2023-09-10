from typing import List
from model import Item
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import sklearn
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
import pickle

# In-memory storage for items
items = []
item_id_counter = 0  # Initialize the item_id counter

# Load the trained model
def load_model(model_filename):
    with open(model_filename, 'rb') as model_file:
        model = pickle.load(model_file)
    return model

# Load the scaler (normalizer)
def load_scaler(scaler_filename):
    with open(scaler_filename, 'rb') as scaler_file:
        scaler = pickle.load(scaler_file)
    return scaler

# Function to preprocess user input and make predictions
def predict_pulsar(model, scaler, input_data):
    # Preprocess the input data using the loaded scaler
    input_data_scaled = scaler.transform(input_data.reshape(1, -1))
    # Predict whether the record is a pulsar or not
    prediction = model.predict(input_data_scaled)
    return prediction


def item_predicted(features) -> int:
    # Load the trained model and scaler
    scaler_filename = "scaler.pkl"
    model_filename = "pulsar_model.pkl"
    trained_model = load_model(model_filename)
    loaded_scaler = load_scaler(scaler_filename)
    # Save prediction result into the model
    list = []
    for feature in features:
        list.append(float(feature))
    input_data = np.array(list)
    #item.prediction = int(predict_pulsar( trained_model, loaded_scaler, input_data))
    return int(predict_pulsar( trained_model, loaded_scaler, input_data))

def create_item(item: Item) -> Item:
    global item_id_counter  # Use the global item_id_counter
    item.prediction = item_predicted(item.features)
    # Assign the item_id to the item
    item.item_id = item_id_counter  
    item_id_counter = item_id_counter + 1
    #saving the complete item
    items.append(item)
    return item

def read_items() -> items:
    return items
    

def read_item(item_id: int) -> Item:
    if 0 <= item_id < len(items):
        return items[item_id]
    else:
        return None

def update_item(item_id: int, updated_item: Item) -> Item:
    # Check if the item with the given item_id exists
    if 0 <= item_id < len(items):
        item = items[item_id]
        # Update the fields of the existing item with the values from updated_item
        item.features = updated_item.features
        item.prediction = item_predicted(updated_item.features)
        return item

    return None

def delete_item(item_id: int) -> Item:
    if 0 <= item_id < len(items):
        deleted_item = items.pop(item_id)
        return deleted_item
    else:
        return None
