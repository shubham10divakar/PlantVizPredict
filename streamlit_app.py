import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
import seaborn as sns
import os
from PIL import Image
import random

initial_dataset_name = "nodataset"
# Predefined datasets
DATASETS = {
    "Iris": "https://raw.githubusercontent.com/mwaskom/seaborn-data/master/iris.csv",
    "PlantVillageDataset": "data/plantvillagedataset/train/color",
    "Crop NPK,Temp and Humidty Content Data": "data/npk.csv",
    "Wine Quality": "https://raw.githubusercontent.com/uciml/red-wine-quality/master/winequality-red.csv"
}

# Load Data
@st.cache_data
def load_data(data_path):
    #data = pd.read_csv(data_path'data/Crop_recommendation.csv')  # Replace with your data file
    data = pd.read_csv(data_path)
    return data



# Load images from the folder path
def load_images_from_folder(folder_path):
    categories = [d for d in os.listdir(folder_path) if os.path.isdir(os.path.join(folder_path, d))]
    images = {}
    for category in categories:
        category_path = os.path.join(folder_path, category)
        images[category] = [os.path.join(category_path, f) for f in os.listdir(category_path) if f.endswith(('png', 'jpg', 'jpeg', 'JPG'))]
    return images

# Display images
def display_images(images, category, num_images=5):
    st.write(f"### {category} - Random Sample of {num_images} Images")
    selected_images = random.sample(images[category], min(num_images, len(images[category])))
    #st.write(images)
    #st.write(images[category])
    for img_path in selected_images:
        img = Image.open(img_path)
        #st.write(img_path)
        st.image(img, caption=os.path.basename(img_path), use_column_width=True)

# Display images in a gallery
def display_images_gallery(images, category, num_images=100000, num_columns=5):
    st.write(f"### {category} - Gallery View")
    selected_images = random.sample(images[category], min(num_images, len(images[category])))
    
    # Calculate number of rows based on number of columns
    num_rows = (len(selected_images) // num_columns) + 1

    # Display images in a grid
    for i in range(num_rows):
        row_images = selected_images[i*num_columns : (i+1)*num_columns]
        st.image(row_images, caption=[os.path.basename(img_path) for img_path in row_images], width=200)

# Display csv data
def display_csv_data(data):
    #st.write("### Plant NPK")
    st.dataframe(data)

# Display data
def display_data():
    st.write("### Plant Disease Data")
    dataset_name = st.selectbox("Choose a dataset to display:", list(DATASETS.keys()))
    #st.write(dataset_name)
    #st.write(DATASETS[dataset_name])
    
    if dataset_name == 'PlantVillageDataset':
        # Summary of PlantVillage dataset
        st.title("PlantVillage Dataset Summary")
        st.write("""
                 The PlantVillage dataset is a comprehensive collection of images related to plant diseases and healthy plant leaves, 
        aimed at advancing research in agricultural technology and plant pathology. It serves as a valuable resource for 
        researchers, developers, and farmers to diagnose and manage crop diseases effectively.
        """)
        # Categories of data
        st.write("""
                 ### Categories of Data:
                     - Fungal diseases
                     - Bacterial diseases
                     - Viral diseases
                     - Nutrient deficiency symptoms
                     - Various stresses affecting plant health
                     """)
        st.write(""" There are total 38 categories of plant leaf images. """)
        
        folder_path = DATASETS[dataset_name]
        images = load_images_from_folder(folder_path)
        #st.write('Loaded images')
        categories = list(images.keys())
        
        st.write(' ')
        
        # Add a download link for the dataset
        st.markdown("[Download PlantVillage Dataset](https://github.com/spMohanty/PlantVillage-Dataset/tree/master/raw/color)")
            
        #Sidebar - Choose category
        st.write("Choose Category")
        selected_category = st.selectbox("Select a category", categories)
            
        if selected_category:
            #display_images(images, selected_category)
            display_images_gallery(images, selected_category)
            
    elif dataset_name=='Crop NPK,Temp and Humidty Content Data':
         display_csv_data(load_data('data/npk.csv'))
         
    elif dataset_name=='Iris':
        # Input field for CSV URL
        csv_url = "https://raw.githubusercontent.com/mwaskom/seaborn-data/master/iris.csv"

        if csv_url:
            try:
                # Read CSV data from URL
                df = pd.read_csv(csv_url)
           
                # Display the data
                st.write("### CSV Data:")
                st.dataframe(df)
       
            except Exception as e:
                st.error(f"Error: {e}")
    
    else:
        # Input field for CSV URL
        csv_url = st.text_input("https://raw.githubusercontent.com/mwaskom/seaborn-data/master/iris.csv")

        if csv_url:
            try:
                # Read CSV data from URL
                df = pd.read_csv(csv_url)
           
                # Display the data
                st.write("### CSV Data:")
                st.dataframe(df)
       
            except Exception as e:
                st.error(f"Error: {e}")
    
    
    

# Train model
def train_model(data):
    X = data.drop('disease', axis=1)
    y = data['disease']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
    model = RandomForestClassifier()
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    return model, accuracy

# Predict disease
def predict_disease(model, input_data):
    prediction = model.predict([input_data])
    return prediction

# Data exploration
def explore_data():
    display_data()

# Visualization
def visualize_data():
    display_data()

# Future works
def future_works():
    st.write("### Future Works")
    st.write("""
        - Improve the model accuracy by experimenting with different algorithms and hyperparameters.
        - Add more features to the dataset.
        - Integrate a user-friendly interface for inputting new data.
        - Provide detailed analysis and insights on the predictions.
    """)
    

# Display app details
def display_app_details():
    st.title("PlantVizPredict Visualizer")
    st.write("""
        Welcome to PlantViz, your tool for predicting and visualizing plant diseases.
        Explore various features such as data analysis, model training, disease prediction, and more.
        Use the sidebar to navigate through different sections.
    """)
    st.write("### Developed by Subham Divakar")

# Streamlit app layout
def main():
    st.title("PlantVizPredict - Plant Disease Data Visualization, Prediction and Analysis Tool")
    
    #display_app_details()
    chosen_dataset_id=0
    
    # Sidebar
    st.sidebar.title("PlantVizPredict")
    st.sidebar.image("images/plantdiseasedetection.jpg", use_column_width=True)  # Replace with your image path
    section = st.sidebar.radio("Go to", ["Home", "Data Explorer", "Model", "Prediction", "Visualization", "Future Works"])
    
    # Sidebar - Show chosen option
    #st.sidebar.title("Chosen Dataset")
    #st.sidebar.write(f"You have selected: **{initial_dataset_name}**")
    
    st.sidebar.title("Social Media Links")
    st.sidebar.write("""
        Connect with me on social media:
        - [My Website](https://shubham10divakar.github.io/showcasehub/)
        - [Github](https://github.com/shubham10divakar)
        - [LinkedIn](https://www.linkedin.com/in/subham-divakar-a7420a12a/)
        - [pip installable apps by me](https://pypi.org/user/subhamdivakar10/)
    """)
    
    
    # Navigation sections
    if section == "Home":
        display_app_details()
    elif section == "Data Explorer":
        #chosen_dataset_id = display_data(data)
        explore_data()
    elif section == "Model":
        st.write("Under Development")
        #model, accuracy = train_model(data)
        #st.write(f"### Model Accuracy: {accuracy * 100:.2f}%")
    elif section == "Prediction":
        #st.write("### Predict Plant Disease")
        #input_data = [st.number_input(f'Feature {i}', min_value=0.0, max_value=1.0, value=0.5) for i in range(data.shape[1] - 1)]
        #if st.button("Predict"):
         #   model, _ = train_model(data)  # Ensure the model is trained
          #  prediction = predict_disease(model, input_data)
           # st.write(f"Predicted Disease: {prediction[0]}")
           st.write('Under Development')
    elif section == "Visualization":
        #visualize_data()
        st.write('Under Development')
    elif section == "Future Works":
        future_works()

if __name__ == "__main__":
    main()
