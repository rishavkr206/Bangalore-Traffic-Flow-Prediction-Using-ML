import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.losses import MeanSquaredError
import joblib
import os
from datetime import datetime
import tensorflow as tf

def check_saved_model():
    model_path = 'traffic_model.h5'
    scaler_path = 'scaler.pkl'
    return os.path.exists(model_path) and os.path.exists(scaler_path)

def initialize_session_state():
    model_exists = check_saved_model()
    
    if 'model_trained' not in st.session_state:
        st.session_state.model_trained = model_exists
    if 'data_loaded' not in st.session_state:
        st.session_state.data_loaded = False
    if 'training_completed' not in st.session_state:
        st.session_state.training_completed = model_exists
    if 'predictor' not in st.session_state:
        st.session_state.predictor = None
        if model_exists:
            try:
                predictor = TrafficPredictor()
                predictor.load_model()
                st.session_state.predictor = predictor
                st.session_state.model_trained = True
                st.session_state.training_completed = True
            except Exception as e:
                st.error(f"Error loading saved model: {str(e)}")

def display_training_status():
    if st.session_state.data_loaded:
        st.sidebar.success("✅ Data loaded")
    else:
        st.sidebar.warning("⚠️ Upload data first")
        
    if st.session_state.training_completed:
        st.sidebar.success("✅ Model trained")
    else:
        st.sidebar.warning("⚠️ Train model")

class TrafficPredictor:
    def __init__(self):
        self.model = None
        self.scaler = StandardScaler()
        self.sequence_length = 10
        self.features = None
        
    def prepare_data(self, df):
        try:
            df['Date'] = pd.to_datetime(df['Date'], dayfirst=True)
        except:
            try:
                df['Date'] = pd.to_datetime(df['Date'], dayfirst=False)
            except:
                try:
                    df['Date'] = pd.to_datetime(df['Date'], format='ISO8601')
                except:
                    df['Date'] = pd.to_datetime(df['Date'], format='mixed')
        
        df['DayOfWeek'] = df['Date'].dt.dayofweek
        df['Month'] = df['Date'].dt.month
        df['Hour'] = df['Date'].dt.hour if 'Hour' in str(df['Date'].dtype) else 0
        
        categorical_columns = ['Area Name', 'Road/Intersection Name', 'Weather Conditions']
        for col in categorical_columns:
            if col in df.columns:
                df[col] = pd.Categorical(df[col]).codes
        
        self.features = ['Traffic Volume', 'Average Speed', 'Travel Time Index', 'Congestion Level',
                        'Road Capacity Utilization', 'Incident Reports', 'Environmental Impact',
                        'Public Transport Usage', 'Traffic Signal Compliance', 'Parking Usage',
                        'Pedestrian and Cyclist Count', 'Weather Conditions', 'Roadwork and Construction Activity',
                        'DayOfWeek', 'Month']
        
        defaults = {
            'Traffic Volume': df['Traffic Volume'].mean() if 'Traffic Volume' in df else 0,
            'Average Speed': df['Average Speed'].mean() if 'Average Speed' in df else 40,
            'Travel Time Index': 1.0,
            'Congestion Level': df['Congestion Level'].mean() if 'Congestion Level' in df else 5,
            'Road Capacity Utilization': df['Road Capacity Utilization'].mean() if 'Road Capacity Utilization' in df else 50,
            'Incident Reports': 0,
            'Environmental Impact': 50,
            'Public Transport Usage': 30,
            'Traffic Signal Compliance': 90,
            'Parking Usage': 70,
            'Pedestrian and Cyclist Count': 100,
            'Weather Conditions': 0,
            'Roadwork and Construction Activity': 0,
            'DayOfWeek': 0,
            'Month': 1
        }
        
        for feature in self.features:
            if feature not in df.columns:
                df[feature] = defaults[feature]
        
        data = df[self.features].values
        scaled_data = self.scaler.fit_transform(data)
        
        X, y = [], []
        for i in range(len(scaled_data) - self.sequence_length):
            X.append(scaled_data[i:(i + self.sequence_length)])
            y.append(scaled_data[i + self.sequence_length, 0])
            
        return np.array(X), np.array(y)
    
    def build_model(self, input_shape):
        model = Sequential([
            LSTM(64, return_sequences=True, input_shape=input_shape),
            Dropout(0.2),
            LSTM(32),
            Dropout(0.2),
            Dense(16, activation='relu'),
            Dense(1)
        ])
        model.compile(optimizer='adam', 
                    loss=MeanSquaredError(),
                    metrics=[MeanSquaredError()])
        return model
    
    def train(self, X_train, y_train, epochs=50, batch_size=32):
        if X_train.shape[0] == 0 or y_train.shape[0] == 0:
            raise ValueError("Empty training data")
            
        input_shape = (X_train.shape[1], X_train.shape[2])
        self.model = self.build_model(input_shape)
        
        history = self.model.fit(
            X_train, y_train,
            epochs=epochs,
            batch_size=batch_size,
            validation_split=0.2,
            verbose=1
        )
        return history
    
    def save_model(self, model_path='traffic_model.h5', scaler_path='scaler.pkl'):
        if self.model is None:
            raise ValueError("No model to save")
        self.model.save(model_path)
        joblib.dump(self.scaler, scaler_path)
        
    def load_model(self, model_path='traffic_model.h5', scaler_path='scaler.pkl'):
        if not (os.path.exists(model_path) and os.path.exists(scaler_path)):
            raise FileNotFoundError("Model or scaler file not found")
        self.model = load_model(model_path, compile=True)
        self.scaler = joblib.load(scaler_path)

def create_streamlit_app():
    st.title('Bangalore Traffic Flow Prediction')
    initialize_session_state()
    
    st.sidebar.header("Status")
    display_training_status()
    
    st.sidebar.header('Model Training')
    st.sidebar.subheader("Step 1: Upload Data")
    uploaded_file = st.sidebar.file_uploader("Upload training data (CSV)", type='csv')
    
    if uploaded_file is not None:
        try:
            with st.spinner("Loading and processing data..."):
                df = pd.read_csv(uploaded_file)
                st.session_state.data_loaded = True
                st.session_state.df = df
                st.sidebar.success(f"✅ Data loaded successfully! Shape: {df.shape}")
                st.sidebar.dataframe(df.head(3))

            st.sidebar.subheader("Step 2: Train Model")
            train_button = st.sidebar.button('Train Model', key='train_model')

            if train_button:
                predictor = TrafficPredictor()
                with st.spinner('Training in progress...'):
                    try:
                        X, y = predictor.prepare_data(df)
                        st.info(f"Prepared {X.shape[0]} sequences for training")
                        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
                        
                        history = predictor.train(X_train, y_train)
                        predictor.save_model()
                        
                        st.session_state.predictor = predictor
                        st.session_state.model_trained = True
                        st.session_state.training_completed = True
                        
                        st.success('✅ Model trained and saved successfully!')
                        st.line_chart(pd.DataFrame(history.history))
                        
                    except Exception as e:
                        st.error(f'❌ Error during training: {str(e)}')
                        
        except Exception as e:
            st.sidebar.error(f"❌ Error loading data: {str(e)}")
    
    st.header('Make Predictions')
    
    col1, col2 = st.columns(2)
    
    with col1:
        area_name = st.selectbox('Area Name', ['Electronic City', 'Whitefield', 'MG Road', 'Koramangala'])
        avg_speed = st.slider('Average Speed (km/h)', 0, 100, 40)
        congestion = st.slider('Congestion Level', 1, 10, 5)
        incidents = st.number_input('Incident Reports', 0, 100, 0)
        
    with col2:
        weather = st.selectbox('Weather Conditions', ['Clear', 'Rainy', 'Cloudy'])
        road_utilization = st.slider('Road Capacity Utilization (%)', 0, 100, 50)
        public_transport = st.slider('Public Transport Usage (%)', 0, 100, 30)
        construction = st.checkbox('Roadwork/Construction Activity')
    
    if st.button('Predict Traffic'):
        if not st.session_state.training_completed:
            st.error('Please complete both steps: 1) Upload Data and 2) Train Model')
        else:
            try:
                predictor = st.session_state.predictor
                
                input_data = pd.DataFrame({
                    'Traffic Volume': [0],
                    'Average Speed': [avg_speed],
                    'Travel Time Index': [1.0],
                    'Congestion Level': [congestion],
                    'Road Capacity Utilization': [road_utilization],
                    'Incident Reports': [incidents],
                    'Environmental Impact': [50],
                    'Public Transport Usage': [public_transport],
                    'Traffic Signal Compliance': [90],
                    'Parking Usage': [70],
                    'Pedestrian and Cyclist Count': [100],
                    'Weather Conditions': [['Clear', 'Rainy', 'Cloudy'].index(weather)],
                    'Roadwork and Construction Activity': [int(construction)],
                    'DayOfWeek': [datetime.now().weekday()],
                    'Month': [datetime.now().month]
                })
                
                scaled_input = predictor.scaler.transform(input_data)
                sequence_input = np.repeat(
                    scaled_input.reshape(1, 1, -1), 
                    predictor.sequence_length, 
                    axis=1
                )

                prediction = predictor.model.predict(sequence_input)
                original_scale_pred = predictor.scaler.inverse_transform(
                    np.array([[prediction[0][0]] + [0] * (input_data.shape[1] - 1)])
                )[0][0]

                st.success(f'Predicted Traffic Volume: {int(original_scale_pred)} vehicles/hour')
                
            except Exception as e:
                st.error(f'Error making prediction: {str(e)}')

if __name__ == '__main__':
    create_streamlit_app()