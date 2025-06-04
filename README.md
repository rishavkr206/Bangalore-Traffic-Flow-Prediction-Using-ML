# Bangalore-Traffic-Flow-Prediction-Using-ML

# Bangalore Traffic Flow Prediction using Machine Learning

A comprehensive machine learning application for predicting traffic flow in Bangalore using LSTM neural networks. This project provides real-time traffic volume predictions based on various environmental and traffic parameters.

## 🚗 Project Overview

This application uses deep learning techniques to predict traffic volume in different areas of Bangalore. Built with Streamlit for an interactive web interface and TensorFlow for the machine learning model, it offers both training capabilities and real-time predictions.

## ✨ Key Features

- **Interactive Web Dashboard**: User-friendly interface built with Streamlit
- **Real-time Predictions**: Instant traffic volume predictions based on current conditions
- **Multiple Input Parameters**: Considers weather, road conditions, incidents, and more
- **LSTM Neural Network**: Advanced deep learning model for time-series prediction
- **Model Persistence**: Save and load trained models for future use
- **Visual Analytics**: Interactive charts and gauge visualizations
- **Multi-area Support**: Predictions for different Bangalore locations

## 🛠️ Technologies Used

- **Python 3.7+**
- **Streamlit** - Web application framework
- **TensorFlow/Keras** - Deep learning model
- **Pandas** - Data manipulation
- **NumPy** - Numerical computing
- **Scikit-learn** - Data preprocessing
- **Plotly** - Interactive visualizations
- **Joblib** - Model serialization

## 📋 Prerequisites

Make sure you have Python 3.7 or higher installed on your system.

## 🚀 Installation and Setup

### Step 1: Clone the Repository

```bash
git clone https://github.com/yourusername/bangalore-traffic-prediction.git
cd bangalore-traffic-prediction
```

### Step 2: Create Virtual Environment (Recommended)

```bash
# Create virtual environment
python -m venv traffic_env

# Activate virtual environment
# On Windows:
traffic_env\Scripts\activate

# On macOS/Linux:
source traffic_env/bin/activate
```

### Step 3: Install Required Dependencies

```bash
pip install -r requirements.txt
```

If `requirements.txt` is not available, install packages manually:

```bash
pip install streamlit pandas numpy scikit-learn tensorflow plotly joblib
```

### Step 4: Run the Application

```bash
streamlit run app.py
```

The application will automatically open in your default web browser at `http://localhost:8501`

## 📁 Project Structure

```
bangalore-traffic-prediction/
│
├── app.py                 # Main Streamlit application
├── requirements.txt       # Python dependencies
├── README.md             # Project documentation
├── data/                 # Sample data files (if any)
├── models/               # Saved model files
│   ├── traffic_model.h5  # Trained LSTM model
│   └── scaler.pkl        # Feature scaler
└── screenshots/          # Application screenshots
```

## 📊 How to Use

### 1. Upload Training Data
- Click on "Upload training data (CSV)" in the sidebar
- Upload a CSV file with traffic data containing relevant columns
- The system will automatically process and validate the data

### 2. Train the Model
- Adjust the number of epochs (10-100) using the slider
- Click "Train Model" to start the training process
- Monitor training progress through the displayed metrics

### 3. Make Predictions
- Select area name from dropdown (Electronic City, Whitefield, MG Road, Koramangala)
- Adjust various parameters:
  - Average Speed
  - Congestion Level
  - Weather Conditions
  - Road Capacity Utilization
  - Public Transport Usage
  - Incident Reports
  - Construction Activity
- Click "Predict Traffic" to get instant predictions

## 📈 Input Parameters

| Parameter | Description | Range/Options |
|-----------|-------------|---------------|
| Area Name | Location in Bangalore | Electronic City, Whitefield, MG Road, Koramangala |
| Average Speed | Vehicle speed in km/h | 0-100 |
| Congestion Level | Traffic congestion rating | 1-10 |
| Weather Conditions | Current weather | Clear, Rainy, Cloudy |
| Road Capacity Utilization | Road usage percentage | 0-100% |
| Public Transport Usage | PT usage percentage | 0-100% |
| Incident Reports | Number of reported incidents | 0-100 |
| Construction Activity | Ongoing roadwork | Yes/No |

## 🔧 Model Architecture

The application uses an LSTM (Long Short-Term Memory) neural network with the following architecture:

- **Input Layer**: Sequence of 10 time steps with multiple features
- **LSTM Layer 1**: 64 units with return sequences
- **Dropout Layer 1**: 20% dropout rate
- **LSTM Layer 2**: 32 units
- **Dropout Layer 2**: 20% dropout rate
- **Dense Layer 1**: 16 units with ReLU activation
- **Output Layer**: 1 unit for traffic volume prediction

## 📊 Data Format

Your training CSV should include the following columns:

```
Date, Traffic Volume, Average Speed, Travel Time Index, Congestion Level,
Road Capacity Utilization, Incident Reports, Environmental Impact,
Public Transport Usage, Traffic Signal Compliance, Parking Usage,
Pedestrian and Cyclist Count, Weather Conditions, 
Roadwork and Construction Activity, Area Name, Road/Intersection Name
```

## 🎯 Results and Visualization

The application provides:

1. **Training Metrics**: Real-time loss and accuracy curves during model training
2. **Prediction Gauge**: Interactive circular gauge showing predicted traffic volume
3. **Numerical Output**: Exact traffic volume in vehicles/hour
4. **Status Indicators**: Clear indicators for data loading and model training status

## 🚨 Troubleshooting

### Common Issues:

1. **Port Already in Use**:
   ```bash
   streamlit run app.py --server.port 8502
   ```

2. **Module Not Found Error**:
   ```bash
   pip install --upgrade -r requirements.txt
   ```

3. **Memory Issues**:
   - Reduce the number of epochs
   - Use smaller batch sizes
   - Close other applications

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 👨‍💻 Author


Project Link: https://github.com/rishavkr206/Bangalore-Traffic-Flow-Prediction-Using-ML.git
## 🙏 Acknowledgments

- Bangalore Traffic Police for inspiration
- Streamlit community for excellent documentation
- TensorFlow team for the robust ML framework

## 📱 Screenshots
        
*Add screenshots of your application here*

## 🔮 Future Enhancements

- [ ] Integration with real-time traffic APIs
- [ ] Mobile responsive design
- [ ] Advanced visualization dashboards
- [ ] Multi-city support
- [ ] Historical trend analysis
- [ ] Email/SMS alert system

---

⭐ **If you found this project helpful, please give it a star!** ⭐
