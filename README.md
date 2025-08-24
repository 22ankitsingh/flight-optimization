# 🛩️ Flight Scheduling MVP

A comprehensive Flask-based web application for analyzing and optimizing flight schedules at Mumbai Airport using real flight data, machine learning models, and AI-powered insights.

## ✨ Features

### 📊 **Dashboard Analytics**
- Real-time flight KPIs and performance metrics
- Interactive visualizations using Plotly
- Delay analysis and trend identification

### 🕐 **Best Times Analysis**
- Optimal takeoff and landing time recommendations
- Hour-by-hour delay pattern analysis
- Data-driven scheduling insights

### 📈 **Traffic Analysis**
- Airport congestion patterns
- Busiest time slots identification
- Capacity planning insights

### ⚙️ **Schedule Tuning Simulator**
- ML-powered delay prediction
- Schedule change impact assessment
- Real-time optimization recommendations

### 🔗 **Cascading Impact Analysis**
- Network analysis of delay propagation
- High-impact flight identification
- Dependency mapping

### 🤖 **AI Assistant (Google Gemini)**
- Natural language query processing
- Context-aware flight insights
- Intelligent scheduling recommendations

## 🛠️ Tech Stack

- **Backend:** Flask (Python)
- **Frontend:** HTML, CSS, JavaScript, Bootstrap
- **Data Visualization:** Plotly
- **Machine Learning:** Scikit-learn, NetworkX
- **AI:** Google Gemini API
- **Data:** Real Mumbai Airport flight data

## 🚀 Quick Start

### Prerequisites
- Python 3.8+
- Google Gemini API key (optional, for AI features)

### Installation

1. **Clone the repository**
   ```bash
   git clone <repository-url>
   cd honeywell
   ```

2. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

3. **Set up environment variables**
   ```bash
   # Copy the example environment file
   cp env_example.txt .env
   
   # Edit .env and add your Gemini API key
   # Get your API key from: https://makersuite.google.com/app/apikey
   ```

4. **Run the application**
   ```bash
   python app.py
   ```

5. **Access the application**
   - Open your browser and go to `http://localhost:5000`
   - The application will automatically load real flight data from `data/` directory

## 📁 Project Structure

```
honeywell/
├── app.py                 # Main Flask application
├── models/
│   └── ml_models.py      # ML models and analytics
├── templates/            # HTML templates
│   ├── base.html
│   ├── index.html
│   ├── best_times.html
│   ├── busiest_slots.html
│   ├── schedule_tuning.html
│   ├── cascading_impact.html
│   └── chatbot.html
├── static/               # Static files
│   ├── css/
│   │   └── style.css
│   └── js/
│       └── app.js
├── data/                 # Flight data files
│   ├── arrival_flights.csv
│   └── departure_flights.csv
├── requirements.txt      # Python dependencies
└── README.md
```

## 🔧 Configuration

### Environment Variables

Create a `.env` file in the root directory:

```env
# Google Gemini API Key (optional)
GEMINI_API_KEY=your_gemini_api_key_here

# Flask Configuration
FLASK_ENV=development
FLASK_DEBUG=True
SECRET_KEY=your_secret_key_here
```

### Data Files

The application uses real Mumbai Airport flight data:
- `data/arrival_flights.csv` - Arrival flight data
- `data/departure_flights.csv` - Departure flight data

## 🤖 AI Features

### Google Gemini Integration

The AI assistant uses Google Gemini API to provide intelligent responses about:
- Flight scheduling optimization
- Delay pattern analysis
- Airport congestion insights
- Schedule tuning recommendations

**Setup:**
1. Get your API key from [Google AI Studio](https://makersuite.google.com/app/apikey)
2. Add it to your `.env` file
3. Restart the application

**Fallback Mode:**
If no API key is provided, the chatbot uses rule-based responses with the same flight data context.

## 📊 Machine Learning Models

The application includes several ML models:

1. **Delay Prediction Models**
   - Random Forest Regressor
   - Gradient Boosting Regressor
   - Support Vector Regression
   - Linear Regression

2. **Cascading Impact Analysis**
   - Network analysis using NetworkX
   - Betweenness centrality calculation
   - Degree centrality analysis

3. **Feature Engineering**
   - Time-based features (hour, day, month)
   - Categorical encoding
   - Delay pattern extraction

## 🎯 Usage Examples

### Dashboard
- View overall flight performance metrics
- Analyze delay patterns and trends
- Monitor airport capacity utilization

### Best Times Analysis
- Find optimal scheduling windows
- Avoid peak congestion hours
- Optimize for minimal delays

### Schedule Tuning
- Input flight number and proposed time
- Get ML-powered delay predictions
- Receive optimization recommendations

### AI Assistant
- Ask natural language questions
- Get contextual flight insights
- Receive scheduling recommendations

## 🔍 API Endpoints

- `GET /` - Dashboard
- `GET /best_times` - Best times analysis
- `GET /busiest_slots` - Traffic analysis
- `GET /schedule_tuning` - Schedule tuning simulator
- `GET /cascading_impact` - Cascading impact analysis
- `GET /chatbot` - AI assistant interface
- `POST /chat` - Chatbot API endpoint

## 🧪 Testing

Run the application and test all features:

1. **Dashboard:** Verify KPIs and charts load correctly
2. **Best Times:** Check delay analysis by hour
3. **Traffic Analysis:** Verify congestion patterns
4. **Schedule Tuning:** Test ML predictions
5. **Cascading Impact:** Check network analysis
6. **AI Assistant:** Test chatbot functionality

## 🚀 Deployment

### Local Development
```bash
python app.py
```

### Production Deployment
```bash
# Set production environment
export FLASK_ENV=production
export FLASK_DEBUG=False

# Run with production server
gunicorn -w 4 -b 0.0.0.0:5000 app:app
```

## 📈 Performance

- **Data Processing:** Handles 1000+ flight records efficiently
- **ML Models:** Trained on real flight data with cross-validation
- **Visualizations:** Interactive charts with real-time updates
- **AI Responses:** Context-aware with fallback mechanisms

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests if applicable
5. Submit a pull request

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🆘 Support

For issues and questions:
1. Check the documentation
2. Review the code comments
3. Open an issue on GitHub

## 🔮 Future Enhancements

- Real-time data integration
- Advanced ML models (Deep Learning)
- Mobile application
- API for external integrations
- Multi-airport support
- Weather integration
- Predictive maintenance

---

**Built with ❤️ for efficient flight scheduling and optimization**
