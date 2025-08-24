import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import json
import plotly.graph_objs as go
import plotly.utils
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.linear_model import LinearRegression
from sklearn.svm import SVR
import networkx as nx
import os
import warnings
warnings.filterwarnings('ignore')

class FlightAnalytics:
    def __init__(self):
        self.arrival_model = None
        self.departure_model = None
        self.scaler = StandardScaler()
        self.label_encoders = {}
        self.models = {}
        self._ensure_data_files()
        self._train_models()
    
    def _ensure_data_files(self):
        """Check if data files exist"""
        if not os.path.exists('data'):
            os.makedirs('data')
        
        # Check if real data files exist, if not create sample data
        if not os.path.exists('data/arrival_flights.csv'):
            self._create_sample_arrival_data()
        
        if not os.path.exists('data/departure_flights.csv'):
            self._create_sample_departure_data()
    
    def _create_sample_arrival_data(self):
        """Generate realistic arrival flight data for Mumbai Airport"""
        np.random.seed(42)
        
        # Airlines operating in Mumbai
        airlines = ['AI', 'SG', '6E', 'UK', 'G8', 'I5', 'QR', 'EK', 'SV', 'TK']
        origins = ['DEL', 'BLR', 'CCU', 'MAA', 'HYD', 'GOI', 'DXB', 'DOH', 'LHR', 'SIN']
        
        data = []
        start_date = datetime.now() - timedelta(days=7)
        
        for day in range(7):
            current_date = start_date + timedelta(days=day)
            
            # Generate 80-120 flights per day
            num_flights = np.random.randint(80, 121)
            
            for i in range(num_flights):
                flight_num = f"{np.random.choice(airlines)}{np.random.randint(100, 999)}"
                origin = np.random.choice(origins)
                
                # Scheduled time (more flights during peak hours)
                hour_weights = [0.5, 0.3, 0.2, 0.3, 0.5, 1.0, 2.0, 3.0, 2.5, 2.0, 1.5, 1.8, 
                              2.2, 2.0, 2.5, 3.0, 2.8, 2.2, 1.8, 1.5, 1.2, 1.0, 0.8, 0.6]
                hour = np.random.choice(range(24), p=np.array(hour_weights)/sum(hour_weights))
                minute = np.random.choice([0, 15, 30, 45])
                
                scheduled_time = current_date.replace(hour=hour, minute=minute, second=0)
                
                # Generate delays (more likely during peak hours and bad weather days)
                base_delay = np.random.exponential(scale=15) if np.random.random() < 0.3 else 0
                
                # Peak hour penalty
                if hour in [7, 8, 9, 17, 18, 19, 20]:
                    base_delay += np.random.exponential(scale=10)
                
                # Weather factor (random bad weather days)
                if day in [2, 5] and np.random.random() < 0.4:
                    base_delay += np.random.exponential(scale=25)
                
                actual_delay = max(0, base_delay + np.random.normal(0, 5))
                actual_time = scheduled_time + timedelta(minutes=actual_delay)
                
                data.append({
                    'flight_number': flight_num,
                    'origin': origin,
                    'scheduled_arrival': scheduled_time.strftime('%Y-%m-%d %H:%M:%S'),
                    'actual_arrival': actual_time.strftime('%Y-%m-%d %H:%M:%S'),
                    'status': 'Delayed' if actual_delay > 15 else 'On Time',
                    'gate': f"A{np.random.randint(1, 25)}"
                })
        
        df = pd.DataFrame(data)
        df.to_csv('data/arrival_flights.csv', index=False)
    
    def _create_sample_departure_data(self):
        """Generate realistic departure flight data for Mumbai Airport"""
        np.random.seed(43)
        
        airlines = ['AI', 'SG', '6E', 'UK', 'G8', 'I5', 'QR', 'EK', 'SV', 'TK']
        destinations = ['DEL', 'BLR', 'CCU', 'MAA', 'HYD', 'GOI', 'DXB', 'DOH', 'LHR', 'SIN']
        
        data = []
        start_date = datetime.now() - timedelta(days=7)
        
        for day in range(7):
            current_date = start_date + timedelta(days=day)
            num_flights = np.random.randint(85, 125)
            
            for i in range(num_flights):
                flight_num = f"{np.random.choice(airlines)}{np.random.randint(100, 999)}"
                destination = np.random.choice(destinations)
                
                # Scheduled departure times
                hour_weights = [1.0, 0.5, 0.3, 0.5, 1.0, 2.5, 3.0, 2.0, 1.5, 1.2, 1.0, 1.5,
                              2.0, 1.8, 2.2, 2.5, 3.0, 2.8, 2.5, 2.0, 1.8, 1.5, 1.2, 1.0]
                hour = np.random.choice(range(24), p=np.array(hour_weights)/sum(hour_weights))
                minute = np.random.choice([0, 15, 30, 45])
                
                scheduled_time = current_date.replace(hour=hour, minute=minute, second=0)
                
                # Generate delays
                base_delay = np.random.exponential(scale=12) if np.random.random() < 0.35 else 0
                
                # Peak hour penalty
                if hour in [6, 7, 8, 16, 17, 18, 19]:
                    base_delay += np.random.exponential(scale=8)
                
                # Weather and operational delays
                if day in [2, 5] and np.random.random() < 0.3:
                    base_delay += np.random.exponential(scale=20)
                
                actual_delay = max(0, base_delay + np.random.normal(0, 4))
                actual_time = scheduled_time + timedelta(minutes=actual_delay)
                
                data.append({
                    'flight_number': flight_num,
                    'destination': destination,
                    'scheduled_departure': scheduled_time.strftime('%Y-%m-%d %H:%M:%S'),
                    'actual_departure': actual_time.strftime('%Y-%m-%d %H:%M:%S'),
                    'status': 'Delayed' if actual_delay > 15 else 'On Time',
                    'gate': f"B{np.random.randint(1, 30)}"
                })
        
        df = pd.DataFrame(data)
        df.to_csv('data/departure_flights.csv', index=False)
    
    def load_arrival_data(self):
        """Load arrival flight data"""
        df = pd.read_csv('data/arrival_flights.csv')
        
        # Check if this is real data (has different column names)
        if 'FlightNumber' in df.columns:
            # Real data format - convert to expected format
            df = df.rename(columns={
                'FlightNumber': 'flight_number',
                'Date': 'date',
                'From': 'origin',
                'To': 'destination',
                'Aircraft': 'aircraft',
                'FlightTime': 'flight_time',
                'STD': 'scheduled_departure',
                'ATD': 'actual_departure',
                'STA': 'scheduled_arrival',
                'ATA': 'actual_arrival'
            })
            
            # Convert time columns to datetime
            # Handle different time formats (with and without seconds)
            def parse_time_with_fallback(date_str, time_str):
                try:
                    # Try with seconds first
                    combined = f"{date_str} {time_str}"
                    return pd.to_datetime(combined, format='%d-%b-%y %H:%M:%S', errors='coerce')
                except:
                    try:
                        # Try without seconds
                        combined = f"{date_str} {time_str}"
                        return pd.to_datetime(combined, format='%d-%b-%y %H:%M', errors='coerce')
                    except:
                        # Try with mixed format
                        return pd.to_datetime(f"{date_str} {time_str}", errors='coerce')
            
            # Parse scheduled times (usually have seconds)
            df['scheduled_arrival'] = pd.to_datetime(df['date'] + ' ' + df['scheduled_arrival'], format='%d-%b-%y %H:%M:%S', errors='coerce')
            
            # Parse actual times (may not have seconds)
            actual_times = []
            for _, row in df.iterrows():
                try:
                    # Try with seconds first
                    combined = f"{row['date']} {row['actual_arrival']}"
                    dt = pd.to_datetime(combined, format='%d-%b-%y %H:%M:%S', errors='coerce')
                    if pd.isna(dt):
                        # Try without seconds
                        dt = pd.to_datetime(combined, format='%d-%b-%y %H:%M', errors='coerce')
                    actual_times.append(dt)
                except:
                    actual_times.append(pd.NaT)
            
            df['actual_arrival'] = actual_times
            
            # Add status column
            df['status'] = 'On Time'
            df.loc[df['actual_arrival'] > df['scheduled_arrival'], 'status'] = 'Delayed'
            
            # Add gate column (simulated)
            df['gate'] = 'A' + (df.index % 25 + 1).astype(str)
            
        return df
    
    def load_departure_data(self):
        """Load departure flight data"""
        df = pd.read_csv('data/departure_flights.csv')
        
        # Check if this is real data (has different column names)
        if 'FlightNumber' in df.columns:
            # Real data format - convert to expected format
            df = df.rename(columns={
                'FlightNumber': 'flight_number',
                'Date': 'date',
                'From': 'origin',
                'To': 'destination',
                'Aircraft': 'aircraft',
                'FlightTime': 'flight_time',
                'STD': 'scheduled_departure',
                'ATD': 'actual_departure',
                'STA': 'scheduled_arrival',
                'ATA': 'actual_arrival'
            })
            
            # Convert time columns to datetime
            # Handle different time formats (with and without seconds)
            def parse_time_with_fallback(date_str, time_str):
                try:
                    # Try with seconds first
                    combined = f"{date_str} {time_str}"
                    return pd.to_datetime(combined, format='%d-%b-%y %H:%M:%S', errors='coerce')
                except:
                    try:
                        # Try without seconds
                        combined = f"{date_str} {time_str}"
                        return pd.to_datetime(combined, format='%d-%b-%y %H:%M', errors='coerce')
                    except:
                        # Try with mixed format
                        return pd.to_datetime(f"{date_str} {time_str}", errors='coerce')
            
            # Parse scheduled times (usually have seconds)
            df['scheduled_departure'] = pd.to_datetime(df['date'] + ' ' + df['scheduled_departure'], format='%d-%b-%y %H:%M:%S', errors='coerce')
            
            # Parse actual times (may not have seconds)
            actual_times = []
            for _, row in df.iterrows():
                try:
                    # Try with seconds first
                    combined = f"{row['date']} {row['actual_departure']}"
                    dt = pd.to_datetime(combined, format='%d-%b-%y %H:%M:%S', errors='coerce')
                    if pd.isna(dt):
                        # Try without seconds
                        dt = pd.to_datetime(combined, format='%d-%b-%y %H:%M', errors='coerce')
                    actual_times.append(dt)
                except:
                    actual_times.append(pd.NaT)
            
            df['actual_departure'] = actual_times
            
            # Add status column
            df['status'] = 'On Time'
            df.loc[df['actual_departure'] > df['scheduled_departure'], 'status'] = 'Delayed'
            
            # Add gate column (simulated)
            df['gate'] = 'B' + (df.index % 30 + 1).astype(str)
            
        return df
    
    def _train_models(self):
        """Train enhanced ML models for delay prediction"""
        try:
            arrivals = self.load_arrival_data()
            departures = self.load_departure_data()
            
            # Prepare features for arrival delays
            arrival_features = self._prepare_features(arrivals, 'arrival')
            departure_features = self._prepare_features(departures, 'departure')
            
            # Train arrival delay model
            if not arrival_features.empty:
                self.models['arrival_delay'] = self._train_delay_model(arrival_features, 'arrival')
            
            # Train departure delay model
            if not departure_features.empty:
                self.models['departure_delay'] = self._train_delay_model(departure_features, 'departure')
            
            # Train cascading impact model
            self.models['cascading_impact'] = self._train_cascading_model(arrivals, departures)
            
        except Exception as e:
            print(f"Error training models: {e}")
    
    def _prepare_features(self, df, flight_type):
        """Prepare features for ML models"""
        try:
            if df.empty:
                return pd.DataFrame()
            
            # Create features
            df = df.copy()
            
            # Extract time-based features
            time_col = 'scheduled_arrival' if flight_type == 'arrival' else 'scheduled_departure'
            actual_col = 'actual_arrival' if flight_type == 'arrival' else 'actual_departure'
            
            df[time_col] = pd.to_datetime(df[time_col])
            df[actual_col] = pd.to_datetime(df[actual_col])
            
            df['hour'] = df[time_col].dt.hour
            df['day_of_week'] = df[time_col].dt.day_name()
            df['month'] = df[time_col].dt.month
            df['is_weekend'] = df[time_col].dt.weekday >= 5
            
            # Calculate delays
            df['delay_minutes'] = (df[actual_col] - df[time_col]).dt.total_seconds() / 60
            df['delay_minutes'] = df['delay_minutes'].fillna(0)
            
            # Encode categorical variables
            categorical_cols = ['origin', 'destination', 'day_of_week']
            for col in categorical_cols:
                if col in df.columns:
                    le = LabelEncoder()
                    df[f'{col}_encoded'] = le.fit_transform(df[col].astype(str))
                    self.label_encoders[f'{col}_{flight_type}'] = le
            
            # Select features for modeling
            feature_cols = ['hour', 'month', 'is_weekend', 'delay_minutes']
            for col in categorical_cols:
                if f'{col}_encoded' in df.columns:
                    feature_cols.append(f'{col}_encoded')
            
            return df[feature_cols].dropna()
            
        except Exception as e:
            print(f"Error preparing features: {e}")
            return pd.DataFrame()
    
    def _train_delay_model(self, features_df, flight_type):
        """Train delay prediction model"""
        try:
            if features_df.empty or 'delay_minutes' not in features_df.columns:
                return None
            
            # Prepare X and y
            X = features_df.drop('delay_minutes', axis=1)
            y = features_df['delay_minutes']
            
            # Split data
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
            
            # Train multiple models
            models = {
                'random_forest': RandomForestRegressor(n_estimators=100, random_state=42),
                'gradient_boosting': GradientBoostingRegressor(n_estimators=100, random_state=42),
                'linear_regression': LinearRegression(),
                'svr': SVR(kernel='rbf')
            }
            
            best_model = None
            best_score = -float('inf')
            
            for name, model in models.items():
                try:
                    model.fit(X_train, y_train)
                    score = model.score(X_test, y_test)
                    
                    if score > best_score:
                        best_score = score
                        best_model = model
                        
                except Exception as e:
                    print(f"Error training {name}: {e}")
                    continue
            
            return best_model
            
        except Exception as e:
            print(f"Error training delay model: {e}")
            return None
    
    def _train_cascading_model(self, arrivals, departures):
        """Train model for cascading impact analysis"""
        try:
            # Create network features
            G = nx.Graph()
            
            # Add flights as nodes
            for _, flight in arrivals.iterrows():
                G.add_node(flight['flight_number'], 
                          type='arrival', 
                          gate=flight.get('gate', 'A1'),
                          scheduled=flight['scheduled_arrival'])
            
            for _, flight in departures.iterrows():
                G.add_node(flight['flight_number'], 
                          type='departure',
                          gate=flight.get('gate', 'B1'), 
                          scheduled=flight['scheduled_departure'])
            
            # Calculate centrality measures
            centrality = nx.betweenness_centrality(G)
            degree_centrality = nx.degree_centrality(G)
            
            return {
                'graph': G,
                'betweenness_centrality': centrality,
                'degree_centrality': degree_centrality
            }
            
        except Exception as e:
            print(f"Error training cascading model: {e}")
            return None
    
    def calculate_delays(self, df, flight_type):
        """Calculate delay statistics for flights"""
        time_col = 'scheduled_arrival' if flight_type == 'arrival' else 'scheduled_departure'
        actual_col = 'actual_arrival' if flight_type == 'arrival' else 'actual_departure'
        
        # The datetime columns should already be parsed in load_arrival_data/load_departure_data
        # Just ensure they are datetime objects
        df[time_col] = pd.to_datetime(df[time_col], errors='coerce')
        df[actual_col] = pd.to_datetime(df[actual_col], errors='coerce')
        
        # Calculate delay in minutes
        df['delay_minutes'] = (df[actual_col] - df[time_col]).dt.total_seconds() / 60
        
        # Handle NaN values (flights with missing actual times)
        df['delay_minutes'] = df['delay_minutes'].fillna(0)
        
        # Handle date boundary issues (flights that cross midnight)
        # If delay is more than 12 hours, it's likely a date boundary issue
        # Adjust by adding/subtracting 24 hours
        df.loc[df['delay_minutes'] > 720, 'delay_minutes'] = df.loc[df['delay_minutes'] > 720, 'delay_minutes'] - 1440
        df.loc[df['delay_minutes'] < -720, 'delay_minutes'] = df.loc[df['delay_minutes'] < -720, 'delay_minutes'] + 1440
        
        # Determine status based on delay
        df['status'] = 'On Time'
        df.loc[df['delay_minutes'] > 15, 'status'] = 'Delayed'
        
        # Add time features
        df['hour'] = df[time_col].dt.hour
        df['day_of_week'] = df[time_col].dt.day_name()
        
        return df[['flight_number', 'delay_minutes', 'hour', 'day_of_week', 'status']]
    
    def create_delay_summary_chart(self, delays_df):
        """Create summary chart for dashboard"""
        if delays_df.empty:
            return json.dumps({})
        
        # Calculate counts directly from delay_minutes, similar to KPI calculation
        total_flights = len(delays_df)
        delayed_flights_count = len(delays_df[delays_df['delay_minutes'] > 15])
        on_time_flights_count = total_flights - delayed_flights_count
        
        labels = ['Delayed', 'On Time']
        values = [delayed_flights_count, on_time_flights_count]
        
        # Define colors based on the legend (Green for Delayed, Red for On Time)
        colors = ['#28a745', '#dc3545'] # Green for Delayed, Red for On Time
        
        fig = go.Figure(data=[
            go.Pie(labels=labels, 
                  values=values,
                  hole=0.4,
                  marker_colors=colors,
                  hoverinfo='label+percent+value',
                  textinfo='percent',
                  textposition='inside')
        ])
        
        from datetime import datetime
        current_time = datetime.now().strftime("%H:%M:%S")
        
        fig.update_layout(
            title=f"Flight Status Distribution (Updated: {current_time})",
            showlegend=True,
            height=400
        )
        
        return json.dumps(fig, cls=plotly.utils.PlotlyJSONEncoder)
    
    def analyze_best_times(self):
        """Analyze best times for takeoff and landing"""
        arrivals = self.load_arrival_data()
        departures = self.load_departure_data()
        
        arrival_delays = self.calculate_delays(arrivals, 'arrival')
        departure_delays = self.calculate_delays(departures, 'departure')
        
        # Group by hour and calculate average delays
        arrival_by_hour = arrival_delays.groupby('hour')['delay_minutes'].agg(['mean', 'count']).round(2)
        departure_by_hour = departure_delays.groupby('hour')['delay_minutes'].agg(['mean', 'count']).round(2)
        
        # Find best times (lowest average delays)
        best_arrival_hour = arrival_by_hour['mean'].idxmin()
        best_departure_hour = departure_by_hour['mean'].idxmin()
        
        return {
            'best_arrival_time': f"{best_arrival_hour:02d}:00",
            'best_departure_time': f"{best_departure_hour:02d}:00",
            'arrival_by_hour': arrival_by_hour.to_dict('index'),
            'departure_by_hour': departure_by_hour.to_dict('index')
        }
    
    def create_best_times_chart(self, analysis):
        """Create visualization for best times analysis"""
        if not analysis or 'arrival_by_hour' not in analysis:
            return json.dumps({})
        
        hours = list(range(24))
        arrival_delays = [analysis['arrival_by_hour'].get(h, {}).get('mean', 0) for h in hours]
        departure_delays = [analysis['departure_by_hour'].get(h, {}).get('mean', 0) for h in hours]
        
        fig = go.Figure()
        
        fig.add_trace(go.Bar(
            x=hours,
            y=arrival_delays,
            name='Arrival Delays',
            marker_color='rgba(55, 128, 191, 0.7)'
        ))
        
        fig.add_trace(go.Bar(
            x=hours,
            y=departure_delays,
            name='Departure Delays',
            marker_color='rgba(255, 153, 51, 0.7)'
        ))
        
        fig.update_layout(
            title='Average Delay by Hour of Day',
            xaxis_title='Hour of Day',
            yaxis_title='Average Delay (minutes)',
            barmode='group',
            height=500
        )
        
        return json.dumps(fig, cls=plotly.utils.PlotlyJSONEncoder)
    
    def analyze_congestion(self):
        """Analyze airport congestion patterns"""
        arrivals = self.load_arrival_data()
        departures = self.load_departure_data()
        
        arrivals['scheduled_arrival'] = pd.to_datetime(arrivals['scheduled_arrival'])
        departures['scheduled_departure'] = pd.to_datetime(departures['scheduled_departure'])
        
        arrivals['hour'] = arrivals['scheduled_arrival'].dt.hour
        departures['hour'] = departures['scheduled_departure'].dt.hour
        
        arrival_counts = arrivals.groupby('hour').size()
        departure_counts = departures.groupby('hour').size()
        total_counts = arrival_counts.add(departure_counts, fill_value=0)
        
        busiest_hour = total_counts.idxmax()
        max_flights = total_counts.max()
        
        return {
            'busiest_hour': f"{busiest_hour:02d}:00",
            'max_flights_per_hour': int(max_flights),
            'hourly_traffic': total_counts.to_dict(),
            'arrival_counts': arrival_counts.to_dict(),
            'departure_counts': departure_counts.to_dict()
        }
    
    def create_congestion_chart(self, analysis):
        """Create congestion visualization"""
        if not analysis or 'hourly_traffic' not in analysis:
            return json.dumps({})
        
        hours = list(range(24))
        arrivals = [analysis['arrival_counts'].get(h, 0) for h in hours]
        departures = [analysis['departure_counts'].get(h, 0) for h in hours]
        
        fig = go.Figure()
        
        fig.add_trace(go.Scatter(
            x=hours,
            y=arrivals,
            mode='lines+markers',
            name='Arrivals',
            line=dict(color='rgba(55, 128, 191, 1)', width=3),
            marker=dict(size=8)
        ))
        
        fig.add_trace(go.Scatter(
            x=hours,
            y=departures,
            mode='lines+markers',
            name='Departures',
            line=dict(color='rgba(255, 153, 51, 1)', width=3),
            marker=dict(size=8)
        ))
        
        fig.update_layout(
            title='Airport Traffic by Hour',
            xaxis_title='Hour of Day',
            yaxis_title='Number of Flights',
            height=500,
            hovermode='x'
        )
        
        return json.dumps(fig, cls=plotly.utils.PlotlyJSONEncoder)
    
    def predict_schedule_impact(self, flight_number, new_time):
        """Predict impact of schedule changes using trained ML models"""
        try:
            new_hour = int(new_time.split(':')[0])
            
            # Load historical data
            arrivals = self.load_arrival_data()
            departures = self.load_departure_data()
            
            # Determine if it's arrival or departure based on flight number
            flight_type = 'arrival' if flight_number in arrivals['flight_number'].values else 'departure'
            
            # Use trained model if available
            model_key = f'{flight_type}_delay'
            if model_key in self.models and self.models[model_key] is not None:
                # Create feature vector for prediction
                features = self._create_prediction_features(new_hour, flight_type)
                
                if features is not None:
                    predicted_delay = self.models[model_key].predict([features])[0]
                else:
                    predicted_delay = 15.0  # fallback
            else:
                # Fallback to statistical approach
                arrival_delays = self.calculate_delays(arrivals, 'arrival')
                departure_delays = self.calculate_delays(departures, 'departure')
                all_delays = pd.concat([arrival_delays, departure_delays])
                
                avg_delay_by_hour = all_delays.groupby('hour')['delay_minutes'].mean()
                predicted_delay = avg_delay_by_hour.get(new_hour, 15.0)
            
            # Calculate impact
            all_delays = pd.concat([
                self.calculate_delays(arrivals, 'arrival'),
                self.calculate_delays(departures, 'departure')
            ])
            current_avg = all_delays['delay_minutes'].mean()
            impact = predicted_delay - current_avg
            
            # Generate recommendation
            if impact < -5:
                recommendation = "Excellent choice - significantly reduces delays"
            elif impact < 0:
                recommendation = "Good choice - reduces delays"
            elif impact < 10:
                recommendation = "Acceptable - minimal impact"
            elif impact < 20:
                recommendation = "Consider alternative times - moderate delays expected"
            else:
                recommendation = "Avoid this time - high delays expected"
            
            return {
                'flight_number': flight_number,
                'new_time': new_time,
                'predicted_delay': round(predicted_delay, 1),
                'impact_vs_average': round(impact, 1),
                'recommendation': recommendation,
                'model_used': 'ML Model' if model_key in self.models else 'Statistical'
            }
        
        except Exception as e:
            return {
                'flight_number': flight_number,
                'new_time': new_time,
                'predicted_delay': 15.0,
                'impact_vs_average': 0.0,
                'recommendation': f'Unable to calculate precise impact: {str(e)}',
                'model_used': 'Error'
            }
    
    def _create_prediction_features(self, hour, flight_type):
        """Create feature vector for ML prediction"""
        try:
            # Create a basic feature vector
            features = [hour, 8, 0]  # hour, month (August), is_weekend (False)
            
            # Add encoded categorical features if available
            for col in ['origin', 'destination', 'day_of_week']:
                encoder_key = f'{col}_{flight_type}'
                if encoder_key in self.label_encoders:
                    # Use most common value as default
                    features.append(0)
                else:
                    features.append(0)
            
            return features
            
        except Exception as e:
            print(f"Error creating prediction features: {e}")
            return None
    
    def create_tuning_chart(self, prediction):
        """Create chart for schedule tuning prediction"""
        if not prediction:
            return json.dumps({})
        
        hours = list(range(24))
        # Simulate delay predictions for all hours
        base_delays = [10 + 5*abs(h-12)/12 + np.random.normal(0, 3) for h in hours]
        
        fig = go.Figure()
        
        fig.add_trace(go.Scatter(
            x=hours,
            y=base_delays,
            mode='lines+markers',
            name='Predicted Delay by Hour',
            line=dict(color='rgba(55, 128, 191, 1)', width=2)
        ))
        
        # Highlight the selected time
        new_hour = int(prediction['new_time'].split(':')[0])
        fig.add_trace(go.Scatter(
            x=[new_hour],
            y=[prediction['predicted_delay']],
            mode='markers',
            name=f"Your Selected Time ({prediction['new_time']})",
            marker=dict(size=15, color='red', symbol='star')
        ))
        
        fig.update_layout(
            title=f'Delay Prediction for {prediction["flight_number"]}',
            xaxis_title='Hour of Day',
            yaxis_title='Predicted Delay (minutes)',
            height=400
        )
        
        return json.dumps(fig, cls=plotly.utils.PlotlyJSONEncoder)
    
    def analyze_cascading_impact(self):
        """Analyze cascading impact of delays using network analysis"""
        arrivals = self.load_arrival_data()
        departures = self.load_departure_data()
        
        # Create a simple network based on gate usage and timing
        G = nx.Graph()
        
        # Add flights as nodes
        for _, flight in arrivals.iterrows():
            G.add_node(flight['flight_number'], 
                      type='arrival', 
                      gate=flight['gate'],
                      scheduled=flight['scheduled_arrival'])
        
        for _, flight in departures.iterrows():
            G.add_node(flight['flight_number'], 
                      type='departure',
                      gate=flight['gate'], 
                      scheduled=flight['scheduled_departure'])
        
        # Add edges based on gate conflicts and timing
        nodes = list(G.nodes(data=True))
        for i, (node1, data1) in enumerate(nodes):
            for node2, data2 in nodes[i+1:]:
                # Connect flights that might interfere with each other
                if (data1['gate'] == data2['gate'] or
                    abs(hash(node1) - hash(node2)) % 100 < 20):  # Simplified connection logic
                    G.add_edge(node1, node2, weight=np.random.random())
        
        # Calculate centrality measures to find high-impact flights
        centrality = nx.betweenness_centrality(G)
        degree_centrality = nx.degree_centrality(G)
        
        # Find top impact flights
        high_impact_flights = sorted(centrality.items(), key=lambda x: x[1], reverse=True)[:10]
        
        return {
            'network_size': len(G.nodes()),
            'connections': len(G.edges()),
            'high_impact_flights': high_impact_flights,
            'avg_centrality': np.mean(list(centrality.values()))
        }
    
    def create_impact_network_chart(self, analysis):
        """Create network visualization for cascading impact"""
        if not analysis or 'high_impact_flights' not in analysis:
            return json.dumps({})
        
        # Create a simplified network visualization
        flights = [f[0] for f in analysis['high_impact_flights']]
        impacts = [f[1] for f in analysis['high_impact_flights']]
        
        fig = go.Figure()
        
        fig.add_trace(go.Bar(
            x=flights,
            y=impacts,
            marker_color='rgba(255, 99, 71, 0.8)',
            name='Impact Score'
        ))
        
        fig.update_layout(
            title='High-Impact Flights (Network Centrality)',
            xaxis_title='Flight Number',
            yaxis_title='Impact Score',
            height=500,
            xaxis={'tickangle': 45}
        )
        
        return json.dumps(fig, cls=plotly.utils.PlotlyJSONEncoder)
    
    def get_flight_context(self):
        """Get flight data context for chatbot"""
        try:
            arrivals = self.load_arrival_data()
            departures = self.load_departure_data()
            
            arrival_delays = self.calculate_delays(arrivals, 'arrival')
            departure_delays = self.calculate_delays(departures, 'departure')
            
            all_delays = pd.concat([arrival_delays, departure_delays])
            
            best_times = self.analyze_best_times()
            congestion = self.analyze_congestion()
            
            context = f"""
            Mumbai Airport Flight Data Summary:
            - Total flights: {len(arrivals) + len(departures)}
            - Average delay: {all_delays['delay_minutes'].mean():.1f} minutes
            - Best arrival time: {best_times['best_arrival_time']}
            - Best departure time: {best_times['best_departure_time']}
            - Busiest hour: {congestion['busiest_hour']}
            - On-time performance: {(all_delays['delay_minutes'] <= 15).mean()*100:.1f}%
            """
            
            return context
            
        except Exception as e:
            return f"Flight data context unavailable: {str(e)}"
    
    def get_route_delay_info(self, route_query):
        """Get specific delay information for routes mentioned in queries"""
        try:
            arrivals = self.load_arrival_data()
            departures = self.load_departure_data()
            
            arrival_delays = self.calculate_delays(arrivals, 'arrival')
            departure_delays = self.calculate_delays(departures, 'departure')
            
            # Add route information
            arrival_delays['route'] = arrivals['origin'] + '-' + arrivals['destination']
            departure_delays['route'] = departures['origin'] + '-' + departures['destination']
            
            all_delays = pd.concat([arrival_delays, departure_delays])
            
            # Extract city names from query (case insensitive)
            query_lower = route_query.lower()
            cities = []
            
            # Common city patterns
            city_patterns = ['mumbai', 'delhi', 'varanasi', 'bangalore', 'hyderabad', 'chennai', 'kolkata', 'pune', 'ahmedabad', 'goa']
            
            for city in city_patterns:
                if city in query_lower:
                    cities.append(city.title())
            
            if len(cities) >= 2:
                # Look for routes between the mentioned cities
                route_info = {}
                
                for i, city1 in enumerate(cities):
                    for city2 in cities[i+1:]:
                        # Check both directions
                        route1 = f"{city1}-{city2}"
                        route2 = f"{city2}-{city1}"
                        
                        # Filter flights for these routes
                        route1_flights = all_delays[all_delays['route'].str.contains(route1, case=False, na=False)]
                        route2_flights = all_delays[all_delays['route'].str.contains(route2, case=False, na=False)]
                        
                        if not route1_flights.empty:
                            route_info[route1] = {
                                'total_flights': len(route1_flights),
                                'avg_delay': route1_flights['delay_minutes'].mean(),
                                'delay_rate': (route1_flights['delay_minutes'] > 15).mean() * 100,
                                'max_delay': route1_flights['delay_minutes'].max()
                            }
                        
                        if not route2_flights.empty:
                            route_info[route2] = {
                                'total_flights': len(route2_flights),
                                'avg_delay': route2_flights['delay_minutes'].mean(),
                                'delay_rate': (route2_flights['delay_minutes'] > 15).mean() * 100,
                                'max_delay': route2_flights['delay_minutes'].max()
                            }
                
                return route_info
            
            return {}
            
        except Exception as e:
            print(f"Error getting route delay info: {e}")
            return {}
    
    def analyze_delays_by_city(self):
        """Analyze delays by origin/destination city"""
        try:
            arrivals = self.load_arrival_data()
            departures = self.load_departure_data()
            
            # Calculate delays for arrivals and departures
            arrival_delays = self.calculate_delays(arrivals, 'arrival')
            departure_delays = self.calculate_delays(departures, 'departure')
            
            # Add city information
            arrival_delays['city'] = arrivals['origin']
            departure_delays['city'] = departures['destination']
            
            # Combine all delays
            all_delays = pd.concat([arrival_delays, departure_delays])
            
            # Group by city and calculate statistics
            city_stats = all_delays.groupby('city').agg({
                'delay_minutes': ['count', 'mean', 'std', 'min', 'max'],
                'status': lambda x: (x == 'Delayed').sum()
            }).round(2)
            
            # Flatten column names
            city_stats.columns = ['total_flights', 'avg_delay', 'std_delay', 'min_delay', 'max_delay', 'delayed_flights']
            city_stats['delay_rate'] = (city_stats['delayed_flights'] / city_stats['total_flights'] * 100).round(1)
            
            # Filter out cities with less than 10 flights
            city_stats = city_stats[city_stats['total_flights'] >= 10]
            city_stats = city_stats.sort_values('delay_rate', ascending=False)
            
            return {
                'city_stats': city_stats.to_dict('index'),
                'top_cities': city_stats.head(10).to_dict('index'),
                'total_cities': len(city_stats)
            }
            
        except Exception as e:
            print(f"Error analyzing delays by city: {e}")
            return {}
    
    def get_city_delay_details(self, city_name):
        """Get detailed delay analysis for a specific city"""
        try:
            arrivals = self.load_arrival_data()
            departures = self.load_departure_data()
            
            # Filter flights for the specific city - handle city name variations
            # Remove common suffixes like (ABC) from city names for matching
            clean_city_name = city_name.split(' (')[0].strip()
            
            city_arrivals = arrivals[arrivals['origin'].str.contains(clean_city_name, case=False, na=False)]
            city_departures = departures[departures['destination'].str.contains(clean_city_name, case=False, na=False)]
            
            # Calculate delays
            arrival_delays = self.calculate_delays(city_arrivals, 'arrival')
            departure_delays = self.calculate_delays(city_departures, 'departure')
            
            # Combine delays
            all_city_delays = pd.concat([arrival_delays, departure_delays])
            
            # Debug: Print city matching info
            print(f"City lookup: '{city_name}' -> '{clean_city_name}'")
            print(f"Found {len(city_arrivals)} arrival flights and {len(city_departures)} departure flights")
            print(f"Total city flights: {len(all_city_delays)}")
            
            if all_city_delays.empty:
                return {
                    'city_name': city_name,
                    'total_flights': 0,
                    'avg_delay': 0,
                    'delay_rate': 0,
                    'hourly_delays': {},
                    'daily_delays': {}
                }
            
            # Calculate statistics
            total_flights = len(all_city_delays)
            avg_delay = all_city_delays['delay_minutes'].mean()
            delay_rate = (all_city_delays['status'] == 'Delayed').mean() * 100
            
            # Hourly delay patterns
            hourly_delays = all_city_delays.groupby('hour')['delay_minutes'].mean().to_dict()
            
            # Daily delay patterns
            daily_delays = all_city_delays.groupby('day_of_week')['delay_minutes'].mean().to_dict()
            
            # Sample flights
            sample_flights = all_city_delays.head(10)[['flight_number', 'delay_minutes', 'status', 'hour']].to_dict('records')
            
            return {
                'city_name': city_name,
                'total_flights': total_flights,
                'avg_delay': round(avg_delay, 1),
                'delay_rate': round(delay_rate, 1),
                'max_delay': round(all_city_delays['delay_minutes'].max(), 1),
                'min_delay': round(all_city_delays['delay_minutes'].min(), 1),
                'hourly_delays': hourly_delays,
                'daily_delays': daily_delays,
                'sample_flights': sample_flights
            }
            
        except Exception as e:
            print(f"Error getting city delay details: {e}")
            return {}
    
    def create_city_delay_chart(self, analysis):
        """Create chart for city delay analysis"""
        if not analysis or 'top_cities' not in analysis:
            return json.dumps({})
        
        cities = list(analysis['top_cities'].keys())
        delay_rates = [analysis['top_cities'][city]['delay_rate'] for city in cities]
        avg_delays = [analysis['top_cities'][city]['avg_delay'] for city in cities]
        
        fig = go.Figure()
        
        fig.add_trace(go.Bar(
            x=cities,
            y=delay_rates,
            name='Delay Rate (%)',
            marker_color='rgba(255, 99, 71, 0.8)',
            yaxis='y'
        ))
        
        fig.add_trace(go.Scatter(
            x=cities,
            y=avg_delays,
            name='Avg Delay (min)',
            marker_color='rgba(55, 128, 191, 1)',
            yaxis='y2'
        ))
        
        fig.update_layout(
            title='Flight Delays by City',
            xaxis_title='City',
            yaxis=dict(title='Delay Rate (%)', side='left'),
            yaxis2=dict(title='Average Delay (minutes)', side='right', overlaying='y'),
            height=500,
            xaxis={'tickangle': 45}
        )
        
        return json.dumps(fig, cls=plotly.utils.PlotlyJSONEncoder)
    
    def create_city_detail_chart(self, analysis):
        """Create detailed chart for specific city"""
        if not analysis or 'hourly_delays' not in analysis:
            return json.dumps({})
        
        # Create subplot with hourly and daily patterns
        fig = go.Figure()
        
        # Hourly delays
        hours = list(range(24))
        hourly_values = [analysis['hourly_delays'].get(h, 0) for h in hours]
        
        fig.add_trace(go.Scatter(
            x=hours,
            y=hourly_values,
            mode='lines+markers',
            name='Hourly Average Delay',
            line=dict(color='rgba(55, 128, 191, 1)', width=3),
            marker=dict(size=8)
        ))
        
        fig.update_layout(
            title=f"Delay Patterns for {analysis['city_name']}",
            xaxis_title='Hour of Day',
            yaxis_title='Average Delay (minutes)',
            height=400
        )
        
        return json.dumps(fig, cls=plotly.utils.PlotlyJSONEncoder)
    
# ...existing code...
    def get_rule_based_response(self, message, context):
        """Enhanced rule-based responses with comprehensive analytics integration"""
        message_lower = message.lower()

        # Initialize response parts
        response_parts = [f"Based on Mumbai Airport data analysis: {context}"]

        # Route-specific analysis
        route_info = self.get_route_delay_info(message)
        if route_info:
            route_response = "\n**Route Analysis:**\n"
            for route, data in route_info.items():
                route_response += f"• {route}: {data['total_flights']} flights, {data['avg_delay']:.1f} min avg delay, {data['delay_rate']:.1f}% delay rate\n"
            response_parts.append(route_response)

        # Best times analysis
        if any(keyword in message_lower for keyword in ['best time', 'optimal', 'when to', 'schedule']):
            try:
                best_times = self.analyze_best_times()
                if best_times:
                    best_response = "\n**Optimal Times:**\n"
                    best_response += f"🛬 Best arrival time: {best_times['best_arrival_time']}\n"
                    best_response += f"🛫 Best departure time: {best_times['best_departure_time']}\n"
                    best_response += "\n**Recommendation:** Schedule arrivals in early morning (5-7 AM) and departures during mid-morning (9-11 AM) or late evening (10 PM-12 AM) for minimal delays.\n"
                    response_parts.append(best_response)
            except Exception as e:
                response_parts.append("\nFor minimal delays, consider scheduling arrivals around early morning hours (5-7 AM) and departures during mid-morning (9-11 AM) or late evening (10 PM-12 AM).\n")

        # Congestion analysis
        elif any(keyword in message_lower for keyword in ['busiest', 'congestion', 'busy', 'avoid', 'peak']):
            try:
                congestion = self.analyze_congestion()
                if congestion:
                    congestion_response = "\n**Congestion Analysis:**\n"
                    congestion_response += f"🔴 Busiest hour: {congestion['busiest_hour']} ({congestion['max_flights_per_hour']} flights)\n"
                    congestion_response += "\n**Recommendation:** Avoid peak hours (7-9 AM, 5-8 PM) when possible. Consider off-peak scheduling for better on-time performance.\n"
                    response_parts.append(congestion_response)
            except Exception as e:
                response_parts.append("\nAirport congestion varies throughout the day. Avoid scheduling during peak hours (7-9 AM, 5-8 PM) if possible.\n")

        # Delay analysis
        elif any(keyword in message_lower for keyword in ['delay', 'late', 'on time', 'punctual']):
            try:
                arrivals = self.load_arrival_data()
                departures = self.load_departure_data()

                if not arrivals.empty or not departures.empty:
                    arrival_delays = self.calculate_delays(arrivals, 'arrival') if not arrivals.empty else pd.DataFrame()
                    departure_delays = self.calculate_delays(departures, 'departure') if not departures.empty else pd.DataFrame()
                    all_delays = pd.concat([arrival_delays, departure_delays]) if not arrival_delays.empty or not departure_delays.empty else pd.DataFrame()

                    if not all_delays.empty:
                        delay_response = "\n**Delay Statistics:**\n"
                        delay_response += f"• Average delay: {all_delays['delay_minutes'].mean():.1f} minutes\n"
                        delay_response += f"• On-time performance: {(all_delays['delay_minutes'] <= 15).mean()*100:.1f}%\n"
                        delay_response += f"• Flights delayed >15 min: {(all_delays['delay_minutes'] > 15).sum()}/{len(all_delays)}\n"
                        response_parts.append(delay_response)
            except Exception as e:
                response_parts.append("\nFlight delays are typically caused by weather, air traffic congestion, and operational issues. Peak hours (7-9 AM, 5-8 PM) tend to have higher delays.\n")

        # City-specific analysis
        city_keywords = ['mumbai', 'delhi', 'bangalore', 'chennai', 'kolkata', 'hyderabad', 'pune', 'ahmedabad', 'goa', 'dubai', 'doha', 'london', 'singapore']
        mentioned_cities = [city for city in city_keywords if city in message_lower]

        if mentioned_cities:
            try:
                city_analysis = self.analyze_delays_by_city()
                if city_analysis and 'city_stats' in city_analysis:
                    city_response = "\n**City Analysis:**\n"
                    for city in mentioned_cities:
                        # Find matching city in analysis
                        matched_city = None
                        for analyzed_city in city_analysis['city_stats'].keys():
                            if city.lower() in analyzed_city.lower():
                                matched_city = analyzed_city
                                break

                        if matched_city:
                            city_data = city_analysis['city_stats'][matched_city]
                            city_response += f"• {matched_city}: {city_data['avg_delay']:.1f} min avg delay, {city_data['delay_rate']:.1f}% delay rate\n"

                    response_parts.append(city_response)
            except Exception as e:
                pass

        # Schedule optimization
        elif any(keyword in message_lower for keyword in ['optimize', 'improve', 'reduce delay', 'schedule impact']):
            response_parts.append("\n**Optimization Tips:**\n• Use off-peak hours for better punctuality\n• Monitor weather forecasts for proactive scheduling\n• Consider alternative routes during high-delay periods\n• Implement buffer times for critical connections\n")

        # General recommendations
        response_parts.append("\n**Quick Tips:**\n")
        response_parts.append("💡 Ask about specific routes (e.g., 'Mumbai to Delhi delays')\n")
        response_parts.append("💡 Check best times (e.g., 'best time to depart')\n")
        response_parts.append("💡 Analyze congestion (e.g., 'busiest hours at airport')\n")

        return "".join(response_parts)
# ...existing code...