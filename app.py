import os
import json
import pandas as pd
import numpy as np
from flask import Flask, render_template, request, jsonify
from datetime import datetime, timedelta
import plotly.graph_objs as go
import plotly.utils
from models.ml_models import FlightAnalytics
import google.generativeai as genai
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

app = Flask(__name__)
app.secret_key = 'flight_scheduling_secret_key_2024'

# Initialize ML models
analytics = FlightAnalytics()

# Configure Gemini AI
GEMINI_API_KEY = os.getenv('GEMINI_API_KEY', 'demo-key')
if GEMINI_API_KEY != 'demo-key':
    genai.configure(api_key=GEMINI_API_KEY)
    gemini_model = genai.GenerativeModel('gemini-1.5-flash')
else:
    gemini_model = None

@app.route('/')
def index():
    """Dashboard with flight KPIs and summary statistics"""
    try:
        # Load and analyze data
        arrivals = analytics.load_arrival_data()
        departures = analytics.load_departure_data()
        
        # Calculate KPIs
        total_flights = len(arrivals) + len(departures)
        
        # Calculate delays
        arrival_delays = analytics.calculate_delays(arrivals, 'arrival')
        departure_delays = analytics.calculate_delays(departures, 'departure')
        
        all_delays = pd.concat([arrival_delays, departure_delays])
        delayed_flights = len(all_delays[all_delays['delay_minutes'] > 15])
        delay_percentage = (delayed_flights / total_flights * 100) if total_flights > 0 else 0
        avg_delay = all_delays['delay_minutes'].mean() if len(all_delays) > 0 else 0
        
        # Create summary chart
        delay_summary = analytics.create_delay_summary_chart(all_delays)
        
        kpis = {
            'total_flights': total_flights,
            'delayed_flights': delayed_flights,
            'delay_percentage': round(delay_percentage, 1),
            'avg_delay_minutes': round(avg_delay, 1)
        }
        
        return render_template('index.html', kpis=kpis, chart_json=delay_summary)
    
    except Exception as e:
        return render_template('index.html', 
                             kpis={'total_flights': 0, 'delayed_flights': 0, 'delay_percentage': 0, 'avg_delay_minutes': 0},
                             chart_json=json.dumps({}),
                             error=f"Error loading dashboard: {str(e)}")

@app.route('/best_times')
def best_times():
    """Analyze best times for takeoff and landing"""
    try:
        best_times_analysis = analytics.analyze_best_times()
        chart_json = analytics.create_best_times_chart(best_times_analysis)
        
        return render_template('best_times.html', 
                             analysis=best_times_analysis,
                             chart_json=chart_json)
    except Exception as e:
        return render_template('best_times.html', 
                             analysis={}, 
                             chart_json=json.dumps({}),
                             error=f"Error analyzing best times: {str(e)}")

@app.route('/busiest_slots')
def busiest_slots():
    """Show busiest time slots with congestion analysis"""
    try:
        congestion_analysis = analytics.analyze_congestion()
        chart_json = analytics.create_congestion_chart(congestion_analysis)
        
        return render_template('busiest_slots.html',
                             analysis=congestion_analysis,
                             chart_json=chart_json)
    except Exception as e:
        return render_template('busiest_slots.html',
                             analysis={},
                             chart_json=json.dumps({}),
                             error=f"Error analyzing congestion: {str(e)}")

# ...existing code...
@app.route('/schedule_tuning', methods=['GET', 'POST'])
def schedule_tuning():
    """Schedule tuning simulator with ML predictions and flight dropdown"""
    if request.method == 'POST':
        try:
            flight_number = request.json.get('flight_number')
            new_time = request.json.get('new_time')
            
            prediction = analytics.predict_schedule_impact(flight_number, new_time)
            chart_json = analytics.create_tuning_chart(prediction)
            
            return jsonify({
                'success': True,
                'prediction': prediction,
                'chart_json': chart_json
            })
        except Exception as e:
            return jsonify({
                'success': False,
                'error': f"Prediction failed: {str(e)}"
            })

    # GET request - load flight numbers for dropdown
    try:
        arrivals = analytics.load_arrival_data()
        departures = analytics.load_departure_data()

        # Get unique flight numbers from both datasets
        arrival_flights = arrivals['flight_number'].dropna().unique().tolist() if 'flight_number' in arrivals.columns else []
        departure_flights = departures['flight_number'].dropna().unique().tolist() if 'flight_number' in departures.columns else []

        # Combine and sort flight numbers
        all_flight_numbers = sorted(list(set(arrival_flights + departure_flights)))

        # Clean the flight numbers (remove any problematic entries)
        clean_flight_numbers = []
        for fn in all_flight_numbers:
            if 'E+' not in str(fn) and 'nan' not in str(fn).lower() and len(str(fn)) <= 10 and str(fn).strip():
                clean_flight_numbers.append(str(fn))

        return render_template('schedule_tuning.html', flight_numbers=clean_flight_numbers)
    except Exception as e:
        # Fallback if data loading fails
        return render_template('schedule_tuning.html', flight_numbers=[])
# ...existing code...

@app.route('/cascading_impact')
def cascading_impact():
    """Analyze cascading impact of flight delays"""
    try:
        impact_analysis = analytics.analyze_cascading_impact()
        chart_json = analytics.create_impact_network_chart(impact_analysis)
        
        return render_template('cascading_impact.html',
                             analysis=impact_analysis,
                             chart_json=chart_json)
    except Exception as e:
        return render_template('cascading_impact.html',
                             analysis={},
                             chart_json=json.dumps({}),
                             error=f"Error analyzing cascading impact: {str(e)}")

@app.route('/city_analysis')
def city_analysis():
    """Analyze delays by city"""
    try:
        city_analysis = analytics.analyze_delays_by_city()
        chart_json = analytics.create_city_delay_chart(city_analysis)
        
        return render_template('city_analysis.html',
                             analysis=city_analysis,
                             chart_json=chart_json)
    except Exception as e:
        return render_template('city_analysis.html',
                             analysis={},
                             chart_json=json.dumps({}),
                             error=f"Error analyzing city delays: {str(e)}")

@app.route('/city/<city_name>')
def city_detail(city_name):
    """Show detailed analysis for a specific city"""
    try:
        city_detail = analytics.get_city_delay_details(city_name)
        chart_json = analytics.create_city_detail_chart(city_detail)
        
        return render_template('city_detail.html',
                             city_name=city_name,
                             analysis=city_detail,
                             chart_json=chart_json)
    except Exception as e:
        return render_template('city_detail.html',
                             city_name=city_name,
                             analysis={},
                             chart_json=json.dumps({}),
                             error=f"Error analyzing {city_name}: {str(e)}")

@app.route('/chatbot')
def chatbot():
    """AI-powered chatbot interface"""
    return render_template('chatbot.html')

@app.route('/chat', methods=['POST'])
def chat():
    """Enhanced chatbot that fully utilizes all analytics methods"""
    try:
        user_message = request.json.get('message', '')

        if not user_message:
            return jsonify({'error': 'No message provided'})

        # Get comprehensive flight context
        context = analytics.get_flight_context()

        # Analyze user intent and extract relevant information
        message_lower = user_message.lower()

        # Initialize response components
        response_parts = []

# ...existing code...
        # 1. Handle route-specific queries (Mumbai-Delhi, Dubai-Mumbai, etc.)
        route_info = analytics.get_route_delay_info(user_message)
        if route_info:
            route_response = "\n**Route-Specific Analysis:**\n"
            for route, data in route_info.items():
                route_response += f"📍 **{route}**: {data['total_flights']} flights, {data['avg_delay']:.1f} min avg delay, {data['delay_rate']:.1f}% delay rate\n"
            response_parts.append(route_response)

        # 2. Handle best times queries
        if any(keyword in message_lower for keyword in ['best time', 'optimal', 'when to', 'schedule']):
            best_times = analytics.analyze_best_times()
            if best_times:
                best_times_response = "\n**⏰ Optimal Scheduling Times:**\n"
                best_times_response += f"🛬 **Best Arrival Time**: {best_times['best_arrival_time']} (minimal delays)\n"
                best_times_response += f"🛫 **Best Departure Time**: {best_times['best_departure_time']} (minimal delays)\n"

                # Add hourly breakdown for top 3 best and worst hours
                arrival_hours = best_times.get('arrival_by_hour', {})
                departure_hours = best_times.get('departure_by_hour', {})

                if arrival_hours:
                    sorted_arrival = sorted(arrival_hours.items(), key=lambda x: x[1]['mean'])
                    best_times_response += "\n**🛬 Arrival Delay Patterns:**\n"
                    best_times_response += f"• Least delays: {sorted_arrival[0][0]:02d}:00 ({sorted_arrival[0][1]['mean']:.1f} min avg)\n"
                    best_times_response += f"• Most delays: {sorted_arrival[-1][0]:02d}:00 ({sorted_arrival[-1][1]['mean']:.1f} min avg)\n"

                response_parts.append(best_times_response)

        # 3. Handle busiest times / congestion queries  
        if any(keyword in message_lower for keyword in ['busiest', 'congestion', 'busy', 'avoid', 'peak']):
            congestion = analytics.analyze_congestion()
            if congestion:
                congestion_response = "\n**📊 Airport Congestion Analysis:**\n"
                congestion_response += f"🔴 **Busiest Hour**: {congestion['busiest_hour']} ({congestion['max_flights_per_hour']} flights)\n"

                # Add hourly traffic breakdown
                hourly_traffic = congestion.get('hourly_traffic', {})
                if hourly_traffic:
                    sorted_traffic = sorted(hourly_traffic.items(), key=lambda x: x[1], reverse=True)
                    congestion_response += "\n**Peak Traffic Hours:**\n"
                    for i, (hour, flights) in enumerate(sorted_traffic[:3]):
                        congestion_response += f"• {hour:02d}:00 - {flights} flights\n"

                    congestion_response += "\n**Quietest Hours (Recommended):**\n"
                    for i, (hour, flights) in enumerate(sorted_traffic[-3:]):
                        congestion_response += f"• {hour:02d}:00 - {flights} flights\n"

                response_parts.append(congestion_response)

        # 4. Handle city-specific queries
        city_keywords = ['mumbai', 'delhi', 'bangalore', 'chennai', 'kolkata', 'hyderabad', 'pune', 'ahmedabad', 'goa', 'dubai', 'doha', 'london', 'singapore']
        mentioned_cities = [city for city in city_keywords if city in message_lower]

        if mentioned_cities:
            city_analysis = analytics.analyze_delays_by_city()
            if city_analysis and 'city_stats' in city_analysis:
                city_response = "\n**🏙️ City-wise Delay Analysis:**\n"

                for city in mentioned_cities:
                    # Find matching city in analysis (case-insensitive partial match)
                    matched_city = None
                    for analyzed_city in city_analysis['city_stats'].keys():
                        if city.lower() in analyzed_city.lower():
                            matched_city = analyzed_city
                            break

                    if matched_city:
                        city_data = city_analysis['city_stats'][matched_city]
                        city_response += f"📍 **{matched_city}**: {city_data['total_flights']} flights, {city_data['avg_delay']:.1f} min avg delay, {city_data['delay_rate']:.1f}% delay rate\n"

                        # Get detailed city analysis
                        city_details = analytics.get_city_delay_details(matched_city)
                        if city_details and city_details['total_flights'] > 0:
                            city_response += f"   • Max delay: {city_details['max_delay']:.1f} min\n"
                            city_response += f"   • Min delay: {city_details['min_delay']:.1f} min\n"

                response_parts.append(city_response)

        # 5. Handle delay-related queries
        if any(keyword in message_lower for keyword in ['delay', 'late', 'on time', 'punctual']):
            arrivals = analytics.load_arrival_data()
            departures = analytics.load_departure_data()

            if not arrivals.empty or not departures.empty:
                arrival_delays = analytics.calculate_delays(arrivals, 'arrival') if not arrivals.empty else pd.DataFrame()
                departure_delays = analytics.calculate_delays(departures, 'departure') if not departures.empty else pd.DataFrame()
                all_delays = pd.concat([arrival_delays, departure_delays]) if not arrival_delays.empty or not departure_delays.empty else pd.DataFrame()

                if not all_delays.empty:
                    delay_response = "\n**⏱️ Delay Statistics:**\n"
                    delay_response += f"• Average delay: {all_delays['delay_minutes'].mean():.1f} minutes\n"
                    delay_response += f"• On-time performance: {(all_delays['delay_minutes'] <= 15).mean()*100:.1f}%\n"
                    delay_response += f"• Flights delayed >15 min: {(all_delays['delay_minutes'] > 15).sum()}/{len(all_delays)}\n"
                    delay_response += f"• Maximum delay recorded: {all_delays['delay_minutes'].max():.1f} minutes\n"

                    # Add day-wise delay patterns
                    daily_delays = all_delays.groupby('day_of_week')['delay_minutes'].mean()
                    if not daily_delays.empty:
                        best_day = daily_delays.idxmin()
                        worst_day = daily_delays.idxmax()
                        delay_response += f"• Best day to fly: {best_day} ({daily_delays[best_day]:.1f} min avg)\n"
                        delay_response += f"• Worst day to fly: {worst_day} ({daily_delays[worst_day]:.1f} min avg)\n"

                    response_parts.append(delay_response)

        # 6. Handle schedule optimization queries
        if any(keyword in message_lower for keyword in ['optimize', 'improve', 'reduce delay', 'schedule impact']):
            # Get cascading impact analysis
            impact_analysis = analytics.analyze_cascading_impact()
            if impact_analysis:
                optimization_response = "\n**🔧 Schedule Optimization Insights:**\n"
                optimization_response += f"• Network size: {impact_analysis['network_size']} flights analyzed\n"
                optimization_response += f"• Flight connections: {impact_analysis['connections']} interdependencies\n"

                if 'high_impact_flights' in impact_analysis:
                    optimization_response += "\n**High-Impact Flights (affecting multiple connections):**\n"
                    for i, (flight, impact) in enumerate(impact_analysis['high_impact_flights'][:3]):
                        optimization_response += f"• {flight} (Impact score: {impact:.3f})\n"

                response_parts.append(optimization_response)

        # Combine all response parts
        if response_parts:
            detailed_response = "".join(response_parts)
        else:
            # Fallback for general queries
            detailed_response = f"\n**📋 Mumbai Airport Overview:**\n{context}"

        # Use Gemini AI if available for natural language enhancement
        if gemini_model:
            enhanced_prompt = f"""
            You are an expert flight scheduling assistant for Mumbai Airport. Based on the comprehensive analysis below, provide a conversational and helpful response to the user's question.

            User Question: {user_message}

            Analysis Results: {detailed_response}

            Instructions:
            - Make the response conversational and user-friendly
            - Highlight the most important insights first
            - Provide actionable recommendations
            - Keep technical jargon minimal
            - If specific routes or cities were mentioned, prioritize that information
            - End with helpful suggestions for further queries
            """

            response = gemini_model.generate_content(enhanced_prompt)
            bot_response = response.text
        
            # Enhanced rule-based response
# ...existing code...
        else:
            # Enhanced rule-based response
            bot_response = f"Based on the latest Mumbai Airport data analysis:{detailed_response}\n\n💡 **Recommendations:**\n"

            if 'best time' in message_lower:
                bot_response += "• Schedule arrivals in early morning (5-7 AM) for minimal delays\n• Plan departures during mid-morning (9-11 AM) or late evening\n"
            elif 'busiest' in message_lower:
                bot_response += "• Avoid peak hours (7-9 AM, 5-8 PM) when possible\n• Consider off-peak scheduling for better on-time performance\n"
            elif any(city in message_lower for city in ['mumbai', 'delhi', 'dubai']):
                bot_response += "• Check route-specific delay patterns before scheduling\n• Consider alternative routes during high-delay periods\n"
            else:
                bot_response += "• Use off-peak hours for better punctuality\n• Monitor real-time updates for dynamic scheduling\n"

            bot_response += "\nAsk me about specific routes, optimal times, or congestion patterns for more detailed insights!"

        return jsonify({
            'response': bot_response,
            'timestamp': datetime.now().isoformat()
        })

    except Exception as e:
        return jsonify({
            'response': f"I encountered an error while analyzing your request: {str(e)}. Please try rephrasing your question or ask about specific flight routes, best times, or airport congestion patterns.",
            'timestamp': datetime.now().isoformat()
        })
# ...existing code...

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)