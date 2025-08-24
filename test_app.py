#!/usr/bin/env python3
"""
Test script for Flight Scheduling MVP
Verifies all components are working correctly
"""

import sys
import os

def test_imports():
    """Test all required imports"""
    print("🔍 Testing imports...")
    
    try:
        import flask
        print("✅ Flask imported successfully")
    except ImportError as e:
        print(f"❌ Flask import failed: {e}")
        return False
    
    try:
        import pandas as pd
        print("✅ Pandas imported successfully")
    except ImportError as e:
        print(f"❌ Pandas import failed: {e}")
        return False
    
    try:
        import plotly
        print("✅ Plotly imported successfully")
    except ImportError as e:
        print(f"❌ Plotly import failed: {e}")
        return False
    
    try:
        import google.generativeai as genai
        print("✅ Google Gemini API imported successfully")
    except ImportError as e:
        print(f"❌ Google Gemini API import failed: {e}")
        return False
    
    try:
        from sklearn.ensemble import RandomForestRegressor
        print("✅ Scikit-learn imported successfully")
    except ImportError as e:
        print(f"❌ Scikit-learn import failed: {e}")
        return False
    
    try:
        import networkx as nx
        print("✅ NetworkX imported successfully")
    except ImportError as e:
        print(f"❌ NetworkX import failed: {e}")
        return False
    
    return True

def test_data_files():
    """Test data files exist"""
    print("\n📁 Testing data files...")
    
    data_files = [
        'data/arrival_flights.csv',
        'data/departure_flights.csv'
    ]
    
    for file_path in data_files:
        if os.path.exists(file_path):
            print(f"✅ {file_path} exists")
        else:
            print(f"❌ {file_path} missing")
            return False
    
    return True

def test_ml_models():
    """Test ML models can be loaded"""
    print("\n🤖 Testing ML models...")
    
    try:
        from models.ml_models import FlightAnalytics
        analytics = FlightAnalytics()
        print("✅ FlightAnalytics class loaded successfully")
        
        # Test data loading
        arrivals = analytics.load_arrival_data()
        departures = analytics.load_departure_data()
        print(f"✅ Loaded {len(arrivals)} arrival flights")
        print(f"✅ Loaded {len(departures)} departure flights")
        
        # Test delay calculation
        arrival_delays = analytics.calculate_delays(arrivals, 'arrival')
        departure_delays = analytics.calculate_delays(departures, 'departure')
        print(f"✅ Calculated delays for {len(arrival_delays)} arrival flights")
        print(f"✅ Calculated delays for {len(departure_delays)} departure flights")
        
        return True
        
    except Exception as e:
        print(f"❌ ML models test failed: {e}")
        return False

def test_flask_app():
    """Test Flask app can be imported"""
    print("\n🌐 Testing Flask app...")
    
    try:
        import app
        print("✅ Flask app imported successfully")
        return True
    except Exception as e:
        print(f"❌ Flask app import failed: {e}")
        return False

def main():
    """Run all tests"""
    print("🚀 Flight Scheduling MVP - Component Test")
    print("=" * 50)
    
    tests = [
        test_imports,
        test_data_files,
        test_ml_models,
        test_flask_app
    ]
    
    passed = 0
    total = len(tests)
    
    for test in tests:
        if test():
            passed += 1
        else:
            print(f"❌ Test failed: {test.__name__}")
    
    print("\n" + "=" * 50)
    print(f"📊 Test Results: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All tests passed! Application is ready to run.")
        print("\n🚀 To start the application:")
        print("   python app.py")
        print("   Then open http://localhost:5000 in your browser")
    else:
        print("⚠️  Some tests failed. Please check the errors above.")
        sys.exit(1)

if __name__ == "__main__":
    main()
