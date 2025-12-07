#!/bin/bash
# Stop all LLM Evaluation UI processes

echo "Stopping LLM Evaluation UI..."

# Stop Streamlit
echo "1. Stopping Streamlit..."
pkill -f "streamlit run app.py"
sleep 1

# Stop MLflow
echo "2. Stopping MLflow..."
pkill -f "mlflow ui"
sleep 1

# Free ports
echo "3. Freeing ports..."
fuser -k 8502/tcp 2>/dev/null
fuser -k 5000/tcp 2>/dev/null

# Check if ports are free
echo ""
echo "Checking ports..."
if lsof -i :8502 > /dev/null 2>&1; then
    echo "⚠️  Port 8502 still in use"
    lsof -i :8502
else
    echo "✅ Port 8502 is free"
fi

if lsof -i :5000 > /dev/null 2>&1; then
    echo "⚠️  Port 5000 still in use"
    lsof -i :5000
else
    echo "✅ Port 5000 is free"
fi

echo ""
echo "✅ Cleanup complete!"
