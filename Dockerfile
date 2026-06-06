FROM python:3.12-slim

WORKDIR /code

# Copy requirements and install
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Create cache directory for openmeteo
RUN mkdir -p .cache && chmod 777 .cache

# Copy project files
COPY . .

# Expose Streamlit port
EXPOSE 7860

# Ensure start script is executable
RUN chmod +x start.sh

CMD ["./start.sh"]
