# Use an official Python runtime as the base image
FROM python:3.8-slim

# Copy the requirements file into the container at /app
COPY requirements.txt .

# Install any needed packages specified in requirements.txt
RUN pip install --upgrade pip
RUN pip install --no-cache-dir -r requirements.txt
RUN pip install opencv-python-headless

# Copy the rest of the application code into the container
COPY . .

# Define the command to run your script
CMD ["python", "roads.py"]
