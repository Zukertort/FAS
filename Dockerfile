# Use an official Python runtime as a parent image
FROM python:3.11.9

# Set the working directory in the container
WORKDIR /app

# Copy the requirements file into the container
COPY requirements.txt .

# Install any needed packages specified in requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# Copy the application code into the container
COPY ./fas_api /app/fas_api

# Expose the port the app runs on
EXPOSE 5000

# Define environment variable
ENV FLASK_APP=fas_api.app

# Command to run the application
CMD ["flask", "run", "--host=0.0.0.0"]