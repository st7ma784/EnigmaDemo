# Use the pytorch base image
FROM pytorch/pytorch:latest

# Copy the requirements.txt file to the container
COPY requirements.txt /app/requirements.txt

# Install the Python dependencies
RUN pip install -r /app/requirements.txt --no-cache-dir

# Set the working directory
WORKDIR /app

# Copy the rest of the files to the container
COPY . /app

# Run the launch command with the num_trials -1 flag
CMD ["python", "CreateWandBSweep.py"]