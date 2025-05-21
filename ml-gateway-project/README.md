# README.md content

# ML Gateway Project

This project is a machine learning API gateway that integrates various models for question answering, text generation, and speech-to-text functionalities.

## Project Structure

- **gateway/**: Contains the FastAPI API Gateway.
  - **app/**: The main application code.
    - **api/**: API routes and handlers.
    - **core/**: Core configuration and initialization.
  - **requirements.txt**: Dependencies for the FastAPI application.

- **model_qa/**: Contains the Question Answering model.
  - **requirements.txt**: Dependencies for the Question Answering model.

- **model_gen/**: Contains the Text Generation model.
  - **requirements.txt**: Dependencies for the Text Generation model.

- **model_stt/**: Contains the Speech-to-Text model.
  - **requirements.txt**: Dependencies for the Speech-to-Text model.

- **tests/**: Contains test modules for the project.

## Setup Instructions

1. Clone the repository:
   ```
   git clone <repository-url>
   cd ml-gateway-project
   ```

2. Install the required dependencies for each module:
   ```
   pip install -r gateway/requirements.txt
   pip install -r model_qa/requirements.txt
   pip install -r model_gen/requirements.txt
   pip install -r model_stt/requirements.txt
   ```

3. Run the FastAPI application:
   ```
   uvicorn gateway.app.main:app --reload
   ```

## Usage Examples

- Access the API documentation at `http://localhost:8000/docs` after running the application.
- Use the endpoints to interact with the models for question answering, text generation, and speech-to-text functionalities.

## License

This project is licensed under the MIT License.