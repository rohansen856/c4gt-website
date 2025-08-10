# Flask Application README

## Prerequisites
Before running this Flask application, ensure you have the following installed on your system:
- Python (3.x recommended)
- pip (Python package installer)
- virtualenv (for creating isolated Python environments)
- PHP (for import/export feature)
- Composer (for import/export feature)

## Setup Instructions

### For Windows

1. **Install virtualenv (if not installed)**:
   Open Command Prompt and run:
   ```bash
   pip install virtualenv
   ```

2. **Install PHP and Composer**:
   - Download PHP from [php.net](https://www.php.net/downloads).
   - Install Composer by following the instructions on [getcomposer.org](https://getcomposer.org/download/).

3. **Clone the repository**:
   ```bash
   git clone https://github.com/ManasMadan/c4gt-website.git
   cd c4gt-website
   ```

4. **Install PHP dependencies for excelinterop**:
   Navigate to the `excelinterop` directory and run:
   ```bash
   cd excelinterop
   composer install
   cd ..
   ```

5. **Create a new virtual environment**:
   ```bash
   venv env
   ```

6. **Activate the virtual environment**:
   ```bash
   .\env\Scripts\activate
   ```

7. **Install Python dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

8. **Configure environment variables**:
   Create a `.env` file in the root directory of the project and add the required variables. You can use the provided `.env.example` file as a reference:
   ```bash
   cp .env.example .env
   ```

9. **Run the application**:
   ```bash
   python main.py
   ```

10. Access the application in your web browser at `http://127.0.0.1:5000`.

### For Mac/Linux

1. **Install virtualenv (if not installed)**:
   Open Terminal and run:
   ```bash
   pip3 install virtualenv
   ```

2. **Install PHP and Composer**:
   - Download PHP from [php.net](https://www.php.net/downloads).
   - Install Composer by following the instructions on [getcomposer.org](https://getcomposer.org/download/).

3. **Clone the repository**:
   ```bash
   git clone https://github.com/ManasMadan/c4gt-website.git
   cd c4gt-website
   ```

4. **Install PHP dependencies for excelinterop**:
   Navigate to the `excelinterop` directory and run:
   ```bash
   cd excelinterop
   composer install
   cd ..
   ```

5. **Create a new virtual environment**:
   ```bash
   python3 -m venv env
   ```

6. **Activate the virtual environment**:
   ```bash
   source env/bin/activate
   ```

7. **Install Python dependencies**:
   ```bash
   pip3 install -r requirements.txt
   ```

8. **Configure environment variables**:
   Create a `.env` file in the root directory of the project and add the required variables. You can use the provided `.env.example` file as a reference:
   ```bash
   cp .env.example .env
   ```

9. **Run the application**:
   ```bash
   python3 main.py
   ```

10. Access the application in your web browser at `http://127.0.0.1:5000`.

## Docker Setup (Recommended)

This application now supports Docker with MinIO for local development:

### Prerequisites for Docker
- Docker
- Docker Compose

### Docker Setup Instructions

1. **Clone the repository**:
   ```bash
   git clone https://github.com/ManasMadan/c4gt-website.git
   cd c4gt-website
   ```

2. **Configure environment variables**:
   Create a `.env` file from the example:
   ```bash
   cp .env.example .env
   ```

3. **Start the services**:
   ```bash
   docker-compose up -d
   ```

4. **Access the services**:
   - **Application**: `http://localhost:5000`
   - **MinIO Console**: `http://localhost:9001` (admin/admin)
   - **MinIO API**: `http://localhost:9000`

5. **Create the storage bucket**:
   - Open MinIO console at `http://localhost:9001`
   - Login with username: `minioadmin`, password: `minioadmin`
   - Create a bucket named `c4gt-storage` (or match your `MINIO_BUCKET_NAME` in `.env`)

### MinIO Configuration

This application uses MinIO as an open-source, S3-compatible object storage solution instead of AWS S3:

- **MinIO Server**: Runs on port 9000
- **MinIO Console**: Web UI on port 9001  
- **Default Credentials**: minioadmin/minioadmin
- **Storage**: Local volume `minio_data`

Environment variables for MinIO:
```env
MINIO_ENDPOINT=http://minio:9000
MINIO_ACCESS_KEY=minioadmin
MINIO_SECRET_KEY=minioadmin
MINIO_BUCKET_NAME=c4gt-storage
```

### Stopping the Services
```bash
docker-compose down
```

To remove volumes as well:
```bash
docker-compose down -v
```

## SEALion AI Integration

This application includes integration with SEALion (Southeast Asian Languages in One Network), an open-source family of Large Language Models specifically designed for Southeast Asian contexts.

### SEALion Features

- **Text Generation**: Generate contextually appropriate content with understanding of Southeast Asian languages and cultures
- **Translation**: Translate text between multiple Southeast Asian languages including:
  - English, Bahasa Indonesia, Bahasa Malaysia
  - Thai, Vietnamese, Tagalog, Mandarin Chinese, Tamil
- **Text Summarization**: Create concise summaries of documents and articles

### Accessing SEALion

1. **Login** to your account
2. **Navigate** to the SEALion AI link in the user menu
3. **Use** any of the three main features:
   - Text Generation with customizable parameters
   - Translation between supported languages
   - Document summarization

### SEALion API Endpoints

- `GET /sealion` - Main SEALion interface
- `POST /sealion` - Text generation API
- `POST /sealion/translate` - Translation API
- `POST /sealion/summarize` - Summarization API

### Technical Requirements

SEALion integration requires additional dependencies:
- `transformers>=4.36.0` - Hugging Face transformers library
- `torch>=2.2.0` - PyTorch for model inference
- `accelerate>=0.24.0` - Model acceleration
- `huggingface_hub>=0.19.0` - Hugging Face model hub

### Model Information

- **Model**: SEALion v3.5 8B parameters
- **Source**: AI Singapore (aisingapore/llama-sea-lion-v3.5-8b-r)
- **Specialization**: Southeast Asian languages and cultural contexts
- **License**: Check model-specific licensing on Hugging Face

### Performance Notes

- First model load may take several minutes depending on hardware
- Model requires significant GPU memory (8GB+ recommended)
- CPU inference is supported but will be slower
- Models are cached locally after first download

## Additional Notes

- If you encounter any issues during installation or execution, please refer to the official documentation for [Flask](https://flask.palletsprojects.com/) and ensure all prerequisites are correctly installed.
- For production deployments, change MinIO default credentials and configure proper security settings.
- SEALion model downloads require stable internet connection and sufficient disk space (15GB+).