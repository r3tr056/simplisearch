# SimpliSearch: Your Simple & Fast Vector Database Server

[![License](https://img.shields.io/badge/License-MIT-blue.svg)](https://opensource.org/licenses/MIT)
[![C++](https://img.shields.io/badge/C++-%2300599C.svg?style=for-the-badge&logo=c%2B%2B&logoColor=white)](https://isocpp.org/)
[![PostgreSQL](https://img.shields.io/badge/PostgreSQL-316192.svg?style=for-the-badge&logo=postgresql&logoColor=white)](https://www.postgresql.org/)
[![ONNX Runtime](https://img.shields.io/badge/ONNX_Runtime-4B44CE.svg?style=for-the-badge&logo=onnx&logoColor=white)](https://onnxruntime.ai/)

**SimpliSearch** is a lightweight, fast, and easy-to-use vector database server built with C++, ONNX Runtime for embedding generation, and PostgreSQL with the `vector` extension for efficient vector storage and similarity search. It's designed to provide a simple yet powerful solution for semantic search and vector-based applications, prioritizing speed and ease of setup.

## ✨ Key Features

*   **Simplicity First:**  Easy to set up and use with a straightforward API. Perfect for quick prototyping and smaller-scale applications.
*   **Fast Similarity Search:** Leverages PostgreSQL's `vector` extension and IVFFLAT indexing for efficient cosine similarity searches.
*   **Local Model Inference:** Uses ONNX Runtime to run pre-converted sentence embedding models locally, ensuring privacy and speed. No dependency on external APIs for embeddings at runtime.
*   **Powered by PostgreSQL:**  Reliable and robust data storage using PostgreSQL, a mature and widely-used database.
*   **Sentence Transformer Ready:** Optimized for sentence embedding models like `sentence-transformers/all-MiniLM-L6-v2` (easily configurable).
*   **CLI Setup Tool:** Includes a Python-based CLI (`cli.py`) for model conversion, database setup, and basic data indexing.
*   **REST API:**  Provides a simple REST API for adding vectors and performing search queries.
*   **Configurable:** Easily configure database connection, server port, model settings through environment variables.

## 🚀 Getting Started

Follow these steps to get SimpliSearch up and running:

### 1. Prerequisites

Before you begin, ensure you have the following installed on your system:

*   **C++ Compiler:**  A modern C++ compiler (like g++ or clang++) with C++17 support.
*   **CMake (>= 3.10):**  Build system.
*   **Make:** Build tool.
*   **PostgreSQL (>= 13) with `vector` extension:**  Database server. Make sure the `vector` extension is installed and enabled in your database.
*   **Python 3.x with `pip`:** Required for the setup CLI tool.
*   **Python Libraries:** Install the necessary Python libraries using pip:
    ```bash
    pip install transformers torch onnx psycopg2 urllib3
    ```

### 2. Clone the Repository

```bash
git clone [https://github.com/r3tr056/simplisearch]  # Replace with the actual repository URL
cd simpli-search
```

### 3. Convert Hugging Face Model to ONNX

Use the provided Python CLI tool (`cli.py`) to convert your desired Hugging Face sentence embedding model to ONNX format. This example uses `sentence-transformers/all-MiniLM-L6-v2`:

```bash
python cli.py convert-model --model-name sentence-transformers/all-MiniLM-L6-v2 --output-path models/model.onnx
```

This command will:

*   Download the `sentence-transformers/all-MiniLM-L6-v2` model from Hugging Face.
*   Convert it to ONNX format.
*   Save the ONNX model to `models/model.onnx`.

You can adjust the `--model-name` and `--output-path` arguments as needed.

### 4. Set Up PostgreSQL Database

Use the CLI tool to set up your PostgreSQL database. **Make sure your PostgreSQL server is running and you have appropriate credentials.**

```bash
python cli.py setup-db --db-url "postgresql://your_user:your_password@your_host:your_port/your_database_name"
```

**Replace `"postgresql://your_user:your_password@your_host:your_port/your_database_name"` with your actual PostgreSQL connection URL.**

This command will:

*   Create the `vector` extension in your database (if it doesn't exist).
*   Create the `embeddings` table with the correct schema and index.

### 5. Build the C++ Server

Navigate to the project directory and use CMake to build the C++ server:

```bash
mkdir build
cd build
cmake ..
make
```

This will compile the `vector_db_server` executable in the `build` directory.

### 6. Run the SimpliSearch Server

Run the compiled server executable from the `build` directory:

```bash
./vector_db_server
```

The server will start and listen for requests on `http://0.0.0.0:8080/api` by default.

## ⚙️ Configuration

SimpliSearch can be configured using environment variables. Here are the key variables you can set:

*   **`DB_HOST` (default: `localhost`):** PostgreSQL database host.
*   **`DB_PORT` (default: `5432`):** PostgreSQL database port.
*   **`DB_NAME` (default: `vector_db`):** PostgreSQL database name.
*   **`DB_USER` (default: `postgres`):** PostgreSQL database username.
*   **`DB_PASSWORD` (default: `password`):** PostgreSQL database password.
*   **`SERVER_PORT` (default: `8080`):** Port for the SimpliSearch server to listen on.
*   **`MODEL_NAME` (default: `sentence-transformers/all-MiniLM-L6-v2`):** Hugging Face model name (used in setup script, but not directly used by the server anymore after ONNX conversion).
*   **`MODEL_CACHE_DIR` (default: `models`):** Directory where the ONNX model is stored.

**Example: Setting environment variables before running the server:**

```bash
export DB_HOST=mydbserver.example.com
export DB_PORT=5433
export DB_NAME=my_search_db
export DB_USER=search_user
export DB_PASSWORD=secure_password
export SERVER_PORT=8081
./vector_db_server
```

## ✉️ API Endpoints

SimpliSearch provides a simple REST API under the `/api` path.

### 1. Add Vector (`POST /api/add`)

Adds a new vector embedding to the database.

**Request Body (JSON):**

```json
{
  "key": "unique_document_id_1",
  "text": "Text content to be embedded (optional, for reference)",
  "metadata": {
    "source": "example_source",
    "other_info": "..."
  }
}
```

*   `key`: A unique identifier for the vector. If the key already exists, the vector and metadata will be updated.
*   `text` (optional): The original text content associated with the vector. This is not used for searching but can be helpful for debugging or metadata.
*   `metadata`:  A JSON object containing any additional metadata you want to associate with the vector.

**Example `curl` command:**

```bash
curl -X POST http://0.0.0.0:8080/api/add \
  -H "Content-Type: application/json" \
  -d '{"key":"doc1","text":"sample text","metadata":{"source":"example"}}'
```

**Response:**

*   `200 OK`: Vector added successfully.
*   `400 Bad Request`:  Error in request data.

### 2. Search Vectors (`POST /api/search`)

Searches for vectors similar to the query vector.

**Request Body (JSON):**

```json
{
  "query": "Your search query text",
  "top_k": 5,       // Optional, default: 5 - Number of top results to return
  "threshold": 0.6  // Optional, default: 0.6 - Minimum similarity score to consider (cosine similarity)
}
```

*   `query`: The text query for semantic search. SimpliSearch will generate an embedding for this query and search for similar vectors in the database.
*   `top_k` (optional): The maximum number of results to return (default is 5).
*   `threshold` (optional):  The minimum cosine similarity score for a result to be included in the response (default is 0.6).  Similarity is calculated as `1 - distance` in PostgreSQL (where distance is cosine distance).

**Example `curl` command:**

```bash
curl -X POST http://0.0.0.0:8080/api/search \
  -H "Content-Type: application/json" \
  -d '{"query":"search text","top_k":5,"threshold":0.6}'
```

**Response (JSON):**

```json
[
  {
    "key": "doc1",
    "similarity": 0.923,
    "metadata": {
      "source": "example"
    }
  },
  {
    "key": "doc2",
    "similarity": 0.887,
    "metadata": {
      "source": "another_source"
    }
  },
  // ... more results up to top_k ...
]
```

*   An array of result objects, sorted by similarity score in descending order.
*   Each result object contains:
    *   `key`: The unique key of the matching vector.
    *   `similarity`: The cosine similarity score (between 0 and 1) indicating the relevance of the result. Higher is better.
    *   `metadata`: The associated metadata for the vector.

## ⚠️ Important Notes

*   **Model Conversion is Required:**  SimpliSearch expects the ONNX model to be pre-converted and placed in the `models` directory. You **must** run the `cli.py convert-model` command before starting the server.
*   **Basic Example:** SimpliSearch is designed for simplicity and speed in basic use cases. For very large-scale, feature-rich search engines, consider more advanced solutions like Elasticsearch or dedicated vector databases.
*   **`embed-and-index` Example in CLI:** The `embed-and-index` command in `cli.py` is a very basic example for demonstration. For production indexing, you will need to implement a more robust data pipeline tailored to your data sources and requirements.
*   **Performance:** Performance will depend on your hardware, model size, database configuration, and data volume. For optimal search speed, ensure proper indexing in PostgreSQL and consider tuning PostgreSQL parameters.

## 🤝 Contributing

Contributions are welcome! If you have ideas for improvements, bug fixes, or new features, please feel free to open issues or submit pull requests.

## 📜 License

SimpliSearch is released under the [MIT License](LICENSE).

---

**Enjoy building simple and fast search applications with SimpliSearch!**
```