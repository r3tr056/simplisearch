#include <iostream>
#include <vector>
#include <string>
#include <unordered_map>
#include <libpq-fe.h>
#include <onnxruntime_cxx_api.h>
#include <Eigen/Dense>
#include <cpprest/http_listener.h>
#include <cpprest/json.h>
#include <nlohmann/json.hpp>
#include <curl/curl.h>
#include <filesystem>
#include <fstream>
#include <thread>
#include <chrono>

using namespace web;
using namespace web::http;
using namespace web::http::experimental::listener;
namespace fs = std::filesystem;

struct ModelConfig {
	std::string model_name = "sentence-transformers/all-MiniLM-L6-v2";
	std::string cache_dir = "models";
	int embedding_dimension = 384;
	int max_sequence_length = 128;
};

class ModelManager {
private:
	ModelConfig config;
	std::unique_ptr<Ort::Session> session;
	Ort::Env env{ORT_LOGGING_LEVEL_WARNING, "EmbeddingModel"};

public:
	ModelManager(const ModelConfig& config) : config(config) {
		fs::create_directories(config.cache_dir);
	}

	bool loadModel() {
		std::string model_path = config.cache_dir + "/model.onnx";
		if (fs::exists(model_path)) {
			std::cout << "Model already exists locally" << std::endl;
			return true;
		} else {
			std::cerr << "Error: ONNX model not found at: " << model_path << std::endl;
            std::cerr << "Please run the `convert_hf_model_to_onnx.py` script to convert and save the model." << std::endl;
            return false;
		}
	}

	bool initialize() {
		if (!loadModel()) return false;

		Ort::SessionOptions session_options;
		session_options.SetIntraOpNumThreads(1);
		session_options.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_ALL);

		try {
			session = std::make_unique<Ort::Session>(
				env,
				(config.cache_dir + "/model.onnx").c_str(),
				session_options
			);
			return true;
		} catch (const Ort::Exception& e) {
			std::cerr << "Error initializing ONNX session: " << e.what() << std::endl;
			return false;
		}
	}

	// std::vector<int64_t> tokenize(const std::string& text) {
    //     // Simple tokenizer (for testing) - Use a proper BERT tokenizer here
    //     std::vector<int64_t> token_ids(config.max_sequence_length, 0);
    //     for (size_t i = 0; i < std::min(text.size(), (size_t)config.max_sequence_length); ++i) {
    //         token_ids[i] = static_cast<int64_t>(text[i]);
    //     }
    //     return token_ids;
    // }

	Eigen::VectorXf getEmbedding(const std::string& text) {
		if (!session) throw std::runtime_error("Model not initialized");

		// prepare input
		std::vector<int64_t> inputShape = {1, static_cast<int64_t>(std::min(
			text.length(),
			static_cast<size_t>(config.max_sequence_length)
		))};
		std::vector<int64_t> inputTensorValues(inputShape[1], 1LL);

		// create input tensor
		Ort::MemoryInfo memoryInfo = Ort::MemoryInfo::CreateCpu(OrtAllocatorType::OrtArenaAllocator, OrtMemType::OrtMemTypeDefault);
		Ort::Value inputTensor = Ort::Value::CreateTensor<int64_t>(
			memoryInfo,
			inputTensorValues.data(),
			inputTensorValues.size(),
			inputShape.data(),
			inputShape.size()
		);

		// run inference
		const char* inputNames[] = {"input"};
		const char* outputNames[] = {"output"};
		auto outputTensors = session->Run(Ort::RunOptions{nullptr}, inputNames, &inputTensor, 1, outputNames, 1);

		// convert to eigen vector
		float* outputData = outputTensors.front().GetTensorMutableData<float>();
	    Eigen::Map<Eigen::VectorXf> embedding(outputData, config.embedding_dimension);
	    return embedding.normalized();
	}
};

class VectorDB {
private:
	PGconn* conn;

	void checkConnection() {
		if (PQstatus(conn) != CONNECTION_OK) {
			PQreset(conn);
			if (PQstatus(conn) != CONNECTION_OK) {
				throw std::runtime_error("Lost connection to database");
			}
		}
	}

public:
	VectorDB(const std::string& conninfo) {
		conn = PQconnectdb(conninfo.c_str());
        if (PQstatus(conn) != CONNECTION_OK) {
            throw std::runtime_error(PQerrorMessage(conn));
        }

		// Create tables and extensions
        const char* setup_queries[] = {
            "CREATE EXTENSION IF NOT EXISTS vector",
            "CREATE TABLE IF NOT EXISTS embeddings ("
            "id SERIAL PRIMARY KEY,"
            "key TEXT UNIQUE,"
            "vector FLOAT8[],"
            "metadata JSONB,"
            "created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP)",
            "CREATE INDEX IF NOT EXISTS embedding_vector_idx ON embeddings USING ivfflat (vector vector_cosine_ops)"
        };

        for (const auto& query : setup_queries) {
            PGresult* res = PQexec(conn, query);
            if (PQresultStatus(res) != PGRES_COMMAND_OK) {
                std::string error = PQerrorMessage(conn);
                PQclear(res);
                throw std::runtime_error(error);
            }
            PQclear(res);
        }
	}

	~VectorDB() {
		if (conn) PQfinish(conn);
	}

	void addVector(const std::string& key, const Eigen::VectorXf& vector, const nlohmann::json& metadata) {
		checkConnection();

		std::stringstream vector_str;
		vector_str << "{";
		for (int i = 0; i < vector.size(); i++) {
			vector_str << vector[i];
			if (i != vector.size() - 1) vector_str << ",";
		}
		vector_str << "}";

		// prepare parameters for the query
		const char* paramValues[3];
		paramValues[0] = key.c_str();
		paramValues[1] = vector_str.str().c_str();

		// serialize metdata to JSON string
		std::string metadata_str = metadata.dump();
		paramValues[2] = metadata_str.c_str();
		if (paramValues[2] == nullptr) {
            paramValues[2] = "{}";
        }

		PGresult* res = PQexecParams(conn,
			"INSERT INTO embeddings (key, vector, metadata) VALUES ($1, $2, $3) "
			"ON CONFLICT (key) DO UPDATE SET vector = EXCLUDED.vector, "
			"metadata = EXCLUDED.metadata",
			3, nullptr, paramValues, nullptr, nullptr, 0
		);

		if (PQresultStatus(res) != PGRES_COMMAND_OK) {
			std::string error = PQerrorMessage(conn);
			PQclear(res);
			throw std::runtime_error(error);
		}

		PQclear(res);
	}

	std::vector<std::tuple<std::string, float, nlohmann::json>> search(const Eigen::VectorXf& query, int topK = 5, float similarity_threshold = 0.6) {
		checkConnection();

		std::stringstream vector_str;
		vector_str << "[";
		for (int i = 0; i < query.size(); i++) {
			vector_str << query[i];
			if (i != query.size() - 1) vector_str << ",";
		}
		vector_str << "]";

		const char* paramValues[3];
        paramValues[0] = vector_str.str().c_str();
        paramValues[1] = std::to_string(similarity_threshold).c_str();
        paramValues[2] = std::to_string(topK).c_str();

        PGresult* res = PQexecParams(conn,
            "SELECT key, vector <=> $1::vector AS distance, metadata "
            "FROM embeddings "
            "WHERE vector <=> $1::vector < $2 "
            "ORDER BY distance ASC "
            "LIMIT $3",
            3, nullptr, paramValues, nullptr, nullptr, 0
        );

        if (PQresultStatus(res) != PGRES_TUPLES_OK) {
        	std::string error = PQerrorMessage(conn);
        	PQclear(res);
        	throw std::runtime_error(error);
        }

        std::vector<std::tuple<std::string, float, nlohmann::json>> results;
        int rows = PQntuples(res);

        for (int i = 0; i < rows; i++) {
        	std::string key = PQgetvalue(res, i, 0);
        	float distance = std::stof(PQgetvalue(res, i, 1));
        	nlohmann::json metadata = nlohmann::json::parse(PQgetvalue(res, i, 2));
        	results.emplace_back(key, 1.0f - distance, metadata);
        }

        PQclear(res);
        return results;
	}
};


class Server {
private:
	http_listener listener;
	VectorDB& db;
	ModelManager& model_manager;

	void handle_search(http_request request) {
		request.extract_json().then([&, request](pplx::task<web::json::value> task) {
			try {
				web::json::value json_obj = task.get();
				std::string query = json_obj["query"].as_string();
				int topK = json_obj.has_field("top_k") ? json_obj["top_k"].as_integer() : 5;
				float threshold = json_obj.has_field("threshold") ? json_obj["threshold"].as_double() : 0.6f;

				Eigen::VectorXf queryEmbedding = model_manager.getEmbedding(query);
				auto results = db.search(queryEmbedding, topK, threshold);

				web::json::value response = web::json::value::array();
				int index = 0;

				for (const auto& [key, similarity, metadata] : results) {
					web::json::value result = web::json::value::object();
					result["key"] = web::json::value::string(key);
					result["similarity"] = web::json::value::number(similarity);
					std::stringstream metadata_ss;
					metadata_ss << metadata;
					result["metadata"] = web::json::value::parse(metadata_ss.str());
					response[index++] = result;
				}
				request.reply(status_codes::OK, response);
			} catch (const std::exception& e) {
				request.reply(status_codes::BadRequest, web::json::value::string(e.what()));
			}
		});
	}

	void handle_add(http_request request) {
		request.extract_json().then([&, request](pplx::task<web::json::value> task) {
			try {
				web::json::value json_obj = task.get();
				std::string key = json_obj["key"].as_string();
				std::string text = json_obj["text"].as_string();

				std::stringstream metadata_ss;
				metadata_ss << json_obj["metadata"];
				nlohmann::json metadata = nlohmann::json::parse(metadata_ss.str());

				Eigen::VectorXf embedding = model_manager.getEmbedding(text);
				db.addVector(key, embedding, metadata);

				request.reply(status_codes::OK, web::json::value::string("Vector added successfully"));
			} catch (const std::exception& e) {
				request.reply(status_codes::BadRequest, web::json::value::string(e.what()));
			}
		});
	}

public:
	Server(const std::string& url, VectorDB& db, ModelManager& model_manager) : listener(url), db(db), model_manager(model_manager) {
		listener.support(methods::POST, [this](http_request request) {
			auto path = uri::split_path(uri::decode(request.relative_uri().path()));
			if (path.empty()) {
				request.reply(status_codes::BadRequest);
				return;
			}

			if (path[0] == "search") {
				handle_search(request);
			} else if (path[0] == "add") {
				handle_add(request);
			} else {
				request.reply(status_codes::NotFound);
			}
		});
	}

	void start() {
		listener.open().wait();
		std::cout << "Server is running..." << std::endl;
	}

	void stop() {
		listener.close().wait();
	}
};

std::unique_ptr<Server> server_ptr;

void signal_handler(int signal) {
	if (server_ptr) {
		std::cout << "\nShutting down server..." << std::endl;
        server_ptr->stop();
	}
	exit(signal);
}

int main() {
	try {
		// Register signal handler
		signal(SIGINT, signal_handler);
		signal(SIGTERM, signal_handler);

		// Load config from env variables or use defaults
		std::string db_host = std::getenv("DB_HOST") ? std::getenv("DB_HOST") : "localhost";
        std::string db_port = std::getenv("DB_PORT") ? std::getenv("DB_PORT") : "5432";
        std::string db_name = std::getenv("DB_NAME") ? std::getenv("DB_NAME") : "simplisearch";
        std::string db_user = std::getenv("DB_USER") ? std::getenv("DB_USER") : "simplisearch_user";
        std::string db_pass = std::getenv("DB_PASSWORD") ? std::getenv("DB_PASSWORD") : "simplisearch";
        std::string server_port = std::getenv("SERVER_PORT") ? std::getenv("SERVER_PORT") : "8080";

		// Intiailize model manager
		ModelConfig config;
		config.model_name = std::getenv("MODEL_NAME") ? std::getenv("MODEL_NAME") : "sentence-transformers/all-MiniLM-L6-v2";
		config.cache_dir = std::getenv("MODEL_CACHE_DIR") ? std::getenv("MODEL_CACHE_DIR") : "models";

		std::cout << "Initializing model manager..." << std::endl;
		ModelManager model_manager(config);
		if (!model_manager.initialize()) {
			throw std::runtime_error("Failed to initialize model");
		}
		std::cout << "Model manager initialized successfully" << std::endl;

		// Initialize database
		std::string db_conn_str = 
			"host=" + db_host + " " +
            "port=" + db_port + " " +
            "dbname=" + db_name + " " +
            "user=" + db_user + " " +
            "password=" + db_pass;

        std::cout << "Connecting to database..." << std::endl;
        VectorDB db(db_conn_str);
        std::cout << "Database connected successfully" << std::endl;

        // Initialize server
        std::string server_address = "http://0.0.0.0:" + server_port;
        server_ptr = std::make_unique<Server>(server_address + "/api", db, model_manager);

        std::cout << "String server on " << server_address << "..." << std::endl;
        server_ptr->start();

        // Print API usage information
        std::cout << "\nAPI Endpoints:\n"
                  << "POST /api/add    - Add new vector\n"
                  << "POST /api/search - Search vectors\n"
                  << "\nExample curl commands:\n"
                  << "Add vector:\n"
                  << "curl -X POST " << server_address << "/api/add \\\n"
                  << "  -H \"Content-Type: application/json\" \\\n"
                  << "  -d '{\"key\":\"doc1\",\"text\":\"sample text\",\"metadata\":{\"source\":\"example\"}}'\n\n"
                  << "Search vectors:\n"
                  << "curl -X POST " << server_address << "/api/search \\\n"
                  << "  -H \"Content-Type: application/json\" \\\n"
                  << "  -d '{\"query\":\"search text\",\"top_k\":5,\"threshold\":0.6}'\n"
                  << std::endl;

        while (true) {
        	std::this_thread::sleep_for(std::chrono::seconds(1));
        }
	} catch (const std::exception& e) {
		std::cerr << "Error: " << e.what() << std::endl;
		return 1;
	}

	return 0;
}