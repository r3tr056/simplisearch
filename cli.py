
import argparse
import json
import os
from sympy import EX
import torch
import psycopg2
import psycopg2.extensions
from urllib.parse import urlparse, parse_qs
from transformers import AutoModel, AutoTokenizer
from pathlib import Path
import onnxruntime as ort
import numpy as np

def convert_hf_to_onnx(model_name, output_path="models/model.onnx"):
	"""Converts a Hugging face Transformer model to ONNX format"""
	try:
		model = AutoModel.from_pretrained(model_name)
		model.eval()

		dummy_input = torch.randint(low=0, high=10000, size=(1, 128), dtype=torch.long)

		output_dir = Path(output_path).parent
		output_dir.mkdir(parents=True, exist_ok=True)

		# Export to ONNX
		torch.onnx.export(
			model,
			(dummy_input,),  # Model input(s) as a tuple
			output_path,
			export_params=True,
			opset_version=14,
			input_names = ['input'],
			output_names = ['output'],
			dynamic_axes={
				'input' : {0 : 'batch_size', 1: 'sequence_length'},
				'output' : {0 : 'batch_size'}
			}
		)

		print(f"Successfully converted and saved ONNX model to: {output_path}")
	except Exception as e:
		print(f"Error during ONNX conversion: {e}")
		exit(1)

def setup_database(db_url):
	try:
		url = urlparse(db_url)
		dbname = url.path[1:]
		user = url.username
		password = url.password
		host = url.hostname
		port = url.port

		conn = psycopg2.connect(dbname=dbname, user=user, password=password, host=host, port=port)
		conn.set_isolation_level(psycopg2.extensions.ISOLATION_LEVEL_AUTOCOMMIT)
		cur = conn.cursor()

		setup_queries = [
			"CREATE EXTENSION IF NOT EXISTS vector",
			"CREATE TABLE IF NOT EXISTS embeddings ("
			"id SERIAL PRIMARY KEY,"
			"key TEXT UNIQUE,"
			"vector FLOAT8[],"
			"metadata JSONB,"
			"created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP)",
			"CREATE INDEX IF NOT EXISTS embedding_vector_idx ON embeddings USING ivfflat (vector vector_cosine_ops)"
		]

		for query in setup_queries:
			cur.execute(query)
		print("Database tables and extensions set up successfully.")

		cur.close()
		conn.close()
	except Exception as e:
		print(f"Error setting up database: {e}")
		print("Make sure PostgreSQL server is running and connection details are correct.")
		exit(1)

def embed_and_index_data(db_url, data_source, batch_size=32):
	"""Embeds and indexes data from a text file into the database using Hugging Face model"""
	try:
		# Setup database connection
		url = urlparse(db_url)
		dbname = url.path[1:]
		user = url.username
		password = url.password
		host = url.hostname
		port = url.port

		conn = psycopg2.connect(dbname=dbname, user=user, password=password, host=host, port=port)
		conn.set_isolation_level(psycopg2.extensions.ISOLATION_LEVEL_AUTOCOMMIT)
		cur = conn.cursor()

		# Load tokenizer and model from Hugging Face
		tokenizer = AutoTokenizer.from_pretrained("sentence-transformers/all-MiniLM-L6-v2")
		model = AutoModel.from_pretrained("sentence-transformers/all-MiniLM-L6-v2")
		model.eval()

		with open(data_source, 'r') as f:
			lines = f.readlines()

		for i in range(0, len(lines), batch_size):
			batch_texts = lines[i:i+batch_size]
			batch_keys = [f"doc_{i+j}" for j in range(len(batch_texts))]

			# Tokenize the batch of texts
			inputs = tokenizer(batch_texts, padding=True, truncation=True, max_length=128, return_tensors="pt")

			# Generate embeddings using the model
			with torch.no_grad():
				outputs = model(**inputs)
				embeddings = outputs.last_hidden_state[:, 0, :].cpu().tolist()  # Get embeddings of the [CLS] token

			# Insert embeddings into the database
			for key, text, embedding in zip(batch_keys, batch_texts, embeddings):
				metadata = {"source": "cli_index", "text_content": text.strip()}
				vector_str = "[" + ",".join(map(str, embedding)) + "]"
				try:
					cur.execute(
						"INSERT INTO embeddings (key, vector, metadata) VALUES (%s, %s, %s) "
						"ON CONFLICT (key) DO UPDATE SET vector = EXCLUDED.vector, metadata = EXCLUDED.metadata",
						(key, vector_str, json.dumps(metadata))
					)
				except Exception as db_err:
					print(f"Database error during indexing key {key}: {db_err}")
					conn.rollback()
					continue

			conn.commit()
			print(f"Indexed batch {i//batch_size + 1}/{len(lines)//batch_size + (1 if len(lines)%batch_size else 0)}")

		print(f"Indexing from '{data_source}' completed.")
		cur.close()
		conn.close()
	except FileNotFoundError:
		print(f"Error: Data source file not found: {data_source}")
		exit(1)
	except Exception as e:
		print(f"Error during data indexing: {e}")
		exit(1)


def main():
	parser = argparse.ArgumentParser(description="SimpliSearch Server Setup CLI")
	subparsers = parser.add_subparsers(title="commands", dest="command", help="Available commands")

	# Convert Model Command
	convert_parser = subparsers.add_parser("convert-model", help="Convert Hugging Face model to ONNX format")
	convert_parser.add_argument("--model-name", type=str, default="sentence-transformers/all-MiniLM-L6-v2", help="Hugging Face model name")
	convert_parser.add_argument("--output-path", type=str, default="models/model.onnx", help="Path to save ONNX model")

	# Setup DB Command
	db_setup_parser = subparsers.add_parser("setup-db", help="Setup PostgreSQL database for vector search")
	db_setup_parser.add_argument("--db-url", type=str, required=True, help="Database connection URL (e.g., postgresql://user:password@host:port/dbname)")

	# Embed and Index Data (Conceptual Example)
	index_parser = subparsers.add_parser("embed-and-index", help="(Example) Embed and index data from a text file")
	index_parser.add_argument("--db-url", type=str, required=True, help="Database connection URL")
	index_parser.add_argument("--data-source", type=str, required=True, help="Path to the text file data source (one document per line)")
	index_parser.add_argument("--batch-size", type=int, default=32, help="Batch size for indexing")


	args = parser.parse_args()

	if args.command == "convert-model":
		convert_hf_to_onnx(args.model_name, args.output_path)
	elif args.command == "setup-db":
		setup_database(args.db_url)
	elif args.command == "embed-and-index":
		embed_and_index_data(args.db_url, args.data_source, args.batch_size)
	elif args.command is None:
		parser.print_help()
	else:
		print(f"Unknown command: {args.command}")
		parser.print_help()
		exit(1)

if __name__ == "__main__":
	main()