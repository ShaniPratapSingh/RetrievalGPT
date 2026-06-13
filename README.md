# 🚀 DeepRetrieval-Gemini

> Enhanced version of DeepRetrieval with Gemini API integration for intelligent query rewriting and retrieval optimization.

![Python](https://img.shields.io/badge/Python-3.9+-blue.svg)
![Gemini](https://img.shields.io/badge/Google-Gemini-orange.svg)
![License](https://img.shields.io/badge/License-MIT-green.svg)

## 📖 Overview

DeepRetrieval-Gemini is a modernized implementation of the DeepRetrieval framework that replaces the original vLLM-based deployment with Google's Gemini API for lightweight, cross-platform query rewriting.

The project generates optimized Boolean search queries from natural language questions, improving retrieval quality for search engines, document collections, and RAG pipelines.

### Example

**Input Query**

```text
What is artificial intelligence?
```

**Generated Query**

```text
("artificial intelligence" OR AI) AND ("machine learning" OR ML)
```

---

## ✨ Features

* 🔍 Intelligent query rewriting
* 🤖 Powered by Gemini API
* ⚡ Fast response generation
* 💻 Works on macOS, Windows, and Linux
* 🔗 Easy integration with retrieval pipelines
* 📚 Optimized Boolean query generation
* 🚀 Lightweight deployment without local LLM hosting

---

## 🏗️ Project Structure

```text
DeepRetrieval/
│
├── query_rewrite.py
├── code/
├── images/
├── README.md
├── LICENSE
└── vllm_host.sh
```

---

## 🛠️ Tech Stack

* Python 3.9+
* Google Gemini API
* Requests
* JSON
* BM25 Query Rewriting
* Information Retrieval

---

## ⚙️ Installation

### Clone Repository

```bash
git clone https://github.com/YOUR_USERNAME/DeepRetrieval-Gemini.git

cd DeepRetrieval-Gemini
```

### Create Environment

```bash
conda create -n deepretrieval python=3.9 -y

conda activate deepretrieval
```

### Install Dependencies

```bash
pip install google-generativeai
pip install requests
```

---

## 🔑 Setup Gemini API Key

Generate an API key from Google AI Studio.

Create an environment variable:

```bash
export GOOGLE_API_KEY="YOUR_API_KEY"
```

Verify:

```bash
echo $GOOGLE_API_KEY
```

---

## ▶️ Usage

Run query rewriting:

```bash
python query_rewrite.py --query "What is artificial intelligence?"
```

Example output:

```text
Original Query:
What is artificial intelligence?

Rewritten Query:
("artificial intelligence" OR AI)
AND
("machine learning" OR ML)
```

---

## 🧠 How It Works

1. User enters a natural language query.
2. Gemini analyzes the query.
3. Important concepts and synonyms are extracted.
4. Boolean operators are applied.
5. An optimized retrieval query is generated.

---

## 📊 Use Cases

* Retrieval-Augmented Generation (RAG)
* Search Engine Optimization
* Academic Literature Search
* Enterprise Knowledge Bases
* Semantic Search Systems
* Information Retrieval Research

---

## 🚀 Future Improvements

* Streamlit Web Interface
* Query Expansion
* Multi-Search Backend Support
* Hybrid Retrieval
* Vector Database Integration
* LangChain Integration

---

## 📸 Demo

```bash
python query_rewrite.py --query "Recent advances in machine learning"
```

Output:

```text
("machine learning" OR ML)
AND
("deep learning" OR neural networks)
AND
recent advances
```

---

## 🤝 Contributing

Contributions are welcome.

1. Fork the repository
2. Create a feature branch
3. Commit your changes
4. Push to GitHub
5. Open a Pull Request

---

## 👨‍💻 Author

**Shani Pratap Singh**

GitHub: https://github.com/ShaniPratapSingh

---

## ⭐ Support

If you found this project useful, please consider giving it a star ⭐.

---

## 📜 License

This project is licensed under the MIT License.
