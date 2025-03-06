# 🚀 **FnReport RAG** 🏦📊  
[![FastAPI](https://img.shields.io/badge/FastAPI-0.95.1-green.svg?style=for-the-badge&logo=fastapi)](https://fastapi.tiangolo.com/)  
[![LangChain](https://img.shields.io/badge/LangChain-RAG-yellow.svg?style=for-the-badge)](https://www.langchain.com/)  
[![OpenAI](https://img.shields.io/badge/OpenAI-GPT4-blue.svg?style=for-the-badge&logo=openai)](https://openai.com/)  
[![AWS](https://img.shields.io/badge/AWS-Cloud-orange.svg?style=for-the-badge&logo=amazon-aws)](https://aws.amazon.com/)  
[![Docker](https://img.shields.io/badge/Dockerized-✔️-lightblue.svg?style=for-the-badge&logo=docker)](https://www.docker.com/)  

## 📌 **Project Overview**

Harnessing advanced **Retrieval-Augmented Generation (RAG) architectures**, **FnReport RAG** is an AI-native financial analytics system engineered with a modular microservices approach, integrating **FastAPI for high-throughput API orchestration, LangChain for dynamic LLM-driven reasoning, OpenAI’s transformer-based models for contextual comprehension, and a high-dimensional vector database for optimized semantic retrieval**. The system is developed within a **fully Agile SDLC**, leveraging **Kanban-driven workflow automation, sprint-based iterative refinement, and CI/CD pipelines with containerized deployments** to ensure continuous model fine-tuning, latency minimization, and scalable inference. By synthesizing complex, unstructured financial disclosures into **high-fidelity, domain-aware insights**, this architecture enables real-time extraction, token-level precision processing, and multi-modal financial intelligence at scale.

--- 

## Front-End 

See https://github.com/alojony/FinReportRAG-FrontendGUI

---

## ✨ **Key Features**
🔹 **📄 Intelligent Document Parsing**: Extracts structured data from complex financial statements (PDFs).  
🔹 **🔍 RAG-powered Query System**: Uses a vector database for accurate financial Q&A.  
🔹 **⚡ FastAPI-powered API**: A scalable and robust backend for seamless integration.  
🔹 **🚀 AWS Cloud Deployment**: Dockerized for cost-effective cloud deployment.  
🔹 **📊 Multi-query RAG Strategies**: Supports RAG-fusion, multi-query, and hybrid retrieval.  
🔹 **🔒 Secure & Scalable**: Built with security, efficiency, and extensibility in mind.  

---

## 🏗 **Tech Stack**
| Category         | Technologies Used  |
|-----------------|------------------|
| **Backend**     | FastAPI, Python  |
| **AI & NLP**    | OpenAI GPT, LangChain |
| **Data Storage**| FAISS, ChromaDB, Pinecone |
| **Deployment**  | Docker, AWS ECS, Lambda |
| **Frontend**    | (Planned) Separate Frontend via API |

---

## 📁 **Project Structure**
```
FnReport-RAG/
│── backend/
│   ├── app/ (FastAPI app, routes, models, services)
│── frontend/ (Planned: Separate repo)
│── docker/ (Docker configs)
│── infrastructure/ (AWS deployment scripts)
│── scripts/ (Data ingestion)
│── data/ (Financial reports)
│── README.md (You are here 📌)
```

## 🏛 **Architecture**

```
+-----------------------------+
|   User Query (Frontend)     |
+-----------------------------+
              │
              ▼
+-----------------------------+
|      FastAPI Backend        |
|-----------------------------|
| - API (main.py, api.py)     |
| - RAG Pipeline (rag.py)     |
| - PDF Processing            |
| - Vector DB & Embeddings    |
+-----------------------------+
              │
              ▼
+-----------------------------+
|     LangChain Integration   |
+-----------------------------+
              │
              ▼
+-----------------------------+
|      Vector Database        |
+-----------------------------+
              │
              ▼
+-----------------------------+
|         OpenAI GPT          |
+-----------------------------+
              │
              ▼
+-----------------------------+
|        AWS Cloud            |
+-----------------------------+
```



---

## 🚀 **Getting Started**

### **1️⃣ Clone the Repository**

```shell
git clone https://github.com/YOUR-USERNAME/FnReport-RAG.git
cd FnReport-RAG
```

### **2️⃣ Install Dependencies**

```shell
pip install -r requirements.txt
```

### **3️⃣ Run the API**

```shell
uvicorn backend.app.main:app --reload
```

The API will be available at **`http://127.0.0.1:8000`**.

### **4️⃣ Test the API**

```shell
curl http://127.0.0.1:8000/query?text="What is the net income?"
```

---

## 📡 **Docker Deployment**
To deploy the API using **Docker**, follow these steps:

### **1️⃣ Build the Docker Image**

```
docker build -t fnreport-rag .
```

### **2️⃣ Run the Container**

```
docker run -p 8000:8000 fnreport-rag
```

### **3️⃣ Test the Running API**

```
curl http://localhost:8000/query?text="What is the revenue?"
```

---

## ☁ **Deploying on AWS**
The project can be deployed on **AWS Lambda, ECS, or EC2**.  
Check the **`infrastructure/`** directory for setup scripts.

### **Planned AWS Architecture**
✅ **AWS S3** → Stores PDF reports  
✅ **AWS Lambda / ECS** → Hosts FastAPI backend  
✅ **AWS RDS or DynamoDB** → Metadata storage  
✅ **AWS API Gateway** → Secure API endpoint  

Here's an example deoloyment with Docker & AWS: 

```bash
$ aws configure  # Set AWS credentials
$ docker build -t fnreport-rag .
$ docker tag fnreport-rag:latest your-aws-repo/fnreport-rag:latest
$ docker push your-aws-repo/fnreport-rag:latest
```

---

## 🛠 **Customization & Configuration**
Modify `.env` to adjust settings:

```
DEBUG=True
DATABASE_URL=postgres://user:password@localhost/db
OPENAI_API_KEY=your-key-here
```

---

## 👨‍💻 **Contributors**

| Name        | Role          | GitHub Profile  |
|------------|--------------|----------------|
| **[Your Name]** | Project Lead | [GitHub](https://github.com/YOUR-USERNAME) |
| **[Contributor 2]** | Backend Engineer | [GitHub](https://github.com/USER) |
| **[Contributor 3]** | AI Specialist | [GitHub](https://github.com/USER) |

Want to contribute? **Feel free to fork and PR!**

---

## 📄 **License**
This project is licensed under the **MIT License**.  

---

## ⭐ **Like This Project? Give It a Star! 🌟**
If you find this useful, consider giving it a **star** ⭐ on GitHub!  
