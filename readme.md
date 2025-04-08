# SHL Assessment Retrieval System

## Overview

The **SHL Assessment Retrieval System** is a web application designed to query and retrieve relevant assessments from the SHL product catalog. The system employs a Retrieval-Augmented Generation (RAG) model that provides users with accurate and contextually relevant test assessments based on their queries. The application is built using Streamlit for the frontend and integrates with Pinecone for efficient vector storage and retrieval.

## Features

- **Data Scraping**: Automatically scrapes assessment data from the SHL product catalog.
- **Data Processing**: Preprocesses and chunks the scraped data for efficient querying.
- **Embedding Model**: Utilizes the `SentenceTransformer` model for encoding queries and documents.
- **Diverse Query Results**: Returns diverse and relevant results based on user queries.
- **User-Friendly Interface**: Built with Streamlit for an interactive user experience.
- **Evaluation**: Includes an evaluation script to assess retrieval performance (using metrics such as Mean Recall@K and MAP@K).

## Technologies Used

- Python
- Streamlit
- Pandas
- Sentence Transformers
- Pinecone (for vector storage and search)
- python-dotenv (for loading environment variables)
- (Optional) BeautifulSoup and Requests for web scraping

## Installation

### Prerequisites

Ensure you have Python 3.7 or higher installed. You can download it from [python.org](https://www.python.org/downloads/).
I used Python 3.10.16.
### Clone the Repository

```bash
git clone https://github.com/pravincoder/shl-assessment-retrieval.git
cd shl-assessment-retrieval
```

### Install Dependencies

You can install the required packages using pip. It is recommended to create a virtual environment first.

```bash
# Create a virtual environment (optional)
python -m venv venv
source venv/bin/activate  # On Windows use `venv\Scripts\activate`

# Install dependencies
pip install -r requirements.txt
```



## Usage

### Scraping Data

Before querying the assessments, you need to scrape the data from the SHL product catalog. You can do this by running the `shl_scraper.py` script:

```bash
python shl_scraper.py
```

This will create a CSV file named `shl_products.csv` containing the scraped assessment data.

### Running the Streamlit App

Once the data is scraped, you can run the Streamlit app:

```bash
streamlit run app.py
```

Open your web browser and navigate to `http://localhost:8501` to access the application.

### Querying Assessments

- Enter your query in the input box and click the "Submit" button.
- The application will display relevant assessments based on your query.

## Code Structure

```
shl-assessment-retrieval/
│
├── app.py # Streamlit application for querying assessments
├── rag.py # RAG model implementation for data processing and querying
├── shl_scraper.py # Web scraper for fetching assessment data
├── evaluate.py # Evaluation script for assessing model performance
├── requirements.txt # List of dependencies
├── .env  # Store Pinecone Key and (Convert text.env to .env and add keys)
└── README.md # Project documentation
```