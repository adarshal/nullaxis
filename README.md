# NYC 311 Data Analytics Agent

A LangGraph-based data analytics agent that answers natural language questions about NYC 311 service requests data.

## Features

- 🤖 **Natural Language Queries**: Ask questions in plain English
- 📊 **Interactive Visualizations**: Automatic chart generation based on query type
- 🏗️ **LangGraph Architecture**: Robust workflow with LLM-based intent analysis
- 🔍 **SQL Generation**: Automatic SQL query generation from natural language
- 📈 **Real-time Analytics**: Live data analysis and insights

## Setup Instructions

### Step 1: Clone Repository

```bash
# Clone the repository
git clone <your-repo-url>
cd nullaxis
```

### Step 2: Create Virtual Environment

```bash
# Create virtual environment
python -m venv venv

# Activate virtual environment
# On Windows:
venv\Scripts\activate
# On macOS/Linux:
source venv/bin/activate
```

### Step 3: Install Dependencies

```bash
# Install required packages
pip install -r requirements.txt
```

### Step 4: Download Data from Kaggle

1. Go to [Kaggle NYC 311 Dataset](https://www.kaggle.com/datasets/nycdata/nyc-311-service-requests)
2. Download the CSV file: `311_Service_Requests_from_2010_to_Present.csv`
3. Place the downloaded CSV file in the root folder of this project

### Step 5: Setup Database

```bash
# Run the setup script to create the database
python setup_existing_data.py
```

This will:
- Check for the CSV file
- Create the SQLite database
- Process and index the data
- Generate a summary of the dataset

### Step 6: Configure API Key

1. Edit the `.env` file (created by setup script)
2. Add your DeepSeek API key:
```
DEEPSEEK_API_KEY=your_deepseek_api_key_here
```

### Step 7: Run the Application

```bash
# Start the Streamlit app
streamlit run app.py
```

### Step 8: Use the Application

1. Open your browser and visit the URL shown in the terminal (usually `http://localhost:8501`)
2. **First-time loading**: Wait for the app to fully load - you'll see a loading indicator in the top right corner of the Streamlit interface
3. Wait for the chat component to load completely
4. for 1st time it may take tie to fully load streamlit app.
5. Start asking questions about the NYC 311 data!

## Sample Questions

Try asking these questions in the chat interface:

- "What are the top 10 complaint types by number of records?"
- "For the top 5 complaint types, what percent were closed within 3 days?"
- "Which ZIP code has the highest number of complaints?"
- "What proportion of complaints include a valid latitude/longitude?"
- "Show me complaints by borough"
- "What's the average closure time by agency?"

## Architecture

The system uses LangGraph to orchestrate the following workflow:

1. **Intent Classification**: LLM analyzes the user query
2. **Schema Understanding**: Retrieves relevant table schemas
3. **Query Generation**: Generates SQL queries from natural language
4. **Query Validation**: Checks for common SQL mistakes
5. **Execution**: Runs the query against the database
6. **Response Formatting**: Formats results with visualizations

## Components

- `src/agents/sql_agent.py`: Main LangGraph agent
- `src/utils/data_processor.py`: Data download and database setup
- `src/utils/deepseek_client.py`: DeepSeek API integration
- `src/executors/`: SQL execution and prebuilt analytics functions
- `app.py`: Streamlit frontend interface

## Configuration

Edit `config.py` to customize:
- Database path
- CSV data path
- API timeouts
- Chart dimensions

## Requirements

- DeepSeek API key
- csv data (311_Service_Requests_from_2010_to_Present.csv)

## Troubleshooting

### API Key Issues
Make sure your DeepSeek API key is set in the `.env` file:
```
DEEPSEEK_API_KEY=your_key_here
```

### Data File Issues
If the setup script can't find the CSV file, make sure:
1. The file is named exactly: `311_Service_Requests_from_2010_to_Present.csv`
2. The file is placed in the root folder of the project (same level as `setup_existing_data.py`)

### Database Issues
If you encounter database errors, delete the existing database file (`data/nyc311.db`) and run `python setup_existing_data.py` again.

## inDevelopment

To extend the system:

1. Add new prebuilt functions in `src/executors/prebuilt_functions.py`
2. Modify the LangGraph workflow in `src/agents/sql_agent.py`
3. Enhance visualizations in `app.py`

   <img width="1875" height="796" alt="image" src="https://github.com/user-attachments/assets/5fad91c3-c6df-4969-9293-18f0bc062f47" />


