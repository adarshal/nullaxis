# NYC 311 Data Analytics Agent

A LangGraph-based data analytics agent that answers natural language questions about NYC 311 service requests data.

## Features

- 🤖 **Natural Language Queries**: Ask questions in plain English
- 📊 **Interactive Visualizations**: Automatic chart generation based on query type
- 🏗️ **LangGraph Architecture**: Robust workflow with LLM-based intent analysis
- 🔍 **SQL Generation**: Automatic SQL query generation from natural language
- 📈 **Real-time Analytics**: Live data analysis and insights

## Quick Start

### Option 1: Quick Setup (Recommended)

```bash
# Clone the repository
git clone <your-repo-url>
cd nullaxis

# Run quick setup (downloads sample data, no Kaggle API needed)
python quick_setup.py

# Edit .env file and add your DeepSeek API key
# Then run the app
streamlit run app.py
```

### Option 2: Full Setup with Kaggle

```bash
# Clone the repository
git clone <your-repo-url>
cd nullaxis

# Run full setup (requires Kaggle API)
python setup.py

# Run the app
streamlit run app.py
```

### Option 3: Manual Setup

```bash
# Install dependencies
pip install -r requirements.txt

# Download data manually
python download_data.py

# Run the app
streamlit run app.py
```

**Note:** The quick setup downloads a sample of 100k records for faster setup. Use the full setup for the complete dataset.

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

- Python 3.8+
- DeepSeek API key
- Internet connection (for initial data download)

## Troubleshooting

### API Key Issues
Make sure your DeepSeek API key is set in the `.env` file:
```
DEEPSEEK_API_KEY=your_key_here
```

### Data Download Issues
If data download fails, you can manually download the NYC 311 dataset and place it in `data/raw/nyc311.csv`.

### Database Issues
If you encounter database errors, delete the existing database file and run `python download_data.py` again.

## Development

To extend the system:

1. Add new prebuilt functions in `src/executors/prebuilt_functions.py`
2. Modify the LangGraph workflow in `src/agents/sql_agent.py`
3. Enhance visualizations in `app.py`

## License

MIT License
