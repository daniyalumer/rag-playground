## Installation

1. Clone the repository:

    ```bash
    git clone https://github.com/yourusername/rag-playground.git
    cd rag-playground
    ```

2. Install the required dependencies using `pipenv`:

    ```bash
    pipenv install
    ```

3. Activate the virtual environment:

    ```bash
    pipenv shell
    ```

## Usage

1. Configure the Elasticsearch endpoint and API key in the [config.py](http://_vscodecontentref_/1) file.

2. Run the main script:

    ```bash
    pipenv run python main.py
    ```

## Configuration

Update the [config.py](http://_vscodecontentref_/2) file with your Elasticsearch endpoint and API key:

```python
ELASTIC_ENDPOINT = "https://your-elasticsearch-endpoint"
ELASTIC_API_KEY = "your-api-key"