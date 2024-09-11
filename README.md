**Abstract**
This GenAI app leverages Retrieval-Augmented Generation (RAG), LlamaIndex, and Streamlit to offer users an intuitive platform for querying real-time financial data from Bank Negara Malaysia (BNM). By combining these technologies, the app allows users to access live exchange rates, interest rates, gold prices, and financial alerts using natural language. It also offers interactive data visualizations and historical trends, making it useful for both the general public and financial professionals. This tool simplifies complex financial insights, providing a user-friendly interface for better decision-making and financial awareness. 

**Introduction**
This proposal outlines the development of a Generative AI (GenAI) application that combines Retrieval-Augmented Generation (RAG), LlamaIndex, and Streamlit to create an intuitive and data-driven platform. The app will utilize real-time and historical data from Bank Negara Malaysia's (BNM) OpenAPI. This app will allow users to query various financial datasets using natural language and receive contextual, data-driven responses. The key datasets include exchange rates, interest rates, and consumer alerts, which are useful for the general public, investors, and businesses.

---

**Objective**
The app aims to simplify how individuals access and understand financial data by using natural language queries, supported by the RAG model, which combines LlamaIndex’s ability to retrieve relevant data with language model-generated insights. The interactive user interface will be built using Streamlit, enabling users to query and visualize data seamlessly.

---

**Key Datasets to Leverage**

1. **Exchange Rates**
   The app will provide users with the ability to query current and historical foreign exchange rates between major global currencies (USD, EUR, MYR, etc.). This will benefit tourists, investors, and international businesses.
   
2. **Interest Rates**
   The app will offer data on current and historical interest rates, including the Overnight Policy Rate (OPR), base lending rates, and Islamic financing rates. This information will be essential for individuals and businesses involved in loans, mortgages, and other financial transactions.
   
3. **Gold Bullion Prices (Kijang Emas)**
   For those interested in commodity investments, users can query the latest gold prices, particularly for Malaysia's Kijang Emas, which is useful for investors in precious metals.
   
4. **Daily Foreign Exchange Turnover**
   Financial analysts and businesses can gain insights into Malaysia’s foreign exchange market by querying the daily foreign exchange turnover, allowing them to monitor liquidity and trading volumes.
   
5. **Islamic Banking Statistics**
   The app will provide Islamic banking data, including profit rates and Shariah-compliant financing details, which will appeal to users interested in Islamic finance.
   
6. **Consumer Financial Alerts**
   A feature offering alerts on financial fraud and unlicensed financial schemes will help users stay informed about potential risks and suspicious activities in the market.

---

**Key Features and Functions**

1. **Natural Language Query Interface**
   - **Functionality**: Users will be able to input questions in natural language such as “What is the current exchange rate for USD to MYR?” or “Show me the base lending rate for the past 5 years.”
   - **Use Case**: This function will allow the general public and financial professionals to access relevant data without requiring in-depth technical knowledge or searching through complex tables.

2. **Real-time Data Retrieval**
   - **Functionality**: The app will retrieve the most up-to-date financial data directly from the BNM OpenAPI, including live exchange rates, interest rates, and daily foreign exchange turnover.
   - **Use Case**: Investors and analysts can make informed decisions by accessing the latest data when making currency trades, investment decisions, or financial planning.

3. **Historical Data Trends**
   - **Functionality**: Users can ask for historical trends on various datasets, such as “What were the interest rate trends over the past decade?” The app will fetch historical data and provide visualizations to show trends.
   - **Use Case**: This feature will be particularly useful for financial analysts, economists, and researchers looking to analyze long-term trends in Malaysia’s economy.

4. **Gold Price Queries**
   - **Functionality**: The app will provide live and historical data on Kijang Emas gold prices. Users can ask for current rates or visualize price fluctuations over a specified period.
   - **Use Case**: Investors in precious metals can track gold prices and make informed buying or selling decisions.

5. **Interactive Financial Visualizations**
   - **Functionality**: Using Streamlit’s visualization tools, the app will generate interactive graphs, charts, and dashboards based on the retrieved data. For example, users can see time-series graphs for exchange rate trends or bar charts comparing interest rates over different periods.
   - **Use Case**: Data visualizations will help users understand complex financial data easily, making the app useful for both financial experts and the general public.

6. **Fraud Alert Notifications**
   - **Functionality**: The app will provide the latest consumer alerts on unlicensed financial schemes or fraudulent activities by querying BNM’s consumer alerts dataset.
   - **Use Case**: This feature will protect users by keeping them informed about scams and illegal financial operations, thus increasing financial awareness and safety.

7. **Customizable Data Export**
   - **Functionality**: Users will have the option to export queried data into CSV or Excel files for further analysis or record-keeping.
   - **Use Case**: Financial professionals, businesses, and researchers can download the data for reporting, presentations, or deeper analysis.

---

**Technical Architecture**

1. **RAG (Retrieval-Augmented Generation)**: This model will enhance the accuracy of responses by retrieving relevant documents and using the language model to generate precise answers based on factual data.

2. **LlamaIndex**: LlamaIndex will connect with the BNM OpenAPI to access structured financial datasets. It will serve as the retrieval system, ensuring that the language model has access to relevant, real-time data.

3. **Streamlit**: Streamlit will serve as the user interface, offering a clean, interactive platform for users to input their queries and visualize the results.

---

**Public Benefits**

- **Accessibility**: Users from all walks of life can access complex financial data with simple language queries.
- **Real-time Decision Making**: Investors and businesses can base their decisions on the latest financial information available.
- **Increased Financial Literacy**: By providing easy access to key financial statistics and fraud alerts, the app can promote financial literacy among the general public.
  
This GenAI app will significantly simplify how individuals and businesses interact with financial data, providing them with actionable insights in an easy-to-use platform.

** Architecture **

```
+--------------------------+
|     User Input            |
| (Natural Language)  |
+---------+----------------+
              |
              v
+----------------------------+             +---------------------------------+
|  RAG Query Engine   |<----------|     Local Data Storage     |
| (Retrieval + LLM)       |              | (Financial Datasets: FX,  |
+----------------------------+              |  Interest Rates, etc.)        |
              |
              v
+------------------------+                  +---------------------------------+
|    LlamaIndex        |---------->    |   Indexed Financial Data   |
|  (Retrieves Data)  |                   |  (Historical, Real-Time)     |
+------------------------+                  +----------------------------------+
              |
              v
+---------------------------+ 
| LLM Response         |
| (Data-driven Output)|
+---------------------------+
             |
             v
+-------------------------+
|  Streamlit UI           |
| (Interactive Display|
|   & Visualizations)  |
+-------------------------+
```
**To-do**
- Decide data to use
- Scrape data https://apikijangportal.bnm.gov.my/openapi
- Prompt construction if required
- Decide loader, vector db, embeddings 
- App development
.
