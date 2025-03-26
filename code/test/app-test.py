import unittest
import sqlite3
import os
import numpy as np
import faiss
from unittest.mock import patch, MagicMock
from typing import List, Dict, Optional
import sys
import json

# Add the directory containing the main script to the Python path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Import the main script functions
from customer_analysis_with_streamlit_ui import (
    init_db, get_customer_details, get_all_customer_ids, generate_similarity_query,
    vector_search, get_llm_recommendations, format_customer_data, PRODUCTS,
    initialize_product_embeddings, DB_FILE, OPENAI_API_KEY
)

class TestCustomerAnalysis(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        """Set up the test environment once for all tests."""
        # Ensure the database file is removed before starting tests
        if os.path.exists(DB_FILE):
            os.remove(DB_FILE)
        
        # Initialize the database
        init_db()
        
        # Mock the OpenAIEmbeddings and FAISS index for vector search
        cls.mock_embedding_model = MagicMock()
        cls.mock_index = MagicMock()
        
        # Mock the embed_documents method to return a fixed set of embeddings
        cls.mock_embedding_model.embed_documents.return_value = [
            [0.1 * i] * 1536 for i in range(len(PRODUCTS))  # 1536 is the dimension of text-embedding-3-large
        ]
        
        # Mock the embed_query method to return a fixed query embedding
        cls.mock_embedding_model.embed_query.return_value = [0.5] * 1536
        
        # Mock the FAISS index search method
        cls.mock_index.search.return_value = (
            np.array([[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]]),  # distances
            np.array([[0, 1, 2, 3, 4, 5, 6, 7, 8, 9]])  # indices (top 10 products)
        )

    def setUp(self):
        """Set up before each test."""
        # Patch the initialize_product_embeddings to return mocked index and embedding model
        self.patcher = patch(
            'customer_analysis_with_streamlit_ui.initialize_product_embeddings',
            return_value=(self.mock_index, self.mock_embedding_model)
        )
        self.patcher.start()

    def tearDown(self):
        """Tear down after each test."""
        self.patcher.stop()

    def test_init_db(self):
        """Test that the database is initialized correctly with all tables and data."""
        conn = sqlite3.connect(DB_FILE)
        cursor = conn.cursor()

        # Check if tables exist
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
        tables = [row[0] for row in cursor.fetchall()]
        expected_tables = [
            'customer_profile_org', 'customer_profile_ind',
            'social_media_sentiment', 'transaction_history'
        ]
        for table in expected_tables:
            self.assertIn(table, tables)

        # Check data in customer_profile_ind
        cursor.execute("SELECT COUNT(*) FROM customer_profile_ind")
        self.assertEqual(cursor.fetchone()[0], 15)  # 15 individual customers

        # Check data in customer_profile_org
        cursor.execute("SELECT COUNT(*) FROM customer_profile_org")
        self.assertEqual(cursor.fetchone()[0], 10)  # 10 organization customers

        # Check data in social_media_sentiment
        cursor.execute("SELECT COUNT(*) FROM social_media_sentiment")
        self.assertEqual(cursor.fetchone()[0], 26)  # 26 sentiment entries

        # Check data in transaction_history
        cursor.execute("SELECT COUNT(*) FROM transaction_history")
        self.assertEqual(cursor.fetchone()[0], 34)  # 34 transaction entries

        conn.close()

    def test_get_customer_details_individual(self):
        """Test retrieving details for an individual customer."""
        customer_id = "CUST2025A"
        customer = get_customer_details(customer_id)
        
        self.assertIsNotNone(customer)
        self.assertEqual(customer['customer_id'], customer_id)
        self.assertEqual(customer['age'], 25)
        self.assertEqual(customer['gender'], 'F')
        self.assertEqual(customer['location'], 'New York')
        self.assertEqual(customer['interests'], 'Luxury Shopping, Travel, Dining')
        self.assertEqual(customer['sentiment_score'], 0.7)
        self.assertEqual(customer['transaction_type'], 'Luxury Shopping')
        self.assertEqual(customer['amount_usd'], 3000)

    def test_get_customer_details_organization(self):
        """Test retrieving details for an organization customer."""
        customer_id = "ORG_US_004"
        customer = get_customer_details(customer_id)
        
        self.assertIsNotNone(customer)
        self.assertEqual(customer['customer_id'], customer_id)
        self.assertEqual(customer['industry'], 'Fashion and Clothing')
        self.assertEqual(customer['revenue_range'], '150M-20M')
        self.assertEqual(customer['employee_count_range'], '800-150')
        self.assertEqual(customer['transaction_type'], 'Retail Space Lease')
        self.assertEqual(customer['amount_usd'], 500000)
        self.assertIsNone(customer.get('sentiment_score'))  # No sentiment data for this org

    def test_get_customer_details_invalid_id(self):
        """Test retrieving details for a non-existent customer."""
        customer_id = "INVALID_ID"
        customer = get_customer_details(customer_id)
        self.assertIsNone(customer)

    def test_get_all_customer_ids(self):
        """Test retrieving all customer IDs."""
        customer_ids = get_all_customer_ids()
        self.assertEqual(len(customer_ids), 25)  # 15 individuals + 10 organizations
        self.assertIn("CUST2025A", customer_ids)
        self.assertIn("ORG_US_004", customer_ids)
        self.assertEqual(customer_ids, sorted(customer_ids))  # Ensure IDs are sorted

    def test_generate_similarity_query_individual(self):
        """Test generating a similarity query for an individual customer."""
        customer_data = {
            'customer_id': 'CUST2025A',
            'age': 25,
            'gender': 'F',
            'occupation': 'Marketing Manager',
            'location': 'New York',
            'income_per_year': 180000,
            'education': "Master's",
            'interests': 'Luxury Shopping, Travel, Dining',
            'preferences': 'Discounts, New Arrivals',
            'transaction_type': 'Luxury Shopping',
            'category': 'Gucci',
            'amount_usd': 3000,
            'purchase_date': '1/5/2025',
            'payment_mode': 'Credit Card',
            'platform': 'Instagram',
            'content': 'Excited to get promoted! Time to plan for wealth creation',
            'timestamp': '11/20/24 19:27',
            'sentiment_score': 0.7,
            'intent': 'Sales and Expansion'
        }
        query = generate_similarity_query(customer_data)
        expected_query = (
            "I’m a 25-year-old F Marketing Manager from New York. "
            "with an income of 180000 per year. "
            "and a Master's education. "
            "My interests are Luxury Shopping, Travel, Dining. "
            "and I prefer Discounts, New Arrivals. "
            "I recently made a Luxury Shopping transaction for Gucci. "
            "costing 3000 USD. "
            "on 1/5/2025. "
            "via Credit Card. "
            "on Instagram. "
            "The content related to this was 'Excited to get promoted! Time to plan for wealth creation' (timestamp: 11/20/24 19:27). "
            "I’m optimistic and seeking premium or growth-oriented solutions. "
            "and my intent is Sales and Expansion. "
            "What banking products match my profile, needs, and behavior?"
        )
        self.assertEqual(query, expected_query)

    def test_generate_similarity_query_organization(self):
        """Test generating a similarity query for an organization customer."""
        customer_data = {
            'customer_id': 'ORG_US_006',
            'industry': 'Luxury Fashion and Apparel',
            'revenue_range': '300M-150M',
            'employee_count_range': '100-250',
            'preferences': 'Limited Edition Collections, Global Marketing',
            'transaction_type': 'Fabric Procurement',
            'category': 'Italian Silk & Cashmere',
            'amount_usd': 1000000,
            'purchase_date': '2/23/2025',
            'payment_mode': 'Bank Wire',
            'platform': 'Twitter',
            'content': 'The rising costs of premium fabrics is impacting price strategy',
            'timestamp': '1/8/25 8:00',
            'sentiment_score': 0.7,
            'intent': 'Travel and work Setup'
        }
        query = generate_similarity_query(customer_data)
        expected_query = (
            "I represent an organization. "
            "in the Luxury Fashion and Apparel industry. "
            "with a revenue range of 300M-150M. "
            "and an employee count of 100-250. "
            "and I prefer Limited Edition Collections, Global Marketing. "
            "I recently made a Fabric Procurement transaction for Italian Silk & Cashmere. "
            "costing 1000000 USD. "
            "on 2/23/2025. "
            "via Bank Wire. "
            "on Twitter. "
            "The content related to this was 'The rising costs of premium fabrics is impacting price strategy' (timestamp: 1/8/25 8:00). "
            "I’m optimistic and seeking premium or growth-oriented solutions. "
            "and my intent is Travel and work Setup. "
            "What banking products match my profile, needs, and behavior?"
        )
        self.assertEqual(query, expected_query)

    def test_vector_search(self):
        """Test the vector search functionality."""
        customer_data = {
            'customer_id': 'CUST2025A',
            'age': 25,
            'gender': 'F',
            'occupation': 'Marketing Manager',
            'location': 'New York'
        }
        retrieved_products = vector_search(customer_data, k=10)
        
        # Check that 10 products are returned
        self.assertEqual(len(retrieved_products), 10)
        
        # Check that the products match the mocked indices (0 to 9)
        expected_product_ids = list(range(10))
        retrieved_product_ids = [prod['id'] - 1 for prod in retrieved_products]  # Adjust for 1-based indexing in PRODUCTS
        self.assertEqual(retrieved_product_ids, expected_product_ids)

    @patch('customer_analysis_with_streamlit_ui.ChatOpenAI')
    @patch('customer_analysis_with_streamlit_ui.ChatPromptTemplate')
    @patch('customer_analysis_with_streamlit_ui.StrOutputParser')
    def test_get_llm_recommendations(self, mock_parser, mock_prompt, mock_chat_openai):
        """Test the LLM recommendation generation."""
        # Mock the LLM chain
        mock_chain = MagicMock()
        mock_parser_instance = MagicMock()
        mock_parser.return_value = mock_parser_instance
        mock_prompt_instance = MagicMock()
        mock_prompt.from_template.return_value = mock_prompt_instance
        mock_prompt_instance.__or__.return_value = mock_chain
        mock_chain.__or__.return_value = mock_chain
        
        # Mock the LLM response
        mock_chain.invoke.return_value = """
Recommended Products:
- Savings Account
  Reason: Suitable for everyday transactions and accumulating funds, aligning with the customer's need for financial management.
- Travel Credit Cards
  Reason: Matches the customer's interest in travel and dining, offering travel benefits and rewards.
- Premium Credit Cards
  Reason: Offers enhanced benefits and higher spending limits, suitable for the customer's high income and luxury preferences.

Business Insights:
- Offer personalized travel and dining offers to engage the customer.
- Highlight premium benefits in marketing campaigns to appeal to their luxury interests.
"""
        
        customer_data = {
            'customer_id': 'CUST2025A',
            'age': 25,
            'gender': 'F',
            'occupation': 'Marketing Manager',
            'location': 'New York',
            'income_per_year': 180000,
            'interests': 'Luxury Shopping, Travel, Dining',
            'sentiment_score': 0.7
        }
        retrieved_products = PRODUCTS[:10]  # First 10 products as mocked by vector_search
        
        response = get_llm_recommendations(customer_data, retrieved_products)
        
        # Check the response format
        self.assertIn("Recommended Products:", response)
        self.assertIn("Savings Account", response)
        self.assertIn("Travel Credit Cards", response)
        self.assertIn("Premium Credit Cards", response)
        self.assertIn("Business Insights:", response)

    def test_get_llm_recommendations_no_api_key(self):
        """Test LLM recommendations when API key is missing."""
        with patch('customer_analysis_with_streamlit_ui.OPENAI_API_KEY', ''):
            customer_data = {'customer_id': 'CUST2025A'}
            retrieved_products = PRODUCTS[:10]
            response = get_llm_recommendations(customer_data, retrieved_products)
            self.assertEqual(response, "OpenAI API key not found in environment variables.")

    def test_format_customer_data(self):
        """Test formatting customer data for LLM prompt."""
        customer_data = {
            'customer_id': 'CUST2025A',
            'age': 25,
            'gender': 'F',
            'occupation': None,
            'location': 'New York'
        }
        formatted = format_customer_data(customer_data)
        expected = (
            "customer_id: CUST2025A\n"
            "age: 25\n"
            "gender: F\n"
            "location: New York"
        )
        self.assertEqual(formatted, expected)

    @patch('customer_analysis_with_streamlit_ui.st')
    def test_main_streamlit_ui(self, mock_st):
        """Test the main Streamlit UI function."""
        # Mock Streamlit methods
        mock_st.set_page_config = MagicMock()
        mock_st.title = MagicMock()
        mock_st.markdown = MagicMock()
        mock_st.selectbox = MagicMock(return_value="CUST2025A")
        mock_st.button = MagicMock(return_value=True)
        mock_st.spinner = MagicMock()
        mock_st.error = MagicMock()
        mock_st.subheader = MagicMock()
        mock_st.json = MagicMock()
        mock_st.write = MagicMock()
        mock_st.markdown = MagicMock()

        # Mock the context manager for spinner
        mock_st.spinner.return_value.__enter__ = MagicMock()
        mock_st.spinner.return_value.__exit__ = MagicMock()

        # Mock the get_customer_details and other functions
        with patch('customer_analysis_with_streamlit_ui.get_customer_details', return_value={
            'customer_id': 'CUST2025A',
            'age': 25,
            'gender': 'F',
            'location': 'New York'
        }), patch('customer_analysis_with_streamlit_ui.vector_search', return_value=PRODUCTS[:10]), \
        patch('customer_analysis_with_streamlit_ui.get_llm_recommendations', return_value="Recommended Products:\n- Savings Account\n  Reason: Test reason\nBusiness Insights:\n- Test insight"):
            from customer_analysis_with_streamlit_ui import main
            main()

        # Verify Streamlit calls
        mock_st.set_page_config.assert_called_once()
        mock_st.title.assert_called_with("🏦 Banking Product Recommender")
        mock_st.selectbox.assert_called_once()
        mock_st.button.assert_called_once_with("Get Recommendations")
        mock_st.subheader.assert_any_call("Customer Details")
        mock_st.json.assert_called_once()
        mock_st.subheader.assert_any_call("AI-Generated Recommendations")
        mock_st.write.assert_called_once()
        mock_st.subheader.assert_any_call("Recommended Banking Products")
        mock_st.markdown.assert_called()

    @classmethod
    def tearDownClass(cls):
        """Clean up after all tests."""
        if os.path.exists(DB_FILE):
            os.remove(DB_FILE)

if __name__ == '__main__':
    unittest.main()