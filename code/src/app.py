import os
import sqlite3
import pandas as pd
import numpy as np
import faiss
from typing import List, Dict, Optional
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
import streamlit as st

# Mock OpenAI API key (replace with actual key in a real environment)
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")

# Product list
PRODUCTS = [
    {"id": 1, "name": "Savings Account", "description": "For everyday transactions and accumulating funds"},
    {"id": 2, "name": "Current Account (Retail)", "description": "Designed for frequent transactions, primarily for business owners and entrepreneurs"},
    {"id": 3, "name": "Salary Account", "description": "Specifically for receiving salary payments"},
    {"id": 4, "name": "Fixed Deposit Account", "description": "Enables earning higher interest rates on a lump sum amount"},
    {"id": 5, "name": "Recurring Deposit Account", "description": "Allows depositing a fixed amount every month to build a lump sum"},
    {"id": 6, "name": "Demat Account", "description": "For holding and trading shares"},
    {"id": 7, "name": "NRI Accounts", "description": "For Non-Resident Indians"},
    {"id": 8, "name": "Current Account (Business)", "description": "Designed for frequent transactions, primarily for business owners and entrepreneurs"},
    {"id": 9, "name": "Overdraft Facility", "description": "Allows businesses to temporarily withdraw more money than they have in their account"},
    {"id": 10, "name": "Cash Credit", "description": "A type of loan that allows businesses to borrow money up to a certain limit"},
    {"id": 11, "name": "Regular Credit Cards", "description": "For general use with rewards and cashback programs"},
    {"id": 12, "name": "Premium Credit Cards", "description": "Offer enhanced benefits and higher spending limits"},
    {"id": 13, "name": "Super Premium Credit Cards", "description": "Provide the highest level of benefits and services"},
    {"id": 14, "name": "Co-branded Credit Cards", "description": "Partnered with specific brands or loyalty programs"},
    {"id": 15, "name": "Commercial/Business Credit Cards", "description": "Designed for business expenses and transactions"},
    {"id": 16, "name": "CashBack Credit Cards", "description": "Offer cashback rewards on purchases"},
    {"id": 17, "name": "Secured Credit Cards", "description": "Require a deposit to secure the credit line"},
    {"id": 18, "name": "Travel Credit Cards", "description": "Provide travel benefits and rewards"},
    {"id": 19, "name": "ATM Cards", "description": "Allow cash withdrawals from ATMs"},
    {"id": 20, "name": "Debit Cards", "description": "For making purchases and withdrawing cash"},
    {"id": 21, "name": "Prepaid Cards", "description": "Loaded with a specific amount of money and can be used for various transactions"},
    {"id": 22, "name": "Virtual Cards", "description": "Online-only cards for secure online transactions"},
    {"id": 23, "name": "Commercial Debit Cards", "description": "For business transactions"},
    {"id": 24, "name": "Transit Cards", "description": "For public transportation"},
    {"id": 25, "name": "Personal Loans", "description": "For personal needs"},
    {"id": 26, "name": "Home Loans", "description": "For purchasing a house"},
    {"id": 27, "name": "Auto Loans", "description": "For purchasing a car"},
    {"id": 28, "name": "Business Loans", "description": "For funding business operations"},
    {"id": 29, "name": "Mutual Funds", "description": "A way to invest in a diversified portfolio"},
    {"id": 30, "name": "Stocks", "description": "Ownership in a company"},
    {"id": 31, "name": "Bank Fixed Deposits (FDs)", "description": "Safe, low-risk savings"},
    {"id": 32, "name": "Bonds", "description": "A form of debt investment"},
    {"id": 33, "name": "Life Insurance", "description": "Provides financial protection to beneficiaries in case of death"},
    {"id": 34, "name": "Health Insurance", "description": "Helps cover medical expenses"},
    {"id": 35, "name": "Property Insurance", "description": "Protects against damage to property"},
    {"id": 36, "name": "UPI (Unified Payments Interface)", "description": "For instant mobile money transfers"},
    {"id": 37, "name": "QR Codes", "description": "For making payments using mobile apps"},
    {"id": 38, "name": "Online Banking", "description": "Accessing bank accounts and performing transactions online"},
    {"id": 39, "name": "ACE Credit Card", "description": "Cashback card with unlimited cashback, dining offers, lounge access"},
    {"id": 40, "name": "Flipkart Shop Credit Card", "description": "Co-branded card with unlimited cashback and shopping discounts"},
    {"id": 41, "name": "Amazon Pay Credit Card", "description": "Co-branded card with 1% cashback on all Amazon purchases"},
    {"id": 42, "name": "Rewards Credit Card", "description": "Offers accelerated points on dining and shopping"},
    {"id": 43, "name": "Privilege Credit Card", "description": "Secured card with EDGE reward points and milestone bonuses"},
    {"id": 44, "name": "SELECT Credit Card", "description": "Premium card with Amazon voucher, grocery discounts, lounge visits"},
    {"id": 45, "name": "Atlas Credit Card", "description": "Super premium travel card with milestone rewards and lounge access"},
    {"id": 46, "name": "Vistara Infinite Credit Card", "description": "Travel card with business class ticket and Vistara Gold membership"},
    {"id": 47, "name": "Magnus Credit Card", "description": "Luxury travel card with concierge services and lounge access"},
    {"id": 48, "name": "Reserve Credit Card", "description": "Commercial card with concierge services and premium hotel stays"},
    {"id": 49, "name": "Platinum Luxury Card", "description": "Luxury card with exclusive travel benefits and hotel discounts"},
    {"id": 50, "name": "SuperSaver Cashback Card", "description": "Cashback card with 5% cashback on all online spends"},
    {"id": 51, "name": "Basic ATM Card", "description": "For cash withdrawals and balance inquiries"},
    {"id": 52, "name": "Delights Debit Card", "description": "Offers 5% cashback on fuel, dining, OTT"},
    {"id": 53, "name": "Priority Platinum Debit Card", "description": "Cashback on movies, ₹1 lakh purchase limit, fuel surcharge waiver"},
    {"id": 54, "name": "Prepaid Travel Card", "description": "Loadable with foreign currency for international use"},
    {"id": 55, "name": "Virtual Debit Card", "description": "Generated via internet banking for online purchases"},
    {"id": 56, "name": "Visa Signature Debit Card", "description": "Commercial card with rewards points and higher limits"},
    {"id": 57, "name": "Metro Travel Card", "description": "Seamless travel payments for public transportation"},
    {"id": 58, "name": "Education Loan", "description": "For funding education expenses with scholarship-linked discounts"},
    {"id": 59, "name": "Gold Loan", "description": "Secured loan against gold with quick approval"},
    {"id": 60, "name": "Term Insurance", "description": "Pure life cover with lower premiums for non-smokers"},
    {"id": 61, "name": "Motor Insurance", "description": "Covers vehicle damage & third-party liability with anti-theft device discounts"},
    {"id": 62, "name": "Home Insurance", "description": "Covers home structure & belongings with bundle discounts"},
    {"id": 63, "name": "Child Plan", "description": "Savings & insurance for child’s future with extra bonus"},
    {"id": 64, "name": "Retirement Plan", "description": "Regular pension after retirement with early investment benefits"},
    {"id": 65, "name": "ULIP (Unit Linked Insurance Plan)", "description": "Insurance + investment growth with loyalty bonus"},
    {"id": 66, "name": "Critical Illness Cover", "description": "Lump sum on diagnosis of major illnesses with premium discounts"},
    {"id": 67, "name": "Travel Insurance", "description": "Covers trip cancellation and medical emergencies with family discounts"},
    {"id": 68, "name": "Annuity Plans", "description": "Regular income post-retirement with bonus annuity options"},
    {"id": 69, "name": "Guaranteed Return Plans", "description": "Assured return on invested amount with guaranteed maturity benefits"},
    {"id": 70, "name": "Capital Guarantee Plans", "description": "Wealth creation + protection with principal protection"},
    {"id": 71, "name": "Pension Plans", "description": "Low risk, long-term savings with higher pension for early investment"},
    {"id": 72, "name": "Senior Citizen Savings Scheme (SCSS)", "description": "Retirement savings with senior citizen benefits"},
    {"id": 73, "name": "Post Office Monthly Income Scheme (POMIS)", "description": "Safe investment with fixed monthly payouts"},
    {"id": 74, "name": "Public Provident Fund (PPF)", "description": "Long-term tax-saving investment with tax-free returns"},
    {"id": 75, "name": "RBI Floating Rate Savings Bonds (FRSB)", "description": "Government-backed floating rate returns with guaranteed interest"},
    {"id": 76, "name": "National Savings Certificate (NSC)", "description": "Secure savings with fixed returns and compounded interest"},
    {"id": 77, "name": "Treasury Bills", "description": "Safe short-term investment with auction-based returns"},
    {"id": 78, "name": "Municipal Bonds", "description": "Invest in municipal projects with municipal tax benefits"},
    {"id": 79, "name": "National Pension Scheme (NPS)", "description": "Pension fund with market-linked returns and extra tax deductions"},
    {"id": 80, "name": "Corporate Bonds", "description": "Corporate debt for stable returns with tax-free options"},
    {"id": 81, "name": "Index Funds", "description": "Passive investment in stock market with low-cost index tracking"},
    {"id": 82, "name": "Debt Mutual Funds", "description": "Stable debt-based returns with no lock-in period"},
    {"id": 83, "name": "Balanced Mutual Funds", "description": "Balanced risk & return with diversified risk management"},
    {"id": 84, "name": "Initial Public Offerings (IPO)", "description": "Long-term high-risk investment with early bird perks"},
    {"id": 85, "name": "Stock Market Trading", "description": "Balancing risk & return with high volatility options"},
    {"id": 86, "name": "Equity Mutual Funds", "description": "Equity-based long-term investment with tax benefits under ELSS"},
    {"id": 87, "name": "Exchange Traded Funds (ETFs)", "description": "Trading in stock market indices with no entry/exit loads"},
    {"id": 88, "name": "Money Market Funds", "description": "Short-term stable income with safe cash-equivalent investment"},
    {"id": 89, "name": "Hedge Funds", "description": "High-risk high-return fund with exclusive access"},
    {"id": 90, "name": "Angel Investment", "description": "Startup investment with potential unicorn returns"},
    {"id": 91, "name": "Real Estate", "description": "Property investment for appreciation with rental yield"},
    {"id": 92, "name": "Forex Trading", "description": "Currency market trading with high leverage options"},
    {"id": 93, "name": "Gold", "description": "Store of value, inflation hedge with no additional charges"},
    {"id": 94, "name": "Cryptocurrencies", "description": "High volatility trading with 30% taxable profits"},
    {"id": 95, "name": "Health Insurance (General)", "description": "Covers medical expenses, hospitalization, critical illnesses, maternity, accidents"},
    {"id": 96, "name": "Motor Insurance (General)", "description": "Provides financial protection for vehicles against accidents, theft, damages"},
    {"id": 97, "name": "Home Insurance (General)", "description": "Protects home structure and contents against calamities, theft, and liabilities"},
    {"id": 98, "name": "Fire Insurance", "description": "Covers fire damages and associated risks like riots, wars, and natural disasters"},
    {"id": 99, "name": "Travel Insurance (General)", "description": "Covers financial loss due to trip cancellations, baggage loss, medical emergencies"},
    {"id": 100, "name": "Term Life Insurance", "description": "Provides financial protection to the family in case of policyholder’s death"},
    {"id": 101, "name": "Whole Life Insurance", "description": "Offers lifelong coverage with a savings component"},
    {"id": 102, "name": "Endowment Plans", "description": "Combination of insurance and savings for financial security"},
    {"id": 103, "name": "Unit-Linked Insurance Plans (ULIPs)", "description": "Part investment in market-linked funds and part insurance"},
    {"id": 104, "name": "Child Plans (Life)", "description": "Savings + insurance for securing child’s financial future"},
    {"id": 105, "name": "Pension Plans (Life)", "description": "Helps build financial security post-retirement through annuities"}
]

# Initialize product embeddings
def initialize_product_embeddings() -> tuple[faiss.IndexFlatL2, OpenAIEmbeddings]:
    if not OPENAI_API_KEY:
        raise ValueError("OpenAI API key not found in environment variables.")
    embedding_model = OpenAIEmbeddings(model="text-embedding-3-large", openai_api_key=OPENAI_API_KEY)
    product_texts = [product["description"] for product in PRODUCTS]
    product_embeddings = embedding_model.embed_documents(product_texts)
    product_embeddings_np = np.array(product_embeddings, dtype='float32')
    dimension = product_embeddings_np.shape[1]
    index = faiss.IndexFlatL2(dimension)
    index.add(product_embeddings_np)
    return index, embedding_model

INDEX, EMBEDDING_MODEL = initialize_product_embeddings()

# Initialize SQLite database
DB_FILE = "customer_data_expanded.db"

def init_db():
    conn = sqlite3.connect(DB_FILE)
    cursor = conn.cursor()

    # Create Customer Profile (Organization) table
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS customer_profile_org (
            customer_id TEXT PRIMARY KEY,
            industry TEXT,
            financial_needs TEXT,
            preferences TEXT,
            revenue_in_dollars TEXT,
            no_of_employees TEXT
        )
    ''')

    # Create Customer Profile (Individual) table
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS customer_profile_ind (
            customer_id TEXT PRIMARY KEY,
            age INTEGER,
            gender TEXT,
            location TEXT,
            interests TEXT,
            preferences TEXT,
            income_per_year INTEGER,
            education TEXT,
            occupation TEXT
        )
    ''')

    # Create Social Media Sentiment table
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS social_media_sentiment (
            customer_id TEXT,
            post_id TEXT,
            platform TEXT,
            content TEXT,
            timestamp TEXT,
            sentiment_score REAL,
            intent TEXT,
            PRIMARY KEY (customer_id, post_id)
        )
    ''')

    # Create Transaction History table
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS transaction_history (
            customer_id TEXT,
            product_id INTEGER,
            transaction_type TEXT,
            category TEXT,
            amount_in_dollars INTEGER,
            purchase_date TEXT,
            payment_mode TEXT,
            PRIMARY KEY (customer_id, product_id)
        )
    ''')

    # Populate Customer Profile (Organization) table with expanded data
    org_data = [
        ('ORG_US_004', 'Fashion and Clothing', 'Supply Chain Financing, Inventory Loans, Retail Banking, Treasury Services, Cloud Platform', 'Direct-To-Customer eCommerce Platform', '150M-20M', '800-150'),
        ('ORG_US_002', 'Sports Equipment and Apparel', 'Business Loans, Sponsorship Financing, Digital Pay Athlete Services', 'Employee Benefits, International Expansion', '50M-80M', '500-1000'),
        ('ORG_US_005', 'Agriculture and Organic Food Production', 'Business Loans, Sponsorship Financing', 'Employee Benefits, International Expansion', '80M-120M', '500-1000'),
        ('ORG_US_007', 'Textile and Sustainable Fabrics', 'Green Loans, Supply Chain Financing, POS Bank Equipment', 'Distribution Channels, R&D on Sustainable Farming', '20M-30M', '200-500'),
        ('ORG_US_006', 'Luxury Fashion and Apparel', 'High-Net Worth Banking, Investment Management, Market Research, Alternate Investments, Private Equity', 'Limited Edition Collections, Global Marketing', '300M-150M', '100-250'),
        ('ORG_US_008', 'Healthcare and Pharmaceuticals', 'Corporate Loans, R&D Funding, Treasury Services', 'Digital Health Solutions, Global Expansion', '200M-250M', '1000-1500'),
        ('ORG_US_009', 'Automotive Manufacturing', 'Supply Chain Financing, Green Loans, Equipment Leasing', 'Sustainable Manufacturing, Electric Vehicle R&D', '500M-600M', '2000-3000'),
        ('ORG_US_010', 'Tech Startups', 'Venture Capital Funding, Business Loans, Cloud Credits', 'AI Integration, Scalable Infrastructure', '10M-50M', '50-100'),
        ('ORG_US_011', 'Renewable Energy', 'Green Bonds, Project Financing, Treasury Services', 'Solar and Wind Projects, Carbon Neutrality', '100M-150M', '300-500'),
        ('ORG_US_012', 'Hospitality and Tourism', 'Business Loans, Revenue Management Tools, Digital Marketing', 'Luxury Experiences, Global Outreach', '80M-100M', '600-800')
    ]
    cursor.executemany('INSERT OR REPLACE INTO customer_profile_org VALUES (?, ?, ?, ?, ?, ?)', org_data)

    # Populate Customer Profile (Individual) table with expanded data
    ind_data = [
        ('CUST2025A', 25, 'F', 'New York', 'Luxury Shopping, Travel, Dining', 'Discounts, New Arrivals', 180000, "Master's", 'Marketing Manager'),
        ('CUST2025B', 22, 'F', 'Los Angeles', 'Flights, Hotels, Adventure Activities, Cameras', 'Home Loan, Retirement Savings, ETFs, Travel Credit Cards', 90000, 'Graduate', 'Software Engineer'),
        ('CUST2025C', 27, 'M', 'Austin', 'Family Vacations, Kids, Education, Home Essentials', 'Family Insurance, Digital Banking, International Travel', 100000, 'MBA', 'Travel Blogger'),
        ('CUST2025D', 45, 'M', 'New York', 'Tech Gadgets, Professional Development', 'Health Insurance, Travel Credit Cards, Gym Subscription', 70000, 'Under-Graduate', 'HR Manager'),
        ('CUST2025E', 66, 'M', 'Chicago', 'Healthcare, Fixed Deposits, Insurance', 'Certificates of Deposits, Medicare Plans, Pension', 120000, 'MBA', 'Financial Advisor'),
        ('CUST2025F', 36, 'F', 'Boston', 'Finance Investments, Deposits, Insurance', 'Wealth Management, Home Loans, Small Business Financing', 55000, 'Graduate', 'Retired with Pension + 401(k)'),
        ('CUST2025G', 42, 'M', 'Denver', 'Gaming, Tech Gadgets, Streaming Subscriptions', 'BNPL, Crypto, Digital Banks, Tax Savings', 110000, "Master's", 'Bank Manager'),
        ('CUST2025H', 26, 'F', 'Portland', 'Fine Dining, Luxury Travel, High-End Gadgets', 'Private Banking, Subscription Services', 60000, 'Graduate', 'Insurance Agent'),
        ('CUST2025I', 42, 'M', 'Chicago', 'Online Shopping, Food Delivery', 'Crypto, Digital Banks, BNPL', 225000, 'MBA', 'Software Engineer and Twitch Streamer'),
        ('CUST2025J', 35, 'F', 'Los Angeles', 'Fashion, Wellness', 'Wealth Management, Tax Advisory', 67000, 'MBA', 'Wealth Manager'),
        ('CUST2025K', 29, 'M', 'Miami', 'Sports, Fitness, Tech Gadgets', 'Fitness Subscriptions, Tech Financing', 95000, 'Graduate', 'Fitness Trainer'),
        ('CUST2025L', 31, 'F', 'Seattle', 'Books, Education, Travel', 'Education Loans, Travel Rewards', 85000, "Master's", 'Teacher'),
        ('CUST2025M', 38, 'M', 'San Francisco', 'Startups, Investments, Tech', 'Venture Capital, Crypto Investments', 200000, 'MBA', 'Entrepreneur'),
        ('CUST2025N', 50, 'F', 'Boston', 'Art, Culture, Travel', 'Art Investments, Luxury Travel Cards', 150000, 'Graduate', 'Art Curator'),
        ('CUST2025O', 33, 'M', 'Austin', 'Gaming, Streaming, Tech', 'Gaming Subscriptions, BNPL', 78000, 'Graduate', 'Content Creator')
    ]
    cursor.executemany('INSERT OR REPLACE INTO customer_profile_ind VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)', ind_data)

    # Populate Social Media Sentiment table with expanded data
    sentiment_data = [
        ('ORG_US_007', '8810', 'Twitter', 'Navigating fluctuations raw material prices!! Cash Flow planning is KEY!', '1/16/25 19:27', -0.4, 'Financial Management Concern'),
        ('CUST2025I', '4656', 'Facebook', 'Just finished a 5K run! Need new running shoes. Any suggestions?', '1/3/25 12:45', 0.8, 'Fashion Interest'),
        ('ORG_US_006', '1155', 'Facebook', 'Exciting collaborations coming soon!! Guess which celeb is joining our campaign?', '3/7/25 18:22', 0.7, 'Audience Engagement'),
        ('ORG_US_007', '4477', 'Reddit', 'Whats the best way to integrate blockchain for textile supply chain transparency?', '10/13/24 29:34', 0.6, 'Tech Innovation Interest'),
        ('CUST2025SP', '3454', 'Instagram', 'Struggling to stick to my budget this month. Trying out the new Tesla self-driving update. Feels like future!!', '1/18/25 12:15', -0.5, 'Budget Concern'),
        ('CUST2025H', '1244', 'Twitter', 'Why do banks charge so many hidden fees? I need a no-fee checking account', '11/27/24 11:10', -0.6, 'Bank Fee Complaint'),
        ('ORG_US_005', '3021', 'Twitter', 'Sponsoring athletes is costly. Any creative financing options to support brand amt?', '2/10/25 14:30', 0.5, 'Sponsorship Concern'),
        ('CUST2025J', '4344', 'Reddit', 'Why is my gym membership so expensive? Thinking of switching.', '2/20/25 17:27', -0.5, 'Subscription Change'),
        ('ORG_US_002', '9999', 'Twitter', 'Looking for efficient supply chain financing to handle seasonal demand spikes', '1/7/25 16:30', 0.8, 'BNPL Usage'),
        ('CUST2025C', '5433', 'Twitter', 'Capturing beautiful moments with newly bought iPhone! Love BNPL!', '1/7/25 16:30', 0.1, 'Wealth Management'),
        ('CUST2025A', '3267', 'Instagram', 'Excited to get promoted! Time to plan for wealth creation', '11/20/24 19:27', 0.7, 'Sales and Expansion'),
        ('ORG_US_002', '7986', 'Facebook', 'Our latest organic products are now available in Whole Foods!', '2/17/25 7:31', -0.2, 'Generational Gap Complaint'),
        ('CUST2025D', '8576', 'Instagram', 'Old movies were a delight to eyes. These days, its just action!!', '1/17/25 16:00', -0.5, 'Cost & Financing Concern'),
        ('ORG_US_006', '7869', 'Twitter', 'The rising costs of premium fabrics is impacting price strategy', '1/8/25 8:00', 0.7, 'Travel and work Setup'),
        ('CUST2025E', '3428', 'Facebook', 'Trying out new beauty products!', '2/20/25 15:27', 0.6, 'Risk Mitigation'),
        ('ORG_US_003', '6549', 'Twitter', 'Marketing volatility ahead', '1/17/25 15:45', 0.7, 'Fashion Focus'),
        ('CUST2025K', '1234', 'Instagram', 'Just got a new fitness tracker! Loving the stats!', '3/1/25 10:00', 0.9, 'Fitness Interest'),
        ('CUST2025L', '5678', 'Twitter', 'Planning a trip to Europe. Any travel tips?', '2/15/25 14:20', 0.8, 'Travel Interest'),
        ('CUST2025M', '9012', 'LinkedIn', 'Investing in a new startup. Exciting times ahead!', '1/10/25 09:30', 0.7, 'Investment Interest'),
        ('CUST2025N', '3456', 'Instagram', 'Visited an art gallery today. So inspiring!', '3/5/25 16:45', 0.9, 'Art Interest'),
        ('CUST2025O', '7890', 'Twitter', 'Streaming my new gaming setup tonight! Join me!', '2/25/25 20:00', 0.8, 'Gaming Interest'),
        ('ORG_US_008', '2345', 'LinkedIn', 'Launching a new telemedicine platform. Stay tuned!', '3/10/25 11:00', 0.9, 'Digital Health Interest'),
        ('ORG_US_009', '6789', 'Twitter', 'Electric vehicle production is ramping up. Exciting times!', '2/20/25 13:15', 0.8, 'Sustainability Interest'),
        ('ORG_US_010', '1236', 'Reddit', 'Looking for AI solutions to scale our startup. Suggestions?', '1/25/25 17:30', 0.6, 'Tech Innovation Interest'),
        ('ORG_US_011', '4567', 'LinkedIn', 'New solar project underway. Aiming for carbon neutrality!', '3/15/25 10:45', 0.9, 'Sustainability Interest'),
        ('ORG_US_012', '8901', 'Instagram', 'Our new luxury resort is now open! Book your stay!', '2/28/25 12:00', 0.9, 'Luxury Travel Interest')
    ]
    cursor.executemany('INSERT OR REPLACE INTO social_media_sentiment VALUES (?, ?, ?, ?, ?, ?, ?)', sentiment_data)

    # Populate Transaction History table with expanded data
    transaction_data = [
        ('CUST2025A', 201, 'Luxury Shopping', 'Gucci', 3000, '1/5/2025', 'Credit Card'),
        ('ORG_US_004', 202, 'Retail Space Lease', 'New Flagship store', 500000, '1/5/2025', 'Wire Transfer'),
        ('CUST2025H', 203, 'Luxury Travel Booking', 'Luxury Business Trip', 4500, '2/9/2025', 'Wire Transfer'),
        ('ORG_US_007', 204, 'Research & Development', 'Sustainable Fabric Innovations', 2500000, '2/9/2025', 'Wire Transfer'),
        ('CUST2025A', 205, 'Stock Investment', 'Equity', 25000, '1/2/2025', 'Auto Debit'),
        ('CUST2025A', 206, 'Travel Booking', 'International Flight', 5000, '2/17/2025', 'Credit Card'),
        ('ORG_US_002', 207, 'Marketing Promotions', 'Branding and Social Media Ads', 1500000, '3/30/2025', 'ACH Debit'),
        ('ORG_US_002', 208, 'Investment', 'Hedge Fund', 5000, '12/10/2024', 'Wire Transfer'),
        ('ORG_US_006', 209, 'Fabric Procurement', 'Italian Silk & Cashmere', 1000000, '2/23/2025', 'Bank Wire'),
        ('CUST2025E', 210, 'Flight Booking', 'New York to Tokyo', 1200, '3/1/2025', 'Chase Sapphire Travel Card'),
        ('ORG_US_006', 211, 'Expansion Loan', 'New Flagship stores', 5000000, '1/15/2025', 'Business Loan'),
        ('CUST2025A', 212, 'Loan EMI', 'Car Loan', 1000, '1/5/2025', 'Auto Debit'),
        ('CUST2025B', 213, 'IRA Contribution', 'Vanguard', 500, '1/10/2025', 'Net Banking'),
        ('ORG_US_006', 214, 'E-Commerce Tech Investment', 'AI-Powered Personalization', 7000000, '3/5/2025', 'Bank Transfer'),
        ('CUST2025D', 215, 'Grocery Shopping', 'Costco', 100, '1/20/2025', 'Credit Card'),
        ('CUST2025G', 216, 'Cloud Services', 'AWS and Microsoft Azure', 1000, '2/20/2025', 'Corporate Card'),
        ('CUST2025I', 217, 'Mortgage Payment', 'Home Loan Repayment', 3500, '3/5/2025', 'Auto Debit'),
        ('CUST2025SP', 218, 'BNPL Purchase', 'PlayStation', 800, '1/25/2025', 'Affirm'),
        ('ORG_US_005', 219, 'Technology Investment', 'AI-Powered E-commerce Platform', 3500000, '9/13/2024', 'Business Loan'),
        ('CUST2025K', 220, 'Fitness Subscription', 'Peloton Membership', 600, '3/1/2025', 'Credit Card'),
        ('CUST2025L', 221, 'Education Loan Payment', 'Student Loan', 1200, '2/15/2025', 'Auto Debit'),
        ('CUST2025M', 222, 'Crypto Investment', 'Bitcoin', 10000, '1/10/2025', 'Bank Transfer'),
        ('CUST2025N', 223, 'Art Purchase', 'Modern Art Piece', 5000, '3/5/2025', 'Credit Card'),
        ('CUST2025O', 224, 'Gaming Subscription', 'Xbox Game Pass', 150, '2/25/2025', 'BNPL'),
        ('ORG_US_008', 225, 'R&D Investment', 'Telemedicine Platform', 4000000, '3/10/2025', 'Wire Transfer'),
        ('ORG_US_009', 226, 'Equipment Leasing', 'EV Manufacturing Equipment', 6000000, '2/20/2025', 'Business Loan'),
        ('ORG_US_010', 227, 'Cloud Credits Purchase', 'AWS Credits', 200000, '1/25/2025', 'Bank Transfer'),
        ('ORG_US_011', 228, 'Project Financing', 'Solar Farm', 8000000, '3/15/2025', 'Green Bonds'),
        ('ORG_US_012', 229, 'Marketing Campaign', 'Luxury Resort Promotion', 1000000, '2/28/2025', 'ACH Debit'),
        ('CUST2025F', 230, 'Fixed Deposit', 'Bank FD', 20000, '1/15/2025', 'Bank Transfer'),
        ('CUST2025J', 231, 'Wellness Retreat', 'Yoga Retreat', 800, '3/20/2025', 'Credit Card'),
        ('CUST2025C', 232, 'Family Vacation', 'Disney World Package', 4000, '2/10/2025', 'Credit Card'),
        ('ORG_US_004', 233, 'Inventory Loan', 'Seasonal Stock', 2000000, '1/20/2025', 'Business Loan'),
        ('CUST2025B', 234, 'Travel Booking', 'Adventure Trip', 3000, '3/25/2025', 'Travel Credit Card')
    ]
    cursor.executemany('INSERT OR REPLACE INTO transaction_history VALUES (?, ?, ?, ?, ?, ?, ?)', transaction_data)

    conn.commit()
    conn.close()

# Initialize the database and populate it
init_db()

# Function to get all customer IDs for the dropdown
def get_all_customer_ids() -> List[str]:
    conn = sqlite3.connect(DB_FILE)
    cursor = conn.cursor()
    
    # Get individual customer IDs
    cursor.execute("SELECT customer_id FROM customer_profile_ind")
    ind_ids = [row[0] for row in cursor.fetchall()]
    
    # Get organization customer IDs
    cursor.execute("SELECT customer_id FROM customer_profile_org")
    org_ids = [row[0] for row in cursor.fetchall()]
    
    conn.close()
    return sorted(ind_ids + org_ids)

# Function to get customer details (combining individual and organization data with sentiment and transaction history)
def get_customer_details(customer_id: str) -> Optional[Dict]:
    conn = sqlite3.connect(DB_FILE)
    conn.row_factory = sqlite3.Row
    cursor = conn.cursor()

    # Check if customer is an individual
    cursor.execute("SELECT * FROM customer_profile_ind WHERE customer_id = ?", (customer_id,))
    ind_row = cursor.fetchone()
    if ind_row:
        customer_data = dict(ind_row)
        # Add sentiment data
        cursor.execute("SELECT * FROM social_media_sentiment WHERE customer_id = ?", (customer_id,))
        sentiment_row = cursor.fetchone()
        if sentiment_row:
            customer_data.update({
                'platform': sentiment_row['platform'],
                'content': sentiment_row['content'],
                'timestamp': sentiment_row['timestamp'],
                'sentiment_score': sentiment_row['sentiment_score'],
                'intent': sentiment_row['intent']
            })
        # Add transaction data
        cursor.execute("SELECT * FROM transaction_history WHERE customer_id = ?", (customer_id,))
        transaction_row = cursor.fetchone()
        if transaction_row:
            customer_data.update({
                'transaction_type': transaction_row['transaction_type'],
                'category': transaction_row['category'],
                'amount_usd': transaction_row['amount_in_dollars'],
                'purchase_date': transaction_row['purchase_date'],
                'payment_mode': transaction_row['payment_mode']
            })
        conn.close()
        return customer_data

    # Check if customer is an organization
    cursor.execute("SELECT * FROM customer_profile_org WHERE customer_id = ?", (customer_id,))
    org_row = cursor.fetchone()
    if org_row:
        customer_data = dict(org_row)
        customer_data['revenue_range'] = customer_data.pop('revenue_in_dollars')
        customer_data['employee_count_range'] = customer_data.pop('no_of_employees')
        # Add sentiment data
        cursor.execute("SELECT * FROM social_media_sentiment WHERE customer_id = ?", (customer_id,))
        sentiment_row = cursor.fetchone()
        if sentiment_row:
            customer_data.update({
                'platform': sentiment_row['platform'],
                'content': sentiment_row['content'],
                'timestamp': sentiment_row['timestamp'],
                'sentiment_score': sentiment_row['sentiment_score'],
                'intent': sentiment_row['intent']
            })
        # Add transaction data
        cursor.execute("SELECT * FROM transaction_history WHERE customer_id = ?", (customer_id,))
        transaction_row = cursor.fetchone()
        if transaction_row:
            customer_data.update({
                'transaction_type': transaction_row['transaction_type'],
                'category': transaction_row['category'],
                'amount_usd': transaction_row['amount_in_dollars'],
                'purchase_date': transaction_row['purchase_date'],
                'payment_mode': transaction_row['payment_mode']
            })
        conn.close()
        return customer_data

    conn.close()
    return None

# Function to generate a similarity query for vector search
def generate_similarity_query(customer_data: Dict) -> str:
    query_parts = []
    is_organization = not (customer_data.get('age') or customer_data.get('gender') or customer_data.get('occupation'))
    if is_organization:
        query_parts.append("I represent an organization")
        if customer_data.get('industry'):
            query_parts.append(f"in the {customer_data['industry']} industry")
        if customer_data.get('revenue_range'):
            query_parts.append(f"with a revenue range of {customer_data['revenue_range']}")
        if customer_data.get('employee_count_range'):
            query_parts.append(f"and an employee count of {customer_data['employee_count_range']}")
    else:
        if customer_data.get('age') and customer_data.get('gender') and customer_data.get('occupation') and customer_data.get('location'):
            query_parts.append(f"I’m a {customer_data['age']}-year-old {customer_data['gender']} {customer_data['occupation']} from {customer_data['location']}")
        if customer_data.get('income_per_year'):
            query_parts.append(f"with an income of {customer_data['income_per_year']} per year")
        if customer_data.get('education'):
            query_parts.append(f"and a {customer_data['education']} education")
    if customer_data.get('interests'):
        query_parts.append(f"My interests are {customer_data['interests']}")
    if customer_data.get('preferences'):
        query_parts.append(f"and I prefer {customer_data['preferences']}")
    if customer_data.get('transaction_type') and customer_data.get('category'):
        query_parts.append(f"I recently made a {customer_data['transaction_type']} transaction for {customer_data['category']}")
        if customer_data.get('amount_usd'):
            query_parts.append(f"costing {customer_data['amount_usd']} USD")
        if customer_data.get('purchase_date'):
            query_parts.append(f"on {customer_data['purchase_date']}")
        if customer_data.get('payment_mode'):
            query_parts.append(f"via {customer_data['payment_mode']}")
    if customer_data.get('platform'):
        query_parts.append(f"on {customer_data['platform']}")
    if customer_data.get('content'):
        content_str = f"The content related to this was '{customer_data['content']}'" + (f" (timestamp: {customer_data['timestamp']})" if customer_data.get('timestamp') else "")
        query_parts.append(content_str)
    sentiment_score = customer_data.get('sentiment_score')
    if sentiment_score is not None:
        if sentiment_score <= -0.3:
            query_parts.append("I’m feeling concerned about costs and need affordable or supportive solutions")
        elif -0.3 < sentiment_score <= 0.3:
            query_parts.append("I’m looking for practical and straightforward options")
        elif sentiment_score > 0.3:
            query_parts.append("I’m optimistic and seeking premium or growth-oriented solutions")
    if customer_data.get('intent'):
        query_parts.append(f"and my intent is {customer_data['intent']}")
    query = ". ".join(query_parts) + ". What banking products match my profile, needs, and behavior?"
    return query

# Function to perform vector search for product recommendations
def vector_search(customer_data: Dict, k: int = 10) -> List[Dict]:
    if not customer_data:
        raise ValueError("Customer data is required for vector search.")
    query_text = generate_similarity_query(customer_data)
    query_embedding = EMBEDDING_MODEL.embed_query(query_text)
    query_embedding_np = np.array([query_embedding], dtype='float32')
    distances, indices = INDEX.search(query_embedding_np, k)
    return [PRODUCTS[i] for i in indices[0]]

# Function to format customer data for LLM prompt
def format_customer_data(customer_data: Dict) -> str:
    formatted_lines = []
    for key, value in customer_data.items():
        if value is not None:
            formatted_lines.append(f"{key}: {value}")
    return "\n".join(formatted_lines)

# Function to get LLM-based recommendations
def get_llm_recommendations(customer_data: Dict, retrieved_products: List[Dict]) -> str:
    if not OPENAI_API_KEY:
        return "OpenAI API key not found in environment variables."
    llm = ChatOpenAI(model="gpt-4o", openai_api_key=OPENAI_API_KEY)
    product_list = "\n".join([f"- {p['name']}: {p['description']}" for p in retrieved_products])
    sentiment_score = customer_data.get('sentiment_score', 0.0)
    if sentiment_score <= -0.3:
        sentiment_context = f"The customer has a negative sentiment (score: {sentiment_score}), indicating potential dissatisfaction or concerns."
    elif sentiment_score >= 0.3:
        sentiment_context = f"The customer has a positive sentiment (score: {sentiment_score}), indicating satisfaction or optimism."
    else:
        sentiment_context = f"The customer has a neutral sentiment (score: {sentiment_score})."
    customer_data_str = format_customer_data(customer_data)
    prompt = ChatPromptTemplate.from_template(
        """
        You are an expert banking product recommender. Your task is to analyze the following customer details and recommend 3-5 banking products from the provided list that best match the customer's profile, needs, and preferences. Additionally, provide insights for the business on how to optimize customer engagement based on this customer's data.

        ### Customer Details
        {customer_data_str}

        ### Customer Type
        - **Individual** if age, gender, occupation are present.
        - **Organization** if industry, revenue_range, employee_count_range are present.

        ### Available Products
        {product_list}

        ### Customer's sentiment
        {sentiment_context}

        ### Instructions
        1. **Identify Customer Type**:
           - Determine if the customer is an individual or an organization based on the provided details.

        2. **Summarize Customer Needs and Preferences**:
           - Based on the customer data, create a concise summary of what the customer is looking for in banking products.
           - Consider their interests, preferences, transaction history, social media activity, and sentiment.

        3. **Evaluate Products**:
           - Review each product in the available list and assess how well it aligns with the customer's summary.
           - Consider the product's description and how it addresses the customer's needs.

        4. **Select Top Recommendations**:
           - Choose 3-5 products that are the best match.
           - Prioritize products that directly cater to the customer's specific requirements and preferences.

        5. **Provide Recommendations with Reasons**:
           - List the recommended product names.
           - For each recommendation, explain why it is suitable for the customer based on their data.

        6. **Business Insights**:
           - Based on the customer's data, suggest ways the bank can optimize customer engagement.
           - This could include personalized marketing strategies, customer service approaches, or other engagement tactics.

        ### Output Format
        - Start with "Recommended Products:"
        - For each product:
          - Product Name
          - Reason: [Explanation]
        - Then, "Business Insights:"
          - [Suggestions for optimizing customer engagement]
        """
    )
    chain = prompt | llm | StrOutputParser()
    return chain.invoke({
        "customer_data_str": customer_data_str,
        "product_list": product_list,
        "sentiment_context": sentiment_context
    }).strip()

# Streamlit UI
def main():
    st.set_page_config(page_title="Banking Product Recommender", page_icon="🏦", layout="wide")
    st.title("🏦 Banking Product Recommender")
    st.markdown("""
        <style>
        .big-font {font-size: 18px; color: #555555;}
        .stButton>button {background-color: #4CAF50; color: white; border-radius: 8px; padding: 8px 16px;}
        </style>
        <p class="big-font">Select a Customer ID from the dropdown below to get personalized banking product recommendations.</p>
    """, unsafe_allow_html=True)

    # Get all customer IDs
    customer_ids = get_all_customer_ids()
    
    # Dropdown to select customer ID
    customer_id = st.selectbox("Select Customer ID", options=customer_ids, index=0)

    # Button to fetch recommendations
    if st.button("Get Recommendations"):
        with st.spinner("Fetching customer data..."):
            customer = get_customer_details(customer_id)
            if not customer:
                st.error(f"No customer found with ID '{customer_id}'. Please select a different ID.")
            else:
                st.subheader("Customer Details")
                st.json(customer)

                with st.spinner("Retrieving relevant products..."):
                    retrieved_products = vector_search(customer, k=10)

                with st.spinner("Generating recommendations with GPT-4o..."):
                    llm_response = get_llm_recommendations(customer, retrieved_products)
                    if "API key not found" in llm_response:
                        st.error(llm_response)
                    else:
                        st.subheader("AI-Generated Recommendations")
                        st.write(llm_response)

                        st.subheader("Recommended Banking Products")
                        for prod in retrieved_products:
                            if prod["name"] in llm_response:
                                st.markdown(f"""
                                    <div style="border: 1px solid #ddd; padding: 15px; border-radius: 5px; margin: 10px 0;">
                                        <h4 style="color: #4CAF50; margin: 0;">{prod['name']}</h4>
                                        <p style="margin: 5px 0 0 0;">{prod['description']}</p>
                                    </div>
                                """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()