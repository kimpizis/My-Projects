# personal_finance_dashboard.py

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st

# Load Data
@st.cache
def load_data(file_path):
    df = pd.read_csv(file_path)
    df['Date'] = pd.to_datetime(df['Date'])
    return df

# Processing and Summarizing
def summary(df):
    total_income = df[df['Type'] == 'Income']['Amount'].sum()
    total_expenses = df[df['Type'] == 'Expense']['Amount'].sum()

    category_expenses = df[df['Type'] == 'Expense'].groupby('Category')['Amount'].sum()
    monthly_summary = df.groupby(df['Date'].dt.to_period('M')).agg({'Amount': 'sum', 'Type': lambda x: x.iloc[0]}).reset_index()
    
    return total_income, total_expenses, category_expenses, monthly_summary

# Plot Functions
def plot_category_expenses(category_expenses):
    plt.figure(figsize=(8, 6))
    category_expenses.plot.pie(autopct='%1.1f%%', startangle=90, cmap='Set3')
    plt.title('Expenses by Category')
    plt.ylabel('')
    st.pyplot(plt.gcf())

def plot_monthly_summary(monthly_summary):
    plt.figure(figsize=(10, 6))
    sns.lineplot(x=monthly_summary['Date'].astype(str), y='Amount', hue='Type', data=monthly_summary)
    plt.title('Monthly Income and Expenses')
    plt.xticks(rotation=45)
    st.pyplot(plt.gcf())

# Main Streamlit App
def main():
    st.title("Personal Finance Dashboard")

    # Upload CSV File
    file_upload = st.file_uploader("Upload a CSV file", type="csv")
    if file_upload is not None:
        df = load_data(file_upload)

        # Show Dataframe
        st.subheader("Uploaded Data")
        st.dataframe(df)

        # Summarize Data
        total_income, total_expenses, category_expenses, monthly_summary = summary(df)

        # Display Summary Metrics
        st.subheader("Summary")
        st.write(f"**Total Income:** ${total_income:.2f}")
        st.write(f"**Total Expenses:** ${total_expenses:.2f}")

        # Plot Category Expenses
        st.subheader("Expenses by Category")
        plot_category_expenses(category_expenses)

        # Plot Monthly Summary
        st.subheader("Monthly Income vs Expenses")
        plot_monthly_summary(monthly_summary)

if __name__ == "__main__":
    main()
