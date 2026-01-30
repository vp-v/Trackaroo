import streamlit as st
import pandas as pd
import torch
from transformers import DistilBertTokenizer, DistilBertForSequenceClassification
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime
import numpy as np

# Page configuration
st.set_page_config(
    page_title="Trackaroo",
    layout="wide",    
)

# Custom CSS for better styling
st.markdown("""
<style>
/* --- App background --- */
.stApp {
    background-color: #F4F6F8;  /* Light neutral gray */
    color: #1F2937;             /* Dark gray text */
    font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
}

/* --- Sidebar styling --- */
[data-testid="stSidebar"] {
    background-color: #FFFFFF;   /* White sidebar */
    color: #1F2937;
    border-right: 1px solid #E5E7EB;
    padding: 20px;
}

/* Sidebar headers and text */
[data-testid="stSidebar"] h1,
[data-testid="stSidebar"] h2,
[data-testid="stSidebar"] h3,
[data-testid="stSidebar"] p,
[data-testid="stSidebar"] li,
[data-testid="stSidebar"] span {
    color: #1F2937;
}

/* Sidebar info box */
div[data-testid="stAlert"][kind="info"] {
    background-color: #E0F2FE !important;   /* Light blue */
    border-left: 5px solid #0284C7 !important; /* Blue accent */
    color: #0369A1 !important;
    font-weight: 500;
    padding: 15px;
    border-radius: 8px;
}

/* Success, warning, recommendation boxes */
.success-box {
    background-color: #ECFDF5;
    border-left: 5px solid #10B981;
    color: #065F46;
    padding: 15px;
    border-radius: 8px;
    margin: 10px 0;
}
.warning-box {
    background-color: #FEF3F2;
    border-left: 5px solid #EF4444;
    color: #B91C1C;
    padding: 15px;
    border-radius: 8px;
    margin: 10px 0;
}
.recommendation-box {
    background-color: #FFFBEB;
    border-left: 5px solid #F59E0B;
    color: #78350F;
    padding: 15px;
    border-radius: 8px;
    margin: 10px 0;
}
/* All metric cards */
.stMetric {
    border-radius: 12px !important;
    padding: 15px !important;
    box-shadow: 0 2px 6px rgba(0,0,0,0.08) !important;
    margin-bottom: 10px !important;
}

/* Col-specific styling using nth-child */
.stColumns > div:nth-child(1) .stMetric {
    background-color: #E0F2FE !important;  /* Light blue */
    color: #0C4A6E !important;             /* Dark blue text */
}
.stColumns > div:nth-child(2) .stMetric {
    background-color: #FEF3C7 !important;  /* Light yellow */
    color: #78350F !important;             /* Dark yellow/brown text */
}
.stColumns > div:nth-child(3) .stMetric {
    background-color: #DCFCE7 !important;  /* Light green */
    color: #166534 !important;             /* Dark green text */
}
.stColumns > div:nth-child(4) .stMetric {
    background-color: #FEE2E2 !important;  /* Light red/pink */
    color: #991B1B !important;             /* Dark red text */
}

/* Ensure metric value and delta text is colored */
.stMetric span, .stMetric > div > div {
    color: inherit !important;
}
/* --- Headings --- */
h1, .stMarkdown h1, [data-testid="stTitle"] h1 {
    color: #1F2937 !important;
    font-weight: 700;
    letter-spacing: 1px;
}
h2, h3, .stMarkdown h2, .stMarkdown h3 {
    color: #374151;
    font-weight: 600;
}

/* Paragraphs and lists */
p, li, span {
    color: #4B5563;
    font-size: 15px;
}

/* --- Buttons --- */
div.stButton > button {
    background-color: #000000;
    color: #FFFFFF;
    font-weight: 600;
    border-radius: 8px;
    padding: 0.5rem 1rem;
    border: none;
    transition: all 0.2s ease;
}
div.stButton > button:hover {
    background-color: #DADADA;
}

/* File uploader */
div.stFileUploader {
    border: 1px solid #D1D5DB;
    border-radius: 8px;
    padding: 10px;
    background-color: #FFFFFF;
}

/* Number input fields */
input[type=number] {
    border-radius: 6px;
    border: 1px solid #D1D5DB;
    padding: 6px 10px;
    color: #1F2937;
}

/* Selectbox, radio, multiselect */
div.stSelectbox, div.stRadio, div.stMultiSelect {
    border-radius: 6px;
}

/* Expanders */
.stExpander {
    background-color: #FFFFFF;
    border-radius: 8px;
    box-shadow: 0 1px 3px rgba(0,0,0,0.05);
}

/* Dataframe table header */
thead th {
    background-color: #F3F4F6 !important;
    color: #111827 !important;
    font-weight: 600 !important;
}

/* SVG icons */
svg {
    fill: #1F2937 !important;
    color: #1F2937 !important;
}
</style>
""", unsafe_allow_html=True)


# Cache the model loading
@st.cache_resource
def load_model(model_path='./transaction_classifier'):
    """Load the trained DistilBERT model"""
    try:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        tokenizer = DistilBertTokenizer.from_pretrained(model_path)
        model = DistilBertForSequenceClassification.from_pretrained(model_path)
        model.to(device)
        model.eval()
        return model, tokenizer, device
    except Exception as e:
        return None, None, None

def classify_transaction(text, model, tokenizer, device):
    """Classify a single transaction"""
    encoding = tokenizer.encode_plus(
        text.lower(),
        add_special_tokens=True,
        max_length=128,
        padding='max_length',
        truncation=True,
        return_attention_mask=True,
        return_tensors='pt'
    )
    
    input_ids = encoding['input_ids'].to(device)
    attention_mask = encoding['attention_mask'].to(device)
    
    with torch.no_grad():
        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
        prediction = torch.argmax(outputs.logits, dim=1).item()
        probabilities = torch.softmax(outputs.logits, dim=1).cpu().numpy()[0]
    
    return 'Non-Essential' if prediction == 1 else 'Essential', probabilities[prediction]

def classify_dataframe(df, model, tokenizer, device, description_col='description'):
    """Classify all transactions in a dataframe"""
    classifications = []
    confidences = []
    
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    for idx, desc in enumerate(df[description_col]):
        label, confidence = classify_transaction(desc, model, tokenizer, device)
        classifications.append(label)
        confidences.append(confidence)
        
        progress = (idx + 1) / len(df)
        progress_bar.progress(progress)
        status_text.text(f'Classifying transactions: {idx + 1}/{len(df)}')
    
    progress_bar.empty()
    status_text.empty()
    
    df['category'] = classifications
    df['confidence'] = confidences
    return df

def generate_insights(expenses_df, essential_total, non_essential_total, total_expenses):
    """Generate detailed insights and recommendations"""
    insights = {
        'recommendations': [],
        'highlights': [],
        'warnings': []
    }
    
    essential_pct = (essential_total / total_expenses * 100) if total_expenses > 0 else 0
    non_essential_pct = (non_essential_total / total_expenses * 100) if total_expenses > 0 else 0
    
    # Spending pattern analysis
    if non_essential_pct > 40:
        insights['warnings'].append({
            'title':'High Non-Essential Spending',
            'message': f'Your non-essential spending is {non_essential_pct:.1f}% of total expenses. Financial experts recommend keeping this below 30-35%.'
        })
    elif non_essential_pct > 30:
        insights['highlights'].append({
            'title': 'Balanced Spending',
            'message': f'Your spending balance is reasonable at {non_essential_pct:.1f}% non-essential.'
        })
    else:
        insights['highlights'].append({
            'title': 'Excellent Financial Discipline',
            'message': f'Outstanding! Only {non_essential_pct:.1f}% goes to non-essential expenses.'
        })
    
    # Top spending categories
    non_essential = expenses_df[expenses_df['category'] == 'Non-Essential']
    if len(non_essential) > 0:
        top_5 = non_essential.nlargest(5, 'amount')
        insights['recommendations'].append({
            'title': 'Top Non-Essential Expenses',
            'items': [f"${row['amount']:.2f} - {row['description']}" for _, row in top_5.iterrows()]
        })
    
    # Savings potential
    if non_essential_pct > 30:
        potential_savings = non_essential_total * 0.20
        insights['recommendations'].append({
            'title': 'Savings Opportunity',
            'items': [
                f"Reducing non-essential spending by 20% could save ${potential_savings:.2f}",
                "Review and cancel unused subscriptions",
                "Plan weekly meals to reduce food delivery costs",
                "Set a monthly budget for discretionary spending"
            ]
        })
    
    return insights

# Main App
def main():
    # Header
    st.markdown("""
        <div style='text-align: center; padding: 20px 0;'>
            <h1>Trackaroo</h1>
            <p style='color: #4b5563; font-size: 15px;'>Track your expenses like how you would track your calories!</p>
        </div>
    """, unsafe_allow_html=True)
    
    # Sidebar
    with st.sidebar:
        st.image("/Users/vyshnavisarathy/Desktop/Trackaroo/logo.png", width=150)
        st.markdown("About This App")
        st.info("""
        This app uses a fine-tuned **DistilBERT** model to automatically classify your bank transactions into:
        
        **Essential** - Groceries, utilities, healthcare, transport
        
         **Non-Essential** - Entertainment, dining out, subscriptions
        """)
        
        st.markdown("---")
        st.markdown("Upload Your Data")
        st.markdown("CSV should contain:")
        st.markdown("- Date column")
        st.markdown("- Description column")
        st.markdown("- Amount column")
        
        st.markdown("---")
        st.markdown("Features")
        st.markdown(" AI-powered classification")
        st.markdown("Interactive visualizations")
        st.markdown("Personalized recommendations")
        st.markdown("Export classified data")
    
    # Load model
    with st.spinner("Loading AI model..."):
        model, tokenizer, device = load_model()
    
    if model is None:
        st.error("Model not found! Please train the model first using `train_complete.py`")
        st.code("python train_complete.py", language="bash")
        st.stop()
    
    # File upload section
    st.markdown("---")
    st.markdown("Upload Your Transactions")
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        uploaded_file = st.file_uploader(
            "Choose a CSV file",
            type=['csv'],
            help="Upload your bank statement in CSV format"
        )
    
    with col2:
        st.markdown("Sample Format")
        sample_df = pd.DataFrame({
            'date': ['2024-01-01', '2024-01-02'],
            'description': ['WOOLWORTHS', 'NETFLIX'],
            'amount': [-150.50, -15.99]
        })
        st.dataframe(sample_df, use_container_width=True)
    
    if uploaded_file is not None:
        # Load data
        try:
            df = pd.read_csv(uploaded_file)
            st.success(f" Loaded {len(df)} transactions")
        except Exception as e:
            st.error(f"Error reading file: {e}")
            st.stop()
        
        # Show preview
        with st.expander(" Preview Raw Data", expanded=False):
            st.dataframe(df.head(10), use_container_width=True)
        
        st.markdown("---")
        st.markdown("Configure Columns")
        
        # Column selection
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            date_col = st.selectbox("Date Column", df.columns, index=0)
        with col2:
            desc_col = st.selectbox("Description Column", df.columns, 
                                   index=1 if len(df.columns) > 1 else 0)
        with col3:
            amount_col = st.selectbox("Amount Column", df.columns,
                                     index=2 if len(df.columns) > 2 else 0)
        with col4:
            # Optional transaction type column
            type_options = ['None (use amount sign)'] + list(df.columns)
            type_col = st.selectbox("Type Column (Optional)", type_options, index=0)
        
        # Income detection method
        st.markdown("Income Detection Method")
        detection_method = st.radio(
            "How should income be identified?",
            options=[
                "By amount sign (positive = income, negative = expense)",
                "By transaction type column (Debit/Credit or similar)",
                "By keywords in description (SALARY, DEPOSIT, etc.)"
            ],
            index=0
        )
        
        # Process button
        st.markdown("---")
        
        if st.button("Analyze My Expenses", type="primary", use_container_width=True):
            with st.spinner("Analyzing your transactions..."):
                
                # Standardize column names
                df_processed = df[[date_col, desc_col, amount_col]].copy()
                df_processed.columns = ['date', 'description', 'amount']
                df_processed['amount'] = pd.to_numeric(df_processed['amount'], errors='coerce')
                
                # Add transaction type if specified
                if type_col != 'None (use amount sign)':
                    df_processed['type'] = df[type_col]
                
                # Remove any NaN amounts
                df_processed = df_processed.dropna(subset=['amount'])
                
                # Determine income based on selected method
                if detection_method == "By amount sign (positive = income, negative = expense)":
                    # Standard method - positive is income
                    df_processed['is_income'] = df_processed['amount'] > 0
                    
                elif detection_method == "By transaction type column (Debit/Credit or similar)":
                    if type_col == 'None (use amount sign)':
                        st.error("Please select a transaction type column for this method!")
                        st.stop()
                    
                    # Identify income based on type column
                    # Common patterns: Credit, Deposit, CR, etc.
                    income_keywords = ['credit', 'cr', 'deposit', 'income', 'received']
                    df_processed['is_income'] = df_processed['type'].astype(str).str.lower().apply(
                        lambda x: any(keyword in x for keyword in income_keywords)
                    )
                    # Make all amounts positive for consistency
                    df_processed['amount'] = df_processed['amount'].abs()
                    
                elif detection_method == "By keywords in description (SALARY, DEPOSIT, etc.)":
                    # Identify income by description keywords
                    income_keywords = ['salary', 'wage', 'payment received', 'deposit', 'direct deposit', 
                                     'income', 'refund', 'reimbursement', 'transfer in', 'credit']
                    df_processed['is_income'] = df_processed['description'].astype(str).str.lower().apply(
                        lambda x: any(keyword in x for keyword in income_keywords)
                    )
                    # Make all amounts positive for consistency
                    df_processed['amount'] = df_processed['amount'].abs()
                
                # Classify transactions
                df_classified = classify_dataframe(df_processed, model, tokenizer, device)
                
                # Separate income and expenses based on detection method
                income_df = df_classified[df_classified['is_income']].copy()
                expenses_df = df_classified[~df_classified['is_income']].copy()
                
                # Ensure amounts are positive for expenses
                expenses_df['amount'] = expenses_df['amount'].abs()
                
                # Calculate metrics
                total_income = income_df['amount'].sum()
                total_expenses = expenses_df['amount'].sum()
                
                essential_total = expenses_df[expenses_df['category'] == 'Essential']['amount'].sum()
                non_essential_total = expenses_df[expenses_df['category'] == 'Non-Essential']['amount'].sum()
                
                essential_pct = (essential_total / total_expenses * 100) if total_expenses > 0 else 0
                non_essential_pct = (non_essential_total / total_expenses * 100) if total_expenses > 0 else 0
                
                net_savings = total_income - total_expenses
                savings_rate = (net_savings / total_income * 100) if total_income > 0 else 0
                
                # Store in session state
                st.session_state['df_classified'] = df_classified
                st.session_state['expenses_df'] = expenses_df
                st.session_state['metrics'] = {
                    'total_income': total_income,
                    'total_expenses': total_expenses,
                    'essential_total': essential_total,
                    'non_essential_total': non_essential_total,
                    'essential_pct': essential_pct,
                    'non_essential_pct': non_essential_pct,
                    'net_savings': net_savings,
                    'savings_rate': savings_rate
                }
        
        # Display results if available
        if 'metrics' in st.session_state:
            metrics = st.session_state['metrics']
            df_classified = st.session_state['df_classified']
            expenses_df = st.session_state['expenses_df']
            
            st.markdown("---")
            st.markdown("Financial Overview")
            
            # Key metrics
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.metric(
                    "Total Income",
                    f"${metrics['total_income']:,.2f}",
                    delta=None
                )
            
            with col2:
                st.metric(
                    "Total Expenses",
                    f"${metrics['total_expenses']:,.2f}",
                    delta=f"{-metrics['total_expenses']/metrics['total_income']*100:.1f}%" if metrics['total_income'] > 0 else None,
                    delta_color="inverse"
                )
            
            with col3:
                st.metric(
                    "Essential",
                    f"${metrics['essential_total']:,.2f}",
                    delta=f"{metrics['essential_pct']:.1f}%"
                )
            
            with col4:
                st.metric(
                    "Non-Essential",
                    f"${metrics['non_essential_total']:,.2f}",
                    delta=f"{metrics['non_essential_pct']:.1f}%",
                    delta_color="inverse"
                )
            
            # Savings metric
            st.markdown("Net Savings")
            col1, col2 = st.columns(2)
            
            with col1:
                delta_color = "normal" if metrics['net_savings'] >= 0 else "inverse"
                st.metric(
                    "Monthly Savings",
                    f"${metrics['net_savings']:,.2f}",
                    delta=f"{metrics['savings_rate']:.1f}% savings rate",
                    delta_color=delta_color
                )
            
            with col2:
                if metrics['savings_rate'] >= 20:
                    st.success("Excellent! You're saving over 20% of your income.")
                elif metrics['savings_rate'] >= 10:
                    st.info("Good job! You're saving 10-20% of your income.")
                elif metrics['savings_rate'] > 0:
                    st.warning("Try to increase your savings rate to at least 10%.")
                else:
                    st.error(" You're spending more than you earn. Time to review expenses!")
            
            # Visualizations
            st.markdown("---")
            st.markdown("Visual Analysis")
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("Expense Distribution")
                
                fig_pie = px.pie(
                    values=[metrics['essential_total'], metrics['non_essential_total']],
                    names=['Essential', 'Non-Essential'],
                    color=['Essential', 'Non-Essential'],
                    color_discrete_map={
                        'Essential': "#008055",
                        'Non-Essential': "#a11f1f"
                    },
                    hole=0.4
                )
                fig_pie.update_traces(
                    textposition='inside',
                    textinfo='percent+label',
                    textfont_size=14,
                    marker=dict(line=dict(color='white', width=2))
                )
                fig_pie.update_layout(
                    showlegend=True,
                    height=400,
                    margin=dict(t=20, b=20, l=20, r=20)
                )
                st.plotly_chart(fig_pie, use_container_width=True)
            
            with col2:
                st.markdown("Income vs Expenses")
                
                fig_bar = go.Figure()
                
                fig_bar.add_trace(go.Bar(
                    name='Income',
                    x=['Overview'],
                    y=[metrics['total_income']],
                    marker_color="#003a97",
                    text=f"${metrics['total_income']:,.0f}",
                    textposition='outside'
                ))
                
                fig_bar.add_trace(go.Bar(
                    name='Essential',
                    x=['Overview'],
                    y=[metrics['essential_total']],
                    marker_color="#009f6a",
                    text=f"${metrics['essential_total']:,.0f}",
                    textposition='outside'
                ))
                
                fig_bar.add_trace(go.Bar(
                    name='Non-Essential',
                    x=['Overview'],
                    y=[metrics['non_essential_total']],
                    marker_color="#822323",
                    text=f"${metrics['non_essential_total']:,.0f}",
                    textposition='outside'
                ))
                
                fig_bar.update_layout(
                    barmode='group',
                    yaxis_title='Amount ($)',
                    showlegend=True,
                    height=400,
                    margin=dict(t=20, b=20, l=20, r=20),
                    yaxis=dict(showgrid=True, gridcolor='lightgray')
                )
                st.plotly_chart(fig_bar, use_container_width=True)
            
            # Insights and Recommendations
            st.markdown("---")
            st.markdown("Personalized Insights")
            
            insights = generate_insights(
                expenses_df,
                metrics['essential_total'],
                metrics['non_essential_total'],
                metrics['total_expenses']
            )
            
            # Display highlights
            if insights['highlights']:
                for highlight in insights['highlights']:
                    st.markdown(f"""
                        <div class='success-box'>
                            <h4>{highlight['title']}</h4>
                            <p>{highlight['message']}</p>
                        </div>
                    """, unsafe_allow_html=True)
            
            # Display warnings
            if insights['warnings']:
                for warning in insights['warnings']:
                    st.markdown(f"""
                        <div class='warning-box'>
                            <h4>{warning['title']}</h4>
                            <p>{warning['message']}</p>
                        </div>
                    """, unsafe_allow_html=True)
            
            # Display recommendations
            if insights['recommendations']:
                for rec in insights['recommendations']:
                    st.markdown(f"""
                        <div class='recommendation-box'>
                            <h4>{rec['title']}</h4>
                            <ul>
                    """, unsafe_allow_html=True)
                    for item in rec['items']:
                        st.markdown(f"<li>{item}</li>", unsafe_allow_html=True)
                    st.markdown("</ul></div>", unsafe_allow_html=True)
            
            # Transaction details
            st.markdown("---")
            st.markdown("Transaction Details")
            
            # Filters
            col1, col2, col3 = st.columns(3)
            
            with col1:
                category_filter = st.multiselect(
                    "Filter by Category",
                    options=['Essential', 'Non-Essential'],
                    default=['Essential', 'Non-Essential']
                )
            
            with col2:
                min_amount = st.number_input(
                    "Minimum Amount ($)",
                    min_value=0.0,
                    value=0.0
                )
            
            with col3:
                sort_by = st.selectbox(
                    "Sort by",
                    options=['Amount (High to Low)', 'Amount (Low to High)', 'Date', 'Category']
                )
            
            # Apply filters
            filtered_df = df_classified[
                (df_classified['category'].isin(category_filter)) &
                (df_classified['amount'].abs() >= min_amount)
            ].copy()
            
            # Apply sorting
            if sort_by == 'Amount (High to Low)':
                filtered_df = filtered_df.sort_values('amount', ascending=False)
            elif sort_by == 'Amount (Low to High)':
                filtered_df = filtered_df.sort_values('amount', ascending=True)
            elif sort_by == 'Date':
                filtered_df = filtered_df.sort_values('date')
            elif sort_by == 'Category':
                filtered_df = filtered_df.sort_values('category')
            
            # Display transactions
            st.dataframe(
                filtered_df.style.format({
                    'amount': '${:,.2f}',
                    'confidence': '{:.1%}'
                }),
                use_container_width=True,
                height=400
            )
            
            # Download options
            st.markdown("---")
            st.markdown("Export Data")
            
            col1, col2 = st.columns(2)
            
            with col1:
                csv = df_classified.to_csv(index=False)
                st.download_button(
                    label="Download Classified Transactions (CSV)",
                    data=csv,
                    file_name=f"classified_transactions_{datetime.now().strftime('%Y%m%d')}.csv",
                    mime="text/csv",
                    use_container_width=True
                )
            
            with col2:
                # Create summary report
                summary = f"""
EXPENSE TRACKER SUMMARY REPORT
Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

FINANCIAL OVERVIEW
==================
Total Income:        ${metrics['total_income']:,.2f}
Total Expenses:      ${metrics['total_expenses']:,.2f}
Net Savings:         ${metrics['net_savings']:,.2f}
Savings Rate:        {metrics['savings_rate']:.1f}%

EXPENSE BREAKDOWN
=================
Essential:           ${metrics['essential_total']:,.2f} ({metrics['essential_pct']:.1f}%)
Non-Essential:       ${metrics['non_essential_total']:,.2f} ({metrics['non_essential_pct']:.1f}%)

TRANSACTION COUNT
=================
Total Transactions:  {len(df_classified)}
Income Entries:      {len(df_classified[df_classified['amount'] > 0])}
Expense Entries:     {len(df_classified[df_classified['amount'] < 0])}
                """
                
                st.download_button(
                    label="Download Summary Report (TXT)",
                    data=summary,
                    file_name=f"expense_summary_{datetime.now().strftime('%Y%m%d')}.txt",
                    mime="text/plain",
                    use_container_width=True
                )

if __name__ == "__main__":
    main()