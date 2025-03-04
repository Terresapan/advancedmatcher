import math
from supabase import create_client, Client
import streamlit as st
import pandas as pd
from datetime import datetime
import os
from langsmith import traceable

os.environ["SUPABASE_URL"] = st.secrets["database"]["SUPABASE_URL"]
os.environ["SUPABASE_KEY"] = st.secrets["database"]["SUPABASE_KEY"]


# Initialize Supabase client
SUPABASE_URL = os.environ["SUPABASE_URL"]  
SUPABASE_KEY = os.environ["SUPABASE_KEY"]  
supabase: Client = create_client(SUPABASE_URL, SUPABASE_KEY)


@st.cache_resource
@traceable(
    metadata={"vectordb": "SUPABASE"}
)
def sync_consultant_data_to_supabase(_df, _embeddings):
    """Sync consultant data to Supabase with embeddings."""
    if _embeddings is None:
        st.error("❌ Embeddings model is not initialized")
        return False
    if _df is None or _df.empty:
        st.error("❌ Consultant DataFrame is empty or not provided")
        return False

    try:
        # Generate text for embeddings
        # Add deduplication here
        _df = _df.drop_duplicates(subset='name', keep='last') 
        
        text_data = []
        for _, row in _df.iterrows():
            combined_text = (
                f"Name: {row.get('name', '')}; "
                f"Age: {row.get('age', '')}; "
                f"Industry Expertise: {row.get('industry_expertise', '')}; "
                f"Education: {row.get('education', '')}; "
                f"Bio: {row.get('bio', '')}; "
            )
            text_data.append(combined_text)
        
        # Generate embeddings
        embeddings_list = _embeddings.embed_documents(text_data)

        # Prepare data for upsert (using name as a unique identifier)
        data_to_insert = []
        for i, (_, row) in enumerate(_df.iterrows()):
            # Map DataFrame columns to Supabase table columns
            # Validate and clean embedding values
            cleaned_embedding = []
            for value in embeddings_list[i]:
                if not isinstance(value, (int, float)):
                    cleaned_embedding.append(0.0)
                elif not math.isfinite(value):
                    cleaned_embedding.append(0.0)
                else:
                    # Ensure value is within JSON-compliant range
                    # JSON can't handle values outside approximately ±1.0e308
                    try:
                        float_val = float(value)
                        # Check if the value is too large or too small for JSON
                        if abs(float_val) > 1.0e308:
                            cleaned_embedding.append(0.0)
                        else:
                            cleaned_embedding.append(float_val)
                    except (ValueError, OverflowError):
                        cleaned_embedding.append(0.0)

            consultant_data = {
                "name": row.get('name'),  # Changed back to lowercase
                "age": row.get('age'),    # Changed back to lowercase
                "finance_expertise": row.get('finance_expertise'),
                "strategy_expertise": row.get('strategy_expertise'),
                "operations_expertise": row.get('operations_expertise'),
                "marketing_expertise": row.get('marketing_expertise'),
                "entrepreneurship_expertise": row.get('entrepreneurship_expertise'),
                "education": row.get('education'),
                "industry_expertise": row.get('industry_expertise'),
                "bio": row.get('bio'),
                "anticipated_availability_date": row.get('anticipated_availability_date'),
                "availability": row.get('availability'),
                "embedding": cleaned_embedding,
                "updated_at": datetime.now().isoformat()
            }

            # Replace np.nan with None and check for non-scalar values
            for key in consultant_data:
                if key != 'embedding':  # Skip embedding
                    value = consultant_data[key]
                    if not pd.api.types.is_scalar(value):
                        st.warning(f"Non-scalar value detected in '{key}': {value} (type: {type(value)})")
                    if pd.isna(value):
                        consultant_data[key] = None

            # Remove None values to prevent SQL errors
            consultant_data = {k: v for k, v in consultant_data.items() if v is not None}
            data_to_insert.append(consultant_data)

        # Clear existing data and insert new data
        supabase.table('consultants').upsert(data_to_insert, on_conflict='name').execute()
        # st.success(f"✅ Successfully synced {len(data_to_insert)} consultant records to Supabase")
        return True

    except Exception as e:
        st.error(f"❌ Error syncing data to Supabase: {str(e)}")
        return False
