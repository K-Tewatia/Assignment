# src/pinecone_manager.py
import pandas as pd
import numpy as np
from typing import List, Dict, Optional, Any, Union
from sentence_transformers import SentenceTransformer
import json

# Try to import new Pinecone API (v3.x)
try:
    from pinecone import Pinecone, ServerlessSpec
    PINECONE_V3 = True
except ImportError:
    # Fall back to old Pinecone API (v2.x)
    import pinecone  # type: ignore
    PINECONE_V3 = False


class PineconeManager:
    def __init__(self, api_key: str, index_name: str = "marketing-roi-data"):
        """Initialize Pinecone Manager"""
        if not api_key:
            raise ValueError("Pinecone API key cannot be empty")
        
        self.api_key = api_key
        self.index_name = index_name
        self.embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
        self.dimension = 384
        self.index: Any = None
        
        # Initialize Pinecone based on version
        if PINECONE_V3:
            self.pc: Any = Pinecone(api_key=api_key)
        else:
            pinecone.init(api_key=api_key, environment="us-east-1-aws")  # type: ignore
            self.pc: Any = pinecone
        
    def create_index(self) -> Any:
        """Create Pinecone index if it doesn't exist"""
        if PINECONE_V3:
            # New API (v3.x)
            existing_indexes = [index.name for index in self.pc.list_indexes()]
            
            if self.index_name not in existing_indexes:
                print(f"Creating new index: {self.index_name}")
                self.pc.create_index(
                    name=self.index_name,
                    dimension=self.dimension,
                    metric='cosine',
                    spec=ServerlessSpec(
                        cloud='aws',
                        region='us-east-1'
                    )
                )
            else:
                print(f"Index {self.index_name} already exists")
            
            self.index = self.pc.Index(self.index_name)
        else:
            # Old API (v2.x)
            existing_indexes = self.pc.list_indexes()
            if self.index_name not in existing_indexes:
                print(f"Creating new index: {self.index_name}")
                self.pc.create_index(
                    name=self.index_name,
                    dimension=self.dimension,
                    metric='cosine'
                )
            else:
                print(f"Index {self.index_name} already exists")
            
            self.index = self.pc.Index(self.index_name)
        
        return self.index
    
    def clean_value(self, value: Any) -> Union[float, int, str]:
        """Clean value to handle NaN, inf, and None"""
        if pd.isna(value) or value is None:
            return 0.0
        if isinstance(value, (float, np.floating)) and np.isinf(float(value)):
            return 0.0
        if isinstance(value, (np.integer, np.floating)):
            return float(value)
        return value
    
    def clean_metadata(self, metadata: Dict[str, Any]) -> Dict[str, Any]:
        """Ensure no NaN or inf values in metadata"""
        cleaned = {}
        for key, value in metadata.items():
            if isinstance(value, float):
                if np.isnan(value) or np.isinf(value):
                    cleaned[key] = 0.0
                else:
                    cleaned[key] = value
            else:
                cleaned[key] = value
        return cleaned
    
    def prepare_marketing_data(self, df: pd.DataFrame) -> List[Dict[str, Any]]:
        """Convert marketing data to embeddings with metadata"""
        vectors_to_upsert: List[Dict[str, Any]] = []
        
        # Clean the dataframe first
        df = df.copy()
        
        # Replace NaN, inf with 0 for numeric columns
        numeric_cols = ['cost', 'revenue', 'conversions', 'clicks', 'impressions', 'roi', 'conversion_rate']
        for col in numeric_cols:
            if col in df.columns:
                df[col] = df[col].replace([np.inf, -np.inf], 0)
                df[col] = df[col].fillna(0)
        
        for idx, row in df.iterrows():
            # Clean all values
            cost = float(self.clean_value(row['cost']))
            revenue = float(self.clean_value(row['revenue']))
            conversions = int(self.clean_value(row['conversions']))
            clicks = int(self.clean_value(row['clicks']))
            impressions = int(self.clean_value(row['impressions']))
            roi = float(self.clean_value(row.get('roi', 0)))
            conversion_rate = float(self.clean_value(row.get('conversion_rate', 0)))
            
            # Create rich text representation for embedding
            text = f"""
            Campaign: {row['campaign_name']} (ID: {row['campaign_id']})
            Channel: {row['channel']}
            Date: {row['date']}
            Cost: ${cost:.2f}
            Revenue: ${revenue:.2f}
            Conversions: {conversions}
            Clicks: {clicks}
            Impressions: {impressions}
            ROI: {roi:.2f}%
            Conversion Rate: {conversion_rate:.2f}%
            """
            
            # Generate embedding
            embedding = self.embedding_model.encode(text).tolist()
            
            # Prepare metadata - ensure no NaN values
            metadata: Dict[str, Any] = {
                'campaign_id': str(row['campaign_id']),
                'campaign_name': str(row['campaign_name']),
                'channel': str(row['channel']),
                'date': str(row['date']),
                'cost': float(cost),
                'revenue': float(revenue),
                'conversions': int(conversions),
                'clicks': int(clicks),
                'impressions': int(impressions),
                'roi': float(roi),
                'conversion_rate': float(conversion_rate),
                'text': text.strip()
            }
            
            # Add currency if present
            if 'currency' in row and pd.notna(row['currency']):
                metadata['currency'] = str(row['currency'])
            
            # Clean metadata to ensure no NaN/inf values
            metadata = self.clean_metadata(metadata)

            vectors_to_upsert.append({
                'id': f"record_{idx}",
                'values': embedding,
                'metadata': metadata
            })
        
        return vectors_to_upsert
    
    def upload_data(self, df: pd.DataFrame, batch_size: int = 100) -> bool:
        """Upload marketing data to Pinecone"""
        if self.index is None:
            self.create_index()
        
        print(f"Preparing {len(df)} records for upload...")
        
        # Clean data before processing
        print("Cleaning data (removing NaN, inf values)...")
        df = df.copy()
        
        # Replace NaN and inf values
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        for col in numeric_cols:
            df[col] = df[col].replace([np.inf, -np.inf], np.nan)
            df[col] = df[col].fillna(0)
        
        # Prepare vectors (already cleaned in prepare_marketing_data)
        vectors = self.prepare_marketing_data(df)
        
        print(f"Uploading to Pinecone in batches of {batch_size}...")
        for i in range(0, len(vectors), batch_size):
            batch = vectors[i:i + batch_size]
            try:
                if self.index is not None:
                    self.index.upsert(vectors=batch)
                print(f"✓ Uploaded batch {i//batch_size + 1}/{(len(vectors)-1)//batch_size + 1}")
            except Exception as e:
                print(f"✗ Error in batch {i//batch_size + 1}: {str(e)}")
                # Print problematic record for debugging
                if batch:
                    print(f"First record in failed batch:")
                    print(f"  ID: {batch[0].get('id', 'N/A')}")
                    print(f"  Metadata: {batch[0].get('metadata', {})}")
                raise
        
        print(f"✓ Successfully uploaded {len(vectors)} vectors to Pinecone!")
        
        # Verify upload
        try:
            if self.index is not None:
                stats = self.index.describe_index_stats()
                print(f"Index stats: {stats}")
        except Exception as e:
            print(f"Could not get index stats: {e}")
        
        return True
    def fetch_all_vectors(self, batch_size: int = 1000) -> Dict[str, Dict[str, Any]]:
        """Fetch all vectors from index using pagination"""
        if self.index is None:
            self.index = self.pc.Index(self.index_name)
        
        print("Fetching all vectors from Pinecone...")
        
        all_metadata = []
        
        try:
            # Get index stats to know total count
            stats = self.index.describe_index_stats()
            print(f"Index stats: {stats}")
            
            # Try to fetch using query with dummy vector
            dummy_vector = [0.0] * self.dimension
            
            # Fetch in batches
            for offset in range(0, 10000, batch_size):
                results = self.index.query(
                    vector=dummy_vector,
                    top_k=min(batch_size, 10000 - offset),
                    include_metadata=True
                )
                
                if isinstance(results, dict) and 'matches' in results and results['matches']:
                    for match in results['matches']:
                        if 'metadata' in match:
                            all_metadata.append(match['metadata'])
                    
                    print(f"  Fetched {len(all_metadata)} records so far...")
                    
                    # If we got less than batch_size, we're done
                    if len(results['matches']) < batch_size:
                        break
                else:
                    break
            
            print(f"✓ Total records fetched: {len(all_metadata)}")
            
        except Exception as e:
            print(f"Error fetching vectors: {e}")
        
        # Aggregate by channel
        channel_data: Dict[str, Dict[str, Any]] = {}
        
        for metadata in all_metadata:
            channel = metadata.get('channel')
            
            if channel and channel not in channel_data:
                channel_data[str(channel)] = {
                    'cost': 0.0,
                    'revenue': 0.0,
                    'conversions': 0,
                    'clicks': 0,
                    'impressions': 0,
                    'records': 0
                }
            
            if channel:
                channel_str = str(channel)
                channel_data[channel_str]['cost'] += float(self.clean_value(metadata.get('cost', 0)))
                channel_data[channel_str]['revenue'] += float(self.clean_value(metadata.get('revenue', 0)))
                channel_data[channel_str]['conversions'] += int(self.clean_value(metadata.get('conversions', 0)))
                channel_data[channel_str]['clicks'] += int(self.clean_value(metadata.get('clicks', 0)))
                channel_data[channel_str]['impressions'] += int(self.clean_value(metadata.get('impressions', 0)))
                channel_data[channel_str]['records'] += 1
        
        # Calculate derived metrics
        for channel, data in channel_data.items():
            if data['cost'] > 0:
                data['roi'] = ((data['revenue'] - data['cost']) / data['cost']) * 100
                data['cpa'] = data['cost'] / data['conversions'] if data['conversions'] > 0 else 0.0
                data['ctr'] = (data['clicks'] / data['impressions']) * 100 if data['impressions'] > 0 else 0.0
                data['conversion_rate'] = (data['conversions'] / data['clicks']) * 100 if data['clicks'] > 0 else 0.0
            else:
                data['roi'] = 0.0
                data['cpa'] = 0.0
                data['ctr'] = 0.0
                data['conversion_rate'] = 0.0
        
        return channel_data

    def query_data(self, query_text: str, top_k: int = 50, filter_dict: Optional[Dict[str, Any]] = None) -> Any:
        """Query Pinecone for relevant marketing data"""
        if self.index is None:
            self.index = self.pc.Index(self.index_name)
        
        query_embedding = self.embedding_model.encode(query_text).tolist()
        
        results = self.index.query(
            vector=query_embedding,
            top_k=top_k,
            include_metadata=True,
            filter=filter_dict
        )
        
        return results
    
    def get_all_channels(self) -> List[str]:
        """Get list of all unique channels"""
        results = self.query_data("marketing channels", top_k=100)
        
        channels: set = set()
        if isinstance(results, dict) and 'matches' in results:
            for match in results['matches']:
                if isinstance(match, dict) and 'metadata' in match:
                    metadata = match['metadata']
                    if isinstance(metadata, dict) and 'channel' in metadata:
                        channels.add(str(metadata['channel']))
        
        return sorted(list(channels))
    
    def get_aggregated_metrics(self) -> Dict[str, Dict[str, Any]]:
        """Get aggregated metrics across all channels"""
        if self.index is None:
            self.index = self.pc.Index(self.index_name)
        
        print("Retrieving data from Pinecone...")
        
        # Strategy 1: Try fetch_all_vectors first
        try:
            channel_data = self.fetch_all_vectors()
            if channel_data:
                print(f"✓ Retrieved data for {len(channel_data)} channels via fetch_all_vectors")
                return channel_data
        except Exception as e:
            print(f"fetch_all_vectors failed: {e}")
        
        # Strategy 2: Try query_data
        try:
            results = self.query_data("all marketing channel performance", top_k=10000)
            channel_data: Dict[str, Dict[str, Any]] = {}
            
            if isinstance(results, dict) and 'matches' in results and results['matches']:
                print(f"Retrieved {len(results['matches'])} matches from query")
                
                for match in results['matches']:
                    if isinstance(match, dict) and 'metadata' in match:
                        metadata = match['metadata']
                        if isinstance(metadata, dict):
                            channel = metadata.get('channel')
                            
                            if channel and channel not in channel_data:
                                channel_data[str(channel)] = {
                                    'cost': 0.0,
                                    'revenue': 0.0,
                                    'conversions': 0,
                                    'clicks': 0,
                                    'impressions': 0,
                                    'records': 0
                                }
                            
                            if channel:
                                channel_str = str(channel)
                                channel_data[channel_str]['cost'] += float(self.clean_value(metadata.get('cost', 0)))
                                channel_data[channel_str]['revenue'] += float(self.clean_value(metadata.get('revenue', 0)))
                                channel_data[channel_str]['conversions'] += int(self.clean_value(metadata.get('conversions', 0)))
                                channel_data[channel_str]['clicks'] += int(self.clean_value(metadata.get('clicks', 0)))
                                channel_data[channel_str]['impressions'] += int(self.clean_value(metadata.get('impressions', 0)))
                                channel_data[channel_str]['records'] += 1
                
                # Calculate derived metrics
                for channel, data in channel_data.items():
                    if data['cost'] > 0:
                        data['roi'] = ((data['revenue'] - data['cost']) / data['cost']) * 100
                        data['cpa'] = data['cost'] / data['conversions'] if data['conversions'] > 0 else 0.0
                        data['ctr'] = (data['clicks'] / data['impressions']) * 100 if data['impressions'] > 0 else 0.0
                        data['conversion_rate'] = (data['conversions'] / data['clicks']) * 100 if data['clicks'] > 0 else 0.0
                    else:
                        data['roi'] = 0.0
                        data['cpa'] = 0.0
                        data['ctr'] = 0.0
                        data['conversion_rate'] = 0.0
                
                if channel_data:
                    print(f"✓ Retrieved data for {len(channel_data)} channels via query_data")
                    return channel_data
        except Exception as e:
            print(f"query_data failed: {e}")
        
        print("⚠ No data found in Pinecone")
        return {}

    
    def delete_index(self) -> None:
        """Delete the index"""
        if PINECONE_V3:
            self.pc.delete_index(self.index_name)
        else:
            self.pc.delete_index(self.index_name)
        print(f"Deleted index: {self.index_name}")
