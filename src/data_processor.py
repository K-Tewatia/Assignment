# src/data_processor.py
import pandas as pd
import numpy as np
from datetime import datetime
from typing import Dict

class DataProcessor:
    def __init__(self):
        self.data = None
        
    def load_csv(self, filepath: str) -> pd.DataFrame:
        """Load and validate CSV data with new schema"""
        self.data = pd.read_csv(filepath)
        self.data['date'] = pd.to_datetime(self.data['date'])
        
        # Validate required columns
        required_cols = ['campaign_id', 'campaign_name', 'channel', 'date', 
                        'impressions', 'clicks', 'conversions', 'cost', 'revenue']
        missing = [col for col in required_cols if col not in self.data.columns]
        
        if missing:
            raise ValueError(f"Missing required columns: {missing}")
        
        # Check for negative values
        numeric_cols = ['impressions', 'clicks', 'conversions', 'cost', 'revenue']
        for col in numeric_cols:
            if (self.data[col] < 0).any():
                raise ValueError(f"Negative values found in column: {col}")
        
        # Calculate ROI if not present - HANDLE DIVISION BY ZERO
        if 'roi' not in self.data.columns:
            self.data['roi'] = np.where(
                self.data['cost'] > 0,
                ((self.data['revenue'] - self.data['cost']) / self.data['cost']) * 100,
                0
            )
        
        # Calculate conversion_rate if not present - HANDLE DIVISION BY ZERO
        if 'conversion_rate' not in self.data.columns:
            self.data['conversion_rate'] = np.where(
                self.data['clicks'] > 0,
                (self.data['conversions'] / self.data['clicks']) * 100,
                0
            )
        
        # Replace any remaining NaN, inf values
        numeric_columns = self.data.select_dtypes(include=[np.number]).columns
        for col in numeric_columns:
            self.data[col] = self.data[col].replace([np.inf, -np.inf], 0)
            self.data[col] = self.data[col].fillna(0)
        
        print(f"✓ Loaded {len(self.data)} records")
        print(f"✓ Date range: {self.data['date'].min()} to {self.data['date'].max()}")
        print(f"✓ Channels: {', '.join(self.data['channel'].unique())}")
        print(f"✓ Campaigns: {self.data['campaign_id'].nunique()}")
        
        return self.data
    
    def format_for_context(self, data: pd.DataFrame) -> str:
        """Format data for LLM context"""
        context_parts = []
        
        # Clean data first
        data = data.copy()
        numeric_columns = data.select_dtypes(include=[np.number]).columns
        for col in numeric_columns:
            data[col] = data[col].replace([np.inf, -np.inf], 0)
            data[col] = data[col].fillna(0)
        
        # Overall summary
        total_cost = data['cost'].sum()
        total_revenue = data['revenue'].sum()
        total_conversions = data['conversions'].sum()
        total_clicks = data['clicks'].sum()
        total_impressions = data['impressions'].sum()
        overall_roi = ((total_revenue - total_cost) / total_cost * 100) if total_cost > 0 else 0
        overall_conversion_rate = (total_conversions / total_clicks * 100) if total_clicks > 0 else 0
        
        context_parts.append(f"""
OVERALL PERFORMANCE SUMMARY:
- Total Cost: ${total_cost:,.2f}
- Total Revenue: ${total_revenue:,.2f}
- Total Conversions: {total_conversions:,}
- Total Clicks: {total_clicks:,}
- Total Impressions: {total_impressions:,}
- Overall ROI: {overall_roi:.2f}%
- Overall Conversion Rate: {overall_conversion_rate:.2f}%
- Average CPA: ${(total_cost / total_conversions):.2f if total_conversions > 0 else 0}
- Number of Campaigns: {data['campaign_id'].nunique()}
""")
        
        # Channel-by-channel breakdown
        context_parts.append("\nCHANNEL PERFORMANCE BREAKDOWN:\n")
        
        for channel in data['channel'].unique():
            channel_data = data[data['channel'] == channel]
            
            cost = channel_data['cost'].sum()
            revenue = channel_data['revenue'].sum()
            conversions = channel_data['conversions'].sum()
            clicks = channel_data['clicks'].sum()
            impressions = channel_data['impressions'].sum()
            roi = ((revenue - cost) / cost * 100) if cost > 0 else 0
            cpa = (cost / conversions) if conversions > 0 else 0
            ctr = (clicks / impressions * 100) if impressions > 0 else 0
            conv_rate = (conversions / clicks * 100) if clicks > 0 else 0
            
            context_parts.append(f"""
{channel}:
  - Cost: ${cost:,.2f}
  - Revenue: ${revenue:,.2f}
  - Conversions: {conversions:,}
  - Clicks: {clicks:,}
  - Impressions: {impressions:,}
  - ROI: {roi:.2f}%
  - CPA: ${cpa:.2f}
  - CTR: {ctr:.2f}%
  - Conversion Rate: {conv_rate:.2f}%
  - Number of Campaigns: {channel_data['campaign_id'].nunique()}
""")
        
        return "\n".join(context_parts)
    
    def aggregate_channel_metrics(self, data: pd.DataFrame) -> Dict:
        """Calculate aggregated metrics by channel"""
        # Clean data first
        data = data.copy()
        numeric_columns = data.select_dtypes(include=[np.number]).columns
        for col in numeric_columns:
            data[col] = data[col].replace([np.inf, -np.inf], 0)
            data[col] = data[col].fillna(0)
        
        metrics = {}
        
        for channel in data['channel'].unique():
            channel_data = data[data['channel'] == channel]
            
            cost = channel_data['cost'].sum()
            revenue = channel_data['revenue'].sum()
            conversions = channel_data['conversions'].sum()
            clicks = channel_data['clicks'].sum()
            impressions = channel_data['impressions'].sum()
            
            metrics[channel] = {
                'cost': float(cost),
                'revenue': float(revenue),
                'conversions': int(conversions),
                'clicks': int(clicks),
                'impressions': int(impressions),
                'roi': float(((revenue - cost) / cost) * 100) if cost > 0 else 0.0,
                'cpa': float(cost / conversions) if conversions > 0 else 0.0,
                'ctr': float((clicks / impressions) * 100) if impressions > 0 else 0.0,
                'conversion_rate': float((conversions / clicks) * 100) if clicks > 0 else 0.0,
                'roi_ratio': float(revenue / cost) if cost > 0 else 0.0,
                'num_campaigns': int(channel_data['campaign_id'].nunique())
            }
        
        return metrics
