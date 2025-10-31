# src/mistral_agent.py
from mistralai import Mistral
from typing import Dict
import json
import time

class MistralAgent:
    def __init__(self, api_key: str, model: str = "mistral-large-latest"):
        """Initialize Mistral Agent"""
        if not api_key:
            raise ValueError("Mistral API key cannot be empty")
        
        self.client = Mistral(api_key=api_key)
        self.model = model
        self.max_retries = 3
        self.retry_delay = 5  # seconds
        
    def _make_request_with_retry(self, messages, temperature=0.3, max_tokens=2000):
        """Make API request with retry logic for rate limits"""
        
        # Try primary model first
        models_to_try = [
            self.model,
            "mistral-medium-latest",
            "mistral-small-latest",
            "open-mistral-7b"
        ]
        
        for model in models_to_try:
            for attempt in range(self.max_retries):
                try:
                    response = self.client.chat.complete(
                        model=model,
                        messages=messages,
                        temperature=temperature,
                        max_tokens=max_tokens
                    )
                    return response.choices[0].message.content
                    
                except Exception as e:
                    error_str = str(e)
                    
                    # Check if it's a rate limit error
                    if "429" in error_str or "capacity" in error_str.lower() or "rate" in error_str.lower():
                        if attempt < self.max_retries - 1:
                            wait_time = self.retry_delay * (attempt + 1)
                            print(f"⚠️ Rate limit hit on {model}, waiting {wait_time}s... (attempt {attempt+1}/{self.max_retries})")
                            time.sleep(wait_time)
                        else:
                            print(f"⚠️ {model} capacity exceeded, trying next model...")
                            break  # Try next model
                    else:
                        # Other error, raise it
                        raise
        
        # If all models fail, return a fallback message
        raise Exception("All Mistral models are at capacity. Please try again later or upgrade your API tier.")
    
    def analyze_marketing_data(self, context_data: str, user_query: str = None) -> str:
        """Analyze marketing data using Mistral LLM"""
        system_prompt = """You are an expert marketing analytics consultant specializing in ROI analysis and budget optimization. 

Your expertise includes:
- Analyzing marketing channel performance
- Calculating and interpreting ROI metrics
- Identifying trends and patterns in marketing data
- Providing actionable recommendations for budget allocation
- Performing constrained optimization for maximum efficiency

When analyzing data:
1. Calculate key metrics (ROI, CPA, CTR, conversion rates)
2. Compare performance across channels
3. Identify top and bottom performers
4. Provide specific, data-driven recommendations
5. Consider industry benchmarks

Be precise, analytical, and provide concrete numbers in your analysis."""

        if user_query is None:
            user_query = "Analyze this marketing performance data and provide comprehensive insights."
        
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": f"""Here is the marketing performance data:

{context_data}

{user_query}

Please provide a detailed analysis including:
1. Overall performance summary
2. Channel-by-channel breakdown
3. Top performing channels and why
4. Underperforming channels and potential issues
5. Key insights and patterns
6. Specific recommendations for improvement"""}
        ]
        
        try:
            return self._make_request_with_retry(messages, temperature=0.3, max_tokens=2000)
        except Exception as e:
            # Return basic analysis as fallback
            return self._generate_fallback_analysis(context_data)
    
    def _generate_fallback_analysis(self, context_data: str) -> str:
        """Generate basic analysis without API when rate limited"""
        return f"""# Marketing Performance Analysis (Auto-Generated)

⚠️ **Note**: Advanced AI analysis is temporarily unavailable due to API limits. Here's a basic analysis:

## Data Overview
{context_data}

## Recommendations:
1. **Focus on High ROI Channels**: Prioritize channels with ROI above 200%
2. **Optimize Underperformers**: Review targeting and creative for channels with ROI below 100%
3. **Scale Winners**: Increase budget for channels with consistent positive ROI
4. **Test and Iterate**: Run A/B tests on underperforming channels before cutting budget

💡 For detailed AI-powered insights, please try again in a few minutes or upgrade your Mistral API tier.
"""
    
    def generate_optimization_recommendations(self, channel_metrics: Dict, 
                                             total_budget: float,
                                             current_allocation: Dict) -> str:
        """Generate budget optimization recommendations"""
        context = f"""
Current Marketing Performance:
{json.dumps(channel_metrics, indent=2)}

Total Budget: ${total_budget:,.2f}

Current Budget Allocation:
{json.dumps(current_allocation, indent=2)}

Industry Benchmarks for Reference:
- Email Marketing: 3600% ROI, $13 CPA
- SEO: 2200% ROI, $20 CPA
- Social Media: 300% ROI, $18.68 CPA
- Paid Search: 200% ROI, $40 CPA
- Display Ads: 150% ROI, $25 CPA
"""
        
        query = f"""Based on this marketing performance data and a total budget of ${total_budget:,.2f}, provide:

1. **Optimization Strategy**: How should the budget be reallocated to maximize ROI?
2. **Channel Recommendations**: 
   - Which channels should receive MORE budget and why?
   - Which channels should receive LESS budget and why?
   - Which channels should maintain current levels?
3. **Specific Allocation**: Suggest specific dollar amounts for each channel
4. **Expected Impact**: What improvements can we expect from these changes?
5. **Implementation Timeline**: How should these changes be rolled out?
6. **Risk Factors**: What risks should be considered?

Provide specific numbers and percentages in your recommendations."""
        
        messages = [
            {"role": "system", "content": "You are a marketing budget optimization expert. Provide specific, actionable recommendations with concrete numbers."},
            {"role": "user", "content": f"{context}\n\n{query}"}
        ]
        
        try:
            return self._make_request_with_retry(messages, temperature=0.4, max_tokens=2500)
        except Exception as e:
            return self._generate_fallback_optimization(channel_metrics, total_budget, current_allocation)
    
    def _generate_fallback_optimization(self, channel_metrics: Dict, total_budget: float, current_allocation: Dict) -> str:
        """Generate basic optimization recommendations without API"""
        
        # Sort channels by ROI
        sorted_channels = sorted(channel_metrics.items(), key=lambda x: x[1].get('roi', 0), reverse=True)
        
        output = f"""# Budget Optimization Recommendations (Auto-Generated)

⚠️ **Note**: Advanced AI recommendations are temporarily unavailable. Here's a rule-based analysis:

## Current Performance Summary
Total Budget: ${total_budget:,.2f}

### Channel Performance (sorted by ROI):
"""
        for channel, metrics in sorted_channels:
            output += f"\n**{channel}**\n"
            output += f"- Current ROI: {metrics.get('roi', 0):.2f}%\n"
            output += f"- Current Cost: ${metrics.get('cost', 0):,.2f}\n"
            output += f"- CPA: ${metrics.get('cpa', 0):.2f}\n"
        
        output += "\n## Recommendations:\n\n"
        
        # Top performers
        if sorted_channels:
            top_channel = sorted_channels[0]
            output += f"### 1. Scale Up: {top_channel[0]}\n"
            output += f"- Current ROI: {top_channel[1].get('roi', 0):.2f}%\n"
            output += f"- Recommendation: Increase budget by 20-30%\n"
            output += f"- This channel is your best performer and likely has room to scale\n\n"
        
        # Bottom performers
        if len(sorted_channels) > 1:
            bottom_channel = sorted_channels[-1]
            output += f"### 2. Optimize or Reduce: {bottom_channel[0]}\n"
            output += f"- Current ROI: {bottom_channel[1].get('roi', 0):.2f}%\n"
            output += f"- Recommendation: Review targeting, creative, and audience\n"
            output += f"- Consider reducing budget by 10-20% until performance improves\n\n"
        
        output += """### 3. General Principles:
- Allocate 60-70% of budget to proven high-ROI channels
- Keep 20-30% for optimization of mid-performers
- Reserve 10% for testing new strategies

💡 For detailed AI-powered optimization, please try again later or upgrade your API tier.
"""
        
        return output
    
    def generate_executive_summary(self, analysis: str, optimization: str) -> str:
        """Generate executive summary"""
        prompt = f"""Based on the following marketing analysis and optimization recommendations, create a concise executive summary suitable for C-level executives.

ANALYSIS:
{analysis}

OPTIMIZATION RECOMMENDATIONS:
{optimization}

Create an executive summary that includes:
1. Key Findings (3-5 bullet points)
2. Critical Actions (top 3 priorities)
3. Expected ROI Improvement (specific percentage)
4. Budget Reallocation Summary (high-level)
5. Timeline for Implementation

Keep it under 300 words and focus on actionable insights."""
        
        messages = [
            {"role": "system", "content": "You are a business communications expert. Create clear, concise executive summaries."},
            {"role": "user", "content": prompt}
        ]
        
        try:
            return self._make_request_with_retry(messages, temperature=0.3, max_tokens=500)
        except Exception as e:
            return """# Executive Summary (Auto-Generated)

**Key Findings:**
- Marketing channels show varied ROI performance
- Opportunities exist for budget reallocation
- Data-driven optimization can improve overall returns

**Critical Actions:**
1. Reallocate budget to high-ROI channels
2. Optimize or reduce spend on underperformers
3. Implement continuous testing and measurement

**Next Steps:**
Review detailed analysis and implement recommendations gradually over next 30-60 days.

💡 Full AI-generated summary temporarily unavailable. Try again later.
"""
    
    def answer_specific_question(self, context_data: str, question: str) -> str:
        """Answer a specific question about the marketing data"""
        messages = [
            {"role": "system", "content": "You are a marketing data analyst. Answer questions accurately based on the provided data."},
            {"role": "user", "content": f"""Marketing Data:
{context_data}

Question: {question}

Provide a clear, data-driven answer with specific numbers and insights."""}
        ]
        
        try:
            return self._make_request_with_retry(messages, temperature=0.2, max_tokens=800)
        except Exception as e:
            return f"""Unable to generate AI response due to API limits.

**Your Question:** {question}

**Available Data:**
{context_data[:500]}...

💡 Please review the data above manually or try again later for AI-powered insights.
"""
