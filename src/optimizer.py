# src/optimizer.py
import numpy as np
import pandas as pd
from scipy.optimize import minimize, LinearConstraint, Bounds
from typing import Optional

class BudgetOptimizer:
    def __init__(self, channel_metrics: pd.DataFrame, total_budget: float):
        """Initialize budget optimizer"""
        if len(channel_metrics) == 0:
            raise ValueError("No channel metrics provided")
        
        if total_budget <= 0:
            raise ValueError("Total budget must be positive")
        
        self.channel_metrics = channel_metrics
        self.total_budget = total_budget
        self.optimal_allocation = None
        self.improvement_summary = None
        
    def validate_constraints(self, min_allocation_pct: float, max_allocation_pct: float) -> tuple[float, float]:
        """Validate and adjust constraints if needed"""
        n_channels = len(self.channel_metrics)
        
        # Check if minimum allocation is feasible
        min_total = n_channels * min_allocation_pct
        if min_total > 1.0:
            # Adjust to make it feasible
            adjusted_min = 0.95 / n_channels  # Leave 5% room for optimization
            print(f"⚠️ Minimum allocation too high ({min_allocation_pct*100:.1f}% × {n_channels} channels = {min_total*100:.1f}%)")
            print(f"   Adjusting minimum to {adjusted_min*100:.1f}% per channel")
            min_allocation_pct = adjusted_min
        
        # Check if max allocation is reasonable
        if max_allocation_pct < min_allocation_pct:
            print(f"⚠️ Maximum allocation ({max_allocation_pct*100:.1f}%) is less than minimum ({min_allocation_pct*100:.1f}%)")
            max_allocation_pct = min(min_allocation_pct + 0.2, 1.0)  # Add 20% room
            print(f"   Adjusting maximum to {max_allocation_pct*100:.1f}%")
        
        # Ensure at least one channel can get max allocation
        if max_allocation_pct * n_channels < 1.0:
            print(f"⚠️ Maximum allocation too restrictive")
            max_allocation_pct = min(1.0 / n_channels + 0.3, 1.0)
            print(f"   Adjusting maximum to {max_allocation_pct*100:.1f}%")
        
        return min_allocation_pct, max_allocation_pct
        
    def optimize(self, min_allocation_pct: float = 0.05, max_allocation_pct: float = 0.50) -> pd.DataFrame:
        """Perform constrained optimization to maximize conversions"""
        
        # Validate and adjust constraints
        min_allocation_pct, max_allocation_pct = self.validate_constraints(min_allocation_pct, max_allocation_pct)
        
        channels = self.channel_metrics['channel'].tolist()
        n_channels = len(channels)
        
        print(f"\nOptimizing budget for {n_channels} channels:")
        print(f"  Total Budget: ${self.total_budget:,.2f}")
        print(f"  Min per channel: {min_allocation_pct*100:.1f}% (${self.total_budget * min_allocation_pct:,.2f})")
        print(f"  Max per channel: {max_allocation_pct*100:.1f}% (${self.total_budget * max_allocation_pct:,.2f})")
        
        # Calculate expected conversions per dollar
        conversion_rates = (self.channel_metrics['conversions'] / 
                          self.channel_metrics['cost']).values
        
        # Handle division by zero or invalid rates
        conversion_rates = np.nan_to_num(conversion_rates, nan=0.0, posinf=0.0, neginf=0.0)
        
        # If all conversion rates are zero, use equal distribution
        if np.sum(conversion_rates) == 0:
            print("⚠️ No valid conversion data, using equal distribution")
            equal_allocation = self.total_budget / n_channels
            
            self.optimal_allocation = pd.DataFrame({
                'channel': channels,
                'current_cost': self.channel_metrics['cost'].values,
                'optimal_budget': [equal_allocation] * n_channels,
                'budget_change': equal_allocation - self.channel_metrics['cost'].values,
                'budget_change_pct': ((equal_allocation - self.channel_metrics['cost'].values) / 
                                     self.channel_metrics['cost'].values) * 100,
                'expected_conversions': [0] * n_channels,
                'expected_revenue': [0] * n_channels,
                'expected_roi': [0.0] * n_channels
            })
            
            self._calculate_improvement_summary()
            return self.optimal_allocation
        
        # Objective: Maximize conversions (minimize negative conversions)
        def objective(budget_allocation):
            return -np.sum(budget_allocation * conversion_rates)
        
        # Gradient of objective function
        def jacobian(budget_allocation):
            return -conversion_rates
        
        # Constraints
        # 1. Sum of budgets must equal total budget (with small tolerance)
        budget_constraint = LinearConstraint(
            A=np.ones(n_channels),
            lb=self.total_budget * 0.999,  # 0.1% tolerance
            ub=self.total_budget * 1.001
        )
        
        # 2. Each channel must get between min and max percentage
        min_budget = self.total_budget * min_allocation_pct
        max_budget = self.total_budget * max_allocation_pct
        
        bounds = Bounds(
            lb=np.full(n_channels, min_budget),
            ub=np.full(n_channels, max_budget)
        )
        
        # Initial guess: distribute based on historical ROI with adjustments
        roi_values = self.channel_metrics['roi'].values
        roi_values = np.nan_to_num(roi_values, nan=0.0, posinf=0.0, neginf=0.0)
        
        # Ensure non-negative and add small offset to avoid zeros
        roi_weights = np.maximum(roi_values + 1, 0.1)  # Add 1 to handle negative ROI, minimum 0.1
        roi_weights = roi_weights / roi_weights.sum()
        
        x0 = roi_weights * self.total_budget
        
        # Ensure initial guess respects bounds
        x0 = np.clip(x0, min_budget, max_budget)
        
        # Adjust to meet budget constraint exactly
        x0 = x0 * (self.total_budget / x0.sum())
        
        print(f"  Initial allocation sum: ${x0.sum():,.2f}")
        
        # Try optimization with different methods
        methods = ['SLSQP', 'trust-constr']
        result = None
        
        for method in methods:
            try:
                print(f"  Attempting optimization with {method}...")
                
                if method == 'SLSQP':
                    result = minimize(
                        fun=objective,
                        x0=x0,
                        method=method,
                        jac=jacobian,
                        bounds=bounds,
                        constraints=[budget_constraint],
                        options={'maxiter': 1000, 'ftol': 1e-6}
                    )
                else:  # trust-constr
                    result = minimize(
                        fun=objective,
                        x0=x0,
                        method=method,
                        jac=jacobian,
                        bounds=bounds,
                        constraints=[budget_constraint],
                        options={'maxiter': 1000}
                    )
                
                if result.success:
                    print(f"  ✓ Optimization successful with {method}")
                    break
                else:
                    print(f"  ✗ {method} failed: {result.message}")
                    
            except Exception as e:
                print(f"  ✗ {method} error: {str(e)}")
                continue
        
        # If optimization failed, use proportional allocation
        if result is None or not result.success:
            print("⚠️ Optimization failed, using proportional allocation based on ROI")
            
            # Use ROI-weighted allocation
            weights = np.maximum(roi_values, 0)
            if weights.sum() == 0:
                weights = np.ones(n_channels)
            weights = weights / weights.sum()
            
            optimal_budgets = weights * self.total_budget
            optimal_budgets = np.clip(optimal_budgets, min_budget, max_budget)
            optimal_budgets = optimal_budgets * (self.total_budget / optimal_budgets.sum())
            
        else:
            optimal_budgets = result.x
        
        # Store results
        self.optimal_allocation = pd.DataFrame({
            'channel': channels,
            'current_cost': self.channel_metrics['cost'].values,
            'optimal_budget': optimal_budgets,
            'budget_change': optimal_budgets - self.channel_metrics['cost'].values,
            'budget_change_pct': np.where(
                self.channel_metrics['cost'].values > 0,
                ((optimal_budgets - self.channel_metrics['cost'].values) / 
                 self.channel_metrics['cost'].values) * 100,
                0
            ),
            'expected_conversions': optimal_budgets * conversion_rates,
            'expected_revenue': optimal_budgets * (self.channel_metrics['revenue'] / 
                                           self.channel_metrics['cost']).replace([np.inf, -np.inf], 0).fillna(0).values,
            'expected_roi': np.where(
                optimal_budgets > 0,
                ((optimal_budgets * (self.channel_metrics['revenue'] / 
                 self.channel_metrics['cost']).replace([np.inf, -np.inf], 0).fillna(0).values) - optimal_budgets) / optimal_budgets * 100,
                0
            )
        })
        
        # Calculate improvement summary
        self._calculate_improvement_summary()
        
        print(f"\n✓ Optimization complete!")
        print(f"  Final allocation sum: ${self.optimal_allocation['optimal_budget'].sum():,.2f}")
        
        return self.optimal_allocation
    
    def _calculate_improvement_summary(self):
        """Calculate improvement metrics"""
        current_conversions = self.channel_metrics['conversions'].sum()
        optimal_conversions = self.optimal_allocation['expected_conversions'].sum()
        
        current_cost = self.channel_metrics['cost'].sum()
        current_revenue = self.channel_metrics['revenue'].sum()
        current_roi = ((current_revenue - current_cost) / current_cost * 100) if current_cost > 0 else 0
        
        optimal_revenue = self.optimal_allocation['expected_revenue'].sum()
        optimal_roi = ((optimal_revenue - self.total_budget) / self.total_budget * 100) if self.total_budget > 0 else 0
        
        conversion_improvement = optimal_conversions - current_conversions
        conversion_improvement_pct = ((conversion_improvement / current_conversions) * 100) if current_conversions > 0 else 0
        
        self.improvement_summary = {
            'current_conversions': float(current_conversions),
            'optimal_conversions': float(optimal_conversions),
            'conversion_improvement': float(conversion_improvement),
            'conversion_improvement_pct': float(conversion_improvement_pct),
            'current_roi': float(current_roi),
            'optimal_roi': float(optimal_roi),
            'roi_improvement': float(optimal_roi - current_roi)
        }
