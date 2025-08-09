"""
Query optimization for efficient data access and processing.
"""

import logging
import time
from typing import Dict, Any, List, Optional, Callable, Union, Tuple
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
import hashlib
import json

logger = logging.getLogger(__name__)


class QueryType(Enum):
    """Types of queries"""
    SELECT = "select"
    FILTER = "filter"
    AGGREGATE = "aggregate"
    JOIN = "join"
    SEARCH = "search"
    ANALYTICS = "analytics"


@dataclass
class QueryPlan:
    """Optimized query execution plan"""
    query_id: str
    query_type: QueryType
    operations: List[Dict[str, Any]]
    estimated_cost: float
    estimated_time: float
    cache_key: Optional[str] = None
    parallel_operations: List[List[int]] = field(default_factory=list)
    optimization_hints: Dict[str, Any] = field(default_factory=dict)


@dataclass
class QueryStats:
    """Query execution statistics"""
    total_queries: int = 0
    cache_hits: int = 0
    avg_execution_time: float = 0.0
    queries_by_type: Dict[str, int] = field(default_factory=dict)
    optimization_savings: float = 0.0


class QueryOptimizer:
    """Intelligent query optimizer with caching and parallel execution"""
    
    def __init__(self, cache_optimizer=None):
        self.cache_optimizer = cache_optimizer
        self.query_cache = None
        if cache_optimizer:
            self.query_cache = cache_optimizer.create_cache(
                name="query_cache",
                max_size=1000,
                max_memory_mb=50,
                ttl_seconds=300  # 5 minutes
            )
            
        self.stats = QueryStats()
        self._query_patterns = {}
        self._optimization_rules = []
        self._initialize_optimization_rules()
        
    def optimize_query(self, query: Dict[str, Any]) -> QueryPlan:
        """Optimize query and create execution plan"""
        
        query_id = self._generate_query_id(query)
        query_type = self._identify_query_type(query)
        
        # Check cache first
        if self.query_cache:
            cached_plan = self.query_cache.get(f"plan_{query_id}")
            if cached_plan:
                self.stats.cache_hits += 1
                return cached_plan
                
        # Generate optimization plan
        operations = self._extract_operations(query)
        optimized_ops = self._optimize_operations(operations, query_type)
        parallel_ops = self._identify_parallel_operations(optimized_ops)
        
        # Estimate costs
        estimated_cost = self._estimate_cost(optimized_ops)
        estimated_time = self._estimate_time(optimized_ops, parallel_ops)
        
        # Create query plan
        plan = QueryPlan(
            query_id=query_id,
            query_type=query_type,
            operations=optimized_ops,
            estimated_cost=estimated_cost,
            estimated_time=estimated_time,
            cache_key=f"result_{query_id}",
            parallel_operations=parallel_ops,
            optimization_hints=self._generate_optimization_hints(query, optimized_ops)
        )
        
        # Cache the plan
        if self.query_cache:
            self.query_cache.put(f"plan_{query_id}", plan, priority=2)
            
        # Update statistics
        self.stats.total_queries += 1
        query_type_str = query_type.value
        self.stats.queries_by_type[query_type_str] = self.stats.queries_by_type.get(query_type_str, 0) + 1
        
        return plan
        
    def execute_plan(self, plan: QueryPlan, data_source: Any, executor: Optional[Callable] = None) -> Any:
        """Execute optimized query plan"""
        
        start_time = time.perf_counter()
        
        # Check result cache
        if self.query_cache and plan.cache_key:
            cached_result = self.query_cache.get(plan.cache_key)
            if cached_result:
                return cached_result
                
        # Execute query plan
        if executor:
            result = executor(plan, data_source)
        else:
            result = self._default_executor(plan, data_source)
            
        execution_time = time.perf_counter() - start_time
        
        # Update statistics
        self._update_execution_stats(execution_time, plan)
        
        # Cache result if beneficial
        if self.query_cache and plan.cache_key and execution_time > 0.1:  # Cache slow queries
            cache_ttl = min(300, max(60, int(execution_time * 100)))  # Dynamic TTL
            self.query_cache.put(
                plan.cache_key, 
                result, 
                ttl_seconds=cache_ttl,
                cost_to_compute=execution_time
            )
            
        return result
        
    def add_optimization_rule(self, rule_func: Callable, priority: int = 1):
        """Add custom optimization rule"""
        self._optimization_rules.append({
            "func": rule_func,
            "priority": priority
        })
        
        # Sort by priority
        self._optimization_rules.sort(key=lambda x: x["priority"], reverse=True)
        
    def analyze_query_patterns(self) -> Dict[str, Any]:
        """Analyze query patterns for optimization insights"""
        
        pattern_analysis = {
            "common_patterns": {},
            "optimization_opportunities": [],
            "performance_insights": {}
        }
        
        # Analyze common query patterns
        for pattern, count in self._query_patterns.items():
            if count > 10:  # Frequent pattern
                pattern_analysis["common_patterns"][pattern] = count
                
        # Identify optimization opportunities
        if self.stats.cache_hits / max(1, self.stats.total_queries) < 0.3:
            pattern_analysis["optimization_opportunities"].append(
                "Low cache hit rate - consider longer TTL or better cache keys"
            )
            
        if self.stats.avg_execution_time > 1.0:
            pattern_analysis["optimization_opportunities"].append(
                "High average execution time - consider query restructuring"
            )
            
        # Performance insights
        pattern_analysis["performance_insights"] = {
            "total_queries": self.stats.total_queries,
            "cache_hit_rate": self.stats.cache_hits / max(1, self.stats.total_queries),
            "avg_execution_time": self.stats.avg_execution_time,
            "optimization_savings": self.stats.optimization_savings
        }
        
        return pattern_analysis
        
    def get_query_statistics(self) -> QueryStats:
        """Get query execution statistics"""
        return self.stats
        
    def _generate_query_id(self, query: Dict[str, Any]) -> str:
        """Generate unique query identifier"""
        query_str = json.dumps(query, sort_keys=True)
        return hashlib.md5(query_str.encode()).hexdigest()[:16]
        
    def _identify_query_type(self, query: Dict[str, Any]) -> QueryType:
        """Identify the type of query"""
        
        if "select" in query or "fields" in query:
            return QueryType.SELECT
        elif "filter" in query or "where" in query:
            return QueryType.FILTER
        elif "group_by" in query or "aggregate" in query:
            return QueryType.AGGREGATE
        elif "join" in query:
            return QueryType.JOIN
        elif "search" in query or "text" in query:
            return QueryType.SEARCH
        else:
            return QueryType.ANALYTICS
            
    def _extract_operations(self, query: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Extract operations from query"""
        
        operations = []
        
        # Data loading operation
        if "source" in query:
            operations.append({
                "type": "load",
                "source": query["source"],
                "estimated_cost": 1.0
            })
            
        # Filter operations
        if "filter" in query:
            for filter_condition in query["filter"]:
                operations.append({
                    "type": "filter",
                    "condition": filter_condition,
                    "estimated_cost": 0.5
                })
                
        # Selection operations
        if "select" in query:
            operations.append({
                "type": "select",
                "fields": query["select"],
                "estimated_cost": 0.2
            })
            
        # Aggregation operations
        if "aggregate" in query:
            operations.append({
                "type": "aggregate",
                "functions": query["aggregate"],
                "estimated_cost": 2.0
            })
            
        # Join operations
        if "join" in query:
            operations.append({
                "type": "join",
                "tables": query["join"],
                "estimated_cost": 5.0
            })
            
        # Sort operations
        if "order_by" in query:
            operations.append({
                "type": "sort",
                "fields": query["order_by"],
                "estimated_cost": 1.5
            })
            
        return operations
        
    def _optimize_operations(self, operations: List[Dict[str, Any]], query_type: QueryType) -> List[Dict[str, Any]]:
        """Apply optimization rules to operations"""
        
        optimized = operations.copy()
        
        # Apply built-in optimization rules
        for rule in self._optimization_rules:
            try:
                optimized = rule["func"](optimized, query_type)
            except Exception as e:
                logger.warning(f"Optimization rule failed: {e}")
                
        return optimized
        
    def _identify_parallel_operations(self, operations: List[Dict[str, Any]]) -> List[List[int]]:
        """Identify operations that can be executed in parallel"""
        
        parallel_groups = []
        
        # Simple parallelization: filters can often run in parallel
        filter_ops = [i for i, op in enumerate(operations) if op["type"] == "filter"]
        if len(filter_ops) > 1:
            parallel_groups.append(filter_ops)
            
        # Aggregations on different fields can be parallel
        agg_ops = [i for i, op in enumerate(operations) if op["type"] == "aggregate"]
        if len(agg_ops) > 1:
            parallel_groups.append(agg_ops)
            
        return parallel_groups
        
    def _estimate_cost(self, operations: List[Dict[str, Any]]) -> float:
        """Estimate computational cost of operations"""
        
        total_cost = 0.0
        
        for op in operations:
            base_cost = op.get("estimated_cost", 1.0)
            
            # Apply complexity multipliers
            if op["type"] == "join":
                base_cost *= 2  # Joins are expensive
            elif op["type"] == "sort":
                base_cost *= 1.5  # Sorting is moderately expensive
                
            total_cost += base_cost
            
        return total_cost
        
    def _estimate_time(self, operations: List[Dict[str, Any]], parallel_ops: List[List[int]]) -> float:
        """Estimate execution time considering parallelization"""
        
        # Calculate sequential time
        sequential_time = sum(op.get("estimated_cost", 1.0) for op in operations)
        
        # Account for parallelization savings
        parallel_savings = 0.0
        for parallel_group in parallel_ops:
            if len(parallel_group) > 1:
                group_cost = sum(operations[i].get("estimated_cost", 1.0) for i in parallel_group)
                max_op_cost = max(operations[i].get("estimated_cost", 1.0) for i in parallel_group)
                parallel_savings += group_cost - max_op_cost
                
        optimized_time = max(0.1, sequential_time - parallel_savings * 0.7)  # 70% parallel efficiency
        return optimized_time
        
    def _generate_optimization_hints(self, query: Dict[str, Any], operations: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Generate optimization hints for the query"""
        
        hints = {}
        
        # Index hints
        if any(op["type"] == "filter" for op in operations):
            hints["consider_indexes"] = "Adding indexes on filtered fields could improve performance"
            
        # Caching hints
        if len(operations) > 5:
            hints["enable_caching"] = "Complex query - results should be cached"
            
        # Parallelization hints
        filter_count = sum(1 for op in operations if op["type"] == "filter")
        if filter_count > 2:
            hints["parallel_filters"] = f"Consider parallelizing {filter_count} filter operations"
            
        return hints
        
    def _default_executor(self, plan: QueryPlan, data_source: Any) -> Any:
        """Default query plan executor"""
        
        result = data_source
        
        for operation in plan.operations:
            if operation["type"] == "load":
                # Data loading simulation
                result = self._simulate_data_load(operation["source"])
            elif operation["type"] == "filter":
                result = self._simulate_filter(result, operation["condition"])
            elif operation["type"] == "select":
                result = self._simulate_select(result, operation["fields"])
            elif operation["type"] == "aggregate":
                result = self._simulate_aggregate(result, operation["functions"])
            elif operation["type"] == "join":
                result = self._simulate_join(result, operation["tables"])
            elif operation["type"] == "sort":
                result = self._simulate_sort(result, operation["fields"])
                
        return result
        
    def _simulate_data_load(self, source: str) -> Dict[str, Any]:
        """Simulate data loading"""
        return {"data": f"loaded_from_{source}", "rows": 1000}
        
    def _simulate_filter(self, data: Any, condition: Dict[str, Any]) -> Any:
        """Simulate filtering operation"""
        if isinstance(data, dict) and "rows" in data:
            # Simulate filtering reducing row count
            data["rows"] = int(data["rows"] * 0.7)  # 30% filtered out
        return data
        
    def _simulate_select(self, data: Any, fields: List[str]) -> Any:
        """Simulate field selection"""
        if isinstance(data, dict):
            data["selected_fields"] = fields
        return data
        
    def _simulate_aggregate(self, data: Any, functions: List[str]) -> Any:
        """Simulate aggregation"""
        if isinstance(data, dict):
            data["aggregated"] = True
            data["functions"] = functions
        return data
        
    def _simulate_join(self, data: Any, tables: List[str]) -> Any:
        """Simulate join operation"""
        if isinstance(data, dict):
            data["joined_tables"] = tables
        return data
        
    def _simulate_sort(self, data: Any, fields: List[str]) -> Any:
        """Simulate sorting"""
        if isinstance(data, dict):
            data["sorted_by"] = fields
        return data
        
    def _update_execution_stats(self, execution_time: float, plan: QueryPlan):
        """Update execution statistics"""
        
        # Update average execution time
        if self.stats.total_queries > 0:
            self.stats.avg_execution_time = (
                (self.stats.avg_execution_time * (self.stats.total_queries - 1) + execution_time) 
                / self.stats.total_queries
            )
        else:
            self.stats.avg_execution_time = execution_time
            
        # Calculate optimization savings
        estimated_original_time = plan.estimated_cost  # Without optimization
        if estimated_original_time > execution_time:
            savings = estimated_original_time - execution_time
            self.stats.optimization_savings += savings
            
    def _initialize_optimization_rules(self):
        """Initialize built-in optimization rules"""
        
        def filter_pushdown(operations: List[Dict[str, Any]], query_type: QueryType) -> List[Dict[str, Any]]:
            """Push filter operations earlier in the pipeline"""
            
            optimized = []
            filters = []
            others = []
            
            for op in operations:
                if op["type"] == "filter":
                    filters.append(op)
                else:
                    others.append(op)
                    
            # Put filters first (except after load)
            load_ops = [op for op in others if op["type"] == "load"]
            non_load_ops = [op for op in others if op["type"] != "load"]
            
            optimized.extend(load_ops)
            optimized.extend(filters)
            optimized.extend(non_load_ops)
            
            return optimized
            
        def combine_filters(operations: List[Dict[str, Any]], query_type: QueryType) -> List[Dict[str, Any]]:
            """Combine multiple filter operations"""
            
            optimized = []
            combined_filters = []
            
            for op in operations:
                if op["type"] == "filter":
                    combined_filters.append(op["condition"])
                else:
                    if combined_filters:
                        # Add combined filter operation
                        optimized.append({
                            "type": "filter",
                            "condition": {"and": combined_filters},
                            "estimated_cost": len(combined_filters) * 0.4  # Slight efficiency gain
                        })
                        combined_filters = []
                    optimized.append(op)
                    
            # Handle trailing filters
            if combined_filters:
                optimized.append({
                    "type": "filter", 
                    "condition": {"and": combined_filters},
                    "estimated_cost": len(combined_filters) * 0.4
                })
                
            return optimized
            
        # Register optimization rules
        self.add_optimization_rule(filter_pushdown, priority=10)
        self.add_optimization_rule(combine_filters, priority=8)