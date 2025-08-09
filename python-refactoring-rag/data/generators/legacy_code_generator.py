"""Legacy code generator for creating refactoring examples."""

import json
import random
import uuid
from typing import List, Dict, Any, Tuple
from dataclasses import dataclass
import itertools
from datetime import datetime

from config.settings import REFACTORING_PATTERNS


@dataclass
class RefactoringPattern:
    """Refactoring pattern definition."""
    name: str
    description: str
    category: str
    complexity_reduction: float
    readability_improvement: float
    common_triggers: List[str]


class LegacyCodeGenerator:
    """Generates realistic Python legacy code examples with refactoring opportunities."""
    
    def __init__(self):
        self.domains = [
            "web_development", "data_processing", "file_management", "user_management",
            "e_commerce", "financial_calculations", "scientific_computing", "api_integration",
            "database_operations", "machine_learning", "system_administration", "game_development"
        ]
        
        self.refactoring_patterns = [
            RefactoringPattern(
                "extract_method", 
                "Break down long functions into smaller, focused methods",
                "complexity_reduction", 
                0.6, 
                0.8, 
                ["long_function", "nested_loops"]
            ),
            RefactoringPattern(
                "extract_class", 
                "Create classes to group related functionality",
                "organization", 
                0.4, 
                0.7, 
                ["god_function", "data_clumps"]
            ),
            RefactoringPattern(
                "replace_conditional_with_polymorphism", 
                "Use polymorphism instead of complex conditionals",
                "design_patterns", 
                0.5, 
                0.6, 
                ["complex_conditionals", "type_checking"]
            ),
            RefactoringPattern(
                "introduce_parameter_object", 
                "Group related parameters into objects",
                "parameter_management", 
                0.3, 
                0.9, 
                ["long_parameter_list", "data_clumps"]
            ),
            RefactoringPattern(
                "replace_loop_with_comprehension", 
                "Use Python list/dict comprehensions",
                "pythonic_patterns", 
                0.4, 
                0.8, 
                ["manual_loops", "accumulator_pattern"]
            ),
            RefactoringPattern(
                "add_type_hints", 
                "Add type annotations for better code clarity",
                "type_safety", 
                0.2, 
                0.9, 
                ["missing_types", "unclear_interfaces"]
            ),
            RefactoringPattern(
                "improve_error_handling", 
                "Add proper exception handling and validation",
                "robustness", 
                0.3, 
                0.7, 
                ["bare_except", "missing_validation"]
            ),
            RefactoringPattern(
                "eliminate_code_duplication", 
                "Remove duplicated code through extraction",
                "maintainability", 
                0.5, 
                0.8, 
                ["copy_paste", "similar_logic"]
            ),
            RefactoringPattern(
                "improve_naming", 
                "Use descriptive, meaningful variable and function names",
                "readability", 
                0.1, 
                0.9, 
                ["unclear_names", "abbreviations"]
            ),
            RefactoringPattern(
                "optimize_performance", 
                "Improve algorithm efficiency and resource usage",
                "performance", 
                0.3, 
                0.5, 
                ["inefficient_algorithms", "resource_waste"]
            )
        ]
        
        self.code_smells = [
            "long_function", "long_parameter_list", "data_clumps", "primitive_obsession",
            "large_class", "divergent_change", "feature_envy", "inappropriate_intimacy",
            "message_chains", "middle_man", "parallel_inheritance", "lazy_class",
            "speculative_generality", "temporary_field", "refused_bequest", "comments",
            "duplicate_code", "dead_code", "shotgun_surgery", "god_class"
        ]
        
    def generate_dataset(self, num_examples: int = 1000) -> List[Dict[str, Any]]:
        """Generate a comprehensive dataset of legacy code examples."""
        dataset = []
        
        # Ensure balanced distribution across patterns
        examples_per_pattern = num_examples // len(self.refactoring_patterns)
        
        for pattern in self.refactoring_patterns:
            for i in range(examples_per_pattern):
                example = self._generate_example_for_pattern(pattern, len(dataset) + 1)
                dataset.append(example)
        
        # Fill remaining examples with random patterns
        while len(dataset) < num_examples:
            pattern = random.choice(self.refactoring_patterns)
            example = self._generate_example_for_pattern(pattern, len(dataset) + 1)
            dataset.append(example)
        
        # Shuffle to randomize order
        random.shuffle(dataset)
        
        return dataset
    
    def _generate_example_for_pattern(self, pattern: RefactoringPattern, example_id: int) -> Dict[str, Any]:
        """Generate a specific example for a refactoring pattern."""
        
        generators = {
            "extract_method": self._generate_extract_method_example,
            "extract_class": self._generate_extract_class_example,
            "replace_conditional_with_polymorphism": self._generate_polymorphism_example,
            "introduce_parameter_object": self._generate_parameter_object_example,
            "replace_loop_with_comprehension": self._generate_comprehension_example,
            "add_type_hints": self._generate_type_hints_example,
            "improve_error_handling": self._generate_error_handling_example,
            "eliminate_code_duplication": self._generate_deduplication_example,
            "improve_naming": self._generate_naming_example,
            "optimize_performance": self._generate_performance_example
        }
        
        generator = generators.get(pattern.name, self._generate_generic_example)
        original_code, refactored_code, description = generator()
        
        # Calculate complexity metrics
        original_complexity = self._estimate_complexity(original_code)
        refactored_complexity = max(1, int(original_complexity * (1 - pattern.complexity_reduction)))
        
        return {
            "id": f"legacy_example_{example_id:04d}",
            "original_code": original_code,
            "refactored_code": refactored_code,
            "description": description,
            "refactoring_type": pattern.name,
            "complexity_before": original_complexity,
            "complexity_after": refactored_complexity,
            "benefits": self._get_benefits_for_pattern(pattern),
            "tags": self._get_tags_for_pattern(pattern),
            "context": {
                "domain": random.choice(self.domains),
                "function_purpose": description.split('.')[0].lower().replace(' ', '_'),
                "common_pattern": random.choice([True, False]),
                "legacy_indicators": random.sample(pattern.common_triggers, min(2, len(pattern.common_triggers))),
                "generated_timestamp": datetime.now().isoformat()
            },
            "code_smells_detected": random.sample(self.code_smells, random.randint(1, 3)),
            "maintainability_score_before": round(random.uniform(3.0, 6.0), 2),
            "maintainability_score_after": round(random.uniform(7.0, 9.5), 2)
        }
    
    def _generate_extract_method_example(self) -> Tuple[str, str, str]:
        """Generate extract method refactoring example."""
        function_names = ["process_user_data", "calculate_report", "handle_file_upload", "validate_form"]
        function_name = random.choice(function_names)
        
        original = f"""def {function_name}(data):
    # Validation
    if not data:
        return None
    if not isinstance(data, dict):
        return None
    if 'id' not in data:
        return None
    if not data['id']:
        return None
    
    # Processing
    results = []
    for item in data.get('items', []):
        if item.get('active'):
            processed_item = {{}}
            processed_item['id'] = item['id']
            processed_item['name'] = item['name'].strip().lower()
            processed_item['value'] = float(item.get('value', 0))
            if processed_item['value'] > 0:
                processed_item['category'] = 'positive'
            else:
                processed_item['category'] = 'negative'
            processed_item['timestamp'] = item.get('timestamp')
            results.append(processed_item)
    
    # Sorting and filtering
    results.sort(key=lambda x: x['value'], reverse=True)
    filtered_results = []
    for result in results:
        if result['value'] > 10:
            filtered_results.append(result)
    
    # Final formatting
    formatted_results = []
    for result in filtered_results:
        formatted_item = {{
            'id': result['id'],
            'display_name': result['name'].title(),
            'amount': f"${{result['value']:.2f}}",
            'type': result['category']
        }}
        formatted_results.append(formatted_item)
    
    return formatted_results"""

        refactored = f'''def {function_name}(data):
    """Process user data with validation, processing, and formatting."""
    if not _is_valid_data(data):
        return None
    
    processed_items = _process_items(data.get('items', []))
    filtered_items = _filter_valuable_items(processed_items)
    return _format_results(filtered_items)

def _is_valid_data(data):
    """Validate input data structure."""
    return (data and 
            isinstance(data, dict) and 
            data.get('id'))

def _process_items(items):
    """Process individual items and calculate derived values."""
    processed = []
    for item in items:
        if not item.get('active'):
            continue
            
        processed_item = {{
            'id': item['id'],
            'name': item['name'].strip().lower(),
            'value': float(item.get('value', 0)),
            'category': 'positive' if float(item.get('value', 0)) > 0 else 'negative',
            'timestamp': item.get('timestamp')
        }}
        processed.append(processed_item)
    
    return sorted(processed, key=lambda x: x['value'], reverse=True)

def _filter_valuable_items(items):
    """Filter items based on value threshold."""
    return [item for item in items if item['value'] > 10]

def _format_results(items):
    """Format items for display."""
    return [{{
        'id': item['id'],
        'display_name': item['name'].title(),
        'amount': f"${{item['value']:.2f}}",
        'type': item['category']
    }} for item in items]'''
        
        description = f"Break down the monolithic {function_name} function into smaller, focused methods"
        return original, refactored, description
    
    def _generate_extract_class_example(self) -> Tuple[str, str, str]:
        """Generate extract class refactoring example."""
        original = '''def manage_inventory(action, item_data):
    """Manages inventory operations - adding, removing, updating items"""
    inventory = []  # This would be loaded from database
    
    if action == 'add':
        # Validate item data
        if not item_data.get('name') or not item_data.get('price'):
            return False
        if item_data['price'] < 0:
            return False
        
        # Check for duplicates
        for item in inventory:
            if item['name'] == item_data['name']:
                return False
        
        # Add item
        new_item = {
            'id': len(inventory) + 1,
            'name': item_data['name'],
            'price': item_data['price'],
            'quantity': item_data.get('quantity', 0),
            'category': item_data.get('category', 'general')
        }
        inventory.append(new_item)
        return True
    
    elif action == 'remove':
        item_id = item_data.get('id')
        for i, item in enumerate(inventory):
            if item['id'] == item_id:
                del inventory[i]
                return True
        return False
    
    return False'''

        refactored = '''from typing import List, Dict, Optional, Any
from dataclasses import dataclass

@dataclass
class InventoryItem:
    id: int
    name: str
    price: float
    quantity: int = 0
    category: str = 'general'
    
    def __post_init__(self):
        if self.price < 0:
            raise ValueError("Price cannot be negative")

class InventoryManager:
    """Manages inventory operations with proper separation of concerns."""
    
    def __init__(self):
        self.items: List[InventoryItem] = []
        self._next_id = 1
    
    def add_item(self, item_data: Dict[str, Any]) -> bool:
        """Add a new item to inventory."""
        if not self._validate_item_data(item_data):
            return False
        
        if self._item_exists(item_data['name']):
            return False
        
        try:
            item = InventoryItem(
                id=self._next_id,
                name=item_data['name'],
                price=item_data['price'],
                quantity=item_data.get('quantity', 0),
                category=item_data.get('category', 'general')
            )
            self.items.append(item)
            self._next_id += 1
            return True
        except ValueError:
            return False
    
    def remove_item(self, item_id: int) -> bool:
        """Remove an item from inventory."""
        for i, item in enumerate(self.items):
            if item.id == item_id:
                del self.items[i]
                return True
        return False
    
    def _validate_item_data(self, data: Dict[str, Any]) -> bool:
        """Validate item data before adding."""
        return bool(data.get('name') and data.get('price') is not None)
    
    def _item_exists(self, name: str) -> bool:
        """Check if item with given name already exists."""
        return any(item.name == name for item in self.items)'''
        
        description = "Extract inventory management logic into a dedicated class with proper data structures"
        return original, refactored, description
    
    def _generate_polymorphism_example(self) -> Tuple[str, str, str]:
        """Generate polymorphism refactoring example."""
        original = '''def process_payment(payment_type, amount, details):
    """Process different types of payments"""
    if payment_type == 'credit_card':
        # Validate credit card
        if not details.get('card_number') or len(details['card_number']) != 16:
            return False
        if not details.get('cvv') or len(details['cvv']) != 3:
            return False
        
        # Process credit card payment
        fee = amount * 0.03  # 3% fee
        total = amount + fee
        
        result = {
            'status': 'success',
            'transaction_id': f"cc_{details['card_number'][-4:]}_{amount}",
            'amount': amount,
            'fee': fee,
            'total': total
        }
        return result
    
    elif payment_type == 'paypal':
        # Validate PayPal
        if not details.get('email'):
            return False
        
        # Process PayPal payment
        fee = amount * 0.025  # 2.5% fee
        total = amount + fee
        
        result = {
            'status': 'success',
            'transaction_id': f"pp_{details['email'].split('@')[0]}_{amount}",
            'amount': amount,
            'fee': fee,
            'total': total
        }
        return result
    
    else:
        return {'status': 'error', 'message': 'Unsupported payment type'}'''

        refactored = '''from abc import ABC, abstractmethod
from typing import Dict, Any

class PaymentProcessor(ABC):
    """Abstract base class for payment processors."""
    
    @abstractmethod
    def validate_details(self, details: Dict[str, Any]) -> bool:
        """Validate payment details."""
        pass
    
    @abstractmethod
    def calculate_fee(self, amount: float) -> float:
        """Calculate processing fee."""
        pass
    
    @abstractmethod
    def generate_transaction_id(self, amount: float, details: Dict[str, Any]) -> str:
        """Generate unique transaction ID."""
        pass
    
    def process_payment(self, amount: float, details: Dict[str, Any]) -> Dict[str, Any]:
        """Process payment with common logic."""
        if not self.validate_details(details):
            return {'status': 'error', 'message': 'Invalid payment details'}
        
        fee = self.calculate_fee(amount)
        total = amount + fee
        transaction_id = self.generate_transaction_id(amount, details)
        
        return {
            'status': 'success',
            'transaction_id': transaction_id,
            'amount': amount,
            'fee': fee,
            'total': total
        }

class CreditCardProcessor(PaymentProcessor):
    """Credit card payment processor."""
    
    def validate_details(self, details: Dict[str, Any]) -> bool:
        card_number = details.get('card_number', '')
        cvv = details.get('cvv', '')
        return len(card_number) == 16 and len(cvv) == 3
    
    def calculate_fee(self, amount: float) -> float:
        return amount * 0.03
    
    def generate_transaction_id(self, amount: float, details: Dict[str, Any]) -> str:
        return f"cc_{details['card_number'][-4:]}_{amount}"

class PayPalProcessor(PaymentProcessor):
    """PayPal payment processor."""
    
    def validate_details(self, details: Dict[str, Any]) -> bool:
        return bool(details.get('email'))
    
    def calculate_fee(self, amount: float) -> float:
        return amount * 0.025
    
    def generate_transaction_id(self, amount: float, details: Dict[str, Any]) -> str:
        return f"pp_{details['email'].split('@')[0]}_{amount}"'''
        
        description = "Replace complex conditional logic with polymorphic payment processors"
        return original, refactored, description
    
    def _generate_parameter_object_example(self) -> Tuple[str, str, str]:
        """Generate parameter object refactoring example."""
        original = '''def create_user_account(first_name, last_name, email, phone, address_line1, 
                      address_line2, city, state, zip_code, country, birth_date, 
                      subscription_type, payment_method, marketing_consent, 
                      terms_accepted, referral_code):
    """Create a new user account with all the details"""
    
    # Validate required fields
    if not first_name or not last_name or not email:
        return None
    
    if not terms_accepted:
        return None
    
    # Format data
    full_name = f"{first_name} {last_name}"
    full_address = f"{address_line1}, {address_line2 or ''}, {city}, {state} {zip_code}, {country}"
    
    # Create account
    account = {
        'id': generate_user_id(),
        'personal_info': {
            'first_name': first_name,
            'last_name': last_name,
            'full_name': full_name,
            'email': email,
            'phone': phone,
            'birth_date': birth_date
        },
        'address': {
            'line1': address_line1,
            'line2': address_line2,
            'city': city,
            'state': state,
            'zip_code': zip_code,
            'country': country,
            'full_address': full_address
        },
        'subscription': {
            'type': subscription_type,
            'payment_method': payment_method
        },
        'preferences': {
            'marketing_consent': marketing_consent,
            'terms_accepted': terms_accepted
        },
        'referral_code': referral_code
    }
    
    return account'''

        refactored = '''from dataclasses import dataclass
from typing import Optional
from datetime import date

@dataclass
class PersonalInfo:
    first_name: str
    last_name: str
    email: str
    phone: Optional[str] = None
    birth_date: Optional[date] = None
    
    @property
    def full_name(self) -> str:
        return f"{self.first_name} {self.last_name}"

@dataclass
class Address:
    line1: str
    city: str
    state: str
    zip_code: str
    country: str
    line2: Optional[str] = None
    
    @property
    def full_address(self) -> str:
        line2_part = f", {self.line2}" if self.line2 else ""
        return f"{self.line1}{line2_part}, {self.city}, {self.state} {self.zip_code}, {self.country}"

@dataclass
class SubscriptionInfo:
    type: str
    payment_method: str

@dataclass
class UserPreferences:
    marketing_consent: bool
    terms_accepted: bool

@dataclass
class UserAccountData:
    personal_info: PersonalInfo
    address: Address
    subscription: SubscriptionInfo
    preferences: UserPreferences
    referral_code: Optional[str] = None

def create_user_account(account_data: UserAccountData) -> Optional[dict]:
    """Create a new user account with structured data"""
    
    # Validate required fields
    if not account_data.personal_info.first_name or not account_data.personal_info.last_name:
        return None
    
    if not account_data.personal_info.email:
        return None
    
    if not account_data.preferences.terms_accepted:
        return None
    
    return {
        'id': generate_user_id(),
        'personal_info': {
            'first_name': account_data.personal_info.first_name,
            'last_name': account_data.personal_info.last_name,
            'full_name': account_data.personal_info.full_name,
            'email': account_data.personal_info.email,
            'phone': account_data.personal_info.phone,
            'birth_date': account_data.personal_info.birth_date
        },
        'address': {
            'line1': account_data.address.line1,
            'line2': account_data.address.line2,
            'city': account_data.address.city,
            'state': account_data.address.state,
            'zip_code': account_data.address.zip_code,
            'country': account_data.address.country,
            'full_address': account_data.address.full_address
        },
        'subscription': {
            'type': account_data.subscription.type,
            'payment_method': account_data.subscription.payment_method
        },
        'preferences': {
            'marketing_consent': account_data.preferences.marketing_consent,
            'terms_accepted': account_data.preferences.terms_accepted
        },
        'referral_code': account_data.referral_code
    }

def generate_user_id():
    """Generate unique user ID"""
    import uuid
    return str(uuid.uuid4())'''
        
        description = "Group related parameters into structured data objects to reduce parameter list complexity"
        return original, refactored, description
    
    def _generate_comprehension_example(self) -> Tuple[str, str, str]:
        """Generate list comprehension refactoring example."""
        original = '''def process_orders(orders):
    """Process orders and calculate totals"""
    valid_orders = []
    for order in orders:
        if order.get('status') == 'pending' and order.get('amount', 0) > 0:
            valid_orders.append(order)
    
    order_totals = []
    for order in valid_orders:
        total = order['amount'] * (1 + order.get('tax_rate', 0.1))
        order_totals.append({
            'id': order['id'],
            'customer': order['customer'],
            'total': round(total, 2)
        })
    
    return order_totals'''
             
        refactored = '''def process_orders(orders):
    """Process orders and calculate totals using comprehensions"""
    return [
        {
            'id': order['id'],
            'customer': order['customer'],
            'total': round(order['amount'] * (1 + order.get('tax_rate', 0.1)), 2)
        }
        for order in orders
        if order.get('status') == 'pending' and order.get('amount', 0) > 0
    ]'''
        
        description = "Replace manual loops with Python comprehensions for more concise and readable code"
        return original, refactored, description
    
    def _generate_type_hints_example(self) -> Tuple[str, str, str]:
        """Generate type hints refactoring example."""
        original = '''def calculate_statistics(data, include_median=False):
    """Calculate basic statistics for a dataset"""
    if not data:
        return None
    
    total = sum(data)
    count = len(data)
    mean = total / count
    
    sorted_data = sorted(data)
    minimum = sorted_data[0]
    maximum = sorted_data[-1]
    
    result = {
        'mean': mean,
        'min': minimum,
        'max': maximum,
        'count': count,
        'sum': total
    }
    
    if include_median:
        mid = count // 2
        if count % 2 == 0:
            median = (sorted_data[mid-1] + sorted_data[mid]) / 2
        else:
            median = sorted_data[mid]
        result['median'] = median
    
    return result'''

        refactored = '''from typing import List, Dict, Union, Optional

def calculate_statistics(
    data: List[Union[int, float]], 
    include_median: bool = False
) -> Optional[Dict[str, float]]:
    """Calculate basic statistics for a dataset.
    
    Args:
        data: List of numeric values
        include_median: Whether to include median in results
        
    Returns:
        Dictionary with statistical measures or None if data is empty
    """
    if not data:
        return None
    
    total = sum(data)
    count = len(data)
    mean = total / count
    
    sorted_data = sorted(data)
    minimum = sorted_data[0]
    maximum = sorted_data[-1]
    
    result: Dict[str, float] = {
        'mean': mean,
        'min': minimum,
        'max': maximum,
        'count': count,
        'sum': total
    }
    
    if include_median:
        mid = count // 2
        if count % 2 == 0:
            median = (sorted_data[mid-1] + sorted_data[mid]) / 2
        else:
            median = sorted_data[mid]
        result['median'] = median
    
    return result'''
        
        description = "Add comprehensive type hints and improve docstrings for better code clarity and IDE support"
        return original, refactored, description
    
    def _generate_error_handling_example(self) -> Tuple[str, str, str]:
        """Generate error handling refactoring example."""
        original = '''def read_config_file(filename):
    """Read configuration from file"""
    try:
        file = open(filename, 'r')
        content = file.read()
        file.close()
        
        config = {}
        lines = content.split('\\n')
        for line in lines:
            if '=' in line:
                key, value = line.split('=')
                config[key.strip()] = value.strip()
        
        return config
    except:
        return {}'''

        refactored = '''import json
import logging
from typing import Dict, Optional
from pathlib import Path

logger = logging.getLogger(__name__)

def read_config_file(filename: str) -> Dict[str, str]:
    """Read configuration from file with proper error handling.
    
    Args:
        filename: Path to configuration file
        
    Returns:
        Dictionary of configuration values
        
    Raises:
        FileNotFoundError: If config file doesn't exist
        PermissionError: If file cannot be read
        ValueError: If file format is invalid
    """
    config_path = Path(filename)
    
    if not config_path.exists():
        raise FileNotFoundError(f"Configuration file not found: {filename}")
    
    if not config_path.is_file():
        raise ValueError(f"Path is not a file: {filename}")
    
    try:
        with config_path.open('r', encoding='utf-8') as file:
            content = file.read()
    except PermissionError:
        logger.error(f"Permission denied reading config file: {filename}")
        raise
    except UnicodeDecodeError as e:
        logger.error(f"Invalid encoding in config file: {filename}")
        raise ValueError(f"Invalid file encoding: {e}")
    
    config = {}
    for line_num, line in enumerate(content.split('\\n'), 1):
        line = line.strip()
        if not line or line.startswith('#'):
            continue
            
        if '=' not in line:
            logger.warning(f"Invalid config line {line_num}: {line}")
            continue
            
        try:
            key, value = line.split('=', 1)
            config[key.strip()] = value.strip()
        except ValueError:
            logger.warning(f"Could not parse config line {line_num}: {line}")
    
    return config'''
        
        description = "Add comprehensive error handling, validation, and logging for robust code"
        return original, refactored, description
    
    def _generate_deduplication_example(self) -> Tuple[str, str, str]:
        """Generate code deduplication refactoring example."""
        original = '''def calculate_employee_bonus(employee_type, salary, performance_rating):
    """Calculate bonus for different employee types"""
    if employee_type == 'manager':
        base_bonus = salary * 0.15
        if performance_rating >= 4.5:
            performance_multiplier = 1.5
        elif performance_rating >= 4.0:
            performance_multiplier = 1.2
        elif performance_rating >= 3.5:
            performance_multiplier = 1.0
        else:
            performance_multiplier = 0.8
        
        final_bonus = base_bonus * performance_multiplier
        if final_bonus > 50000:
            final_bonus = 50000
        
        return final_bonus
    
    elif employee_type == 'developer':
        base_bonus = salary * 0.12
        if performance_rating >= 4.5:
            performance_multiplier = 1.5
        elif performance_rating >= 4.0:
            performance_multiplier = 1.2
        elif performance_rating >= 3.5:
            performance_multiplier = 1.0
        else:
            performance_multiplier = 0.8
        
        final_bonus = base_bonus * performance_multiplier
        if final_bonus > 40000:
            final_bonus = 40000
        
        return final_bonus
    
    else:
        return 0'''

        refactored = '''from typing import Dict

def calculate_employee_bonus(employee_type: str, salary: float, performance_rating: float) -> float:
    """Calculate bonus for different employee types using configurable parameters."""
    
    bonus_config = {
        'manager': {'base_rate': 0.15, 'max_bonus': 50000},
        'developer': {'base_rate': 0.12, 'max_bonus': 40000},
        'sales': {'base_rate': 0.18, 'max_bonus': 60000}
    }
    
    if employee_type not in bonus_config:
        return 0
    
    config = bonus_config[employee_type]
    base_bonus = salary * config['base_rate']
    performance_multiplier = _get_performance_multiplier(performance_rating)
    final_bonus = base_bonus * performance_multiplier
    
    return min(final_bonus, config['max_bonus'])

def _get_performance_multiplier(rating: float) -> float:
    """Calculate performance multiplier based on rating."""
    if rating >= 4.5:
        return 1.5
    elif rating >= 4.0:
        return 1.2
    elif rating >= 3.5:
        return 1.0
    else:
        return 0.8'''
        
        description = "Eliminate code duplication by extracting common logic and using configuration data"
        return original, refactored, description
    
    def _generate_naming_example(self) -> Tuple[str, str, str]:
        """Generate naming improvement refactoring example."""
        original = '''def proc_usr_data(d):
    """Process user data"""
    res = []
    for x in d:
        if x['st'] == 'a':
            tmp = {}
            tmp['id'] = x['uid']
            tmp['n'] = x['fn'] + ' ' + x['ln']
            tmp['e'] = x['em']
            tmp['ph'] = x['p']
            tmp['addr'] = x['a1'] + ', ' + x['ct'] + ', ' + x['st_code']
            
            # Calculate age
            from datetime import datetime
            bd = datetime.strptime(x['bd'], '%Y-%m-%d')
            today = datetime.now()
            age = today.year - bd.year - ((today.month, today.day) < (bd.month, bd.day))
            tmp['ag'] = age
            
            res.append(tmp)
    
    return res'''

        refactored = '''from datetime import datetime
from typing import List, Dict, Any

def process_active_users(user_data_list: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Process and format data for active users.
    
    Args:
        user_data_list: List of raw user data dictionaries
        
    Returns:
        List of processed user records with standardized fields
    """
    processed_users = []
    
    for user_record in user_data_list:
        if user_record['status'] != 'active':
            continue
            
        processed_user = {
            'user_id': user_record['user_id'],
            'full_name': f"{user_record['first_name']} {user_record['last_name']}",
            'email': user_record['email'],
            'phone': user_record['phone'],
            'address': _format_user_address(user_record),
            'age': _calculate_age_from_birthdate(user_record['birth_date'])
        }
        
        processed_users.append(processed_user)
    
    return processed_users

def _format_user_address(user_record: Dict[str, Any]) -> str:
    """Format user address from individual components."""
    return f"{user_record['address_line1']}, {user_record['city']}, {user_record['state_code']}"

def _calculate_age_from_birthdate(birth_date_string: str) -> int:
    """Calculate current age from birth date string."""
    birth_date = datetime.strptime(birth_date_string, '%Y-%m-%d')
    current_date = datetime.now()
    
    age = current_date.year - birth_date.year
    if (current_date.month, current_date.day) < (birth_date.month, birth_date.day):
        age -= 1
    
    return age'''
        
        description = "Improve variable and function names for better code readability and maintainability"
        return original, refactored, description
    
    def _generate_performance_example(self) -> Tuple[str, str, str]:
        """Generate performance optimization refactoring example."""
        original = '''def find_duplicates_in_lists(list1, list2):
    """Find common elements between two lists"""
    duplicates = []
    for item1 in list1:
        for item2 in list2:
            if item1 == item2 and item1 not in duplicates:
                duplicates.append(item1)
    return duplicates'''

        refactored = '''from typing import List, Any

def find_duplicates_in_lists(list1: List[Any], list2: List[Any]) -> List[Any]:
    """Find common elements between two lists using set intersection.
    
    Time complexity: O(n + m) instead of O(n * m)
    """
    return list(set(list1) & set(list2))'''
        
        description = "Optimize performance using efficient algorithms and Python built-ins"
        return original, refactored, description
    
    def _generate_generic_example(self) -> Tuple[str, str, str]:
        """Generate a generic refactoring example."""
        return self._generate_extract_method_example()
    
    def _estimate_complexity(self, code: str) -> int:
        """Estimate cyclomatic complexity of code."""
        complexity = 1  # Base complexity
        
        # Count decision points
        decision_keywords = ['if', 'elif', 'while', 'for', 'except', 'and', 'or']
        for keyword in decision_keywords:
            complexity += code.count(keyword)
        
        return max(1, complexity)
    
    def _get_benefits_for_pattern(self, pattern: RefactoringPattern) -> List[str]:
        """Get benefits list for a refactoring pattern."""
        benefit_mapping = {
            "extract_method": ["reduced_complexity", "improved_readability", "better_testability"],
            "extract_class": ["better_organization", "single_responsibility", "reduced_coupling"],
            "replace_conditional_with_polymorphism": ["extensibility", "reduced_complexity", "open_closed_principle"],
            "introduce_parameter_object": ["parameter_clarity", "data_cohesion", "reduced_coupling"],
            "replace_loop_with_comprehension": ["pythonic_code", "better_performance", "improved_readability"],
            "add_type_hints": ["type_safety", "better_ide_support", "documentation"],
            "improve_error_handling": ["robustness", "better_debugging", "user_experience"],
            "eliminate_code_duplication": ["maintainability", "consistency", "reduced_bugs"],
            "improve_naming": ["readability", "self_documenting_code", "reduced_confusion"],
            "optimize_performance": ["better_performance", "resource_efficiency", "scalability"]
        }
        
        return benefit_mapping.get(pattern.name, ["improved_code_quality", "better_maintainability"])
    
    def _get_tags_for_pattern(self, pattern: RefactoringPattern) -> List[str]:
        """Get tags for a refactoring pattern."""
        tag_mapping = {
            "extract_method": ["functions", "complexity", "organization"],
            "extract_class": ["classes", "organization", "oop"],
            "replace_conditional_with_polymorphism": ["polymorphism", "design_patterns", "conditionals"],
            "introduce_parameter_object": ["parameters", "data_structures", "organization"],
            "replace_loop_with_comprehension": ["loops", "comprehensions", "pythonic"],
            "add_type_hints": ["types", "documentation", "static_analysis"],
            "improve_error_handling": ["exceptions", "robustness", "validation"],
            "eliminate_code_duplication": ["duplication", "dry_principle", "maintainability"],
            "improve_naming": ["naming", "readability", "conventions"],
            "optimize_performance": ["performance", "algorithms", "efficiency"]
        }
        
        base_tags = tag_mapping.get(pattern.name, ["refactoring", "code_quality"])
        base_tags.append("legacy_code")
        return base_tags


def save_dataset_to_json(dataset: List[Dict[str, Any]], filename: str) -> None:
    """Save the generated dataset to a JSON file."""
    with open(filename, 'w', encoding='utf-8') as f:
        json.dump(dataset, f, indent=2, ensure_ascii=False, default=str)


def generate_dataset_statistics(dataset: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Generate statistics about the dataset."""
    from collections import Counter
    
    stats = {
        'total_examples': len(dataset),
        'refactoring_types': Counter(item['refactoring_type'] for item in dataset),
        'domains': Counter(item['context']['domain'] for item in dataset),
        'complexity_distribution': {
            'before': {
                'mean': sum(item['complexity_before'] for item in dataset) / len(dataset),
                'min': min(item['complexity_before'] for item in dataset),
                'max': max(item['complexity_before'] for item in dataset)
            },
            'after': {
                'mean': sum(item['complexity_after'] for item in dataset) / len(dataset),
                'min': min(item['complexity_after'] for item in dataset),
                'max': max(item['complexity_after'] for item in dataset)
            }
        },
        'benefits_frequency': Counter(
            benefit for item in dataset for benefit in item['benefits']
        ),
        'code_smells_frequency': Counter(
            smell for item in dataset for smell in item['code_smells_detected']
        )
    }
    
    return stats


if __name__ == "__main__":
    # Set random seed for reproducible results
    random.seed(42)
    
    # Generate dataset
    generator = LegacyCodeGenerator()
    dataset = generator.generate_dataset(2200)
    Dataset_Path = "Inputs/Dataset/python_legacy_refactoring_dataset.json"
    Stats_Path = "Inputs/Dataset/dataset_statistics.json"
    # Save dataset
    save_dataset_to_json(dataset, Dataset_Path)
    
    # Generate statistics
    stats = generate_dataset_statistics(dataset)
    save_dataset_to_json(stats, Stats_Path)
    
    print(f"Generated {len(dataset)} examples")
    print(f"Refactoring types: {list(stats['refactoring_types'].keys())}")
    print(f"Average complexity reduction: {stats['complexity_distribution']['before']['mean'] - stats['complexity_distribution']['after']['mean']:.2f}")
    print(f"Dataset saved to {Dataset_Path}")
    print(f"Statistics saved to {Stats_Path}")