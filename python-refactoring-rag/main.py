"""Enhanced main entry point with comprehensive evaluation and improved interactive mode."""

import logging
import os
import sys
from pathlib import Path
from typing import Dict, Any, List

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler('rag_system.log')
    ]
)

logger = logging.getLogger(__name__)

# Load environment variables
try:
    from dotenv import load_dotenv
    load_dotenv(dotenv_path="key.env")
except ImportError:
    logger.warning("python-dotenv not available. Ensure environment variables are set manually.")

# Import system components
from config.model_configs import get_default_llm_configs
from config.settings import PDFConfig, get_default_config
from services.rag_service import RefactoringRAGSystem
from models.evaluation.rag_evaluator import RAGEvaluator
from models.optimization.nsga2_optimizer import run_nsga2_optimization
from fixed_nsga2_optimizer import run_fixed_nsga2_optimization
from utils.logging_utils import setup_enhanced_logging

# Try to import enhanced display manager
try:
    from interactive_display_manager import create_interactive_session
    ENHANCED_DISPLAY_AVAILABLE = True
except ImportError:
    ENHANCED_DISPLAY_AVAILABLE = False
    logger.warning("Enhanced display manager not found. Using built-in interactive mode.")


def get_enhanced_test_cases():
    """
    Complete test cases covering all 10 refactoring patterns from the dataset generator.
    This ensures comprehensive evaluation against the actual dataset patterns.
    """
    return [
        # 1. EXTRACT_METHOD - Break down long functions
        {
            'query': 'How can I refactor this long function to be more readable and maintainable?',
            'original_code': '''def process_customer_order(order_data):
    # Input validation
    if not order_data:
        return {'error': 'No order data provided'}
    if not order_data.get('customer_id'):
        return {'error': 'Customer ID required'}
    if not order_data.get('items') or len(order_data['items']) == 0:
        return {'error': 'Order must contain items'}
    
    # Calculate totals
    subtotal = 0
    for item in order_data['items']:
        if item.get('quantity', 0) <= 0:
            return {'error': f'Invalid quantity for item {item.get("name", "unknown")}'}
        if item.get('price', 0) <= 0:
            return {'error': f'Invalid price for item {item.get("name", "unknown")}'}
        item_total = item['quantity'] * item['price']
        subtotal += item_total
    
    # Apply discounts
    discount = 0
    if order_data.get('discount_code'):
        if order_data['discount_code'] == 'SAVE10':
            discount = subtotal * 0.10
        elif order_data['discount_code'] == 'SAVE20':
            discount = subtotal * 0.20
        elif order_data['discount_code'] == 'NEWCUSTOMER':
            discount = min(subtotal * 0.15, 50)
    
    # Calculate tax
    tax_rate = 0.08
    if order_data.get('shipping_state') == 'CA':
        tax_rate = 0.10
    elif order_data.get('shipping_state') == 'NY':
        tax_rate = 0.09
    
    discounted_subtotal = subtotal - discount
    tax = discounted_subtotal * tax_rate
    total = discounted_subtotal + tax
    
    # Process payment
    result = {
        'order_id': f"ORD-{order_data['customer_id']}-{len(order_data['items'])}",
        'subtotal': round(subtotal, 2),
        'discount': round(discount, 2),
        'tax': round(tax, 2),
        'total': round(total, 2),
        'status': 'processed'
    }
    
    return result''',
            'reference_answer': """Break down this monolithic function into smaller, focused functions:

```python
def process_customer_order(order_data):
    if not _validate_order_data(order_data):
        return _get_validation_error(order_data)
    
    subtotal = _calculate_subtotal(order_data['items'])
    if isinstance(subtotal, dict):  # Error case
        return subtotal
    
    discount = _calculate_discount(subtotal, order_data.get('discount_code'))
    tax = _calculate_tax(subtotal - discount, order_data.get('shipping_state'))
    
    return _build_order_result(order_data, subtotal, discount, tax)

def _validate_order_data(order_data):
    return (order_data and 
            order_data.get('customer_id') and 
            order_data.get('items') and 
            len(order_data['items']) > 0)

def _calculate_subtotal(items):
    subtotal = 0
    for item in items:
        if item.get('quantity', 0) <= 0 or item.get('price', 0) <= 0:
            return {'error': f'Invalid item data for {item.get("name", "unknown")}'}
        subtotal += item['quantity'] * item['price']
    return subtotal
```

This refactoring improves readability by giving each function a single responsibility."""
        },

        # 2. EXTRACT_CLASS - Create classes to group functionality
        {
            'query': 'This function is handling too many responsibilities. How can I organize it into a proper class structure?',
            'original_code': '''def manage_library_system(action, book_data=None, member_data=None, book_id=None, member_id=None):
    books = load_books()
    members = load_members()
    
    if action == 'add_book':
        if not book_data.get('title') or not book_data.get('author'):
            return {'error': 'Title and author required'}
        
        new_book = {
            'id': len(books) + 1,
            'title': book_data['title'],
            'author': book_data['author'],
            'isbn': book_data.get('isbn'),
            'available': True,
            'borrowed_by': None
        }
        books.append(new_book)
        save_books(books)
        return {'success': True, 'book': new_book}
    
    elif action == 'checkout_book':
        book = next((b for b in books if b['id'] == book_id), None)
        member = next((m for m in members if m['id'] == member_id), None)
        
        if not book or not member:
            return {'error': 'Book or member not found'}
        if not book['available']:
            return {'error': 'Book not available'}
        
        book['available'] = False
        book['borrowed_by'] = member_id
        member['borrowed_books'].append(book_id)
        
        save_books(books)
        save_members(members)
        return {'success': True}
    
    elif action == 'add_member':
        if not member_data.get('name') or not member_data.get('email'):
            return {'error': 'Name and email required'}
        
        new_member = {
            'id': len(members) + 1,
            'name': member_data['name'],
            'email': member_data['email'],
            'borrowed_books': []
        }
        members.append(new_member)
        save_members(members)
        return {'success': True, 'member': new_member}''',
            'reference_answer': """Extract this into separate classes with clear responsibilities:

```python
class Book:
    def __init__(self, id, title, author, isbn=None):
        self.id = id
        self.title = title
        self.author = author
        self.isbn = isbn
        self.available = True
        self.borrowed_by = None

class Member:
    def __init__(self, id, name, email):
        self.id = id
        self.name = name
        self.email = email
        self.borrowed_books = []

class LibraryManager:
    def __init__(self):
        self.books = self._load_books()
        self.members = self._load_members()
    
    def add_book(self, book_data):
        if not self._validate_book_data(book_data):
            return {'error': 'Title and author required'}
        
        book = Book(
            len(self.books) + 1,
            book_data['title'],
            book_data['author'],
            book_data.get('isbn')
        )
        self.books.append(book)
        self._save_books()
        return {'success': True, 'book': book}
```

This separates data models from business logic and makes the code more maintainable."""
        },

        # 3. REPLACE_CONDITIONAL_WITH_POLYMORPHISM
        {
            'query': 'How can I replace these complex conditional statements with a more extensible design?',
            'original_code': '''def calculate_shipping_cost(package_type, weight, distance, delivery_speed):
    if package_type == 'standard':
        base_cost = 5.00
        if delivery_speed == 'express':
            base_cost *= 1.5
        elif delivery_speed == 'overnight':
            base_cost *= 2.0
        
        weight_cost = weight * 0.50
        distance_cost = distance * 0.02
        return base_cost + weight_cost + distance_cost
    
    elif package_type == 'fragile':
        base_cost = 8.00
        if delivery_speed == 'express':
            base_cost *= 1.8
        elif delivery_speed == 'overnight':
            base_cost *= 2.5
        
        weight_cost = weight * 0.75
        distance_cost = distance * 0.03
        fragile_surcharge = 3.00
        return base_cost + weight_cost + distance_cost + fragile_surcharge
    
    elif package_type == 'hazardous':
        base_cost = 15.00
        if delivery_speed == 'standard':
            # Hazardous can only be express or overnight
            return None
        elif delivery_speed == 'express':
            base_cost *= 2.0
        elif delivery_speed == 'overnight':
            base_cost *= 3.0
        
        weight_cost = weight * 1.00
        distance_cost = distance * 0.05
        hazmat_fee = 10.00
        return base_cost + weight_cost + distance_cost + hazmat_fee''',
            'reference_answer': """Use polymorphism to eliminate complex conditionals:

```python
from abc import ABC, abstractmethod

class ShippingCalculator(ABC):
    @abstractmethod
    def calculate_cost(self, weight, distance, delivery_speed):
        pass

class StandardShipping(ShippingCalculator):
    def calculate_cost(self, weight, distance, delivery_speed):
        base_cost = 5.00 * self._get_speed_multiplier(delivery_speed)
        return base_cost + (weight * 0.50) + (distance * 0.02)
    
    def _get_speed_multiplier(self, speed):
        return {'standard': 1.0, 'express': 1.5, 'overnight': 2.0}[speed]

class FragileShipping(ShippingCalculator):
    def calculate_cost(self, weight, distance, delivery_speed):
        base_cost = 8.00 * self._get_speed_multiplier(delivery_speed)
        return base_cost + (weight * 0.75) + (distance * 0.03) + 3.00
```

This approach makes adding new package types easy without modifying existing code."""
        },

        # 4. INTRODUCE_PARAMETER_OBJECT
        {
            'query': 'This function has too many parameters. How can I simplify the parameter list?',
            'original_code': '''def create_employee_profile(first_name, last_name, email, phone, department, 
                            position, salary, start_date, manager_id, office_location,
                            emergency_contact_name, emergency_contact_phone, 
                            benefits_plan, vacation_days, sick_days, remote_work_eligible):
    
    # Validate required fields
    if not first_name or not last_name or not email:
        return None
    
    if not department or not position:
        return None
    
    # Format data
    full_name = f"{first_name} {last_name}"
    employee_id = f"{department[:3].upper()}{len(all_employees) + 1:04d}"
    
    # Create profile
    profile = {
        'employee_id': employee_id,
        'personal_info': {
            'first_name': first_name,
            'last_name': last_name,
            'full_name': full_name,
            'email': email,
            'phone': phone
        },
        'employment_info': {
            'department': department,
            'position': position,
            'salary': salary,
            'start_date': start_date,
            'manager_id': manager_id,
            'office_location': office_location
        },
        'emergency_contact': {
            'name': emergency_contact_name,
            'phone': emergency_contact_phone
        },
        'benefits': {
            'plan': benefits_plan,
            'vacation_days': vacation_days,
            'sick_days': sick_days,
            'remote_eligible': remote_work_eligible
        }
    }
    
    return profile''',
            'reference_answer': """Group related parameters into data structures:

```python
from dataclasses import dataclass
from typing import Optional
from datetime import date

@dataclass
class PersonalInfo:
    first_name: str
    last_name: str
    email: str
    phone: str
    
    @property
    def full_name(self) -> str:
        return f"{self.first_name} {self.last_name}"

@dataclass
class EmploymentInfo:
    department: str
    position: str
    salary: float
    start_date: date
    manager_id: Optional[str]
    office_location: str

@dataclass
class EmergencyContact:
    name: str
    phone: str

@dataclass
class BenefitsInfo:
    plan: str
    vacation_days: int
    sick_days: int
    remote_eligible: bool

@dataclass
class EmployeeData:
    personal: PersonalInfo
    employment: EmploymentInfo
    emergency_contact: EmergencyContact
    benefits: BenefitsInfo

def create_employee_profile(employee_data: EmployeeData) -> Optional[dict]:
    # Now validation and processing is much cleaner
    if not employee_data.personal.first_name or not employee_data.personal.email:
        return None
    
    employee_id = f"{employee_data.employment.department[:3].upper()}{len(all_employees) + 1:04d}"
    # ... rest of processing
```

This eliminates the long parameter list and groups related data logically."""
        },

        # 5. REPLACE_LOOP_WITH_COMPREHENSION
        {
            'query': 'How can I make these loops more Pythonic and concise?',
            'original_code': '''def process_sales_data(sales_records):
    # Filter active sales
    active_sales = []
    for sale in sales_records:
        if sale.get('status') == 'completed' and sale.get('amount', 0) > 0:
            active_sales.append(sale)
    
    # Calculate commission for each sale
    commissioned_sales = []
    for sale in active_sales:
        commission_rate = 0.05 if sale.get('amount', 0) < 1000 else 0.08
        commissioned_sale = {
            'sale_id': sale['id'],
            'amount': sale['amount'],
            'commission': sale['amount'] * commission_rate,
            'salesperson': sale['salesperson']
        }
        commissioned_sales.append(commissioned_sale)
    
    # Group by salesperson
    grouped_sales = {}
    for sale in commissioned_sales:
        person = sale['salesperson']
        if person not in grouped_sales:
            grouped_sales[person] = []
        grouped_sales[person].append(sale)
    
    # Calculate totals per salesperson
    salesperson_totals = {}
    for person, sales in grouped_sales.items():
        total_sales = 0
        total_commission = 0
        for sale in sales:
            total_sales += sale['amount']
            total_commission += sale['commission']
        
        salesperson_totals[person] = {
            'total_sales': total_sales,
            'total_commission': total_commission,
            'sale_count': len(sales)
        }
    
    return salesperson_totals''',
            'reference_answer': """Use Python comprehensions and built-in functions:

```python
from collections import defaultdict

def process_sales_data(sales_records):
    # Filter and transform in one comprehension
    commissioned_sales = [
        {
            'sale_id': sale['id'],
            'amount': sale['amount'],
            'commission': sale['amount'] * (0.05 if sale['amount'] < 1000 else 0.08),
            'salesperson': sale['salesperson']
        }
        for sale in sales_records
        if sale.get('status') == 'completed' and sale.get('amount', 0) > 0
    ]
    
    # Group using defaultdict and comprehension
    grouped_sales = defaultdict(list)
    for sale in commissioned_sales:
        grouped_sales[sale['salesperson']].append(sale)
    
    # Calculate totals with comprehension
    return {
        person: {
            'total_sales': sum(sale['amount'] for sale in sales),
            'total_commission': sum(sale['commission'] for sale in sales),
            'sale_count': len(sales)
        }
        for person, sales in grouped_sales.items()
    }
```

This is more concise and leverages Python's powerful comprehension syntax."""
        },

        # 6. ADD_TYPE_HINTS
        {
            'query': 'How can I add proper type annotations to improve code clarity and catch errors?',
            'original_code': '''def analyze_user_behavior(user_data, start_date, end_date, metrics=None):
    """Analyze user behavior patterns"""
    if not user_data or not start_date:
        return None
    
    if metrics is None:
        metrics = ['page_views', 'session_duration', 'bounce_rate']
    
    results = {}
    for user_id, sessions in user_data.items():
        user_metrics = {}
        filtered_sessions = []
        
        for session in sessions:
            session_date = session['date']
            if start_date <= session_date <= end_date:
                filtered_sessions.append(session)
        
        if not filtered_sessions:
            continue
        
        if 'page_views' in metrics:
            total_views = sum(session.get('page_views', 0) for session in filtered_sessions)
            user_metrics['avg_page_views'] = total_views / len(filtered_sessions)
        
        if 'session_duration' in metrics:
            total_duration = sum(session.get('duration', 0) for session in filtered_sessions)
            user_metrics['avg_session_duration'] = total_duration / len(filtered_sessions)
        
        results[user_id] = user_metrics
    
    return results''',
            'reference_answer': """Add comprehensive type hints for better clarity:

```python
from typing import Dict, List, Optional, Union, Any
from datetime import date, datetime

SessionData = Dict[str, Union[str, int, float, date]]
UserData = Dict[str, List[SessionData]]
MetricsResult = Dict[str, float]
AnalysisResult = Dict[str, MetricsResult]

def analyze_user_behavior(
    user_data: UserData,
    start_date: date,
    end_date: date,
    metrics: Optional[List[str]] = None
) -> Optional[AnalysisResult]:
    \"\"\"Analyze user behavior patterns within date range.
    
    Args:
        user_data: Dictionary mapping user IDs to session data
        start_date: Analysis period start date
        end_date: Analysis period end date  
        metrics: List of metrics to calculate (defaults to standard set)
        
    Returns:
        Dictionary of user metrics or None if invalid input
    \"\"\"
    if not user_data or not start_date:
        return None
    
    if metrics is None:
        metrics = ['page_views', 'session_duration', 'bounce_rate']
    
    results: AnalysisResult = {}
    
    for user_id, sessions in user_data.items():
        filtered_sessions = _filter_sessions_by_date(sessions, start_date, end_date)
        if not filtered_sessions:
            continue
            
        user_metrics = _calculate_user_metrics(filtered_sessions, metrics)
        results[user_id] = user_metrics
    
    return results

def _filter_sessions_by_date(
    sessions: List[SessionData], 
    start_date: date, 
    end_date: date
) -> List[SessionData]:
    \"\"\"Filter sessions within date range.\"\"\"
    return [
        session for session in sessions
        if start_date <= session['date'] <= end_date
    ]
```

Type hints make the function interface clear and enable better IDE support."""
        },

        # 7. IMPROVE_ERROR_HANDLING
        {
            'query': 'How can I add proper error handling and validation to make this code more robust?',
            'original_code': '''def process_payment_data(payment_file, output_file):
    """Process payment data from CSV file"""
    
    file = open(payment_file, 'r')
    lines = file.readlines()
    file.close()
    
    processed_payments = []
    
    for line in lines[1:]:  # Skip header
        parts = line.strip().split(',')
        
        payment_id = parts[0]
        amount = float(parts[1])
        currency = parts[2]
        date_str = parts[3]
        
        # Convert to USD
        if currency == 'EUR':
            usd_amount = amount * 1.1
        elif currency == 'GBP':
            usd_amount = amount * 1.3
        else:
            usd_amount = amount
        
        processed_payment = {
            'id': payment_id,
            'original_amount': amount,
            'currency': currency,
            'usd_amount': usd_amount,
            'date': date_str
        }
        
        processed_payments.append(processed_payment)
    
    # Save results
    output = open(output_file, 'w')
    for payment in processed_payments:
        output.write(f"{payment['id']},{payment['usd_amount']},{payment['date']}\\n")
    output.close()
    
    return len(processed_payments)''',
            'reference_answer': """Add comprehensive error handling and validation:

```python
import csv
import logging
from typing import List, Dict, Any
from pathlib import Path
from decimal import Decimal, InvalidOperation

logger = logging.getLogger(__name__)

class PaymentProcessingError(Exception):
    \"\"\"Custom exception for payment processing errors.\"\"\"
    pass

def process_payment_data(payment_file: str, output_file: str) -> int:
    \"\"\"Process payment data with proper error handling.
    
    Args:
        payment_file: Path to input CSV file
        output_file: Path to output CSV file
        
    Returns:
        Number of successfully processed payments
        
    Raises:
        FileNotFoundError: If input file doesn't exist
        PaymentProcessingError: If processing fails
        PermissionError: If files cannot be accessed
    \"\"\"
    
    # Validate input files
    input_path = Path(payment_file)
    if not input_path.exists():
        raise FileNotFoundError(f"Payment file not found: {payment_file}")
    
    if not input_path.is_file():
        raise PaymentProcessingError(f"Input path is not a file: {payment_file}")
    
    try:
        processed_payments = _read_and_process_payments(input_path)
        _write_processed_payments(processed_payments, output_file)
        return len(processed_payments)
        
    except PermissionError:
        logger.error(f"Permission denied accessing files")
        raise
    except Exception as e:
        logger.error(f"Unexpected error processing payments: {e}")
        raise PaymentProcessingError(f"Processing failed: {e}")

def _read_and_process_payments(file_path: Path) -> List[Dict[str, Any]]:
    \"\"\"Read and validate payment data.\"\"\"
    processed_payments = []
    
    try:
        with file_path.open('r', encoding='utf-8') as file:
            reader = csv.DictReader(file)
            
            for row_num, row in enumerate(reader, 2):  # Start at 2 for header
                try:
                    payment = _process_payment_row(row, row_num)
                    processed_payments.append(payment)
                except ValueError as e:
                    logger.warning(f"Skipping invalid row {row_num}: {e}")
                    continue
                    
    except UnicodeDecodeError:
        raise PaymentProcessingError("Invalid file encoding")
    
    return processed_payments

def _process_payment_row(row: Dict[str, str], row_num: int) -> Dict[str, Any]:
    \"\"\"Process and validate a single payment row.\"\"\"
    try:
        payment_id = row.get('payment_id', '').strip()
        if not payment_id:
            raise ValueError("Missing payment ID")
        
        amount_str = row.get('amount', '').strip()
        if not amount_str:
            raise ValueError("Missing amount")
            
        amount = Decimal(amount_str)
        if amount <= 0:
            raise ValueError("Amount must be positive")
        
        currency = row.get('currency', '').strip().upper()
        if currency not in ['USD', 'EUR', 'GBP']:
            raise ValueError(f"Unsupported currency: {currency}")
        
        usd_amount = _convert_to_usd(amount, currency)
        
        return {
            'id': payment_id,
            'original_amount': float(amount),
            'currency': currency,
            'usd_amount': float(usd_amount),
            'date': row.get('date', '').strip()
        }
        
    except (InvalidOperation, ValueError) as e:
        raise ValueError(f"Row {row_num} validation failed: {e}")
```

This adds proper validation, logging, and graceful error handling."""
        },

        # 8. ELIMINATE_CODE_DUPLICATION (already in original test cases but adding dataset-specific version)
        {
            'query': 'I notice a lot of repeated code patterns. How can I eliminate this duplication?',
            'original_code': '''def generate_user_report(user_type):
    """Generate reports for different user types"""
    if user_type == 'admin':
        # Get admin data
        users = get_admin_users()
        
        # Validate data
        if not users:
            return "No admin users found"
        
        # Format data
        report_data = []
        for user in users:
            formatted_user = {
                'name': user['name'].title(),
                'email': user['email'].lower(),
                'last_login': format_date(user['last_login']),
                'permissions': ', '.join(user['permissions'])
            }
            report_data.append(formatted_user)
        
        # Generate report
        report = "ADMIN USER REPORT\\n"
        report += "=" * 50 + "\\n"
        for user in report_data:
            report += f"Name: {user['name']}\\n"
            report += f"Email: {user['email']}\\n"
            report += f"Last Login: {user['last_login']}\\n"
            report += f"Permissions: {user['permissions']}\\n"
            report += "-" * 30 + "\\n"
        
        return report
    
    elif user_type == 'customer':
        # Get customer data
        users = get_customer_users()
        
        # Validate data
        if not users:
            return "No customer users found"
        
        # Format data
        report_data = []
        for user in users:
            formatted_user = {
                'name': user['name'].title(),
                'email': user['email'].lower(),
                'last_login': format_date(user['last_login']),
                'orders': str(user.get('order_count', 0))
            }
            report_data.append(formatted_user)
        
        # Generate report
        report = "CUSTOMER USER REPORT\\n"
        report += "=" * 50 + "\\n"
        for user in report_data:
            report += f"Name: {user['name']}\\n"
            report += f"Email: {user['email']}\\n"
            report += f"Last Login: {user['last_login']}\\n"
            report += f"Orders: {user['orders']}\\n"
            report += "-" * 30 + "\\n"
        
        return report''',
            'reference_answer': """Extract common patterns and use configuration:

```python
from typing import Dict, List, Callable, Any

def generate_user_report(user_type: str) -> str:
    \"\"\"Generate reports using configurable templates.\"\"\"
    
    report_configs = {
        'admin': {
            'data_source': get_admin_users,
            'title': 'ADMIN USER REPORT',
            'fields': {
                'name': lambda u: u['name'].title(),
                'email': lambda u: u['email'].lower(), 
                'last_login': lambda u: format_date(u['last_login']),
                'permissions': lambda u: ', '.join(u['permissions'])
            }
        },
        'customer': {
            'data_source': get_customer_users,
            'title': 'CUSTOMER USER REPORT',
            'fields': {
                'name': lambda u: u['name'].title(),
                'email': lambda u: u['email'].lower(),
                'last_login': lambda u: format_date(u['last_login']),
                'orders': lambda u: str(u.get('order_count', 0))
            }
        }
    }
    
    if user_type not in report_configs:
        return f"Unknown user type: {user_type}"
    
    config = report_configs[user_type]
    users = config['data_source']()
    
    if not users:
        return f"No {user_type} users found"
    
    return _build_report(users, config)

def _build_report(users: List[Dict], config: Dict[str, Any]) -> str:
    \"\"\"Build report using common template.\"\"\"
    report_lines = [config['title'], "=" * 50]
    
    for user in users:
        for field_name, formatter in config['fields'].items():
            value = formatter(user)
            report_lines.append(f"{field_name.title()}: {value}")
        report_lines.append("-" * 30)
    
    return "\\n".join(report_lines)
```

This eliminates duplication through parameterization and shared templates."""
        },

        # 9. IMPROVE_NAMING (already in original but adding dataset version)
        {
            'query': 'These variable and function names are unclear. How can I improve the naming?',
            'original_code': '''def calc_stuff(d, p, t):
    """Calculate some financial stuff"""
    r = []
    
    for x in d:
        if x['s'] == 'a':
            tmp = {}
            tmp['id'] = x['uid']
            tmp['amt'] = x['val'] * p / 100
            
            if t == 'q':
                tmp['amt'] = tmp['amt'] * 4
            elif t == 'm':
                tmp['amt'] = tmp['amt'] * 12
                
            tmp['desc'] = f"Calc for {x['nm']}"
            r.append(tmp)
    
    return r''',
            'reference_answer': """Use descriptive, meaningful names:

```python
def calculate_financial_projections(
    account_data: List[Dict], 
    interest_rate_percent: float, 
    projection_period: str
) -> List[Dict]:
    \"\"\"Calculate financial projections for active accounts.
    
    Args:
        account_data: List of account information dictionaries
        interest_rate_percent: Annual interest rate as percentage
        projection_period: 'q' for quarterly, 'm' for monthly
        
    Returns:
        List of financial projections for active accounts
    \"\"\"
    financial_projections = []
    
    for account in account_data:
        if account['status'] != 'active':
            continue
            
        projection = {
            'account_id': account['user_id'],
            'projected_amount': _calculate_projected_amount(
                account['value'], 
                interest_rate_percent, 
                projection_period
            ),
            'description': f"Financial projection for {account['name']}"
        }
        
        financial_projections.append(projection)
    
    return financial_projections

def _calculate_projected_amount(
    principal_amount: float, 
    annual_rate_percent: float, 
    period_type: str
) -> float:
    \"\"\"Calculate projected amount based on period type.\"\"\"
    base_amount = principal_amount * annual_rate_percent / 100
    
    period_multipliers = {
        'q': 4,   # quarterly
        'm': 12   # monthly
    }
    
    multiplier = period_multipliers.get(period_type, 1)
    return base_amount * multiplier
```

Clear naming makes the code self-documenting and easier to maintain."""
        },

        # 10. OPTIMIZE_PERFORMANCE (already in original but adding dataset version)
        {
            'query': 'This code is running slowly with large datasets. How can I optimize the performance?',
            'original_code': '''def find_matching_records(dataset1, dataset2, match_field):
    """Find records that match between two datasets"""
    matches = []
    
    for record1 in dataset1:
        for record2 in dataset2:
            if record1.get(match_field) == record2.get(match_field):
                if record1.get(match_field) is not None:
                    match_info = {
                        'record1': record1,
                        'record2': record2,
                        'match_value': record1[match_field]
                    }
                    
                    # Check if we already have this match
                    duplicate = False
                    for existing in matches:
                        if (existing['record1'] == record1 and 
                            existing['record2'] == record2):
                            duplicate = True
                            break
                    
                    if not duplicate:
                        matches.append(match_info)
    
    return matches''',
            'reference_answer': """Optimize using hash-based lookups for O(n) performance:

```python
from typing import List, Dict, Any, Set
from collections import defaultdict

def find_matching_records(
    dataset1: List[Dict[str, Any]], 
    dataset2: List[Dict[str, Any]], 
    match_field: str
) -> List[Dict[str, Any]]:
    \"\"\"Find matching records using optimized hash lookup.
    
    Time complexity: O(n + m) instead of O(n * m)
    \"\"\"
    # Create hash map for dataset2 for O(1) lookups
    dataset2_by_field = defaultdict(list)
    for record in dataset2:
        field_value = record.get(match_field)
        if field_value is not None:
            dataset2_by_field[field_value].append(record)
    
    matches = []
    seen_pairs: Set[tuple] = set()
    
    for record1 in dataset1:
        field_value = record1.get(match_field)
        if field_value is None:
            continue
            
        # O(1) lookup instead of O(n) iteration
        for record2 in dataset2_by_field[field_value]:
            # Use tuple of IDs to avoid duplicate checking
            pair_key = (id(record1), id(record2))
            if pair_key not in seen_pairs:
                matches.append({
                    'record1': record1,
                    'record2': record2,
                    'match_value': field_value
                })
                seen_pairs.add(pair_key)
    
    return matches

# Alternative using set operations for even better performance
def find_matching_records_fast(
    dataset1: List[Dict[str, Any]], 
    dataset2: List[Dict[str, Any]], 
    match_field: str
) -> List[Dict[str, Any]]:
    \"\"\"Ultra-fast version using set intersection.\"\"\"
    values1 = {(i, record.get(match_field)) for i, record in enumerate(dataset1) 
               if record.get(match_field) is not None}
    values2 = {(i, record.get(match_field)) for i, record in enumerate(dataset2) 
               if record.get(match_field) is not None}
    
    # Find common values
    common_values = {v for _, v in values1} & {v for _, v in values2}
    
    # Build matches efficiently
    matches = []
    for i1, val in values1:
        if val in common_values:
            for i2, val2 in values2:
                if val == val2:
                    matches.append({
                        'record1': dataset1[i1],
                        'record2': dataset2[i2], 
                        'match_value': val
                    })
    
    return matches
```

This optimization reduces time complexity from O(n²) to O(n) using hash-based lookups."""
        }
    ]


def setup_system() -> RefactoringRAGSystem:
    """Initialize and setup the RAG system."""
    logger.info("Initializing Python Code Refactoring RAG System...")
    
    # Get LLM configurations
    try:
        llm_configs = get_default_llm_configs()
        if not llm_configs:
            logger.error("No valid LLM configurations found. Please check your API keys.")
            sys.exit(1)
        
        logger.info(f"Loaded {len(llm_configs)} LLM configurations")
        
    except Exception as e:
        logger.error(f"Error loading LLM configurations: {e}")
        sys.exit(1)
    
    # Initialize RAG system
    system = RefactoringRAGSystem(
        llm_configs=llm_configs,
        qdrant_url=os.getenv('QDRANT_URL'),
        qdrant_api_key=os.getenv('QDRANT_API_KEY')
    )
    
    # Setup with enhanced configuration
    config = get_default_config()
    config['pdf'] = PDFConfig(
        max_chunk_size=1000,
        min_chunk_size=100,
        overlap_size=50,
        extract_code_blocks=True,
        use_pymupdf=True
    )
    
    system.setup(config=config)
    
    return system


def process_data(system: RefactoringRAGSystem, 
                dataset_path: str = None,
                pdf_paths: list = None,
                force_reindex: bool = False) -> Dict[str, int]:
    """Process and index data sources with intelligent skip logic."""
    results = {'dataset_chunks': 0, 'pdf_chunks': 0}
    
    # Process dataset if provided
    if dataset_path and Path(dataset_path).exists():
        logger.info(f"Processing dataset: {dataset_path}")
        dataset_chunks = system.process_dataset(dataset_path, force_reindex)
        results['dataset_chunks'] = dataset_chunks
        logger.info(f"Processed {dataset_chunks} chunks from dataset")
    elif dataset_path:
        logger.warning(f"Dataset file not found: {dataset_path}")
    
    # Process PDFs if provided - now with intelligent skip logic
    if pdf_paths:
        valid_pdfs = [path for path in pdf_paths if Path(path).exists()]
        if valid_pdfs:
            logger.info(f"Found {len(valid_pdfs)} valid PDF files: {[Path(p).name for p in valid_pdfs]}")
            
            # Use the enhanced PDF processing with skip logic
            pdf_chunks = system.process_pdfs(valid_pdfs, force_reindex)
            results['pdf_chunks'] = pdf_chunks
            
            if pdf_chunks > 0:
                logger.info(f"Processed {pdf_chunks} chunks from PDFs")
            else:
                logger.info("No new PDF chunks processed (already up to date)")
        else:
            logger.warning("No valid PDF files found")
    
    return results


def run_evaluation(system: RefactoringRAGSystem) -> Dict[str, Any]:
    """Run system evaluation with enhanced test cases."""
    logger.info("Running comprehensive system evaluation...")
    
    # Get enhanced test cases
    test_cases = get_enhanced_test_cases()
    
    # Run evaluation for each available model
    evaluator = RAGEvaluator()
    available_models = system.llm_provider.get_available_models()
    
    evaluation_results = {}
    
    print(f"\n{'='*80}")
    print(f"EVALUATING {len(available_models)} MODELS ON {len(test_cases)} TEST CASES")
    print(f"{'='*80}")
    
    for model_name in available_models:
        logger.info(f"Evaluating model: {model_name}")
        model_results = []
        
        print(f"\n🔍 Testing Model: {model_name}")
        print("-" * 50)
        
        for i, test_case in enumerate(test_cases):
            logger.info(f"Testing case {i+1}: {test_case['query'][:50]}...")
            
            try:
                # Get suggestion from system
                suggestion = system.get_refactoring_suggestions(
                    query=test_case['query'],
                    model_name=model_name,
                    user_code=test_case.get('original_code')
                )
                
                # Get context chunks for evaluation
                similar_chunks = system.retrieval_service.search_with_enhanced_query(
                    test_case['query'], 5
                )
                
                # Evaluate the result
                result = evaluator.evaluate_single_query(
                    query=test_case['query'],
                    context_chunks=similar_chunks,
                    answer=suggestion,
                    reference_answer=test_case.get('reference_answer')
                )
                
                result['model_name'] = model_name
                model_results.append(result)
                
                # Print brief results
                if result['success'] and result['rag_metrics']:
                    rag = result['rag_metrics']
                    print(f"  ✓ Case {i+1}: Context:{rag.context_relevance:.3f} | "
                          f"Answer:{rag.answer_relevance:.3f} | Faithful:{rag.faithfulness:.3f} | "
                          f"BLEU:{rag.bleu_score:.4f} | ROUGE:{rag.rouge_l_score:.4f}")
                else:
                    print(f"  ✗ Case {i+1}: Failed")
                
            except Exception as e:
                logger.error(f"Error evaluating {model_name} on case {i+1}: {e}")
                model_results.append({
                    'query': test_case['query'],
                    'model_name': model_name,
                    'success': False,
                    'error': str(e)
                })
                print(f"  ✗ Case {i+1}: Error - {str(e)[:50]}...")
        
        evaluation_results[model_name] = model_results
    
    return evaluation_results


def display_evaluation_summary(evaluation_results: Dict[str, Any]):
    """Display comprehensive evaluation summary."""
    print(f"\n{'='*80}")
    print("EVALUATION SUMMARY")
    print(f"{'='*80}")
    
    # Calculate aggregate metrics per model
    model_aggregates = {}
    
    for model_name, results in evaluation_results.items():
        successful_results = [r for r in results if r['success'] and r.get('rag_metrics')]
        
        if successful_results:
            # Calculate averages
            avg_context_rel = sum(r['rag_metrics'].context_relevance for r in successful_results) / len(successful_results)
            avg_answer_rel = sum(r['rag_metrics'].answer_relevance for r in successful_results) / len(successful_results)
            avg_faithfulness = sum(r['rag_metrics'].faithfulness for r in successful_results) / len(successful_results)
            avg_completeness = sum(r['rag_metrics'].response_completeness for r in successful_results) / len(successful_results)
            avg_bleu = sum(r['rag_metrics'].bleu_score for r in successful_results) / len(successful_results)
            avg_rouge = sum(r['rag_metrics'].rouge_l_score for r in successful_results) / len(successful_results)
            
            model_aggregates[model_name] = {
                'context_relevance': avg_context_rel,
                'answer_relevance': avg_answer_rel,
                'faithfulness': avg_faithfulness,
                'response_completeness': avg_completeness,
                'bleu_score': avg_bleu,
                'rouge_l_score': avg_rouge,
                'success_rate': len(successful_results) / len(results)
            }
        else:
            model_aggregates[model_name] = {
                'context_relevance': 0.0,
                'answer_relevance': 0.0,
                'faithfulness': 0.0,
                'response_completeness': 0.0,
                'bleu_score': 0.0,
                'rouge_l_score': 0.0,
                'success_rate': 0.0
            }
    
    # Display results table
    print(f"{'Model':<30} {'Context':<8} {'Answer':<8} {'Faith':<8} {'Complete':<8} {'BLEU':<8} {'ROUGE':<8} {'Success':<8}")
    print("-" * 88)
    
    for model_name, metrics in model_aggregates.items():
        print(f"{model_name:<30} "
              f"{metrics['context_relevance']:<8.3f} "
              f"{metrics['answer_relevance']:<8.3f} "
              f"{metrics['faithfulness']:<8.3f} "
              f"{metrics['response_completeness']:<8.3f} "
              f"{metrics['bleu_score']:<8.4f} "
              f"{metrics['rouge_l_score']:<8.4f} "
              f"{metrics['success_rate']:<8.2%}")
    
    return model_aggregates


def run_model_optimization(evaluation_results: Dict[str, Any], 
                          pre_optimization_metrics: Dict[str, Any]) -> Dict[str, Any]:
    """Run NSGA-II optimization and show before/after comparison."""
    logger.info("Running NSGA-II multi-objective optimization...")
    
    try:
        
        optimization_results = run_fixed_nsga2_optimization(evaluation_results)
        
        print(f"\n{'='*80}")
        print("NSGA-II OPTIMIZATION RESULTS & COMPARISON")
        print(f"{'='*80}")
        
        best_model = optimization_results['best_model']
        print(f"🏆 Best Model Selected: {best_model}")
        print(f"📊 Pareto Front Size: {optimization_results['pareto_front_size']}")
        print(f"⏱️  Optimization Time: {optimization_results['optimization_time_seconds']:.2f}s")
        print(f"🔧 Algorithm: {optimization_results['algorithm']}")
        
        # Show before/after metrics comparison
        print(f"\n{'='*50}")
        print("METRICS COMPARISON: BEFORE vs AFTER OPTIMIZATION")
        print(f"{'='*50}")
        
        best_metrics = optimization_results.get('best_model_objectives', {})
        pre_metrics = pre_optimization_metrics.get(best_model, {})
        
        if best_metrics and pre_metrics:
            print(f"{'Metric':<20} {'Before':<10} {'After':<10} {'Change':<10}")
            print("-" * 50)
            
            for metric in ['context_relevance', 'answer_relevance', 'faithfulness', 
                          'response_completeness', 'bleu_score', 'rouge_l_score']:
                if metric in best_metrics and metric in pre_metrics:
                    before_val = pre_metrics[metric]
                    after_val = best_metrics[metric]
                    change = after_val - before_val
                    change_str = f"+{change:.4f}" if change >= 0 else f"{change:.4f}"
                    
                    print(f"{metric.replace('_', ' ').title():<20} "
                          f"{before_val:<10.4f} "
                          f"{after_val:<10.4f} "
                          f"{change_str:<10}")
        
        # Show all Pareto solutions
        print(f"\n{'='*50}")
        print("TOP 5 PARETO OPTIMAL SOLUTIONS")
        print(f"{'='*50}")
        
        # Convert to list of solutions
        pareto_solutions = [
            {'model': model_name, 'objectives': scores}
            for model_name, scores in pre_optimization_metrics.items()
        ]

        # Optional: sort by average score if desired
        pareto_solutions.sort(key=lambda s: sum(s['objectives'].values()) / len(s['objectives']), reverse=True)

        # Print top 5
        for i, solution in enumerate(pareto_solutions[:5], 1):
            model = solution.get('model', 'Unknown')
            objectives = solution.get('objectives', {})
            avg_score = sum(objectives.values()) / len(objectives) if objectives else 0
            print(f"{i}. {model:<30} (Avg Score: {avg_score:.4f})")
        
        return optimization_results
        
    except Exception as e:
        logger.error(f"Error in NSGA-II optimization: {e}")
        return {'error': str(e)}


# Optional: Install rich library for enhanced UI
try:
    from rich.console import Console
    from rich.panel import Panel
    from rich.syntax import Syntax
    from rich.prompt import Prompt
    from rich.text import Text
    from rich.columns import Columns
    RICH_AVAILABLE = True
except ImportError:
    RICH_AVAILABLE = False


def display_comprehensive_response(system: RefactoringRAGSystem, query: str, 
                                 best_model: str = None, user_code: str = None):
    """Display both best model and all models' responses."""
    if not RICH_AVAILABLE:
        return display_comprehensive_response_basic(system, query, best_model, user_code)
    
    console = Console()
    
    console.print(f"\n[bold blue]Processing comprehensive response...[/bold blue]")
    console.print(f"[dim]Query: {query[:100]}{'...' if len(query) > 100 else ''}[/dim]")
    
    # Get suggestions from all models
    all_suggestions = system.get_refactoring_suggestions(query, user_code=user_code)
    
    if isinstance(all_suggestions, str):
        # Single model response
        console.print(Panel(all_suggestions, title="Response"))
        return
    
    # Display best model first if available
    if best_model and best_model in all_suggestions:
        best_suggestion = all_suggestions[best_model]
        
        # Truncate if too long but show full content
        display_text = best_suggestion
        title = f"🏆 BEST MODEL: {best_model}"
        
        console.print(Panel(display_text, title=title, border_style="green"))
        
        # Ask if user wants to see all models
        console.print(f"\n[yellow]Best model response shown above.[/yellow]")
        show_all = Prompt.ask("Show all models' responses? (y/n)", default="n")
        
        if show_all.lower() in ['y', 'yes']:
            console.print(f"\n[bold cyan]ALL MODELS' RESPONSES:[/bold cyan]")
            for model_name, suggestion in all_suggestions.items():
                if model_name != best_model:  # Skip best model as already shown
                    title = f"📋 {model_name}"
                    # Create scrollable content for long responses
                    display_text = suggestion
                    console.print(Panel(display_text, title=title, border_style="blue"))
    else:
        # Show all models
        console.print(f"\n[bold cyan]ALL MODELS' RESPONSES:[/bold cyan]")
        for model_name, suggestion in all_suggestions.items():
            title = f"📋 {model_name}"
            is_best = model_name == best_model
            border_style = "green" if is_best else "blue"
            
            if is_best:
                title = f"🏆 {title} (BEST)"
            
            display_text = suggestion
            console.print(Panel(display_text, title=title, border_style=border_style))


def display_comprehensive_response_basic(system: RefactoringRAGSystem, query: str, 
                                       best_model: str = None, user_code: str = None):
    """Basic display without Rich library."""
    print(f"\n{'='*80}")
    print("COMPREHENSIVE RESPONSE")
    print(f"{'='*80}")
    print(f"Query: {query}")
    
    # Get suggestions from all models
    all_suggestions = system.get_refactoring_suggestions(query, user_code=user_code)
    
    if isinstance(all_suggestions, str):
        print(f"\nResponse:\n{all_suggestions}")
        return
    
    # Display best model first
    if best_model and best_model in all_suggestions:
        print(f"\n🏆 BEST MODEL ({best_model}):")
        print("-" * 50)
        print(all_suggestions[best_model])
        
        response = input(f"\nShow all models' responses? (y/n): ").lower()
        if response in ['y', 'yes']:
            print(f"\n{'='*60}")
            print("ALL MODELS' RESPONSES")
            print(f"{'='*60}")
            
            for model_name, suggestion in all_suggestions.items():
                if model_name != best_model:
                    print(f"\n📋 {model_name}:")
                    print("-" * 40)
                    print(suggestion)
    else:
        for model_name, suggestion in all_suggestions.items():
            marker = "🏆" if model_name == best_model else "📋"
            print(f"\n{marker} {model_name}:")
            print("-" * 40)
            print(suggestion)


def interactive_mode_enhanced(system: RefactoringRAGSystem, best_model: str = None):
    """Enhanced interactive mode with best model optimization."""
    if not RICH_AVAILABLE:
        return interactive_mode_basic_enhanced(system, best_model)

    console = Console()

    # Display welcome message
    console.print(Panel.fit(
        f"[bold blue]Python Code Refactoring RAG System[/bold blue]\n"
        f"[yellow]Enhanced Interactive Mode[/yellow]\n"
        f"[green]Optimized Best Model: {best_model or 'Not determined'}[/green]",
        title="Welcome"
    ))

    # Display available commands
    console.print("\n[bold green]Commands:[/bold green]")
    console.print("  • [cyan]quit/exit[/cyan] - Exit")
    console.print("  • [cyan]stats[/cyan] - System statistics")
    console.print("  • [cyan]health[/cyan] - Health check")
    console.print("  • [cyan]best[/cyan] - Show only best model response")
    console.print("  • [cyan]all[/cyan] - Show all models' responses")
    console.print("  • [cyan]help[/cyan] - Show help")

    console.print("\n[bold green]Multi-line Input:[/bold green]")
    console.print("  • Type [yellow]###END###[/yellow] to submit")
    console.print("  • Press [yellow]CTRL+D[/yellow] to submit")
    console.print("  • Double [yellow]ENTER[/yellow] for short inputs")

    session = 0
    response_mode = "comprehensive"  # "best", "all", or "comprehensive"

    while True:
        try:
            session += 1
            console.print(f"\n[bold magenta]Session {session}[/bold magenta]")

            console.print(f"\n[bold yellow]Enter your query or code:[/bold yellow]")

            lines = []
            line_num = 1
            empty_count = 0

            while True:
                try:
                    prompt_text = f"[dim]{line_num:2d}|[/dim] "
                    line = Prompt.ask(prompt_text, default="")

                    if line.strip() == '###END###':
                        break

                    if not line.strip():
                        empty_count += 1
                        if empty_count >= 2:
                            console.print("[dim]Detected double empty line. Submitting...[/dim]")
                            break
                    else:
                        empty_count = 0

                    lines.append(line)
                    line_num += 1

                except EOFError:
                    console.print("\n[dim]EOF detected. Submitting...[/dim]")
                    break

            query = '\n'.join(lines).strip()

            if not query:
                continue

            # Command handling
            command = query.lower()

            if command in ['quit', 'exit']:
                console.print("[bold red]Exiting interactive mode.[/bold red]")
                break

            elif command == 'stats':
                stats = system.get_system_stats()
                console.print(Panel(str(stats), title="System Statistics"))
                continue

            elif command == 'health':
                health = system.health_check()
                health_text = Text()
                for component, status in health.items():
                    status_text = "HEALTHY" if status else "UNHEALTHY"
                    color = "green" if status else "red"
                    health_text.append(f"{component}: ", style="white")
                    health_text.append(f"{status_text}\n", style=color)
                console.print(Panel(health_text, title="Health Check"))
                continue

            elif command == 'best':
                response_mode = "best"
                console.print("[green]Mode set to: Best model only[/green]")
                continue

            elif command == 'all':
                response_mode = "all"
                console.print("[green]Mode set to: All models[/green]")
                continue

            elif command == 'help':
                help_text = """[bold green]Commands:[/bold green]
• quit/exit - Exit interactive mode
• stats - Show system statistics
• health - Perform system health check
• best - Show only best model responses
• all - Show all models' responses
• help - Display help menu

[bold green]Input Tips:[/bold green]
• Paste code or ask natural language questions
• Use ###END### or CTRL+D to submit multi-line input
• Double ENTER also works for short queries
"""
                console.print(Panel(help_text, title="Help"))
                continue

            # Process the query
            console.print(f"\n[bold blue]Processing your query...[/bold blue]")
            console.print(f"[dim]Input length: {len(query)} characters, {len(lines)} lines[/dim]")
            console.print(f"[dim]Response mode: {response_mode}[/dim]")

            # Detect code in input
            has_code = any(kw in query.lower() for kw in ['def ', 'class ', 'import ', 'for ', 'if ', 'while ', '```'])
            user_code = query if has_code else None

            if response_mode == "best" and best_model:
                # Show only best model
                suggestion = system.get_refactoring_suggestions(query, model_name=best_model, user_code=user_code)
                title = f"🏆 BEST MODEL: {best_model}"
                console.print(Panel(suggestion, title=title, border_style="green"))
            
            elif response_mode == "all":
                # Show all models
                all_suggestions = system.get_refactoring_suggestions(query, user_code=user_code)
                if isinstance(all_suggestions, dict):
                    for model_name, suggestion in all_suggestions.items():
                        is_best = model_name == best_model
                        title = f"🏆 {model_name}" if is_best else f"📋 {model_name}"
                        border_style = "green" if is_best else "blue"
                        console.print(Panel(suggestion, title=title, border_style=border_style))
                else:
                    console.print(Panel(all_suggestions, title="Response"))
            
            else:
                # Comprehensive mode (default)
                display_comprehensive_response(system, query, best_model, user_code)

            console.print("[bold green]Query completed.[/bold green]")

        except KeyboardInterrupt:
            console.print("\n[bold red]Interrupted by user. Exiting...[/bold red]")
            break
        except Exception as error:
            console.print(f"[bold red]Error: {error}[/bold red]")
            console.print("[yellow]Use 'help' for available commands.[/yellow]")


def interactive_mode_basic_enhanced(system: RefactoringRAGSystem, best_model: str = None):
    """Enhanced basic interactive mode without Rich."""
    print(f"\n{'='*80}")
    print("ENHANCED INTERACTIVE MODE - Python Code Refactoring RAG System")
    print(f"{'='*80}")
    print(f"Optimized Best Model: {best_model or 'Not determined'}")

    print("\nAvailable Commands:")
    print("  • quit / exit / q     - Exit interactive mode")
    print("  • stats               - Show system statistics")
    print("  • health              - Perform system health check")
    print("  • best                - Show only best model response")
    print("  • all                 - Show all models' responses")
    print("  • pdf-status          - Show PDF processing status") 
    print("  • help                - Display help message")

    session_count = 0
    response_mode = "comprehensive"

    while True:
        try:
            session_count += 1
            print(f"\n{'='*20} Session {session_count} {'='*20}")

            query = input("\nEnter your refactoring query or code: ").strip()
            
            if not query:
                print("No input provided. Please try again.")
                continue

            command = query.lower()

            if command in ['quit', 'exit', 'q']:
                print("Exiting. Goodbye!")
                break  # This should properly exit the main loop

            elif command == 'best':
                response_mode = "best"
                print("Mode set to: Best model only")
                continue

            elif command == 'all':
                response_mode = "all"
                print("Mode set: All models")
                continue

            elif command == 'stats':
                stats = system.get_system_stats()
                print("\nSystem Statistics")
                print("-" * 40)
                for key, value in stats.items():
                    print(f"{key}: {value}")
                continue

            elif command == 'health':
                health = system.health_check()
                print("\nSystem Health Check")
                print("-" * 40)
                for component, status in health.items():
                    status_text = "HEALTHY" if status else "UNHEALTHY"
                    print(f"{component}: {status_text}")
                continue

            # Process query based on mode
            print(f"\nProcessing query in '{response_mode}' mode...")
            
            has_code = any(kw in query.lower() for kw in ['def ', 'class ', 'import ', 'for ', 'if '])
            user_code = query if has_code else None
            
            if response_mode == "best" and best_model:
                suggestion = system.get_refactoring_suggestions(query, model_name=best_model, user_code=user_code)
                print(f"\n🏆 BEST MODEL ({best_model}):")
                print("-" * 50)
                print(suggestion)
            
            elif response_mode == "all":
                display_comprehensive_response_basic(system, query, best_model, user_code)
            
            else:
                display_comprehensive_response_basic(system, query, best_model, user_code)

            print("\nQuery completed successfully.")

        except KeyboardInterrupt:
            print("\nExiting interactive mode.")
            break
        except Exception as e:
            print(f"\nError occurred: {e}")


def interactive_mode(system: RefactoringRAGSystem, best_model: str = None):
    """Select the most appropriate interactive mode with enhanced display management."""
    if ENHANCED_DISPLAY_AVAILABLE:
        # Use the new enhanced interactive session manager
        session = create_interactive_session(system, best_model)
        session.run()
    elif RICH_AVAILABLE:
        # Fallback to built-in Rich mode
        interactive_mode_enhanced(system, best_model)
    else:
        # Fallback to basic mode
        interactive_mode_basic_enhanced(system, best_model)


def main():
    """Enhanced main entry point."""
    print("Python Code Refactoring RAG System - Enhanced Edition")
    print("=" * 70)
    
    # Configuration
    DATASET_PATH = "E:\Shoban\Shoban-NCI\Practicum\Python_Refactoring\Python_refactoring\python-refactoring-rag\Inputs\python_legacy_refactoring_dataset_2.json"
    PDF_PATHS = [
        "E:\Shoban\Shoban-NCI\Practicum\Python_Refactoring\Python_refactoring\python-refactoring-rag\Inputs\clean-code-in-python.pdf",
        "E:\Shoban\Shoban-NCI\Practicum\Python_Refactoring\Python_refactoring\python-refactoring-rag\Inputs\\the-clean-coder-a-code-of-conduct-for-professional-programmers.pdf" 
    ]
    
    try:
        # Setup system
        system = setup_system()
        
        # Check system health
        health = system.health_check()
        if not health['overall']:
            logger.error("System health check failed. Please check the logs.")
            return
        
        print("System initialized successfully!")
        
        # Process data sources
        processing_results = process_data(
            system, 
            DATASET_PATH, 
            PDF_PATHS, 
            force_reindex=False
        )
        
        total_chunks = processing_results['dataset_chunks'] + processing_results['pdf_chunks']
        print(f"Data processing complete: {total_chunks} total chunks indexed")
        
        # Show system statistics
        stats = system.get_system_stats()
        print(f"\nSystem ready with {stats['vector_store'].get('points_count', 0)} indexed chunks")
        
        # Run comprehensive evaluation
        print(f"\n{'='*70}")
        print("RUNNING COMPREHENSIVE SYSTEM EVALUATION")
        print(f"{'='*70}")
        
        evaluation_results = run_evaluation(system)
        
        # Display evaluation summary and store pre-optimization metrics
        pre_optimization_metrics = display_evaluation_summary(evaluation_results)
        
        # Run optimization with comparison
        optimization_results = run_model_optimization(evaluation_results, pre_optimization_metrics)
        
        # Determine best model and set it in the system
        best_model = None
        if 'error' not in optimization_results:
            best_model = optimization_results['best_model']
            system.set_best_model(best_model, optimization_results)  # Set in system
            print(f"\n🎯 SYSTEM READY!")
            print(f"📊 Recommended Model: {best_model}")
            print("✅ System is optimized for production use!")
        else:
            print(f"\n⚠️  Optimization failed: {optimization_results['error']}")
            print("⚡ System is still functional for basic queries.")
        
        # Enhanced interactive mode with all features
        interactive_mode(system, best_model)
        
    except KeyboardInterrupt:
        print("\nShutdown requested by user")
    except Exception as e:
        logger.error(f"Critical error in main: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()