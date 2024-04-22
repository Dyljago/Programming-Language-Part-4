from antlr4 import *
from JavaLexer import JavaLexer
from JavaParser import JavaParser
from antlr4.tree.Trees import Trees
from antlr4 import ParserRuleContext, TerminalNode
import sys
from PyQt5.QtWidgets import QApplication, QWidget, QVBoxLayout, QTreeWidget, QTreeWidgetItem
import json
from bs4 import BeautifulSoup
import requests
from g4f.client import Client
import spacy
import os
import psycopg2

# Connect to your PostgreSQL database
conn = psycopg2.connect(
    dbname='Test-fabio',
    user='postgres',
    password='301SQL',
    host='localhost',
    port = 5432
)

cur = conn.cursor()
# Execute the SQL query
cur.execute("""
CREATE TABLE IF NOT EXISTS function_table (
    function_name character varying,
    api_name character varying,
    CONSTRAINT unique_function_api_pair UNIQUE (function_name, api_name)
);
""")

conn.commit()

cur.execute('''
CREATE TABLE IF NOT EXISTS API_Function_Specific (
    id SERIAL PRIMARY KEY,
    api_name_fk VARCHAR,
    function_name_fk VARCHAR,
    api_context TEXT,
    api_topic TEXT,
    function_context TEXT,
    function_topic TEXT,
    llm_expert_API TEXT,
    sim_expert_API TEXT,
    llm_expert_function TEXT,
    sim_expert_function TEXT,
    CONSTRAINT unique_function_api_specific_pair UNIQUE (api_name_fk, function_name_fk),
    FOREIGN KEY (api_name_fk, function_name_fk) REFERENCES function_table (api_name, function_name)
)
''')

conn.commit()

# Close the connection
cur.close()
conn.close()

# Fetch the results
# rows = cur.fetchall()

# for row in rows:
#     print(row[0])  # Assuming 'name' is the first (and only) column in the SELECT query


all_imports = set()
nlp = spacy.load("en_core_web_lg")
input_files = []
input_file = ""
input_file_directory = './jabref-5.0-alpha'
output_directory = './program_outputs'
program_output_file = open(f'{input_file}_output.txt', 'w')
file_path = "classification_data.txt"
client = Client()
jabrefRunning = False

api_context = []
api_topic = []
function_context = []
function_topic = []
api_gpt = []
api_sim = []
function_gpt = []
function_sim = []

messages = [
    {"role": "system",
     "content": "You are attempting to classify the inputted description into one of the 31 labels based off of the simularity to it."},
    {"role": "system",
     "content": "As you guess this, your final response should only be one concise paragraph with the label chosen and why."},
    {"role": "system",
     "content": "Please do not add any pleasantries, please only output the result and why it fits it that category."},
    {"role": "system",
     "content": "please answer in the format of: Label: givenlabel"
                "Reason: reasonwhy"}
]

messages_overload_summ = [
    {"role": "system",
     "content": "You are attempting to comprise, summarize, or paraphrase the content of a programming function method with multiple overloads with individual parameters and descriptions into one concise Method title and summary."},
    {"role": "system",
     "content": "Please do not add any pleasantries, please only output the result."},
    {"role": "system",
     "content": "Please answer in the format of: Summary: summary"},

]

messages_description_summ = [
    {"role": "system",
      "content": "You are attempting to summarize or paraphrase the content of a description into a concise and accurate summary of the passed content."},
     {"role": "system",
      "content": "As you guess this, your final response should only be one concise paragraph with your paraphrased summary or description without reference links or acknowledgements of prompts."},
     {"role": "system",
      "content": "Please do not add any pleasantries, please only output the result and why it fits it that category."},
     {"role": "system",
      "content": "Please answer in the format of: description"},
     ]

application_options = [
    {"User Interaction": "Tools and components facilitating user engagement and interaction within software applications, including user interfaces, input validation, and feedback mechanisms."},
    {"Task Management": "Functionality for organizing, scheduling, and tracking tasks and activities within software applications, aiding in productivity and workflow management."},
    {"Workflow Automation": "Tools and frameworks enabling the automation of repetitive tasks and processes within software applications, streamlining workflows and improving efficiency."}
]

application_performance_manager_options = [
    {"Resource Allocation": "Techniques and algorithms for efficiently distributing and managing computational resources within software systems, optimizing performance and resource utilization."},
    {"Bottleneck Identification": "Methods and tools for identifying performance bottlenecks and inefficiencies within software applications, facilitating performance optimization and tuning."},
    {"Performance Tuning": "Strategies and practices for improving the overall performance and responsiveness of software applications, including optimization of code, algorithms, and system configurations."}
]

big_data_options = [
    {"Data Transformation": "Tools and algorithms for converting, processing, and manipulating large volumes of data across diverse formats and structures, facilitating data integration and analysis."},
    {"Distributed Computing": "Frameworks and platforms for parallel processing and distributed computing, enabling scalable and efficient handling of big data sets across multiple nodes or clusters."},
    {"Data Analytics": "Techniques and tools for extracting insights, patterns, and trends from large and complex datasets, enabling data-driven decision-making and analysis."}
]

cloud_options = [
    {"Cloud Storage": "Services and APIs for storing and accessing data in remote cloud-based storage systems, providing scalability, durability, and accessibility."},
    {"Service Integration": "Tools and frameworks for integrating cloud services and applications, facilitating seamless communication and interoperability between different cloud-based resources."},
    {"Resource Orchestration": "Automation tools and platforms for provisioning, managing, and scaling cloud resources dynamically, optimizing resource utilization and workload distribution."}
]

computer_graphics_options = [
    {"3D Rendering": "Techniques and algorithms for generating and rendering three-dimensional graphics and visual effects, enabling the creation of immersive virtual environments and simulations."},
    {"Animation Techniques": "Tools and libraries for creating and manipulating animations, including keyframing, tweening, and skeletal animation, enhancing the visual appeal and interactivity of applications."},
    {"Image Processing Algorithms": "Algorithms and methods for manipulating and enhancing digital images, including filtering, segmentation, and feature extraction, enabling image analysis and manipulation."}
]

data_structure_options = [
    {"Queue Operations": "Methods and operations for managing and manipulating queue data structures, including enqueue, dequeue, and peek, facilitating FIFO (First-In-First-Out) data processing."},
    {"Stack Manipulation": "Functions and operations for managing stack data structures, such as push, pop, and peek, supporting LIFO (Last-In-First-Out) data processing and storage."},
    {"Hashing Functions": "Algorithms and techniques for generating and manipulating hash values, facilitating efficient storage, retrieval, and comparison of data in hash-based data structures."}
]

databases_options = [
    {"CRUD Operations": "Functions and methods for performing CRUD (Create, Read, Update, Delete) operations on database records, enabling data manipulation and management."},
    {"Query Optimization": "Techniques and strategies for optimizing database queries to improve performance and efficiency, including index usage, query planning, and execution optimization."},
    {"Indexing Strategies": "Methods and algorithms for creating and managing database indexes to accelerate data retrieval and query processing, improving overall database performance."}
]

software_development_and_it_options = [
    {"Build Automation": "Tools and frameworks for automating the build and compilation process of software applications, including dependency management, build scripts, and continuous integration."},
    {"Code Review Tools": "Platforms and utilities for conducting code reviews, facilitating collaborative code inspection, feedback, and quality assurance among development teams."},
    {"Deployment Pipelines": "Workflows and processes for automating the deployment of software applications and updates across development, testing, and production environments, ensuring consistency and reliability in deployment processes."}
]

error_handling_options = [
    {"Exception Logging": "Mechanisms and utilities for logging and recording exceptions, errors, and runtime issues within software applications, aiding in debugging, troubleshooting, and analysis."},
    {"Error Reporting": "Systems and tools for reporting and alerting developers or administrators about errors and exceptions occurring within software applications, enabling timely response and resolution."},
    {"Error Recovery Strategies": "Techniques and practices for implementing error recovery and fault tolerance mechanisms within software systems, ensuring graceful degradation and recovery from unexpected failures."}
]

event_handling_options = [
    {"Event Dispatching": "Systems and frameworks for dispatching and routing events within software applications, facilitating event-driven programming paradigms and asynchronous event processing."},
    {"Event Filtering": "Mechanisms and utilities for filtering and processing events based on specific criteria or conditions, enabling selective handling and routing of events within applications."},
    {"Event Driven Architecture": "Architectural patterns and designs centered around event-driven communication and interaction between software components, promoting loose coupling and scalability in distributed systems."}
]

geographic_information_system_options = [
    {"Spatial Analysis": "Techniques and algorithms for analyzing spatial data and geographic information, including spatial queries, proximity analysis, and spatial statistics."},
    {"Geocoding Services": "APIs and services for converting addresses or place names into geographic coordinates, enabling mapping, navigation, and location-based services."},
    {"Cartographic Rendering": "Tools and libraries for rendering and visualizing geographic information and maps, including map styling, labeling, and thematic mapping techniques."}
]

input_output_options = [
    {"File Parsing": "Utilities and libraries for parsing and processing various file formats, facilitating reading from and writing to files in different data formats."},
    {"Stream Serialization": "Mechanisms and techniques for serializing and deserializing data streams, enabling efficient data transmission and storage across different platforms and systems."},
    {"Input Validation": "Methods and tools for validating and sanitizing user input to ensure data integrity, security, and compatibility within software applications."}
]

interpreter_options = [
    {"Script Parsing": "Processes and utilities for parsing and analyzing script files written in scripting languages, enabling interpretation and execution within a runtime environment."},
    {"Bytecode Generation": "Techniques and tools for generating bytecode representations from source code, facilitating execution in interpreted or virtual machine-based environments."},
    {"Runtime Environment": "Environments and frameworks for executing interpreted code or scripts, providing runtime support and resources for script execution and evaluation."}
]

internationalization_options = [
    {"Locale Detection": "Mechanisms and utilities for detecting and identifying user locale preferences and language settings, enabling localization and internationalization of software applications."},
    {"Language Translation": "Tools and services for translating user interface elements, content, and messages into different languages, supporting multilingual applications and global audiences."},
    {"Cultural Adaptation": "Techniques and practices for adapting software applications to diverse cultural norms, preferences, and expectations, ensuring cultural sensitivity and inclusivity in user experiences."}
]

logic_options = [
    {"Decision Trees": "Algorithms and methods for representing and processing decision trees, enabling decision-making and classification tasks in machine learning and artificial intelligence applications."},
    {"Logical Operators": "Operators and functions for performing logical operations and computations, including AND, OR, NOT, and XOR, supporting logical reasoning and inference in software systems."},
    {"Conditional Statements": "Constructs and syntax for defining conditional logic and branching behavior within software applications, facilitating flow control and decision-making based on specific conditions."}
]

language_options = [
    {"Compiler Features": "Language features and constructs related to compiler optimizations, code generation, and language extensions, influencing compilation and execution behavior."},
    {"Language Extensions": "Extensions and enhancements to programming languages, providing additional functionality, syntax, or semantics beyond the core language specifications."},
    {"Syntax Parsing": "Techniques and tools for parsing and analyzing programming language syntax, enabling code comprehension, validation, and transformation within development environments."}
]

logging_options = [
    {"Log Rotation": "Strategies and utilities for rotating and managing log files to control file size and retention, ensuring efficient log storage and maintenance."},
    {"Log Filtering": "Mechanisms and configurations for filtering and routing log messages based on severity, category, or other criteria, enabling selective logging and analysis."},
    {"Log Visualization": "Tools and platforms for visualizing and analyzing log data, including dashboards, charts, and graphs, facilitating monitoring, debugging, and troubleshooting activities."}
]

machine_learning_options = [
    {"Model Training": "Methods and algorithms for training machine learning models on labeled datasets, including supervised, unsupervised, and reinforcement learning techniques."},
    {"Feature Engineering": "Techniques and methods for selecting, extracting, and transforming features from raw data to improve model performance and accuracy in machine learning tasks."},
    {"Model Evaluation": "Metrics and methods for evaluating the performance and effectiveness of machine learning models, including accuracy, precision, recall, and F1-score."}
]

microservices_services_options = [
    {"Service Discovery": "Mechanisms and tools for dynamically locating and connecting to services within a distributed system, enabling service registration, discovery, and load balancing."},
    {"Load Balancing": "Techniques and algorithms for distributing incoming network traffic across multiple servers or resources to ensure optimal resource utilization, performance, and reliability."},
    {"Service Orchestration": "Tools and frameworks for coordinating and managing interactions between microservices or service components within a distributed architecture, facilitating complex workflows and transactions."}
]

multimedia_options = [
    {"Audio Processing": "Techniques and algorithms for processing and manipulating digital audio signals, including filtering, equalization, compression, and synthesis."},
    {"Video Encoding": "Methods and standards for encoding, compressing, and transmitting digital video streams, enabling efficient storage, transmission, and playback of video content."},
    {"Image Compression": "Algorithms and techniques for compressing and reducing the size of digital images while preserving visual quality and fidelity, facilitating efficient storage and transmission of image data."}
]

multithread_options = [
    {"Thread Synchronization": "Mechanisms and techniques for coordinating and synchronizing the execution of multiple threads within a software application, preventing race conditions and ensuring data consistency."},
    {"Thread Pool Management": "Utilities and frameworks for managing and orchestrating a pool of worker threads, enabling efficient thread reuse and resource management."},
    {"Concurrency Control": "Techniques and algorithms for managing concurrent access to shared resources and data structures by multiple threads, ensuring thread safety and avoiding conflicts."}
]

natural_language_processing_options = [
    {"Text Tokenization": "Techniques and algorithms for breaking down text documents into smaller units, such as words or sentences, for further processing and analysis."},
    {"Named Entity Recognition": "Methods and algorithms for identifying and classifying named entities, such as persons, organizations, and locations, within text data."},
    {"Sentiment Analysis": "Techniques and algorithms for determining the sentiment or emotional tone expressed in textual data, enabling opinion mining and sentiment classification."}
]

network_options = [
    {"Protocol Implementation": "Utilities and libraries for implementing network protocols, such as TCP/IP, HTTP, and FTP, facilitating communication between networked devices and systems."},
    {"Socket Programming": "APIs and frameworks for creating and managing network sockets, enabling bidirectional communication and data exchange between client and server applications."},
    {"Network Security": "Technologies and practices for securing network communications and data exchange, including encryption, authentication, and access control mechanisms."}
]

operating_system_options = [
    {"System Configuration": "Tools and utilities for configuring and managing system settings, parameters, and resources on an operating system, including hardware, software, and network configurations."},
    {"Resource Management": "Mechanisms and utilities for managing system resources, such as CPU, memory, and disk space, optimizing resource allocation and utilization."},
    {"File System Operations": "Functions and utilities for interacting with the file system, including file manipulation, directory traversal, and file I/O operations, enabling file management and data storage."}
]

parser_options = [
    {"Syntax Parsing": "Algorithms and techniques for parsing and analyzing the syntax of data or code structures, including parsing expressions, statements, and program structures."},
    {"Semantic Analysis": "Processes and algorithms for analyzing the meaning and semantics of data or code constructs, enabling interpretation, validation, and optimization."},
    {"Code Generation": "Techniques and methods for generating executable code or machine-readable output from parsed data or code representations, facilitating compilation, translation, or transformation."}
]

search_options = [
    {"Information Retrieval": "Techniques and algorithms for retrieving relevant information from large datasets or collections, including indexing, ranking, and relevance scoring methods."},
    {"Indexing Techniques": "Methods and algorithms for creating and maintaining search indexes to enable fast and efficient information retrieval from large datasets or document collections."},
    {"Query Parsing": "Mechanisms and algorithms for parsing and analyzing search queries to identify search terms, operators, and modifiers, enabling query understanding and processing."}
]

security_options = [
    {"Authentication Mechanisms": "Methods and protocols for verifying the identity of users or entities accessing a system or application, ensuring secure and authorized access."},
    {"Access Control Policies": "Rules and configurations for controlling and managing access to resources and data within a system, enforcing security policies and permissions."},
    {"Encryption Techniques": "Algorithms and methods for encrypting and securing sensitive data during transmission or storage, protecting against unauthorized access or disclosure."}
]

setup_options = [
    {"Configuration Management": "Tools and practices for managing and maintaining software configurations, settings, and dependencies across different environments and deployments."},
    {"Installation Scripts": "Scripts and utilities for automating the installation and setup process of software applications, ensuring consistent and reproducible deployments."},
    {"Dependency Resolution": "Methods and tools for resolving and managing dependencies between software components, libraries, and modules, ensuring compatibility and consistency in software environments."}
]

user_interface_options = [
    {"GUI Frameworks": "Libraries and frameworks for creating graphical user interfaces (GUIs), providing widgets, controls, and layout management for building interactive applications."},
    {"UI Design Patterns": "Patterns and best practices for designing intuitive and user-friendly interfaces, including model-view-controller (MVC), observer, and command patterns."},
    {"Accessibility Features": "Features and utilities for enhancing the accessibility of user interfaces, including screen readers, keyboard navigation, and alternative input methods."}
]

utility_options = [
    {"String Manipulation": "Functions and utilities for manipulating and processing strings, including string concatenation, substring extraction, case conversion, and pattern matching."},
    {"Math Functions": "Functions and utilities for performing mathematical operations and calculations, including arithmetic operations, trigonometric functions, and statistical analysis."},
    {"Data Conversion": "Utilities and functions for converting data between different formats, types, and representations, including type casting, data serialization, and data encoding."}
]

test_options = [
    {"Test Case Generation": "Techniques and tools for automatically generating test cases and input data to achieve test coverage and verify software functionality, including random testing, boundary testing, and model-based testing."},
    {"Test Coverage Analysis": "Methods and tools for analyzing and measuring the extent to which software code is exercised by tests, including statement coverage, branch coverage, and path coverage."},
    {"Test Report Generation": "Utilities and frameworks for generating comprehensive test reports and summaries to document test results, findings, and metrics, aiding in test analysis, review, and decision-making."}
]


class ParseTreeViewer(QWidget):
    def __init__(self, parse_tree, tree):
        super().__init__()
        self.setWindowTitle("Parse Tree Viewer")
        self.setGeometry(100, 100, 800, 600)
        layout = QVBoxLayout()
        self.tree_widget = QTreeWidget()
        self.tree_widget.setHeaderLabels(["Parse Tree"])
        layout.addWidget(self.tree_widget)
        self.setLayout(layout)
        self.setup_parse_tree_in_widget(parse_tree)  # Adjusted to directly use parse_tree
        self.export_parse_tree_to_json(tree, f"{input_file}_parse_tree.json")

    def setup_parse_tree_in_widget(self, parse_tree, parent_item=None):
        self.tree_widget.setHeaderLabels(["Parse Tree"])
        parent_item = QTreeWidgetItem(self.tree_widget)
        self.add_children(parent_item, parse_tree)
        QTreeWidgetItem(parent_item, ["<EOF>"])  # Add <EOF> as the final part

    def get_node_text(self, node):
        if isinstance(node, TerminalNode):
            return node.getText()
        elif isinstance(node, ParserRuleContext):
            # Use the rule name as text for ParserRuleContext nodes.
            return type(node).__name__
        else:
            # Fallback for any other types of nodes.
            return 'Unknown Node'

    def add_children(self, parent_item, text):
        if text.startswith("(") and text.endswith(")"):
            text = text[1:-1]
        newText = ""
        start = 0
        for i in range(start, len(text)):
            if text[i] != ")" and text[i] != "(":
                newText += text[i]
            else:
                break
        item = QTreeWidgetItem(parent_item, [newText])
        start_index = 0
        while True:
            start = text.find("(", start_index)
            if start == -1:
                break
            count = 1
            for i in range(start + 1, len(text)):
                if text[i] == "(":
                    count += 1
                elif text[i] == ")":
                    count -= 1
                    if count == 0:
                        end = i
                        break
            else:
                return
            self.add_children(item, text[start:end + 1])
            start_index = end + 1

    def display_parse_tree(self, parse_tree):
        # Set up the tree widget for displaying the parse tree
        self.tree_widget.setHeaderLabels(["Parse Tree"])
        parent_item = QTreeWidgetItem(self.tree_widget)
        self.add_children(parent_item, parse_tree)
        QTreeWidgetItem(parent_item, ["<EOF>"])  # Add <EOF> as the final part

    def determine_role(node_type):
        # Example role determination logic
        if node_type in ['IfStatement', 'ForStatement', 'WhileStatement']:
            return 'conditional'
        elif 'Function' in node_type or 'Method' in node_type:
            return 'function'
        elif 'compilationUnitContext' in node_type or 'compiler' in node_type or 'compilation' in node_type:
            return 'compiler'
        elif 'QualifiedNameContext' in node_type:
            return 'type reference'
        elif 'ImportDeclarationContext' in node_type:
            return 'import'
        elif 'ClassDeclarationContext' in node_type:
            return 'class declaration'
        elif 'MethodDeclarationContext' in node_type:
            return 'method declaration'
        elif 'VariableDeclaratorContext' in node_type or 'FieldDeclarationContext' in node_type:
            return 'variable declaration'
        elif 'Literal' in node_type:
            return 'literal value'
        elif 'ExpressionContext' in node_type:
            return 'expression'
        elif 'Modifier' in node_type:
            return 'modifier'
        elif 'Annotation' in node_type:
            return 'annotation'
        elif 'Parameter' in node_type:
            return 'parameter'
        elif 'GenericType' in node_type or 'TypeParameter' in node_type:
            return 'generic type'
        elif 'Comment' in node_type:
            return 'comment'
        elif 'SwitchCase' in node_type:
            return 'switch case'
        elif 'TryCatch' in node_type:
            return 'try-catch block'
        # Add more conditions here based on your grammar and AST structure
        else:
            return 'unknown'

    @staticmethod
    def parse_tree_to_dict(node):
        # Determine the node type and its role in the AST
        if isinstance(node, TerminalNode):
            # Terminal nodes often represent literals, identifiers, etc.
            text = node.getText()
            node_type = 'TERMINAL'
            role = 'identifier' if text.isidentifier() else 'literal'
            return {'type': node_type, 'text': text, 'role': role}

        elif isinstance(node, ParserRuleContext):
            node_type = type(node).__name__
            # Here, you would have logic to determine the role based on node_type
            # For example, if node_type indicates a function declaration, role could be 'function'
            role = ParseTreeViewer.determine_role(node_type)  # Placeholder for role determination logic
            children = [ParseTreeViewer.parse_tree_to_dict(child) for child in node.getChildren()]

            return {'type': node_type, 'children': children, 'role': role}

        else:
            return {'type': 'UNKNOWN', 'text': '', 'children': [], 'role': ''}

    @staticmethod
    def export_parse_tree_to_json(parse_tree, filename):
        tree_dict = ParseTreeViewer.parse_tree_to_dict(parse_tree)
        with open(filename, 'w') as file:
            json.dump(tree_dict, file, indent=4)

    # Updated function to traverse the AST and collect found reserved words, found variables, and found node types
    def traverse_ast(self, node, reserved_words, found_reserved_words=[], found_variables=[], found_node_types=[]):
        # Check if the current node is a terminal node
        if isinstance(node, TerminalNode):
            token_text = node.getText()
            if token_text in reserved_words:
                found_reserved_words.append(token_text)
            else:
                # Here you might want to add more conditions to accurately identify variables
                if self.is_java_identifier(token_text) and token_text not in found_variables:
                    found_variables.append(token_text)
            # Regardless of type, the text of a terminal node is its type here
            node_type = 'TERMINAL'
            if node_type not in found_node_types:
                found_node_types.append(node_type)

        # If the node is a ParserRuleContext (non-terminal), collect its type and visit all its children
        elif isinstance(node, ParserRuleContext):
            node_type = type(node).__name__
            if node_type not in found_node_types:
                found_node_types.append(node_type)

            if node.children is not None:  # Check if children is not None
                for child in node.children:
                    self.traverse_ast(child, reserved_words, found_reserved_words, found_variables, found_node_types)

        return found_reserved_words, found_variables, found_node_types

    def is_java_identifier(self, text):
        if not text:  # Check if the text is not empty
            return False
        if text[0].isdigit():  # Check if the first character is not a digit
            return False
        if not (text[0].isalpha() or text[0] == '_' or text[0] == '$'):  # Check if the first character is valid
            return False
        for char in text[1:]:  # Check the rest of the characters
            if not (char.isalnum() or char == '_' or char == '$'):
                return False
        return True


def ask_gpt(description, options):
    # Construct the prompt with the object description and option descriptions
    prompt = f"Object Description: {description}\n"
    prompt += f"Options: {options}\n"
    messages.append({"role": "user", "content": prompt})
    response = client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=messages,
        stream=True
    )
    return response


def ask_gpt_summary(input_messages):
    response = client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=input_messages,
        stream=True
    )
    return response


def ask_gpt_about_java_file(url):
    def fetch_java_file(url):
        """Fetches raw content of a Java file from GitHub."""
        response = requests.get(url)
        if response.status_code == 200:
            return response.text
        else:
            raise Exception(f"Failed to retrieve the file: HTTP {response.status_code}")

    """Fetches a Java file and asks GPT to analyze or summarize the file."""
    try:
        java_code = fetch_java_file(url)

        # Construct the prompt for the GPT model to analyze or summarize the entire Java file
        prompt = f"Analyze the following Java code:\n{java_code}\n---\nProvide a detailed summary or insight into the code's functionality and structure."

        # OpenAI API call setup (make sure to configure API key properly)
        response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=prompt,
            stream=True  # You can adjust max_tokens as necessary
        )
        return response.choices[0].message['content'] if response.choices else "No response."
    except Exception as e:
        return str(e)

def extract_class_usages_from_file(file_path):
    imported_classes = set()
    variable_to_class = {}
    variable_to_methods = {}
    package_names = set()

    with open(file_path, 'r', encoding='utf-8') as file:
        lines = file.readlines()

    for line in lines:
        if line.strip().startswith('import '):
            full_class_path = line.strip()[7:].rstrip(';')
            all_imports.add(full_class_path)
            if ((full_class_path.startswith('java.') or full_class_path.startswith(
                    'javafx.') or full_class_path.startswith('javax.') or full_class_path.startswith(
                'org.apache.logging.log4j') or full_class_path.startswith('org.slf4j') or
                 full_class_path.startswith('org.controlsfx') or full_class_path.startswith(
                        'com.tobiasdiez.easybind')) or full_class_path.startswith('com.google.common') or
                    full_class_path.startswith('org.fxmisc.richtext') or full_class_path.startswith('org.jabref')):
                imported_classes.add(full_class_path)
                package_name = '.'.join(full_class_path.split('.')[:-1])
                package_names.add(package_name)

    # Map classes to methods
    class_to_methods = {}
    # Initialize methods set for each class in imported_classes
    for cls in imported_classes:
        class_to_methods[cls] = set()

    for line in lines:
        for full_class_path in imported_classes:
            class_name = full_class_path.split('.')[-1]
            # Check for new instance creation
            if f'new {class_name}' in line:
                parts = line.split('=')
                if len(parts) >= 2:
                    variable_name = parts[0].strip().split()[-1]
                    variable_to_class[variable_name] = full_class_path
            # Check for static method calls
            if f'{class_name}.' in line:
                method_start = line.find(f'{class_name}.') + len(class_name) + 1
                method_end = line.find('(', method_start)
                if method_end != -1:
                    method_name = line[method_start:method_end].strip()
                    class_to_methods[full_class_path].add(method_name)

        # Check for method invocations for existing variables
        for variable, assigned_class in variable_to_class.items():
            if f'{variable}.' in line:
                method_start = line.find(f'{variable}.') + len(variable) + 1
                method_end = line.find('(', method_start)
                if method_end != -1:
                    method_name = line[method_start:method_end].strip()
                    variable_to_methods.setdefault(variable, set()).add(method_name)
                    class_to_methods[assigned_class].add(method_name)

    # Convert sets to lists for output consistency
    for class_path, methods in class_to_methods.items():
        class_to_methods[class_path] = list(methods)

    return list(imported_classes), list(package_names), class_to_methods


def find_module_for_package(package_name):
    modules_to_packages = [
        ('java.base', ['java.io', 'java.lang', 'java.lang.annotation', 'java.lang.constant', 'java.lang.foreign',
                       'java.lang.invoke', 'java.lang.module', 'java.lang.ref', 'java.lang.reflect',
                       'java.lang.runtime', 'java.math',
                       'java.net', 'java.net.spi', 'java.nio', 'java.nio.channels', 'java.nio.channels.spi',
                       'java.nio.charset',
                       'java.nio.charset.spi', 'java.nio.file', 'java.nio.file.attribute', 'java.nio.file.spi',
                       'java.security',
                       'java.security.cert', 'java.security.interfaces', 'java.security.spec', 'java.text',
                       'java.text.spi', 'java.time',
                       'java.time.chrono', 'java.time.format', 'java.time.temporal', 'java.time.zone', 'java.util',
                       'java.util.concurrent', 'java.util.concurrent.atomic', 'java.util.concurrent.locks',
                       'java.util.function',
                       'java.util.jar', 'java.util.random', 'java.util.regex', 'java.util.spi', 'java.util.stream',
                       'java.util.zip',
                       'javax.crypto', 'javax.crypto.interfaces', 'javax.crypto.spec', 'javax.net', 'javax.net.ssl',
                       'javax.security.auth', 'javax.security.auth.callback', 'javax.security.auth.login',
                       'javax.security.auth.spi',
                       'javax.security.auth.x500', 'javax.security.cert']),
        ('java.compiler', ['javax.annotation.processing', 'javax.lang.model', 'javax.lang.model.element',
                           'javax.lang.model.type', 'javax.lang.model.util', 'javax.tools']),
        ('java.datatransfer', ['java.awt.datatransfer']),
        ('java.desktop',
         ['java.applet', 'java.awt', 'java.awt.color', 'java.awt.desktop', 'java.awt.dnd', 'java.awt.event',
          'java.awt.font', 'java.awt.geom', 'java.awt.im', 'java.awt.im.spi', 'java.awt.image',
          'java.awt.image.renderable',
          'java.awt.print', 'java.beans', 'java.beans.beancontext', 'javax.accessibility', 'javax.imageio',
          'javax.imageio.event', 'javax.imageio.metadata', 'javax.imageio.plugins.bmp', 'javax.imageio.plugins.jpeg',
          'javax.imageio.plugins.tiff', 'javax.imageio.spi', 'javax.imageio.stream', 'javax.print',
          'javax.print.attribute',
          'javax.print.attribute.standard', 'javax.print.event', 'javax.sound.midi', 'javax.sound.midi.spi',
          'javax.sound.sampled', 'javax.sound.sampled.spi', 'javax.swing', 'javax.swing.border',
          'javax.swing.colorchooser',
          'javax.swing.event', 'javax.swing.filechooser', 'javax.swing.plaf', 'javax.swing.plaf.basic',
          'javax.swing.plaf.metal', 'javax.swing.plaf.multi', 'javax.swing.plaf.nimbus', 'javax.swing.plaf.synth',
          'javax.swing.table', 'javax.swing.text', 'javax.swing.text.html', 'javax.swing.text.html.parser',
          'javax.swing.text.rtf', 'javax.swing.tree', 'javax.swing.undo']),
        ('java.instrument', ['java.lang.instrument']),
        ('java.logging', ['java.util.logging']),
        ('java.management', ['java.lang.management', 'javax.management', 'javax.management.loading',
                             'javax.management.modelmbean', 'javax.management.monitor', 'javax.management.openmbean',
                             'javax.management.relation', 'javax.management.remote', 'javax.management.timer']),
        ('java.management.rmi', ['javax.management.remote.rmi']),
        ('java.naming', ['javax.naming', 'javax.naming.directory', 'javax.naming.event', 'javax.naming.ldap',
                         'javax.naming.ldap.spi', 'javax.naming.spi']),
        ('java.net.http', ['java.net.http']),
        ('java.prefs', ['java.util.prefs']),
        ('java.rmi', ['java.rmi', 'java.rmi.dgc', 'java.rmi.registry', 'java.rmi.server', 'javax.rmi.ssl']),
        ('java.scripting', ['javax.script']),
        ('java.se', []),
        ('java.security.jgss', ['javax.security.auth.kerberos', 'org.ietf.jgss']),
        ('java.security.sasl', ['javax.security.sasl']),
        ('java.smartcardio', ['javax.smartcardio']),
        ('java.sql', ['java.sql', 'javax.sql']),
        ('java.sql.rowset', ['javax.sql.rowset', 'javax.sql.rowset.serial', 'javax.sql.rowset.spi']),
        ('java.transaction.xa', ['javax.transaction.xa']),
        (
            'java.xml',
            ['javax.xml', 'javax.xml.catalog', 'javax.xml.datatype', 'javax.xml.namespace', 'javax.xml.parsers',
             'javax.xml.stream', 'javax.xml.stream.events', 'javax.xml.stream.util', 'javax.xml.transform',
             'javax.xml.transform.dom', 'javax.xml.transform.sax', 'javax.xml.transform.stax',
             'javax.xml.transform.stream',
             'javax.xml.validation', 'javax.xml.xpath', 'org.w3c.dom', 'org.w3c.dom.bootstrap',
             'org.w3c.dom.events',
             'org.w3c.dom.ls', 'org.w3c.dom.ranges', 'org.w3c.dom.traversal', 'org.w3c.dom.views',
             'org.xml.sax',
             'org.xml.sax.ext', 'org.xml.sax.helpers']),
        ('java.xml.crypto', ['javax.xml.crypto', 'javax.xml.crypto.dom', 'javax.xml.crypto.dsig',
                             'javax.xml.crypto.dsig.dom', 'javax.xml.crypto.dsig.keyinfo',
                             'javax.xml.crypto.dsig.spec']),
        ('jdk.accessibility', ['com.sun.java.accessibility.util']),
        ('jdk.attach', ['com.sun.tools.attach', 'com.sun.tools.attach.spi']),
        ('jdk.charsets', []),
        ('jdk.compiler',
         ['com.sun.source.doctree', 'com.sun.source.tree', 'com.sun.source.util', 'com.sun.tools.javac']),
        ('jdk.crypto.cryptoki', []),
        ('jdk.crypto.ec', []),
        ('jdk.dynalink', ['jdk.dynalink', 'jdk.dynalink.beans', 'jdk.dynalink.linker', 'jdk.dynalink.linker.support',
                          'jdk.dynalink.support']),
        ('jdk.editpad', []),
        ('jdk.hotspot.agent', []),
        ('jdk.httpserver', ['com.sun.net.httpserver', 'com.sun.net.httpserver.spi']),
        ('jdk.incubator.vector', ['jdk.incubator.vector']),
        ('jdk.jartool', ['jdk.security.jarsigner']),
        ('jdk.javadoc', ['jdk.javadoc.doclet']),
        ('jdk.jcmd', []),
        ('jdk.jconsole', ['com.sun.tools.jconsole']),
        ('jdk.jdeps', []),
        ('jdk.jdi', ['com.sun.jdi', 'com.sun.jdi.connect', 'com.sun.jdi.connect.spi', 'com.sun.jdi.event',
                     'com.sun.jdi.request']),
        ('jdk.jdwp.agent', []),
        ('jdk.jfr', ['jdk.jfr', 'jdk.jfr.consumer']),
        ('jdk.jlink', []),
        ('jdk.jpackage', []),
        ('jdk.jshell', ['jdk.jshell', 'jdk.jshell.execution', 'jdk.jshell.spi', 'jdk.jshell.tool']),
        ('jdk.jsobject', ['netscape.javascript']),
        ('jdk.jstatd', []),
        ('jdk.localedata', []),
        ('jdk.management', ['com.sun.management']),
        ('jdk.management.agent', []),
        ('jdk.management.jfr', ['jdk.management.jfr']),
        ('jdk.naming.dns', []),
        ('jdk.naming.rmi', []),
        ('jdk.net', ['jdk.net', 'jdk.nio']),
        ('jdk.nio.mapmode', ['jdk.nio.mapmode']),
        ('jdk.sctp', ['com.sun.nio.sctp']),
        ('jdk.security.auth', ['com.sun.security.auth', 'com.sun.security.auth.callback', 'com.sun.security.auth.login',
                               'com.sun.security.auth.module']),
        ('jdk.security.jgss', ['com.sun.security.jgss']),
        ('jdk.xml.dom', ['org.w3c.dom.css', 'org.w3c.dom.html', 'org.w3c.dom.stylesheets', 'org.w3c.dom.xpath']),
        ('jdk.zipfs', [])
    ]
    # Loop through the modules to find which one includes this package
    for module, packages in modules_to_packages:
        if package_name in packages:
            return module
    return None


# https://docs.oracle.com/en/java/javase/22/docs/api/java.base/java/io/BufferedInputStream.html
def fetch_method_descriptions(module_name, full_class_name, methods):
    use_new_html_structure = module_name is not None  # False if JavaFX (None), true otherwise
    use_log4j_html_structure = full_class_name.startswith('org.apache.logging.log4j')
    use_slf4j_html_structure = full_class_name.startswith('org.slf4j')
    use_controlsfx_html_structure = full_class_name.startswith('org.controlsfx')
    use_tobiasdiez_html_structure = full_class_name.startswith('com.tobiasdiez.easybind')
    use_google_html_structure = full_class_name.startswith('com.google.common')
    use_richtext_html_structure = full_class_name.startswith('org.fxmisc.richtext')

    def construct_base_url(module, package_name):
        # Determine if the class is from Log4J or a regular Java package
        if 'org.apache.logging.log4j' in package_name:
            return f"https://logging.apache.org/log4j/2.x/javadoc/log4j-api/{package_name.replace('.', '/')}/"
        elif 'org.slf4j' in package_name:
            return f"https://www.slf4j.org/api/{package_name.replace('.', '/')}/"
        elif 'org.controlsfx' in package_name:
            return f"https://javadoc.io/static/org.controlsfx/controlsfx/8.40.14/{package_name.replace('.', '/')}/"
        elif 'com.tobiasdiez.easybind' in package_name:
            return f"https://github.com/tobiasdiez/EasyBind/blob/main/src/main/java/com/tobiasdiez/easybind/"
        elif 'com.google.common' in package_name:
            return f"https://guava.dev/releases/19.0/api/docs/{package_name.replace('.', '/')}/"
        elif 'org.fxmisc.richtext' in package_name:
            return f"http://fxmisc.github.io/richtext/javadoc/0.8.1/{package_name.replace('.', '/')}/"
        elif module is None:  # For JavaFX classes
            return f"https://docs.oracle.com/javafx/2/api/{package_name.replace('.', '/')}/"
        else:  # Default to Java SE documentation
            return f"https://docs.oracle.com/en/java/javase/22/docs/api/{module}/{package_name.replace('.', '/')}/"

    # Split the full class name into its package and class name parts
    parts = full_class_name.split('.')
    package_name = '/'.join(parts[:-1])  # Form the package path
    class_name = parts[-1]  # Get the class name
    base_url = construct_base_url(module_name, package_name)

    # Construct the URL to the class documentation page
    class_url = f"{base_url}{class_name}.html"
    print(f"Fetching documentation from: {class_url}")
    program_output_file.write(f"Fetching documentation from: {class_url}")

    # Fetch the content of the class documentation page
    response = requests.get(class_url)
    # print(f"Response status: {response.status_code}")

    # Fetch the content of the class documentation page
    response = requests.get(class_url)
    print(f"Response status: {response.status_code}")
    soup = BeautifulSoup(response.text, 'html.parser')
    method_descriptions = {}

    # Function to search for method descriptions within the provided soup
    def search_method_descriptions(soup, method, use_new_html_structure):
        descriptions = []  # List to store all found descriptions for this method
        # print(f"DEBUG: Starting search for {method} using {'new' if use_new_html_structure else 'old'} HTML structure.")  # Debug statement
        if use_new_html_structure:
            for method_tag in soup.find_all('code'):
                if method in method_tag.text:
                    # print(f"DEBUG: Found method tag for {method}.")  # Debug statement
                    parent_div = method_tag.find_parent('div')
                    if parent_div:
                        next_div = parent_div.find_next_sibling('div')
                        if next_div:
                            description_tags = next_div.find_all('div', class_='block')
                            if description_tags:
                                full_description = ' '.join(d.text.strip() for d in description_tags)
                                descriptions.append(full_description)
                                # print(f"DEBUG: Added description for {method}.")  # Debug statement
        elif use_log4j_html_structure:
            for method_tag in soup.find_all('code'):
                if method in method_tag.text:
                    print(f"DEBUG: Found method tag for {method}.")  # Debug statement
                    parent_div = method_tag.find_parent('div')
                    if parent_div:
                        next_div = parent_div.find_next_sibling('div')
                        if next_div:
                            description_tags = next_div.find_all('div', class_='block')
                            if description_tags:
                                full_description = ' '.join(d.text.strip() for d in description_tags)
                                descriptions.append(full_description)
                                # print(f"DEBUG: Added description for {method}.")  # Debug statement
        elif use_slf4j_html_structure:
            for method_tag in soup.find_all('code'):
                if method in method_tag.text:
                    parent_div = method_tag.find_parent('div')
                    if parent_div:
                        next_div = parent_div.find_next_sibling('div')
                        if next_div:
                            description_tags = next_div.find_all('div', class_='block')
                            if description_tags:
                                full_description = ' '.join(d.text.strip() for d in description_tags)
                                descriptions.append(full_description)
                                # print(f"DEBUG: Added description for {method}.")  # Debug statement
        elif use_controlsfx_html_structure:
            for method_tag in soup.find_all('code'):
                # Navigate to the parent 'td' of this 'a' tag
                parent_td = method_tag.find_parent('td', class_='colLast')
                if parent_td:
                    # Find the 'div' with class "block" within this 'td'
                    description_tag = parent_td.find('div', class_='block')
                    if description_tag:
                        full_description = description_tag.get_text(strip=True)
                        descriptions.append(full_description)
        elif use_google_html_structure:
            # Iterate over all `h4` tags which might contain method names
            for method_tag in soup.find_all('h4'):
                # Check if the method name is in the current `h4` text
                if method in method_tag.text:
                    # Find the next 'pre' sibling (assumes 'pre' is immediately after 'h4')
                    pre_tag = method_tag.find_next_sibling('pre')
                    if pre_tag:
                        # Find the next sibling 'div' of 'pre' with class 'block' that contains the description
                        method_description_div = pre_tag.find_next_sibling('div', class_='block')
                        if method_description_div:
                            # Extract the text and strip whitespace
                            full_description = ' '.join(method_description_div.text.strip().split())
                            descriptions.append(full_description)
                            # Debug output
                            print(f"DEBUG: Added description for {method}.")  # Debug statement
                        else:
                            print("DEBUG: No description div found.")
                    else:
                        print("DEBUG: No 'pre' tag found next to 'h4'.")
        elif use_tobiasdiez_html_structure:
            descriptions = ask_gpt_about_java_file(class_url)
        elif use_richtext_html_structure:
            for method in methods:
                found = False
                for code_tag in soup.find_all('code'):
                    if method in code_tag.text:
                        # Try to find the description for this method
                        next_div = code_tag.parent.find_next_sibling('div', class_='block')
                        if next_div:
                            method_descriptions[method] = next_div.text.strip()
                            found = True
                            break
                if not found:
                    for method_tag in soup.find_all('code'):
                        inherited_methods = {}
                        # Find the section containing inherited methods
                        for h3 in soup.find_all('h3'):
                            if 'Methods inherited from class' in h3.text:
                                inherited_package = h3.text.replace('Methods inherited from class', '').strip()
                                package_name = ".".join(inherited_package.rsplit('.')[:-1])
                                class_name = inherited_package.split('.')[-1]
                                # Now, look for the specific method within the inherited class content
                                description = fetch_method_descriptions(find_module_for_package(package_name))
                                if description:
                                    inherited_methods[method_tag] = description
                                break
                        return inherited_methods
        else:  # Old HTML structure, typically used for JavaFX
            for method_tag in soup.find_all('code'):
                if method in method_tag.text:
                    # print(f"DEBUG: Found method tag for {method} in JavaFX documentation.")  # Debug statement
                    td_tag = method_tag.find_parent('td')
                    if td_tag:
                        description_tags = td_tag.find_all('div', class_='block')
                        if description_tags:
                            full_description = ' '.join(d.text.strip() for d in description_tags)
                            descriptions.append(full_description)
                            # print(f"DEBUG: Added description for {method} in JavaFX documentation.")  # Debug statement
        if not descriptions:
            prompt = f'Please describe the following function {method} from this import file {full_class_name}.'
            messages_description_summ.append({"role": "user", "content": prompt})
            test = ask_gpt_summary(messages_description_summ)
            answer = ""
            for chunk in test:
                if chunk.choices[0].delta.content:
                    answer += (chunk.choices[0].delta.content.strip('*') or "")
            package_description_gpt = answer.strip('#### ')
            package_description = package_description_gpt
            program_output_file.write(answer.strip('#### '))
            messages_description_summ.pop()
            function_context.append({"fullname": full_class_name, "name": method, "other": package_description})

            prompt = f'Please comprise, paraphrase, or summarize all the following method descriptions from {full_class_name} into one encompassing description: {package_description}'
            messages_overload_summ.append({"role": "user", "content": prompt})
            gpt_response = ask_gpt_summary(messages_overload_summ)
            answer = ""
            for chunk in gpt_response:
                if chunk.choices[0].delta.content:
                    answer += (chunk.choices[0].delta.content.strip('*') or "")
            print(f'Class: {full_class_name}, Method : {method},  {answer.strip("#### ")}\n')
            program_output_file.write(f'Class: {full_class_name}, Method : {method},  {answer.strip("#### ")}\n')
            messages_overload_summ.pop()  # Remove last user prompt
            function_topic.append({"fullname": full_class_name, "name": method, "other": answer.strip("#### ")})

            # Find the dictionary where the 'name' key matches the name to find
            gpt_label_dict = next((item for item in api_gpt if item["name"] == full_class_name), None)
            gpt_sim_dict = next((item for item in api_sim if item["name"] == full_class_name), None)
            print(full_class_name)
            gpt_label = gpt_label_dict["other"]
            gpt_sim = gpt_sim_dict["other"]

            # Split the string at the first colon
            sim_received = gpt_sim.split(":", 1)

            # Take the first part
            gpt_sim = sim_received[0].strip()

            # Convert the extracted text to lowercase and replace spaces with underscores
            formatted_label = gpt_sim.lower().replace(" ", "_")
            variable_name = formatted_label + "_options"
            try:
                sim_options_list = globals()[variable_name]
            except KeyError:
                sim_options_list = globals()['utility_options']

            # Find the index of "Label:" and "Reason:"
            label_index = gpt_label.find("Label:")
            reason_index = gpt_label.find("Reason:")

            # Extract the text between "Label:" and "Reason:"
            label_text = gpt_label[label_index + len("Label:"):reason_index].strip()

            # Convert the extracted text to lowercase and replace spaces with underscores
            formatted_label = label_text.lower().replace(" ", "_")
            variable_name = formatted_label + "_options"
            try:
                options_list = globals()[variable_name]
            except KeyError:
                options_list = globals()['utility_options']

            givenOptions = ""
            similarity_results = []

            for options in options_list:
                for option, descriptions in options.items():
                    givenOptions += (option + ": " + descriptions)

            for options in sim_options_list:
                for option, descriptions in options.items():
                    class_desc = nlp(answer.strip("#### "))
                    given_desc = nlp(option + ": " + descriptions)
                    similarity_results.append(class_desc.similarity(given_desc))
            greatest_value = 0
            greatest_value_index = 0
            for i in range(len(similarity_results)):
                if similarity_results[i] > greatest_value:
                    greatest_value = similarity_results[i]
                    greatest_value_index = i
            # Get the option at the greatest_value_index
            option = sim_options_list[greatest_value_index]

            # Extract the label and description from the option dictionary
            label = list(option.keys())[0]
            description = list(option.values())[0]
            function_sim.append({"fullname": full_class_name, "name": method,
                                 "other": label + ": " + description})
            answer = ""
            response = ask_gpt(package_description, givenOptions)
            for chunk in response:
                if chunk.choices[0].delta.content:
                    answer += (chunk.choices[0].delta.content.strip('*') or "")
            answer = answer.strip('#### ')
            words = answer.split()
            answer = ''
            counter = 0
            for i, word in enumerate(words):
                if word == "Reason:":
                    answer += '\n'
                    counter -= i
                answer += word
                counter += 1
                if (counter + 1) % 17 == 0:  # Add newline every x words
                    answer += '\n'
                else:
                    answer += ' '
            print(answer + '\n')
            function_gpt.append({"fullname": full_class_name, "name": method, "other": answer})
            # print(f"No descriptions found for {method}.")  # Debug statement
            # program_output_file.write(f"No descriptions found for {method}.")
        return descriptions

    # Search for the documentation of each method in the list
    for method in methods:
        # print(f"Searching for method: {method}")
        descriptions = search_method_descriptions(soup, method, use_new_html_structure)

        # If no description found and not a JavaFX class, look for inherited methods
        if not descriptions and use_new_html_structure:
            # Find the div that contains inherited methods
            inherited_sections = soup.find_all("div", class_="inherited-list")
            for section in inherited_sections:
                # The class name from which methods are inherited is in the 'h3' tag's 'id' attribute
                class_info = section.find('h3')
                if class_info:
                    class_name_from_id = class_info.get('id', '').replace('methods-inherited-from-class-', '').replace(
                        '.', '/')
                    if class_name_from_id:  # Ensure the class name was successfully extracted
                        # print(f"DEBUG: Found inherited section for class: {class_name_from_id}")  # Debug statement
                        # Iterate over all 'a' tags within 'code' elements
                        for link in section.find_all('a'):
                            if method.lower() == link.text.lower():  # Case-insensitive comparison
                                # Extract just the path to the class documentation, not including method specifics
                                inherited_href = link['href']
                                # Assume inherited_href is like "../AbstractQueue.html#element()", need to remove '../' and method specifics
                                inherited_class_path = inherited_href.split('#')[0].replace('../',
                                                                                            '').strip()  # Now should be "AbstractQueue.html"
                                inherited_class = inherited_class_path.replace('.html',
                                                                               '')  # Now should be just "AbstractQueue"

                                # Now we use the class name from the section's id for package structure, which we have already cleaned in class_name_from_id
                                # Assuming class_name_from_id is like "java/util/AbstractQueue", split and join to format correctly
                                if class_name_from_id:
                                    inherited_package_path = '/'.join(class_name_from_id.split('/')[
                                                                      :-1])  # Get the path without the class name itself
                                    inherited_url = f"https://docs.oracle.com/en/java/javase/22/docs/api/{module_name}/{inherited_package_path}/{inherited_class}.html"
                                    # print(f"DEBUG: Fetching documentation from inherited class at {inherited_url}.")
                                    # Proceed with fetching and parsing the inherited documentation
                                inherited_response = requests.get(inherited_url)
                                inherited_soup = BeautifulSoup(inherited_response.text, 'html.parser')
                                descriptions = search_method_descriptions(inherited_soup, method,
                                                                          use_new_html_structure)
                                if descriptions:
                                    # print(f"DEBUG: Found descriptions for {method} in inherited class.")
                                    break  # Stop searching if descriptions are found
                        if descriptions:
                            # print(f"DEBUG: Descriptions found for {method} after checking inherited classes.")
                            break  # Stop searching other inherited sections if found
            if not descriptions:
                print(f"Descriptions for {method} not found in inherited classes either.")  # Debug statement
                program_output_file.write(f"Descriptions for {method} not found in inherited classes either.")

        # Add the found descriptions to the method descriptions dictionary
        if descriptions:
            method_descriptions[method] = descriptions
        else:
            print(f"Method {method} not found in the documentation.")
            program_output_file.write(f"Method {method} not found in the documentation.")

    # Return the collected method descriptions
    return method_descriptions


def fetch_import_description(full_import_name):
    if(import_in_file(full_import_name)):
        print("Fetching description for " + full_import_name + "\n")
        words = ""
        started = False
        with open('./JabrefDocumentation.txt', 'r') as file:
            for line in file:
                if full_import_name in line:
                    started = True
                elif started:
                    if not line.startswith("org.jabref"):
                        words += " ".join(line.split()) + " "
                    else:
                        started = False
        print(words + "\n")
        api_topic.append({"name": full_import_name, "other": words})

        # Directory where the .txt files are located
        txt_files_directory = './program_outputs'

        # Name of the .java file to check for
        java_file_name_to_check = os.path.splitext(full_import_name)[-1].split('.')[-1] + '.java_output.txt'
        print("SCHESJD: " + java_file_name_to_check)
        # Check if the .java file exists among the .txt files
        java_file_found = False
        documentation = "Not Found"
        for txt_file_name in os.listdir(txt_files_directory):
            # Check if the file is a .txt file
            if txt_file_name.endswith('.txt'):
                # Extract the base name of the .java file (without extension) from the .txt file name
                java_file_base_name = os.path.splitext(txt_file_name)[0]
                # Check if the .java file name matches the one we're looking for
                if java_file_base_name == os.path.splitext(java_file_name_to_check)[0]:
                    java_file_found = True
                    txt_file_path = os.path.join(txt_files_directory, txt_file_name)
                    with open(txt_file_path, 'r') as txt_file:
                        print(f"Contents of {java_file_name_to_check}:")
                        documentation = (txt_file.read())
                    break
        if java_file_found:
            api_context.append({"name": full_import_name, "other": documentation})
        else:
            api_context.append({"name": full_import_name, "other": "Not Found"})

        options = {
            "Application": "Software components designed by third parties or as plugins to enhance specific functionalities within a system.",
            "Application Performance Manager": "Tools or systems dedicated to monitoring, analyzing, and optimizing the performance of various software applications.",
            "Big Data": "APIs tailored for handling and managing vast amounts of data, encompassing diverse formats and structures.",
            "Cloud": "Software tools and services delivered over the Internet, facilitating remote access, scalability, and flexibility.",
            "Computer Graphics": "Technologies focused on creating, editing, and rendering visual content, encompassing various media formats.",
            "Data Structure": "Patterns and frameworks governing the organization, storage, and manipulation of data, including collections, lists, and trees.",
            "Databases": "Systems and tools for storing, managing, and retrieving structured data and associated metadata.",
            "Software Development and IT": "Libraries and frameworks catering to version control, continuous integration, and deployment processes.",
            "Error Handling": "Strategies and mechanisms designed to detect, respond to, and recover from errors or exceptional conditions within software systems.",
            "Event Handling": "Mechanisms and components responsible for detecting, processing, and responding to events triggered within software applications.",
            "Geographic Information System": "Technologies dealing with the storage, analysis, and visualization of spatially referenced data and geographic information.",
            "Input/Output": "Functionalities and interfaces facilitating the reading from and writing to various data sources and destinations.",
            "Interpreter": "Features and functionalities associated with interpreting and executing code or scripts within a software environment.",
            "Internationalization": "Tools and frameworks enabling the adaptation of software applications to diverse linguistic, cultural, and regional contexts.",
            "Logic": "Frameworks and patterns governing the organization and execution flow of software applications, including control structures and architectural paradigms.",
            "Language": "Features and capabilities inherent to programming languages, including syntax, semantics, and data type conversions.",
            "Logging": "Mechanisms for recording and storing activity and status information generated by software applications for monitoring, debugging, and analysis purposes.",
            "Machine Learning": "Tools and libraries supporting the development, training, and deployment of machine learning models based on data analysis and pattern recognition.",
            "Microservices/Services": "Decomposed and independently deployable software components facilitating modular and scalable application architectures and inter-application communication.",
            "Multimedia": "Technologies enabling the representation and manipulation of information across various media formats, including text, audio, and video.",
            "Multithread": "Support for concurrent execution and management of multiple threads within a software application or system.",
            "Natural Language Processing": "Technologies and algorithms enabling the processing, understanding, and generation of human language data within computational systems.",
            "Network": "Protocols, APIs, and tools facilitating communication and data exchange between networked devices and systems.",
            "Operating System": "Interfaces and functionalities providing access to and management of a computer's hardware and software resources, including system-level APIs.",
            "Parser": "Components and algorithms responsible for analyzing and interpreting data or code structures, often breaking them down into identifiable components for further processing.",
            "Search": "APIs and tools facilitating the retrieval and manipulation of information from various data sources, particularly for web-based searching and indexing.",
            "Security": "Technologies, protocols, and practices aimed at safeguarding data, systems, and communications from unauthorized access, breaches, and malicious activities.",
            "Setup": "Configurations, settings, and initialization processes necessary for setting up and configuring software applications or systems.",
            "User Interface": "Components and frameworks defining the visual and interactive elements of software applications, including forms, screens, and graphical controls.",
            "Utility": "Third party libraries for general-purpose functions and utilities.",
            "Test": "Frameworks and tools facilitating the automation, execution, and management of software testing processes and procedures."
        }
        givenOptions = ""
        similarity_results = []
        for option, descriptions in options.items():
            givenOptions += (option + ": " + descriptions)
            class_desc = nlp(documentation)
            given_desc = nlp(option + ": " + descriptions)
            similarity_results.append(class_desc.similarity(given_desc))
        greatest_value = 0
        greatest_value_index = 0
        for i in range(len(similarity_results)):
            if similarity_results[i] > greatest_value:
                greatest_value = similarity_results[i]
                greatest_value_index = i
        # print(greatest_value_index)
        print("\nSimilarity Score: " + str(similarity_results[greatest_value_index].__round__(4)))
        program_output_file.write(
            f"\nSimilarity Score: {str(similarity_results[greatest_value_index].__round__(4))}\n")
        print(
            list(options.keys())[greatest_value_index] + ": " + list(options.values())[greatest_value_index] + '\n')
        api_sim.append({"name": full_import_name,
                        "other": list(options.keys())[greatest_value_index] + ": " + list(options.values())[
                            greatest_value_index]})
        program_output_file.write(
            list(options.keys())[greatest_value_index] + ": " + list(options.values())[greatest_value_index] + '\n')
        answer = ""
        if not check_package_in_file(full_import_name):
            response = ask_gpt(documentation, givenOptions)
            for chunk in response:
                if chunk.choices[0].delta.content:
                    answer += (chunk.choices[0].delta.content.strip('*') or "")
            answer = answer.strip('#### ')
            words = answer.split()
            answer = ''
            counter = 0
            for i, word in enumerate(words):
                if word == "Reason:":
                    answer += '\n'
                    counter -= i
                answer += word
                counter += 1
                if (counter + 1) % 17 == 0:  # Add newline every x words
                    answer += '\n'
                else:
                    answer += ' '
            save_data_to_text_file(full_import_name, answer)
        else:
            start = False
            with open(file_path, 'r') as file:
                for line in file:
                    if not start:
                        words = line.split()
                        if full_import_name in words:
                            start = True
                    elif not 'END' in line:
                        answer += line
                    else:
                        start = False
        print(answer + '\n')
        api_gpt.append({"name": full_import_name, "other": answer})
        program_output_file.write(answer + '\n')

    else:
        # Split the full class name into its package and class name parts
        parts = full_import_name.split('.')
        package_name = '/'.join(parts[:-1])  # Join all but the last part with '/', to form the package path
        class_name = parts[-1]  # The last part is the class name
        # Determine the base URL based on whether the class is part of JavaFX
        if full_import_name.startswith('javafx.'):
            base_url = f"https://docs.oracle.com/javase/8/javafx/api/{package_name}/"
        else:
            base_url = f"https://docs.oracle.com/javase/8/docs/api/{package_name}/"

        class_url = f'{base_url}{class_name}/package-summary.html'
        print(f"Fetching documentation from: {class_url}")  # Debug
        response = requests.get(class_url)
        #print(f"Response status: {response.status_code}")  # Debug
        if response.url != class_url:
            print(f'Unable to retrieve documentation')
            class_url = f"{base_url}{class_name}.html"
            print(f'Fetching documentation from: {class_url}')
            response = requests.get(class_url)
        # print(f"Response status: {response.status_code}")
        if response.url == class_url:
            soup = BeautifulSoup(response.text, 'html.parser')

            print(f'Fetching description for {full_import_name}: ')
            program_output_file.write(f'\n\nDescription for {full_import_name}:\n')
            description_naming_conventions = [('div', 'docSummary'), ('div', 'description')]
            i = 0
            package_description = False
            while not package_description and i < len(description_naming_conventions):
                package_description = soup.find(description_naming_conventions[i][0], description_naming_conventions[i][1])
                i += 1
            if package_description:
                program_output_file.write(f'Documentation URL: {class_url}\n')
                package_description = package_description.text
                print(package_description)
                api_context.append({"name": full_import_name, "other": package_description})
                prompt = f'Please summarize the following description concisely and accurately. Description: {package_description}. End of Description.'
                messages_description_summ.append({"role": "user", "content": prompt})
                test = ask_gpt_summary(messages_description_summ)
                answer = ""
                for chunk in test:
                   if chunk.choices[0].delta.content:
                       answer += (chunk.choices[0].delta.content.strip('*') or "")
                package_description_gpt = answer.strip('#### ')
                package_description = package_description_gpt
                program_output_file.write(answer.strip('#### '))
                messages_description_summ.pop()
                api_topic.append({"name": full_import_name, "other": package_description})

                options = {
                    "Application": "Software components designed by third parties or as plugins to enhance specific functionalities within a system.",
                    "Application Performance Manager": "Tools or systems dedicated to monitoring, analyzing, and optimizing the performance of various software applications.",
                    "Big Data": "APIs tailored for handling and managing vast amounts of data, encompassing diverse formats and structures.",
                    "Cloud": "Software tools and services delivered over the Internet, facilitating remote access, scalability, and flexibility.",
                    "Computer Graphics": "Technologies focused on creating, editing, and rendering visual content, encompassing various media formats.",
                    "Data Structure": "Patterns and frameworks governing the organization, storage, and manipulation of data, including collections, lists, and trees.",
                    "Databases": "Systems and tools for storing, managing, and retrieving structured data and associated metadata.",
                    "Software Development and IT": "Libraries and frameworks catering to version control, continuous integration, and deployment processes.",
                    "Error Handling": "Strategies and mechanisms designed to detect, respond to, and recover from errors or exceptional conditions within software systems.",
                    "Event Handling": "Mechanisms and components responsible for detecting, processing, and responding to events triggered within software applications.",
                    "Geographic Information System": "Technologies dealing with the storage, analysis, and visualization of spatially referenced data and geographic information.",
                    "Input/Output": "Functionalities and interfaces facilitating the reading from and writing to various data sources and destinations.",
                    "Interpreter": "Features and functionalities associated with interpreting and executing code or scripts within a software environment.",
                    "Internationalization": "Tools and frameworks enabling the adaptation of software applications to diverse linguistic, cultural, and regional contexts.",
                    "Logic": "Frameworks and patterns governing the organization and execution flow of software applications, including control structures and architectural paradigms.",
                    "Language": "Features and capabilities inherent to programming languages, including syntax, semantics, and data type conversions.",
                    "Logging": "Mechanisms for recording and storing activity and status information generated by software applications for monitoring, debugging, and analysis purposes.",
                    "Machine Learning": "Tools and libraries supporting the development, training, and deployment of machine learning models based on data analysis and pattern recognition.",
                    "Microservices/Services": "Decomposed and independently deployable software components facilitating modular and scalable application architectures and inter-application communication.",
                    "Multimedia": "Technologies enabling the representation and manipulation of information across various media formats, including text, audio, and video.",
                    "Multithread": "Support for concurrent execution and management of multiple threads within a software application or system.",
                    "Natural Language Processing": "Technologies and algorithms enabling the processing, understanding, and generation of human language data within computational systems.",
                    "Network": "Protocols, APIs, and tools facilitating communication and data exchange between networked devices and systems.",
                    "Operating System": "Interfaces and functionalities providing access to and management of a computer's hardware and software resources, including system-level APIs.",
                    "Parser": "Components and algorithms responsible for analyzing and interpreting data or code structures, often breaking them down into identifiable components for further processing.",
                    "Search": "APIs and tools facilitating the retrieval and manipulation of information from various data sources, particularly for web-based searching and indexing.",
                    "Security": "Technologies, protocols, and practices aimed at safeguarding data, systems, and communications from unauthorized access, breaches, and malicious activities.",
                    "Setup": "Configurations, settings, and initialization processes necessary for setting up and configuring software applications or systems.",
                    "User Interface": "Components and frameworks defining the visual and interactive elements of software applications, including forms, screens, and graphical controls.",
                    "Utility": "Third party libraries for general-purpose functions and utilities.",
                    "Test": "Frameworks and tools facilitating the automation, execution, and management of software testing processes and procedures."
                }
                givenOptions = ""
                similarity_results = []
                for option, descriptions in options.items():
                    givenOptions += (option + ": " + descriptions)
                    class_desc = nlp(package_description)
                    given_desc = nlp(option + ": " + descriptions)
                    similarity_results.append(class_desc.similarity(given_desc))
                greatest_value = 0
                greatest_value_index = 0
                for i in range(len(similarity_results)):
                    if similarity_results[i] > greatest_value:
                        greatest_value = similarity_results[i]
                        greatest_value_index = i
                # print(greatest_value_index)
                print("\nSimilarity Score: " + str(similarity_results[greatest_value_index].__round__(4)))
                program_output_file.write(
                    f"\nSimilarity Score: {str(similarity_results[greatest_value_index].__round__(4))}\n")
                print(
                    list(options.keys())[greatest_value_index] + ": " + list(options.values())[greatest_value_index] + '\n')
                api_sim.append({"name": full_import_name, "other": list(options.keys())[greatest_value_index] + ": " + list(options.values())[greatest_value_index]})
                program_output_file.write(
                    list(options.keys())[greatest_value_index] + ": " + list(options.values())[greatest_value_index] + '\n')
                answer = ""
                if not check_package_in_file(full_import_name):
                    response = ask_gpt(package_description, givenOptions)
                    for chunk in response:
                        if chunk.choices[0].delta.content:
                            answer += (chunk.choices[0].delta.content.strip('*') or "")
                    answer = answer.strip('#### ')
                    words = answer.split()
                    answer = ''
                    counter = 0
                    for i, word in enumerate(words):
                        if word == "Reason:":
                            answer += '\n'
                            counter -= i
                        answer += word
                        counter += 1
                        if (counter + 1) % 17 == 0:  # Add newline every x words
                            answer += '\n'
                        else:
                            answer += ' '
                    save_data_to_text_file(full_import_name, answer)
                else:
                    start = False
                    with open(file_path, 'r') as file:
                        for line in file:
                            if not start:
                                words = line.split()
                                if full_import_name in words:
                                    start = True
                            elif not 'END' in line:
                                answer += line
                            else:
                                start = False
                print(answer + '\n')
                api_gpt.append({"name": full_import_name, "other": answer})
                program_output_file.write(answer + '\n')

            else:
                print(f'Could not retrieve description for {full_import_name}\n')
                program_output_file.write(f'Could not retrieve description for {full_import_name}\n')
        else:
            prompt = f'Please summarize the following class concisely and accurately. Method: {full_import_name}. End of Description.'
            messages_description_summ.append({"role": "user", "content": prompt})
            test = ask_gpt_summary(messages_description_summ)
            answer = ""
            for chunk in test:
                if chunk.choices[0].delta.content:
                    answer += (chunk.choices[0].delta.content.strip('*') or "")
            package_description_gpt = answer.strip('#### ')
            package_description = package_description_gpt
            program_output_file.write(answer.strip('#### '))
            messages_description_summ.pop()
            api_context.append({"name": full_import_name, "other": package_description})
            api_topic.append({"name": full_import_name, "other": package_description})

            options = {
                "Application": "Software components designed by third parties or as plugins to enhance specific functionalities within a system.",
                "Application Performance Manager": "Tools or systems dedicated to monitoring, analyzing, and optimizing the performance of various software applications.",
                "Big Data": "APIs tailored for handling and managing vast amounts of data, encompassing diverse formats and structures.",
                "Cloud": "Software tools and services delivered over the Internet, facilitating remote access, scalability, and flexibility.",
                "Computer Graphics": "Technologies focused on creating, editing, and rendering visual content, encompassing various media formats.",
                "Data Structure": "Patterns and frameworks governing the organization, storage, and manipulation of data, including collections, lists, and trees.",
                "Databases": "Systems and tools for storing, managing, and retrieving structured data and associated metadata.",
                "Software Development and IT": "Libraries and frameworks catering to version control, continuous integration, and deployment processes.",
                "Error Handling": "Strategies and mechanisms designed to detect, respond to, and recover from errors or exceptional conditions within software systems.",
                "Event Handling": "Mechanisms and components responsible for detecting, processing, and responding to events triggered within software applications.",
                "Geographic Information System": "Technologies dealing with the storage, analysis, and visualization of spatially referenced data and geographic information.",
                "Input/Output": "Functionalities and interfaces facilitating the reading from and writing to various data sources and destinations.",
                "Interpreter": "Features and functionalities associated with interpreting and executing code or scripts within a software environment.",
                "Internationalization": "Tools and frameworks enabling the adaptation of software applications to diverse linguistic, cultural, and regional contexts.",
                "Logic": "Frameworks and patterns governing the organization and execution flow of software applications, including control structures and architectural paradigms.",
                "Language": "Features and capabilities inherent to programming languages, including syntax, semantics, and data type conversions.",
                "Logging": "Mechanisms for recording and storing activity and status information generated by software applications for monitoring, debugging, and analysis purposes.",
                "Machine Learning": "Tools and libraries supporting the development, training, and deployment of machine learning models based on data analysis and pattern recognition.",
                "Microservices/Services": "Decomposed and independently deployable software components facilitating modular and scalable application architectures and inter-application communication.",
                "Multimedia": "Technologies enabling the representation and manipulation of information across various media formats, including text, audio, and video.",
                "Multithread": "Support for concurrent execution and management of multiple threads within a software application or system.",
                "Natural Language Processing": "Technologies and algorithms enabling the processing, understanding, and generation of human language data within computational systems.",
                "Network": "Protocols, APIs, and tools facilitating communication and data exchange between networked devices and systems.",
                "Operating System": "Interfaces and functionalities providing access to and management of a computer's hardware and software resources, including system-level APIs.",
                "Parser": "Components and algorithms responsible for analyzing and interpreting data or code structures, often breaking them down into identifiable components for further processing.",
                "Search": "APIs and tools facilitating the retrieval and manipulation of information from various data sources, particularly for web-based searching and indexing.",
                "Security": "Technologies, protocols, and practices aimed at safeguarding data, systems, and communications from unauthorized access, breaches, and malicious activities.",
                "Setup": "Configurations, settings, and initialization processes necessary for setting up and configuring software applications or systems.",
                "User Interface": "Components and frameworks defining the visual and interactive elements of software applications, including forms, screens, and graphical controls.",
                "Utility": "Third party libraries for general-purpose functions and utilities.",
                "Test": "Frameworks and tools facilitating the automation, execution, and management of software testing processes and procedures."
            }
            givenOptions = ""
            similarity_results = []
            for option, descriptions in options.items():
                givenOptions += (option + ": " + descriptions)
                class_desc = nlp(package_description)
                given_desc = nlp(option + ": " + descriptions)
                similarity_results.append(class_desc.similarity(given_desc))
            greatest_value = 0
            greatest_value_index = 0
            for i in range(len(similarity_results)):
                if similarity_results[i] > greatest_value:
                    greatest_value = similarity_results[i]
                    greatest_value_index = i
            # print(greatest_value_index)
            print("\nSimilarity Score: " + str(similarity_results[greatest_value_index].__round__(4)))
            program_output_file.write(
                f"\nSimilarity Score: {str(similarity_results[greatest_value_index].__round__(4))}\n")
            print(
                list(options.keys())[greatest_value_index] + ": " + list(options.values())[greatest_value_index] + '\n')
            api_sim.append({"name": full_import_name,
                            "other": list(options.keys())[greatest_value_index] + ": " + list(options.values())[
                                greatest_value_index]})
            program_output_file.write(
                list(options.keys())[greatest_value_index] + ": " + list(options.values())[greatest_value_index] + '\n')
            answer = ""
            if not check_package_in_file(full_import_name):
                response = ask_gpt(package_description, givenOptions)
                for chunk in response:
                    if chunk.choices[0].delta.content:
                        answer += (chunk.choices[0].delta.content.strip('*') or "")
                answer = answer.strip('#### ')
                words = answer.split()
                answer = ''
                counter = 0
                for i, word in enumerate(words):
                    if word == "Reason:":
                        answer += '\n'
                        counter -= i
                    answer += word
                    counter += 1
                    if (counter + 1) % 17 == 0:  # Add newline every x words
                        answer += '\n'
                    else:
                        answer += ' '
                save_data_to_text_file(full_import_name, answer)
            else:
                start = False
                with open(file_path, 'r') as file:
                    for line in file:
                        if not start:
                            words = line.split()
                            if full_import_name in words:
                                start = True
                        elif not 'END' in line:
                            answer += line
                        else:
                            start = False
            print(answer + '\n')
            api_gpt.append({"name": full_import_name, "other": answer})
            program_output_file.write(answer + '\n')
            print(f'Could not retrieve documentation for {full_import_name}\n')
            program_output_file.write(f'Could not retrieve documentation for {full_import_name}\n')


def save_data_to_text_file(package, description):
    with open(file_path, 'a') as file:
        file.write("\n{}\n".format(package))
        file.write("{}\n".format(description))
        file.write("END\n")


def import_in_file(package_name):
    with open('./JabrefDocumentation.txt', 'r') as file:
        for line in file:
            if package_name in line:
                return True
    return False

def check_package_in_file(package_name):
    with open(file_path, 'r') as file:
        for line in file:
            words = line.split()
            if package_name in words:
                return True
    return False

def find_java_files(directory):
    java_files = []
    for root, dirs, files in os.walk(directory):
        for file in files:
            if file.endswith('.java'):
                java_files.append(os.path.join(root, file))
    return java_files

def main():
    # Connect to your PostgreSQL database
    conn = psycopg2.connect(
        dbname='Test-fabio',
        user='postgres',
        password='301SQL',
        host='localhost',
        port=5432
    )

    cur = conn.cursor()
    # Correctly formatted reserved words list
    reserved_words = [
        "EOF", "abstract", "assert", "boolean", "break", "byte", "case", "catch", "char", "class", "const", "continue",
        "default", "do", "double", "else", "enum", "extends", "final", "finally", "float", "for", "if", "goto",
        "implements", "import", "instanceof", "int", "interface", "long", "native", "new", "package", "private",
        "protected", "public", "return", "short", "static", "strictfp", "super", "switch", "synchronized", "this",
        "throw", "throws", "transient", "try", "void", "volatile", "while", "integerliteral", "floatingpointliteral",
        "booleanliteral", "characterliteral", "stringliteral", "nullliteral", "lparen", "rparen", "lbrace", "rbrace",
        "lbrack", "rbrack", "semi", "comma", "dot", "assign", "gt", "lt", "bang", "tilde", "question", "colon", "equal",
        "le", "ge", "notequal", "and", "or", "inc", "dec", "add", "sub", "mul", "div", "bitand", "bitor", "caret",
        "mod", "add_assign", "sub_assign", "mul_assign", "div_assign", "and_assign", "or_assign", "xor_assign",
        "mod_assign", "lshift_assign", "rshift_assign", "urshift_assign", "identifier", "at", "ellipsis", "ws",
        "comment", "line_comment"
    ]
    Java_modules_to_packages = [
        ('java.base', ['java.io', 'java.lang', 'java.lang.annotation', 'java.lang.constant', 'java.lang.foreign',
                       'java.lang.invoke', 'java.lang.module', 'java.lang.ref', 'java.lang.reflect',
                       'java.lang.runtime', 'java.math',
                       'java.net', 'java.net.spi', 'java.nio', 'java.nio.channels', 'java.nio.channels.spi',
                       'java.nio.charset',
                       'java.nio.charset.spi', 'java.nio.file', 'java.nio.file.attribute', 'java.nio.file.spi',
                       'java.security',
                       'java.security.cert', 'java.security.interfaces', 'java.security.spec', 'java.text',
                       'java.text.spi', 'java.time',
                       'java.time.chrono', 'java.time.format', 'java.time.temporal', 'java.time.zone', 'java.util',
                       'java.util.concurrent', 'java.util.concurrent.atomic', 'java.util.concurrent.locks',
                       'java.util.function',
                       'java.util.jar', 'java.util.random', 'java.util.regex', 'java.util.spi', 'java.util.stream',
                       'java.util.zip',
                       'javax.crypto', 'javax.crypto.interfaces', 'javax.crypto.spec', 'javax.net', 'javax.net.ssl',
                       'javax.security.auth', 'javax.security.auth.callback', 'javax.security.auth.login',
                       'javax.security.auth.spi',
                       'javax.security.auth.x500', 'javax.security.cert']),
        ('java.compiler', ['javax.annotation.processing', 'javax.lang.model', 'javax.lang.model.element',
                           'javax.lang.model.type', 'javax.lang.model.util', 'javax.tools']),
        ('java.datatransfer', ['java.awt.datatransfer']),
        ('java.desktop',
         ['java.applet', 'java.awt', 'java.awt.color', 'java.awt.desktop', 'java.awt.dnd', 'java.awt.event',
          'java.awt.font', 'java.awt.geom', 'java.awt.im', 'java.awt.im.spi', 'java.awt.image',
          'java.awt.image.renderable',
          'java.awt.print', 'java.beans', 'java.beans.beancontext', 'javax.accessibility', 'javax.imageio',
          'javax.imageio.event', 'javax.imageio.metadata', 'javax.imageio.plugins.bmp', 'javax.imageio.plugins.jpeg',
          'javax.imageio.plugins.tiff', 'javax.imageio.spi', 'javax.imageio.stream', 'javax.print',
          'javax.print.attribute',
          'javax.print.attribute.standard', 'javax.print.event', 'javax.sound.midi', 'javax.sound.midi.spi',
          'javax.sound.sampled', 'javax.sound.sampled.spi', 'javax.swing', 'javax.swing.border',
          'javax.swing.colorchooser',
          'javax.swing.event', 'javax.swing.filechooser', 'javax.swing.plaf', 'javax.swing.plaf.basic',
          'javax.swing.plaf.metal', 'javax.swing.plaf.multi', 'javax.swing.plaf.nimbus', 'javax.swing.plaf.synth',
          'javax.swing.table', 'javax.swing.text', 'javax.swing.text.html', 'javax.swing.text.html.parser',
          'javax.swing.text.rtf', 'javax.swing.tree', 'javax.swing.undo']),
        ('java.instrument', ['java.lang.instrument']),
        ('java.logging', ['java.util.logging']),
        ('java.management', ['java.lang.management', 'javax.management', 'javax.management.loading',
                             'javax.management.modelmbean', 'javax.management.monitor', 'javax.management.openmbean',
                             'javax.management.relation', 'javax.management.remote', 'javax.management.timer']),
        ('java.management.rmi', ['javax.management.remote.rmi']),
        ('java.naming', ['javax.naming', 'javax.naming.directory', 'javax.naming.event', 'javax.naming.ldap',
                         'javax.naming.ldap.spi', 'javax.naming.spi']),
        ('java.net.http', ['java.net.http']),
        ('java.prefs', ['java.util.prefs']),
        ('java.rmi', ['java.rmi', 'java.rmi.dgc', 'java.rmi.registry', 'java.rmi.server', 'javax.rmi.ssl']),
        ('java.scripting', ['javax.script']),
        ('java.se', []),
        ('java.security.jgss', ['javax.security.auth.kerberos', 'org.ietf.jgss']),
        ('java.security.sasl', ['javax.security.sasl']),
        ('java.smartcardio', ['javax.smartcardio']),
        ('java.sql', ['java.sql', 'javax.sql']),
        ('java.sql.rowset', ['javax.sql.rowset', 'javax.sql.rowset.serial', 'javax.sql.rowset.spi']),
        ('java.transaction.xa', ['javax.transaction.xa']),
        (
            'java.xml',
            ['javax.xml', 'javax.xml.catalog', 'javax.xml.datatype', 'javax.xml.namespace', 'javax.xml.parsers',
             'javax.xml.stream', 'javax.xml.stream.events', 'javax.xml.stream.util', 'javax.xml.transform',
             'javax.xml.transform.dom', 'javax.xml.transform.sax', 'javax.xml.transform.stax',
             'javax.xml.transform.stream',
             'javax.xml.validation', 'javax.xml.xpath', 'org.w3c.dom', 'org.w3c.dom.bootstrap',
             'org.w3c.dom.events',
             'org.w3c.dom.ls', 'org.w3c.dom.ranges', 'org.w3c.dom.traversal', 'org.w3c.dom.views',
             'org.xml.sax',
             'org.xml.sax.ext', 'org.xml.sax.helpers']),
        ('java.xml.crypto', ['javax.xml.crypto', 'javax.xml.crypto.dom', 'javax.xml.crypto.dsig',
                             'javax.xml.crypto.dsig.dom', 'javax.xml.crypto.dsig.keyinfo',
                             'javax.xml.crypto.dsig.spec']),
        ('jdk.accessibility', ['com.sun.java.accessibility.util']),
        ('jdk.attach', ['com.sun.tools.attach', 'com.sun.tools.attach.spi']),
        ('jdk.charsets', []),
        ('jdk.compiler',
         ['com.sun.source.doctree', 'com.sun.source.tree', 'com.sun.source.util', 'com.sun.tools.javac']),
        ('jdk.crypto.cryptoki', []),
        ('jdk.crypto.ec', []),
        ('jdk.dynalink', ['jdk.dynalink', 'jdk.dynalink.beans', 'jdk.dynalink.linker', 'jdk.dynalink.linker.support',
                          'jdk.dynalink.support']),
        ('jdk.editpad', []),
        ('jdk.hotspot.agent', []),
        ('jdk.httpserver', ['com.sun.net.httpserver', 'com.sun.net.httpserver.spi']),
        ('jdk.incubator.vector', ['jdk.incubator.vector']),
        ('jdk.jartool', ['jdk.security.jarsigner']),
        ('jdk.javadoc', ['jdk.javadoc.doclet']),
        ('jdk.jcmd', []),
        ('jdk.jconsole', ['com.sun.tools.jconsole']),
        ('jdk.jdeps', []),
        ('jdk.jdi', ['com.sun.jdi', 'com.sun.jdi.connect', 'com.sun.jdi.connect.spi', 'com.sun.jdi.event',
                     'com.sun.jdi.request']),
        ('jdk.jdwp.agent', []),
        ('jdk.jfr', ['jdk.jfr', 'jdk.jfr.consumer']),
        ('jdk.jlink', []),
        ('jdk.jpackage', []),
        ('jdk.jshell', ['jdk.jshell', 'jdk.jshell.execution', 'jdk.jshell.spi', 'jdk.jshell.tool']),
        ('jdk.jsobject', ['netscape.javascript']),
        ('jdk.jstatd', []),
        ('jdk.localedata', []),
        ('jdk.management', ['com.sun.management']),
        ('jdk.management.agent', []),
        ('jdk.management.jfr', ['jdk.management.jfr']),
        ('jdk.naming.dns', []),
        ('jdk.naming.rmi', []),
        ('jdk.net', ['jdk.net', 'jdk.nio']),
        ('jdk.nio.mapmode', ['jdk.nio.mapmode']),
        ('jdk.sctp', ['com.sun.nio.sctp']),
        ('jdk.security.auth', ['com.sun.security.auth', 'com.sun.security.auth.callback', 'com.sun.security.auth.login',
                               'com.sun.security.auth.module']),
        ('jdk.security.jgss', ['com.sun.security.jgss']),
        ('jdk.xml.dom', ['org.w3c.dom.css', 'org.w3c.dom.html', 'org.w3c.dom.stylesheets', 'org.w3c.dom.xpath']),
        ('jdk.zipfs', [])
    ]
    # Handle non ascii chars
    with open(input_file, 'r', encoding='utf-8') as file:
        data = file.read()

    # Replace non-ASCII characters with ASCII equivalents or remove them
    ascii_data = ''.join(char if ord(char) < 128 else ' ' for char in data)

    # Write the ASCII data to a new file
    with open(input_file, 'w', encoding='ascii') as output_file:
        output_file.write(ascii_data)

    input_stream = FileStream(input_file)
    lexer = JavaLexer(input_stream)
    token_stream = CommonTokenStream(lexer)
    parser = JavaParser(token_stream)
    tree = parser.compilationUnit()

    parse_tree_str = Trees.toStringTree(tree, None, parser)

    app = QApplication(sys.argv)
    viewer = ParseTreeViewer(parse_tree_str, tree)

    found_reserved_words, found_variables, found_node_types = viewer.traverse_ast(tree, reserved_words, [], [], [])

    print("Found reserved words:", found_reserved_words)
    print("Found variables:", found_variables)
    print("Found node types:", found_node_types)

    program_output_file.write(
        f"Found reserved words:{found_reserved_words}\nFound variables:{found_variables}\nFound node types:{found_node_types}\n")

    imported_classes = []  # This will store fully qualified class names like 'java.awt.Toolkit'
    package_names = []  # This will store unique package names like 'java.awt'

    file_path = input_file  # Adjust this to the path of your Java file
    imported_classes, package_names, class_to_methods = extract_class_usages_from_file(file_path)

    print("\nImported classes:", imported_classes)
    print("Package names:", package_names)
    print()

    program_output_file.write(f"\nImported classes:{imported_classes}\nPackage names:{package_names}\n\n")

    for imported_package in all_imports:
        fetch_import_description(imported_package)

        # Convert full class paths to simple class names for display
    simple_class_names = [cls.split('.')[-1] for cls in imported_classes]

    # Convert keys in class_to_methods from full class paths to simple class names for consistent display
    simple_class_to_methods = {cls.split('.')[-1]: methods for cls, methods in class_to_methods.items()}

    print("Class to Methods mapping:", simple_class_to_methods)
    program_output_file.write(f"Classes to Methods mapping:{simple_class_to_methods}")

    for full_class_name, methods in class_to_methods.items():
        print("\n" + full_class_name)
        program_output_file.write("\n" + full_class_name)
        simple_class_name = full_class_name.split('.')[-1]  # Extract simple class name for display
        package_name = '.'.join(full_class_name.split('.')[:-1])  # Extract the package name
        module_name = find_module_for_package(package_name)
        descriptions = fetch_method_descriptions(module_name, full_class_name, methods)

        if methods:
            # Execute the SQL querysun
            cur.execute("""
            INSERT INTO function_table (function_name, api_name) 
            VALUES (%s, %s)
            ON CONFLICT (function_name, api_name) DO NOTHING;
            """, (methods[0], full_class_name))

            conn.commit()

        for method, description_list in descriptions.items():
            summary_description = ''
            if len(description_list) > 1:  # Check for overloads
                new_desc = ''
                for description in description_list:
                    summary_description += f'Method: {method}, Description: {description}'
                    new_desc += description
                function_context.append({"fullname": full_class_name, "name": method, "other": new_desc})
                # print(f'Summary Description: {summary_description}') # Debug
                prompt = f'Please comprise, paraphrase, or summarize all the following method descriptions from {simple_class_name} into one encompassing description: {summary_description}'
                messages_overload_summ.append({"role": "user", "content": prompt})
                gpt_response = ask_gpt_summary(messages_overload_summ)
                answer = ""
                for chunk in gpt_response:
                    if chunk.choices[0].delta.content:
                        answer += (chunk.choices[0].delta.content.strip('*') or "")
                print(f'Class: {simple_class_name}, Method : {method},  {answer.strip("#### ")}\n')
                program_output_file.write(f'Class: {simple_class_name}, Method : {method},  {answer.strip("#### ")}\n')
                messages_overload_summ.pop()  # Remove last user prompt
                function_topic.append({"fullname": full_class_name, "name": method, "other": answer.strip("#### ")})


                # Find the dictionary where the 'name' key matches the name to find
                gpt_label_dict = next((item for item in api_gpt if item["name"] == full_class_name), None)
                gpt_sim_dict = next((item for item in api_sim if item["name"] == full_class_name), None)
                print(full_class_name)
                gpt_label = gpt_label_dict["other"]
                gpt_sim = gpt_sim_dict["other"]

                # Split the string at the first colon
                sim_received = gpt_sim.split(":", 1)

                # Take the first part
                gpt_sim = sim_received[0].strip()

                # Convert the extracted text to lowercase and replace spaces with underscores
                formatted_label = gpt_sim.lower().replace(" ", "_")
                variable_name = formatted_label + "_options"
                try:
                    sim_options_list = globals()[variable_name]
                except KeyError:
                    sim_options_list = globals()['utility_options']

                # Find the index of "Label:" and "Reason:"
                label_index = gpt_label.find("Label:")
                reason_index = gpt_label.find("Reason:")

                # HERE

                # Extract the text between "Label:" and "Reason:"
                label_text = gpt_label[label_index + len("Label:"):reason_index].strip()

                # Convert the extracted text to lowercase and replace spaces with underscores
                formatted_label = label_text.lower().replace(" ", "_")
                variable_name = formatted_label + "_options"
                try:
                    options_list = globals()[variable_name]
                except KeyError:
                    options_list = globals()['utility_options']

                givenOptions = ""
                similarity_results = []

                for options in options_list:
                    for option, descriptions in options.items():
                        givenOptions += (option + ": " + descriptions)

                for options in sim_options_list:
                    for option, descriptions in options.items():
                        class_desc = nlp(summary_description)
                        given_desc = nlp(option + ": " + descriptions)
                        similarity_results.append(class_desc.similarity(given_desc))
                greatest_value = 0
                greatest_value_index = 0
                for i in range(len(similarity_results)):
                    if similarity_results[i] > greatest_value:
                        greatest_value = similarity_results[i]
                        greatest_value_index = i
                # Get the option at the greatest_value_index
                option = sim_options_list[greatest_value_index]

                # Extract the label and description from the option dictionary
                label = list(option.keys())[0]
                description = list(option.values())[0]
                function_sim.append({"fullname": full_class_name, "name": method,
                                     "other": label + ": " + description})
                answer = ""
                response = ask_gpt(summary_description, givenOptions)
                for chunk in response:
                    if chunk.choices[0].delta.content:
                        answer += (chunk.choices[0].delta.content.strip('*') or "")
                answer = answer.strip('#### ')
                words = answer.split()
                answer = ''
                counter = 0
                for i, word in enumerate(words):
                    if word == "Reason:":
                        answer += '\n'
                        counter -= i
                    answer += word
                    counter += 1
                    if (counter + 1) % 17 == 0:  # Add newline every x words
                        answer += '\n'
                    else:
                        answer += ' '
                print(answer + '\n')
                function_gpt.append({"fullname": full_class_name, "name": method, "other": answer})
            else:
                new_desc = ''
                for description in description_list:
                    summary_description += f'Method: {method}, Description: {description}'
                    new_desc += description
                    program_output_file.write(
                        f"Class: {simple_class_name}, Method: {method}, Description: {description}")
                function_context.append({"fullname": full_class_name, "name": method, "other": new_desc})
                prompt = f'Please comprise, paraphrase, or summarize all the following method descriptions from {simple_class_name} into one encompassing description: {summary_description}'
                messages_overload_summ.append({"role": "user", "content": prompt})
                gpt_response = ask_gpt_summary(messages_overload_summ)
                answer = ""
                for chunk in gpt_response:
                    if chunk.choices[0].delta.content:
                        answer += (chunk.choices[0].delta.content.strip('*') or "")
                print(f'Class: {simple_class_name}, Method : {method},  {answer.strip("#### ")}\n')
                program_output_file.write(f'Class: {simple_class_name}, Method : {method},  {answer.strip("#### ")}\n')
                messages_overload_summ.pop()  # Remove last user prompt
                function_topic.append({"fullname": full_class_name, "name": method, "other": answer.strip("#### ")})

                # Find the dictionary where the 'name' key matches the name to find
                gpt_label_dict = next((item for item in api_gpt if item["name"] == full_class_name), None)
                gpt_sim_dict = next((item for item in api_sim if item["name"] == full_class_name), None)
                print(full_class_name)
                gpt_label = gpt_label_dict["other"]
                gpt_sim = gpt_sim_dict["other"]

                # Split the string at the first colon
                sim_received = gpt_sim.split(":", 1)

                # Take the first part
                gpt_sim = sim_received[0].strip()

                # Convert the extracted text to lowercase and replace spaces with underscores
                formatted_label = gpt_sim.lower().replace(" ", "_")
                variable_name = formatted_label + "_options"
                try:
                    sim_options_list = globals()[variable_name]
                except KeyError:
                    sim_options_list = globals()['utility_options']

                # Find the index of "Label:" and "Reason:"
                label_index = gpt_label.find("Label:")
                reason_index = gpt_label.find("Reason:")

                # HERE

                # Extract the text between "Label:" and "Reason:"
                label_text = gpt_label[label_index + len("Label:"):reason_index].strip()

                # Convert the extracted text to lowercase and replace spaces with underscores
                formatted_label = label_text.lower().replace(" ", "_")
                variable_name = formatted_label + "_options"
                try:
                    options_list = globals()[variable_name]
                except KeyError:
                    options_list = globals()['utility_options']

                givenOptions = ""
                similarity_results = []

                for options in options_list:
                    for option, descriptions in options.items():
                        givenOptions += (option + ": " + descriptions)

                for options in sim_options_list:
                    for option, descriptions in options.items():
                        class_desc = nlp(summary_description)
                        given_desc = nlp(option + ": " + descriptions)
                        similarity_results.append(class_desc.similarity(given_desc))
                greatest_value = 0
                greatest_value_index = 0
                for i in range(len(similarity_results)):
                    if similarity_results[i] > greatest_value:
                        greatest_value = similarity_results[i]
                        greatest_value_index = i
                # Get the option at the greatest_value_index
                option = sim_options_list[greatest_value_index]

                # Extract the label and description from the option dictionary
                label = list(option.keys())[0]
                description = list(option.values())[0]
                function_sim.append({"fullname": full_class_name, "name": method,
                                     "other": label + ": " + description})
                answer = ""
                response = ask_gpt(summary_description, givenOptions)
                for chunk in response:
                    if chunk.choices[0].delta.content:
                        answer += (chunk.choices[0].delta.content.strip('*') or "")
                answer = answer.strip('#### ')
                words = answer.split()
                answer = ''
                counter = 0
                for i, word in enumerate(words):
                    if word == "Reason:":
                        answer += '\n'
                        counter -= i
                    answer += word
                    counter += 1
                    if (counter + 1) % 17 == 0:  # Add newline every x words
                        answer += '\n'
                    else:
                        answer += ' '
                print(answer + '\n')
                function_gpt.append({"fullname": full_class_name, "name": method, "other": answer})

    # Iterate through the lists and insert data into the database
    for i in range(len(function_context)):
        function_name_fk = function_context[i]['name']
        api_name_fk = function_context[i]['fullname']
        api_context_info = next(api['other'] for api in api_context if api['name'] == api_name_fk)
        api_topic_info = next(api['other'] for api in api_topic if api['name'] == api_name_fk)
        api_gpt_info = next(api['other'] for api in api_gpt if api['name'] == api_name_fk)
        api_sim_info = next(api['other'] for api in api_sim if api['name'] == api_name_fk)
        function_context_info = function_context[i]['other']
        function_topic_info = function_topic[i]['other']
        function_gpt_info = function_gpt[i]['other']
        function_sim_info = function_sim[i]['other']
        cur.execute("""
            INSERT INTO API_Function_Specific (
                function_name_fk, api_name_fk, api_context, api_topic, llm_expert_API, 
                sim_expert_API, function_context, function_topic, llm_expert_function, 
                sim_expert_function
            ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
            ON CONFLICT (function_name_fk, api_name_fk) DO NOTHING;
            """, (function_name_fk, api_name_fk, api_context_info, api_topic_info, api_gpt_info,
                  api_sim_info, function_context_info, function_topic_info, function_gpt_info,
                  function_sim_info))
        conn.commit()
    print(api_context)
    print(api_topic)
    print(api_gpt)
    print(api_sim)
    print(function_context)
    print(function_topic)
    print(function_gpt)
    print(function_sim)


    program_output_file.close()
    # Close the connection
    cur.close()
    conn.close()
    if not jabrefRunning:
        viewer.show()
        sys.exit(app.exec())


if __name__ == '__main__':
    jabrefRun = False
    if jabrefRun:
        jabrefRunning = True
        input_files = find_java_files(input_file_directory)
        count = 1
        for found_file in input_files:
            if found_file == "./jabref-5.0-alpha/src/main/java/org/jabref/model/database/BibDatabaseContext.java" or found_file == "./jabref-5.0-alpha/src/main/java/org/jabref/logic/preview/PreviewLayout.java" or found_file == "./jabref-5.0-alpha/src/main/java/org/jabref/logic/util/WebViewStore.java" or found_file == "./jabref-5.0-alpha/src/main/java/org/jabref/gui/actions/StandardActions.java" or found_file == "./jabref-5.0-alpha/src/main/java/org/jabref/gui/auximport/NewSubLibraryAction.java" or found_file == "./jabref-5.0-alpha/src/main/java/org/jabref/gui/bibtexextractor/ExtractBibtexAction.java" or found_file == "./jabref-5.0-alpha/src/main/java/org/jabref/gui/citationkeypattern/GenerateCitationKeyAction.java" or found_file == "./jabref-5.0-alpha/src/main/java/org/jabref/gui/cleanup/CleanupAction.java" or found_file == "./jabref-5.0-alpha/src/main/java/org/jabref/gui/contentselector/ManageContentSelectorAction.java" or found_file == "./jabref-5.0-alpha/src/main/java/org/jabref/gui/copyfiles/CopyFilesAction.java" or found_file == "./jabref-5.0-alpha/src/main/java/org/jabref/gui/customentrytypes/CustomizeEntryAction.java" or found_file == "./jabref-5.0-alpha/src/main/java/org/jabref/gui/documentviewer/ShowDocumentViewerAction.java" or found_file == "./jabref-5.0-alpha/src/main/java/org/jabref/gui/duplicationFinder/DuplicateSearch.java" or found_file == "./jabref-5.0-alpha/src/main/java/org/jabref/gui/edit/automaticfiededitor/AutomaticFieldEditorAction.java" or found_file == "./jabref-5.0-alpha/src/main/java/org/jabref/gui/entryeditor/OpenEntryEditorAction.java" or found_file == "./jabref-5.0-alpha/src/main/java/org/jabref/gui/entryeditor/PreviewSwitchAction.java" or found_file == "./jabref-5.0-alpha/src/main/java/org/jabref/gui/externalfiles/AutoLinkFilesAction.java" or found_file == "./jabref-5.0-alpha/src/main/java/org/jabref/gui/externalfiles/DownloadFullTextAction.java" or found_file == "./jabref-5.0-alpha/src/main/java/org/jabref/gui/externalfiles/FindUnlinkedFilesAction.java" or found_file == "./jabref-5.0-alpha/src/main/java/org/jabref/gui/help/AboutAction.java" or found_file == "./jabref-5.0-alpha/src/main/java/org/jabref/gui/help/ErrorConsoleAction.java" or found_file == "./jabref-5.0-alpha/src/main/java/org/jabref/gui/help/HelpAction.java" or found_file == "./jabref-5.0-alpha/src/main/java/org/jabref/gui/help/SearchForUpdateAction.java" or found_file == "./jabref-5.0-alpha/src/main/java/org/jabref/gui/icon/IconTheme.java" or found_file == "./jabref-5.0-alpha/src/main/java/org/jabref/gui/importer/actions/OpenDatabaseAction.java" or found_file == "./jabref-5.0-alpha/src/main/java/org/jabref/gui/importer/fetcher/LookupIdentifierAction.java" or found_file == "./jabref-5.0-alpha/src/main/java/org/jabref/gui/integrity/IntegrityCheckAction.java" or found_file == "./jabref-5.0-alpha/src/main/java/org/jabref/gui/journals/AbbreviateAction.java" or found_file == "./jabref-5.0-alpha/src/main/java/org/jabref/gui/keyboard/KeyBinding.java" or found_file == "./jabref-5.0-alpha/src/main/java/org/jabref/gui/keyboard/KeyBindingRepository.java" or found_file == "./jabref-5.0-alpha/src/main/java/org/jabref/gui/libraryproperties/LibraryPropertiesAction.java" or found_file == "./jabref-5.0-alpha/src/main/java/org/jabref/gui/menus/FileHistoryMenu.java" or found_file == "./jabref-5.0-alpha/src/main/java/org/jabref/gui/mergeentries/MergeEntriesAction.java" or found_file == "./jabref-5.0-alpha/src/main/java/org/jabref/gui/preferences/ShowPreferencesAction.java" or found_file == "./jabref-5.0-alpha/src/main/java/org/jabref/gui/preview/CopyCitationAction.java" or found_file == "./jabref-5.0-alpha/src/main/java/org/jabref/gui/push/PushToApplicationAction.java" or found_file == "./jabref-5.0-alpha/src/main/java/org/jabref/gui/push/PushToApplicationsManager.java" or found_file == "./jabref-5.0-alpha/src/main/java/org/jabref/gui/search/GlobalSearchBar.java" or found_file == "./jabref-5.0-alpha/src/main/java/org/jabref/gui/search/RebuildFulltextSearchIndexAction.java" or found_file == "./jabref-5.0-alpha/src/main/java/org/jabref/gui/shared/ConnectToSharedDatabaseCommand.java" or found_file == "./jabref-5.0-alpha/src/main/java/org/jabref/gui/shared/PullChangesFromSharedAction.java" or found_file == "./jabref-5.0-alpha/src/main/java/org/jabref/gui/sidepane/SidePane.java" or found_file == "./jabref-5.0-alpha/src/main/java/org/jabref/gui/sidepane/SidePaneType.java" or found_file == "./jabref-5.0-alpha/src/main/java/org/jabref/gui/slr/ExistingStudySearchAction.java" or found_file == "./jabref-5.0-alpha/src/main/java/org/jabref/gui/slr/StartNewStudyAction.java" or found_file == "./jabref-5.0-alpha/src/main/java/org/jabref/gui/specialfields/SpecialFieldMenuItemFactory.java" or found_file == "./jabref-5.0-alpha/src/main/java/org/jabref/gui/texparser/ParseLatexAction.java" or found_file == "./jabref-5.0-alpha/src/main/java/org/jabref/gui/theme/ThemeManager.java" or found_file == "./jabref-5.0-alpha/src/main/java/org/jabref/gui/undo/CountingUndoManager.java" or found_file == "./jabref-5.0-alpha/src/main/java/org/jabref/gui/undo/UndoRedoAction.java" or found_file == "./jabref-5.0-alpha/src/main/java/org/jabref/gui/util/BackgroundTask.java" or found_file == "./jabref-5.0-alpha/src/main/java/org/jabref/gui/util/DefaultTaskExecutor.java" or found_file == "./jabref-5.0-alpha/src/main/java/org/jabref/gui/util/TaskExecutor.java" or found_file == "./jabref-5.0-alpha/src/main/java/org/jabref/logic/autosaveandbackup/AutosaveManager.java" or found_file == "./jabref-5.0-alpha/src/main/java/org/jabref/logic/autosaveandbackup/BackupManager.java" or found_file == "./jabref-5.0-alpha/src/main/java/org/jabref/logic/citationstyle/CitationStyleOutputFormat.java" or found_file == "./jabref-5.0-alpha/src/main/java/org/jabref/logic/help/HelpFile.java" or found_file == "./jabref-5.0-alpha/src/main/java/org/jabref/logic/l10n/Localization.java" or found_file == "./jabref-5.0-alpha/src/main/java/org/jabref/logic/shared/DatabaseLocation.java" or found_file == "./jabref-5.0-alpha/src/main/java/org/jabref/logic/undo/AddUndoableActionEvent.java" or found_file == "./jabref-5.0-alpha/src/main/java/org/jabref/logic/undo/UndoChangeEvent.java" or found_file == "./jabref-5.0-alpha/src/main/java/org/jabref/logic/undo/UndoRedoEvent.java" or found_file == "./jabref-5.0-alpha/src/main/java/org/jabref/logic/util/OS.java" or found_file == "./jabref-5.0-alpha/src/main/java/org/jabref/model/database/BibDatabaseContext.java" or found_file == "./jabref-5.0-alpha/src/main/java/org/jabref/model/entry/BibEntry.java" or found_file == "./jabref-5.0-alpha/src/main/java/org/jabref/model/entry/field/SpecialField.java" or found_file == "./jabref-5.0-alpha/src/main/java/org/jabref/model/entry/types/StandardEntryType.java" or found_file == "./jabref-5.0-alpha/src/main/java/org/jabref/preferences/PreferencesService.java" or found_file == "./jabref-5.0-alpha/src/main/java/org/jabref/preferences/TelemetryPreferences.java" or found_file == "./jabref-5.0-alpha/src/main/java/org/jabref/model/strings/StringUtil.java" or found_file == "./jabref-5.0-alpha/src/main/java/org/jabref/model/util/OptionalUtil.java" or found_file == "./jabref-5.0-alpha/src/main/java/org/jabref/gui/externalfiletype/ExternalFileType.java" or found_file == "./jabref-5.0-alpha/src/main/java/org/jabref/gui/externalfiletype/ExternalFileTypes.java" or found_file == "./jabref-5.0-alpha/src/main/java/org/jabref/gui/externalfiletype/UnknownExternalFileType.java" or found_file == "./jabref-5.0-alpha/src/main/java/org/jabref/logic/cleanup/MoveFilesCleanup.java" or found_file == "./jabref-5.0-alpha/src/main/java/org/jabref/logic/cleanup/RenamePdfCleanup.java" or found_file == "./jabref-5.0-alpha/src/main/java/org/jabref/logic/pdf/search/indexing/IndexingTaskManager.java" or found_file == "./jabref-5.0-alpha/src/main/java/org/jabref/logic/pdf/search/indexing/PdfIndexer.java" or found_file == "./jabref-5.0-alpha/src/main/java/org/jabref/logic/util/io/FileUtil.java" or found_file == "./jabref-5.0-alpha/src/main/java/org/jabref/model/database/BibDatabaseContext.java" or found_file == "./jabref-5.0-alpha/src/main/java/org/jabref/model/entry/BibEntry.java" or found_file == "./jabref-5.0-alpha/src/main/java/org/jabref/model/entry/LinkedFile.java" or found_file == "./jabref-5.0-alpha/src/main/java/org/jabref/preferences/FilePreferences.java" or found_file == "./jabref-5.0-alpha/src/main/java/org/jabref/model/texparser/LatexParserResult.java" or found_file == "./jabref-5.0-alpha/src/main/java/org/jabref/logic/cleanup/FieldFormatterCleanup.java" or found_file == "./jabref-5.0-alpha/src/main/java/org/jabref/logic/formatter/bibtexfields/ClearFormatter.java" or found_file == "./jabref-5.0-alpha/src/main/java/org/jabref/logic/formatter/bibtexfields/RemoveBracesFormatter.java" or found_file == "./jabref-5.0-alpha/src/main/java/org/jabref/logic/importer/fetcher/transformers/DefaultQueryTransformer.java" or found_file == "./jabref-5.0-alpha/src/main/java/org/jabref/logic/importer/util/JsonReader.java" or found_file == "./jabref-5.0-alpha/src/main/java/org/jabref/logic/util/strings/StringSimilarity.java" or found_file == "./jabref-5.0-alpha/src/main/java/org/jabref/model/entry/Author.java" or found_file == "./jabref-5.0-alpha/src/main/java/org/jabref/model/entry/AuthorList.java" or found_file == "./jabref-5.0-alpha/src/main/java/org/jabref/model/entry/BibEntry.java" or found_file == "./jabref-5.0-alpha/src/main/java/org/jabref/model/entry/field/StandardField.java" or found_file == "./jabref-5.0-alpha/src/main/java/org/jabref/model/entry/identifier/DOI.java" or found_file == "./jabref-5.0-alpha/src/main/java/org/jabref/model/entry/types/EntryType.java" or found_file == "./jabref-5.0-alpha/src/main/java/org/jabref/model/entry/types/StandardEntryType.java" or found_file == "./jabref-5.0-alpha/src/main/java/org/jabref/model/util/OptionalUtil.java" or found_file == "./jabref-5.0-alpha/src/main/java/org/jabref/architecture/AllowedToUseAwt.java" or found_file == "./jabref-5.0-alpha/src/main/java/org/jabref/logic/bibtex/BibEntryWriter.java" or found_file == "./jabref-5.0-alpha/src/main/java/org/jabref/logic/bibtex/FieldWriter.java" or found_file == "./jabref-5.0-alpha/src/main/java/org/jabref/model/database/BibDatabaseMode.java" or found_file == "./jabref-5.0-alpha/src/main/java/org/jabref/model/entry/BibEntry.java" or found_file == "./jabref-5.0-alpha/src/main/java/org/jabref/preferences/PreferencesService.java" or found_file == "./jabref-5.0-alpha/src/main/java/org/jabref/logic/util/CoarseChangeFilter.java" or found_file == "./jabref-5.0-alpha/src/main/java/org/jabref/model/database/BibDatabaseContext.java" or found_file == "./jabref-5.0-alpha/src/main/java/org/jabref/model/database/event/AutosaveEvent.java" or found_file == "./jabref-5.0-alpha/src/main/java/org/jabref/model/database/event/BibDatabaseContextChangedEvent.java":
                # output_file_path = f'{found_file}_output.txt'
                # print(f"Current working directory: {os.getcwd()}")
                output_file_path = os.path.join(output_directory, os.path.basename(found_file) + '_output.txt')
                if not os.path.exists(output_file_path):  # Check if output file already exists
                    program_output_file = open(output_file_path, 'w')
                    print(f'\n\nGenerating Output for .java file: {os.path.basename(found_file)}\n\n')
                    print('hi')
                    input_file = found_file
                    # Handle non ascii chars
                    with open(input_file, 'r', encoding='utf-8') as file:
                        data = file.read()

                    # Replace non-ASCII characters with ASCII equivalents or remove them
                    ascii_data = ''.join(char if ord(char) < 128 else ' ' for char in data)

                    # Write the ASCII data to a new file
                    with open(input_file, 'w', encoding='ascii') as output_file:
                        output_file.write(ascii_data)
                    main()
                else:
                    print(f'Already Done With: {os.path.basename(found_file)}: ' + str(count))
                    count += 1
    else:
        input_file = "./test_files/CrossRef.java"  # Change last portion to a test file  # Program output file will be stored in test_files by default
        main()
