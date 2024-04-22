import psycopg2

print("HERE1")
# Connect to your PostgreSQL database
conn = psycopg2.connect(
    dbname='Testing Code',
    user='postgres',
    password='301SQL',
    host='localhost',
    port=5432
)

cur = conn.cursor()


# Execute the SQL query
# Execute the SQL query
cur.execute('''
CREATE TABLE IF NOT EXISTS function_table (
    function_name character varying,
    api_name character varying,
    CONSTRAINT unique_function_api_pair UNIQUE (function_name, api_name)
);
''')

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

cur.execute("""
INSERT INTO function_table (function_name, api_name) 
VALUES ('Add', 'ArrayList'), ('Erase', 'ArrayList'), ('nextInt', 'Random')
ON CONFLICT (function_name, api_name) DO NOTHING;

""")

conn.commit()

api_context = [
    {'name': "ArrayList", 'other': "Documentation for arraylist"},
    {'name': "Random", 'other': "Documentation for random"},
    {'name': "Third", 'other': "Documentation for third"}
]
api_topic = [
    {'name': "ArrayList", 'other': "Summary for arraylist"},
    {'name': "Random", 'other': "Summary for random"},
    {'name': "Third", 'other': "Summary for third"}
]
api_gpt = [
    {'name': "ArrayList", 'other': "Gpt for arraylist"},
    {'name': "Random", 'other': "Label: Utility Reason: This class falls under the 'Utility' label because it provides functionality that can be used across various applications for generating pseudorandom numbers. It's a general-purpose utility that doesn't fit specifically into any other category mentioned in the options. It serves a broad purpose and can be utilized in a variety of scenarios where randomness is required, hence classifying it under the 'Utility' label."},
    {'name': "Third", 'other': "Gpt for third"}
]
api_sim = [
    {'name': "ArrayList", 'other': "Sim for arraylist"},
    {'name': "Random", 'other': "Sim for random"},
    {'name': "Third", 'other': "Sim for third"}
]
function_context = [
    {'fullName': "ArrayList", 'name': "Add", 'other': "Documentation for arraylist"},
    {'fullName': "Random", 'name': "nextInt", 'other': "Documentation for random"},
    {'fullName': "ArrayList", 'name': "Erase", 'other': "Documentation for erase"}
]
function_topic = [
    {'fullName': "ArrayList", 'name': "Add", 'other': "Summary for arraylist"},
    {'fullName': "Random", 'name': "nextInt", 'other': "Summary for random"},
    {'fullName': "ArrayList", 'name': "Erase", 'other': "Summary for erase"}
]
function_gpt = [
    {'fullName': "ArrayList", 'name': "Add", 'other': "Gpt for arraylist"},
    {'fullName': "Random", 'name': "nextInt", 'other': "Gpt for random"},
    {'fullName': "ArrayList", 'name': "Erase", 'other': "Gpt for erase"}
]
function_sim = [
    {'fullName': "ArrayList", 'name': "Add", 'other': "Sim for arraylist"},
    {'fullName': "Random", 'name': "nextInt", 'other': "Sim for random"},
    {'fullName': "ArrayList", 'name': "Erase", 'other': "Sim for erase"}
]


# Find the dictionary where the 'name' key matches the name to find
gpt_label_dict = next((item for item in api_gpt if item["name"] == "Random"), None)
gpt_label = gpt_label_dict["other"]

# Find the index of "Label:" and "Reason:"
label_index = gpt_label.find("Label:")
reason_index = gpt_label.find("Reason:")

# Extract the text between "Label:" and "Reason:"
label_text = gpt_label[label_index + len("Label:"):reason_index].strip()

# Convert the extracted text to lowercase and replace spaces with underscores
formatted_label = label_text.lower().replace(" ", "_")
variable_name = formatted_label + "_options"
options_list = globals()[variable_name]
print(options_list)

# Iterate through the lists and insert data into the database
for i in range(len(function_context)):
    function_name_fk = function_context[i]['name']
    api_name_fk = function_context[i]['fullName']
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

# cur.execute("""
# INSERT INTO API_Function_Specific (function_name_fk, api_name_fk, api_context, api_topic, llm_expert_API, sim_expert_API, function_context, function_topic, llm_expert_function, sim_expert_function)
# VALUES ('Add', 'ArrayList', 'Documentation Text', 'Summary of Documentation', 'Database', 'Utility', 'Function Documentation', 'Function Summary', 'Arithmetic Operator', 'Other option')
# ON CONFLICT (function_name_fk, api_name_fk) DO NOTHING;
# """)
#
# conn.commit()

print("Table created successfully.")

# Close the connection
cur.close()
conn.close()