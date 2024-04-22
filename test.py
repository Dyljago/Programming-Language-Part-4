from g4f.client import Client

client = Client()
messages = [
    {"role": "system",
     "content": "You are attempting to classify the inputted description into one of the 31 labels based off of the simularity to it."},
    {"role": "system",
     "content": "As you guess this, your final response should only be one concise paragraph with the label chosen and why."},
    {"role": "system",
     "content": "Please do not add any pleasantries, please only output the result and why it fits it that category."},
    {"role": "system",
     "content": "please answer in the format of: Label: /givenlabel/"
                                               "Reason: /reasonwhy/"}
]


def ask_gpt3(description, options):
    # Construct the prompt with the object description and option descriptions
    prompt = f"Object Description: {description}\n"
    prompt += f"Options: {options}\n"
    messages.append({"role": "user", "content": prompt})
    response = client.chat.completions.create(
        model="gpt-4-turbo",
        messages=messages,
        stream=True
    )
    return response


description = "A DataFlavor provides meta information about data. DataFlavor is typically used to access data on the clipboard, or during a drag and drop operation."
options = {
    "Application": "third party apps or plugins for specific use attached to the system",
    "Application Performance Manager": "monitors performance or benchmark",
    "Big Data": "API's that deal with storing large amounts of data. with variety of formats",
    "Cloud": "APUs for software and services that run on the Internet",
    "Computer Graphics": "Manipulating visual content",
    "Data Structure": "Data structures patterns (e.g., collections, lists, trees)",
    "Databases": "Databases or metadata",
    "Software Development and IT": "Libraries for version control, continuous integration and continuous delivery",
    "Error Handling": "response and recovery procedures from error conditions",
    "Event Handling": "answers to event like listeners",
    "Geographic Information System": "Geographically referenced information",
    "Input/Output": "read, write data",
    "Interpreter": "compiler or interpreter features",
    "Internationalization": "integrate and infuse international, intercultural, and global dimensions",
    "Logic": "frameworks, patterns like commands, controls, or architecture-oriented classes",
    "Language": "internal language features and conversions",
    "Logging": "log registry for the app",
    "Machine Learning": "ML support like build a model based on training data",
    "Microservices/Services": "Independently deployable smaller services. Interface between two different applications so that they can communicate with each other",
    "Multimedia": "Representation of information with text, audio, video",
    "Multithread": "Support for concurrent execution",
    "Natural Language Processing": "Process and analyze natural language data",
    "Network": "Web protocols, sockets RMI APIs",
    "Operating System": "APIs to access and manage a computer's resources",
    "Parser": "Breaks down data into recognized pieces for further analysis",
    "Search": "API for web searching",
    "Security": "Crypto and secure protocols",
    "Setup": "Internal app configurations",
    "User Interface": "Defines forms, screens, visual controls",
    "Utility": "third party libraries for general use",
    "Test": "test automation"
}
response = ask_gpt3(description, options)
counter = 0
answer = ""
for chunk in response:
    if chunk.choices[0].delta.content:
        answer += (chunk.choices[0].delta.content.strip('*') or "")
answer = answer.strip('#### ')
words = answer.split()
answer = ''
for i, word in enumerate(words):
    if word == "Reason:":
        answer += '\n'
    answer += word
    if (i + 1) % 15 == 0:  # Add newline every x words
        answer += '\n'
    else:
        answer += ' '
print(answer)
