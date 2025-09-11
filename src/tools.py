from langchain.tools import Tool

def save_to_txt():
    print("test")

save_tool = Tool(
    name = "testing_tool",
    func = save_to_txt,
    description = "testing function",
)