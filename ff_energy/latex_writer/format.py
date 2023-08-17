import re

def safe_latex_string(string) -> str:
    """removes all unsave latex characters (e.g. / and _) and return the string"""
    regex = re.compile(r"[^a-zA-Z0-9\s]")
    print(string)
    string = regex.sub("", string)
    print(regex.findall(string))
    return string