import ast

def str_to_int_list(string_list:str):
    list_list = ast.literal_eval(string_list)
    int_list = [int(x) for x in list_list]
    return int_list