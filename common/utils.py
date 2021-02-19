import os


def extract_bool_env(env_variable_name):
    value = os.getenv(env_variable_name, "")
    if value == "true" or value == "True":
        return True
    else:
        return False
