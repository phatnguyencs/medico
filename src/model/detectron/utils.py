import os 

def create_dir(file_dir, is_print=False):
    try:
        os.mkdir(file_dir)
        if is_print:
            print("Directory " , file_dir ,  " Created ") 
    except FileExistsError:
        if is_print:
            print("Directory " , file_dir ,  " already exists")
        else:
            pass