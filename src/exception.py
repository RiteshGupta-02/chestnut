from logger import logging
import sys

def error_message_detail(error, error_details:sys):
    _,_,error_tb = error_details.exc_info()
    error_message = f"Error occurred in script: {error_tb.tb_frame.f_code.co_filename} at line number: {error_tb.tb_lineno} error message: {str(error)}"
    return error_message

class CustomException(Exception):
    def __init__(self, error_message, error_details:sys):
        super().__init__(error_message)
        self.error_message = error_message_detail(error_message, error_details)

    def __str__(self):
        return self.error_message
    
if __name__ == "__main__":
    try:
        a = 1/0
    except Exception as e:
        logging.info(e)
        raise CustomException(e, sys)