import sys 

def function(message , message_detail : sys)-> str:
    _,_,message_tb=message_detail.exc_info()
    file=message_tb.tb_frame.f_code.co_filename
    line=message_tb.tb_lineno
    message=str(message)
    error_message="Error has occured in python script name [{0}] line [{1}] error message [{2}]".format(
        file ,
        line ,
        message        
    )
    return error_message

class CustomException (Exception):
    def __init__(self, message , message_detail : sys):
        super().__init__(message_detail)
        self.error_message=function(message=message , message_detail=message_detail)
        
    def __str__(self):
        return self.error_message
    