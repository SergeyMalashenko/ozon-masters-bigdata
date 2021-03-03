#
#
def filter_cond(line_dict):
    try:
        cond_match = (
            ( 20 < float(line_dict["if1"]) ) and ( float(line_dict["if1"]) < 40 ) 
        )
        return True if cond_match else False
    except ValueError:
        return False
