import inspect
import sys
import argparse

def generate_numpy_template(func_name, func_obj):
    """
    Generate a NumPy style docstring template for a given function.
    """
    sig = inspect.signature(func_obj)
    
    template = [
        '    """',
        '    [Summary: One-line description of the function.]',
        '',
        '    [Extended Summary: (Optional) Detailed explanation of the logic.]',
        '',
        '    Parameters',
        '    ----------'
    ]
    
    for name, param in sig.parameters.items():
        type_str = ""
        if param.annotation != inspect.Parameter.empty:
            type_str = str(param.annotation).replace("typing.", "")
        
        default_str = ""
        if param.default != inspect.Parameter.empty:
            default_str = f", default {repr(param.default)}"
            
        template.append(f'    {name} : {type_str}{default_str}')
        template.append('        [Description of the parameter.]')
        
    template.append('')
    template.append('    Returns')
    template.append('    -------')
    
    ret_type = ""
    if sig.return_annotation != inspect.Signature.empty:
        ret_type = str(sig.return_annotation).replace("typing.", "")
    
    template.append(f'    {ret_type}')
    template.append('        [Description of the return value.]')
    template.append('    """')
    
    return "\n".join(template)

if __name__ == "__main__":
    # This is a basic harness for the script
    # In a real scenario, the agent would use this to get a template
    # for a function it is about to document.
    print("NumPy Template Generator Script")
