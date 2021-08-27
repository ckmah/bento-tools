
import inspect
from functools import wraps

def get_default_args(func):
    signature = inspect.signature(func)
    return {
        k: v.default
        for k, v in signature.parameters.items()
        if v.default is not inspect.Parameter.empty
    }

def track(func):
    """
    Track changes in AnnData object after applying function.
    """
    @wraps(func)
    def wrapper(*args, **kwds):
        kwargs = get_default_args(func)
        kwargs.update(kwds)
        
        adata = args[0]
        old_attr = list_attributes(adata)
        
        if kwargs['copy']:
            out_adata = func(*args, **kwds)
            new_attr = list_attributes(out_adata)
        else:
            func(*args, **kwds)
            new_attr = list_attributes(adata)
            
        # Print differences between new and old adata
        out = ''
        out += 'AnnData object modified:'
        
        if old_attr['n_obs'] != new_attr['n_obs']:
            out += f"\nn_obs: {old_attr['n_obs']} -> {new_attr['n_obs']}"
            
        if old_attr['n_vars'] != new_attr['n_vars']:
             out += f"\nn_vars: {old_attr['n_vars']} -> {new_attr['n_vars']}"

        modified = False
        for attr in old_attr.keys():
            
            if attr == 'n_obs' or attr == 'n_vars':
                continue
            
            removed = list(old_attr[attr] - new_attr[attr])
            added = list(new_attr[attr] - old_attr[attr])
            
            if len(removed) > 0 or len(added) > 0:
                modified = True
                out += "\n    obs:"
                if len(removed) > 0:
                    out += f"\n        - removed: {', '.join(removed)}"
                if len(added) > 0:
                    out += f"\n        + added: {', '.join(added)}"
        
        if modified:
            print(out)
        
        return out_adata if kwargs['copy'] else None
    
    return wrapper
        

def list_attributes(adata):
    found_attr = dict(n_obs=adata.n_obs, n_vars=adata.n_vars)
    for attr in [
        "obs",
        "var",
        "uns",
        "obsm",
        "varm",
        "layers",
        "obsp",
        "varp",
    ]:
        keys = set(getattr(adata, attr).keys())
        found_attr[attr] = keys
    
    return found_attr