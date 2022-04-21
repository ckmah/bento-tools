import matplotlib.pyplot as plt
import inspect
from functools import wraps



def get_default_args(func):
    signature = inspect.signature(func)
    return {
        k: v.default
        for k, v in signature.parameters.items()
        if v.default is not inspect.Parameter.empty
    }


def savefig(plot_fn):
    """
    Save figure from plotting function.
    """

    @wraps(plot_fn)
    def wrapper(*args, **kwds):
        kwargs = get_default_args(plot_fn)
        kwargs.update(kwds)

        plot_fn(*args, **kwds)
        
        fname = kwargs['fname']
        rc = {'svg.fonttype': 'none', 'font.family':'Arial'}
        if fname:
            with plt.rc_context(rc):
                plt.savefig(fname, dpi=400)

                
            print(f'Saved to {fname}')
        
    return wrapper