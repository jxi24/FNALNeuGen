# This file was automatically generated by SWIG (http://www.swig.org).
# Version 3.0.12
#
# Do not make changes to this file unless you know what you are doing--modify
# the SWIG interface file instead.

from sys import version_info as _swig_python_version_info
if _swig_python_version_info >= (2, 7, 0):
    def swig_import_helper():
        import importlib
        pkg = __name__.rpartition('.')[0]
        mname = '.'.join((pkg, '_form_factor')).lstrip('.')
        try:
            return importlib.import_module(mname)
        except ImportError:
            return importlib.import_module('_form_factor')
    _form_factor = swig_import_helper()
    del swig_import_helper
elif _swig_python_version_info >= (2, 6, 0):
    def swig_import_helper():
        from os.path import dirname
        import imp
        fp = None
        try:
            fp, pathname, description = imp.find_module('_form_factor', [dirname(__file__)])
        except ImportError:
            import _form_factor
            return _form_factor
        try:
            _mod = imp.load_module('_form_factor', fp, pathname, description)
        finally:
            if fp is not None:
                fp.close()
        return _mod
    _form_factor = swig_import_helper()
    del swig_import_helper
else:
    import _form_factor
del _swig_python_version_info

try:
    _swig_property = property
except NameError:
    pass  # Python < 2.2 doesn't have 'property'.

try:
    import builtins as __builtin__
except ImportError:
    import __builtin__

def _swig_setattr_nondynamic(self, class_type, name, value, static=1):
    if (name == "thisown"):
        return self.this.own(value)
    if (name == "this"):
        if type(value).__name__ == 'SwigPyObject':
            self.__dict__[name] = value
            return
    method = class_type.__swig_setmethods__.get(name, None)
    if method:
        return method(self, value)
    if (not static):
        if _newclass:
            object.__setattr__(self, name, value)
        else:
            self.__dict__[name] = value
    else:
        raise AttributeError("You cannot add attributes to %s" % self)


def _swig_setattr(self, class_type, name, value):
    return _swig_setattr_nondynamic(self, class_type, name, value, 0)


def _swig_getattr(self, class_type, name):
    if (name == "thisown"):
        return self.this.own()
    method = class_type.__swig_getmethods__.get(name, None)
    if method:
        return method(self)
    raise AttributeError("'%s' object has no attribute '%s'" % (class_type.__name__, name))


def _swig_repr(self):
    try:
        strthis = "proxy of " + self.this.__repr__()
    except __builtin__.Exception:
        strthis = ""
    return "<%s.%s; %s >" % (self.__class__.__module__, self.__class__.__name__, strthis,)

try:
    _object = object
    _newclass = 1
except __builtin__.Exception:
    class _object:
        pass
    _newclass = 0

class SwigPyIterator(_object):
    __swig_setmethods__ = {}
    __setattr__ = lambda self, name, value: _swig_setattr(self, SwigPyIterator, name, value)
    __swig_getmethods__ = {}
    __getattr__ = lambda self, name: _swig_getattr(self, SwigPyIterator, name)

    def __init__(self, *args, **kwargs):
        raise AttributeError("No constructor defined - class is abstract")
    __repr__ = _swig_repr
    __swig_destroy__ = _form_factor.delete_SwigPyIterator
    __del__ = lambda self: None

    def value(self):
        return _form_factor.SwigPyIterator_value(self)

    def incr(self, n=1):
        return _form_factor.SwigPyIterator_incr(self, n)

    def decr(self, n=1):
        return _form_factor.SwigPyIterator_decr(self, n)

    def distance(self, x):
        return _form_factor.SwigPyIterator_distance(self, x)

    def equal(self, x):
        return _form_factor.SwigPyIterator_equal(self, x)

    def copy(self):
        return _form_factor.SwigPyIterator_copy(self)

    def next(self):
        return _form_factor.SwigPyIterator_next(self)

    def __next__(self):
        return _form_factor.SwigPyIterator___next__(self)

    def previous(self):
        return _form_factor.SwigPyIterator_previous(self)

    def advance(self, n):
        return _form_factor.SwigPyIterator_advance(self, n)

    def __eq__(self, x):
        return _form_factor.SwigPyIterator___eq__(self, x)

    def __ne__(self, x):
        return _form_factor.SwigPyIterator___ne__(self, x)

    def __iadd__(self, n):
        return _form_factor.SwigPyIterator___iadd__(self, n)

    def __isub__(self, n):
        return _form_factor.SwigPyIterator___isub__(self, n)

    def __add__(self, n):
        return _form_factor.SwigPyIterator___add__(self, n)

    def __sub__(self, *args):
        return _form_factor.SwigPyIterator___sub__(self, *args)
    def __iter__(self):
        return self
SwigPyIterator_swigregister = _form_factor.SwigPyIterator_swigregister
SwigPyIterator_swigregister(SwigPyIterator)

class dvect(_object):
    __swig_setmethods__ = {}
    __setattr__ = lambda self, name, value: _swig_setattr(self, dvect, name, value)
    __swig_getmethods__ = {}
    __getattr__ = lambda self, name: _swig_getattr(self, dvect, name)
    __repr__ = _swig_repr

    def iterator(self):
        return _form_factor.dvect_iterator(self)
    def __iter__(self):
        return self.iterator()

    def __nonzero__(self):
        return _form_factor.dvect___nonzero__(self)

    def __bool__(self):
        return _form_factor.dvect___bool__(self)

    def __len__(self):
        return _form_factor.dvect___len__(self)

    def __getslice__(self, i, j):
        return _form_factor.dvect___getslice__(self, i, j)

    def __setslice__(self, *args):
        return _form_factor.dvect___setslice__(self, *args)

    def __delslice__(self, i, j):
        return _form_factor.dvect___delslice__(self, i, j)

    def __delitem__(self, *args):
        return _form_factor.dvect___delitem__(self, *args)

    def __getitem__(self, *args):
        return _form_factor.dvect___getitem__(self, *args)

    def __setitem__(self, *args):
        return _form_factor.dvect___setitem__(self, *args)

    def pop(self):
        return _form_factor.dvect_pop(self)

    def append(self, x):
        return _form_factor.dvect_append(self, x)

    def empty(self):
        return _form_factor.dvect_empty(self)

    def size(self):
        return _form_factor.dvect_size(self)

    def swap(self, v):
        return _form_factor.dvect_swap(self, v)

    def begin(self):
        return _form_factor.dvect_begin(self)

    def end(self):
        return _form_factor.dvect_end(self)

    def rbegin(self):
        return _form_factor.dvect_rbegin(self)

    def rend(self):
        return _form_factor.dvect_rend(self)

    def clear(self):
        return _form_factor.dvect_clear(self)

    def get_allocator(self):
        return _form_factor.dvect_get_allocator(self)

    def pop_back(self):
        return _form_factor.dvect_pop_back(self)

    def erase(self, *args):
        return _form_factor.dvect_erase(self, *args)

    def __init__(self, *args):
        this = _form_factor.new_dvect(*args)
        try:
            self.this.append(this)
        except __builtin__.Exception:
            self.this = this

    def push_back(self, x):
        return _form_factor.dvect_push_back(self, x)

    def front(self):
        return _form_factor.dvect_front(self)

    def back(self):
        return _form_factor.dvect_back(self)

    def assign(self, n, x):
        return _form_factor.dvect_assign(self, n, x)

    def resize(self, *args):
        return _form_factor.dvect_resize(self, *args)

    def insert(self, *args):
        return _form_factor.dvect_insert(self, *args)

    def reserve(self, n):
        return _form_factor.dvect_reserve(self, n)

    def capacity(self):
        return _form_factor.dvect_capacity(self)
    __swig_destroy__ = _form_factor.delete_dvect
    __del__ = lambda self: None
dvect_swigregister = _form_factor.dvect_swigregister
dvect_swigregister(dvect)

class map_dict(_object):
    __swig_setmethods__ = {}
    __setattr__ = lambda self, name, value: _swig_setattr(self, map_dict, name, value)
    __swig_getmethods__ = {}
    __getattr__ = lambda self, name: _swig_getattr(self, map_dict, name)
    __repr__ = _swig_repr

    def iterator(self):
        return _form_factor.map_dict_iterator(self)
    def __iter__(self):
        return self.iterator()

    def __nonzero__(self):
        return _form_factor.map_dict___nonzero__(self)

    def __bool__(self):
        return _form_factor.map_dict___bool__(self)

    def __len__(self):
        return _form_factor.map_dict___len__(self)
    def __iter__(self):
        return self.key_iterator()
    def iterkeys(self):
        return self.key_iterator()
    def itervalues(self):
        return self.value_iterator()
    def iteritems(self):
        return self.iterator()

    def __getitem__(self, key):
        return _form_factor.map_dict___getitem__(self, key)

    def __delitem__(self, key):
        return _form_factor.map_dict___delitem__(self, key)

    def has_key(self, key):
        return _form_factor.map_dict_has_key(self, key)

    def keys(self):
        return _form_factor.map_dict_keys(self)

    def values(self):
        return _form_factor.map_dict_values(self)

    def items(self):
        return _form_factor.map_dict_items(self)

    def __contains__(self, key):
        return _form_factor.map_dict___contains__(self, key)

    def key_iterator(self):
        return _form_factor.map_dict_key_iterator(self)

    def value_iterator(self):
        return _form_factor.map_dict_value_iterator(self)

    def __setitem__(self, *args):
        return _form_factor.map_dict___setitem__(self, *args)

    def asdict(self):
        return _form_factor.map_dict_asdict(self)

    def __init__(self, *args):
        this = _form_factor.new_map_dict(*args)
        try:
            self.this.append(this)
        except __builtin__.Exception:
            self.this = this

    def empty(self):
        return _form_factor.map_dict_empty(self)

    def size(self):
        return _form_factor.map_dict_size(self)

    def swap(self, v):
        return _form_factor.map_dict_swap(self, v)

    def begin(self):
        return _form_factor.map_dict_begin(self)

    def end(self):
        return _form_factor.map_dict_end(self)

    def rbegin(self):
        return _form_factor.map_dict_rbegin(self)

    def rend(self):
        return _form_factor.map_dict_rend(self)

    def clear(self):
        return _form_factor.map_dict_clear(self)

    def get_allocator(self):
        return _form_factor.map_dict_get_allocator(self)

    def count(self, x):
        return _form_factor.map_dict_count(self, x)

    def erase(self, *args):
        return _form_factor.map_dict_erase(self, *args)

    def find(self, x):
        return _form_factor.map_dict_find(self, x)

    def lower_bound(self, x):
        return _form_factor.map_dict_lower_bound(self, x)

    def upper_bound(self, x):
        return _form_factor.map_dict_upper_bound(self, x)
    __swig_destroy__ = _form_factor.delete_map_dict
    __del__ = lambda self: None
map_dict_swigregister = _form_factor.map_dict_swigregister
map_dict_swigregister(map_dict)

FormFactorMode_one = _form_factor.FormFactorMode_one
FormFactorMode_two = _form_factor.FormFactorMode_two
FormFactorMode_three = _form_factor.FormFactorMode_three
FormFactorMode_ZExpansion = _form_factor.FormFactorMode_ZExpansion
class FormFactor(_object):
    __swig_setmethods__ = {}
    __setattr__ = lambda self, name, value: _swig_setattr(self, FormFactor, name, value)
    __swig_getmethods__ = {}
    __getattr__ = lambda self, name: _swig_getattr(self, FormFactor, name)
    __repr__ = _swig_repr

    def __init__(self, params):
        this = _form_factor.new_FormFactor(params)
        try:
            self.this.append(this)
        except __builtin__.Exception:
            self.this = this

    def __call__(self, mode, Q2):
        return _form_factor.FormFactor___call__(self, mode, Q2)
    __swig_destroy__ = _form_factor.delete_FormFactor
    __del__ = lambda self: None
FormFactor_swigregister = _form_factor.FormFactor_swigregister
FormFactor_swigregister(FormFactor)

# This file is compatible with both classic and new-style classes.


