from utils import to_hashable
from copy import deepcopy


class Base:

    """
    Base defines properties that are useful for Experiment and its inheritors, the various plotters, and Spreadsheet.
    They all access elements of the context.
    """

    @property
    def data_opts(self):
        return (self.data_type_context.val if self.data_type_context is not None else None) or None

    @data_opts.setter
    def data_opts(self, opts):
        self.data_type_context.set_val(opts)

    @property
    def data_type(self):
        return self.data_opts['data_type']

    @data_type.setter
    def data_type(self, data_type):
        data_opts = deepcopy(self.data_opts)
        data_opts['data_type'] = data_type
        self.data_opts = data_opts

    @property
    def selected_neuron_type(self):
        return self.neuron_type_context.val

    @selected_neuron_type.setter
    def selected_neuron_type(self, neuron_type):
        self.neuron_type_context.set_val(neuron_type)

    @property
    def neuron_types(self):
        return ['IN', 'PN']


class Context:
    """
    A Context stores information about, well, the context and communicates it to its subscribers when it changes.
    """
    def __init__(self, name):
        self.name = name
        self.observers = []
        self.cache_id = None
        self.val = None

    def subscribe(self, observer):
        self.observers.append(observer)

    def notify(self):
        for observer in self.observers:
            observer.update(self)

    def set_val(self, new_val):
        if isinstance(self.val, dict) and isinstance(new_val, dict):
            if self.val.items() == new_val.items():
                return
        else:
            if new_val == self.val:
                return
        self.val = new_val
        self.cache_id = to_hashable(new_val)
        self.notify()


class NeuronTypeMixin:

    """A mixin to confer neuron type updating functionality to Animal and Group."""

    def check_for_new_neuron_type(self, context):
        if context.name == 'neuron_type_context':
            if self.last_neuron_type != context.val:
                self.last_neuron_type = context.val
                self.update_neuron_type()

