from utils import to_hashable


class Context:
    """
    A Context stores information about, well, the context and communicates it to its subscribers when it changes.
    """
    def __init__(self):
        self.observers = []
        self.cache_id = None
        self.vals = {}

    def subscribe(self, observer):
        self.observers.append(observer)

    def notify(self, name):
        for observer in self.observers:
            observer.update(name)

    def set_val(self, name, new_val):
        if isinstance(self.vals.get(name), dict) and isinstance(new_val, dict):
            if self.val.items() == new_val.items():
                return
        elif new_val in [self.vals.get(name), None] and name != 'neuron_type':
            return
        else:
            self.vals[name] = new_val
            self.cache_id = to_hashable(self.vals)
            self.notify(name)


class Subscriber:

    def subscribe(self, context):
        setattr(self, 'context', context)
        context.subscribe(self)


experiment_context = Context()
