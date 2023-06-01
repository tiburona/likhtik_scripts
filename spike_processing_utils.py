import functools
import re


def smart_title_case(s):
    lowercase_words = {'a', 'an', 'the', 'at', 'by', 'for', 'in', 'of', 'on', 'to', 'up', 'and', 'as', 'but', 'or', 'nor', 'is'}
    words = re.split(r'(\W)', s)  # Split string on non-alphanumeric characters, preserving delimiters
    title_words = [word if word.lower() not in lowercase_words or i == 0 or i == len(words) - 1 
                   else word.lower() 
                   for i, word in enumerate(words)]
    title = ''.join(title_words)
    return re.sub(r"[A-Za-z]+('[A-Za-z]+)?",
                  lambda mo: mo.group(0)[0].upper() + mo.group(0)[1:].lower() if not mo.group(0).isupper() else mo.group(0),
                  title)


def cache_method(method):
    cache_name = "_cache_" + method.__name__

    def to_hashable(item):
        if isinstance(item, dict):
            return tuple(sorted(item.items()))
        elif isinstance(item, (list, set)):
            return tuple(to_hashable(i) for i in item)
        else:
            return item

    @functools.wraps(method)
    def wrapper(self, *args, **kwargs):
        cache = getattr(self, cache_name, {})
        cache_key = (tuple(to_hashable(arg) for arg in args), tuple(sorted(kwargs.items())))
        if cache_key not in cache:
            cache[cache_key] = method(self, *args, **kwargs)
            setattr(self, cache_name, cache)
        return cache[cache_key]

    return wrapper


def init_animal(entry, Animal, Unit):
    name = entry[1][0]
    condition = entry[2][0]
    tone_period_onsets = entry[4][0]
    tone_onsets_expanded = entry[6][0]
    animal = Animal(name, condition, tone_period_onsets=tone_period_onsets, tone_onsets_expanded=tone_onsets_expanded)
    categories = entry[3][0][0]
    cat_names = [k for k in categories.dtype.fields.keys()]
    cat_units = dict(zip(cat_names, [category[0] for category in categories]))
    units = {cat: [{'spikes': [spike_time[0] for spike_time in unit[0]]} for unit in cat_units[cat]] for cat in cat_names}
    {cat: [Unit(animal, cat, unit['spikes']) for unit in units[cat]] for cat in units}
    for i, unit in enumerate(animal.units['good']):
        unit.neuron_type = 'PN' if cat_units['good'][i][8][0][0] < 2 else 'IN'
    return animal