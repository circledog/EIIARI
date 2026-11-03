"""Empty stand-in for graphviz: old torch2trt imports it in utils.py (graph drawing only)."""


# torch2trt only references this class when drawing a network
class Digraph(object):
    # No behaviour needed
    pass
