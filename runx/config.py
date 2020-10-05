from .collections import AttrDict


__C = AttrDict()
cfg = __C

# Random note: avoid using '.ON' as a config key since yaml converts it to True;

__C.FARM = None
__C.LOGROOT = None
__C.EXP_NAME = None
