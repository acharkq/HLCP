import os.path
from ctypes import cdll, c_char, c_char_p, cast, POINTER
from utilities.data_config import bin_dir

thulac_model = os.path.join(bin_dir, "thulac_models")


class Thulac(object):
    _lib = None

    def __init__(self, model_path=thulac_model, user_dict_path=os.path.join(thulac_model, 'dict.txt'), pre_alloc_size=1024 * 1024 * 16,
                 t2s=False, seg_only=True):
        if not os.path.exists(user_dict_path):
            user_dict_path = ''
        if self._lib == None:
            self._lib = cdll.LoadLibrary(os.path.join(thulac_model, "libthulac.so"))
        self._lib.init(c_char_p(model_path.encode('utf-8')),
                       c_char_p(user_dict_path.encode('utf-8')),
                       pre_alloc_size, int(t2s), int(seg_only))

    def clear(self):
        if self._lib != None: self._lib.deinit()

    def cut(self, data, text=False):
        assert self._lib != None
        r = self._lib.seg(c_char_p(data.encode('utf-8')))
        assert r > 0
        self._lib.getResult.restype = POINTER(c_char)
        p = self._lib.getResult()
        s = cast(p, c_char_p)
        d = s.value.decode('utf-8')
        self._lib.freeResult()
        if text:
            return d
        return d.split(' ')