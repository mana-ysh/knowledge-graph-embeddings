
import numpy as np
import pickle
import yaml


cdef class BaseModel:
    def zerograds(self):
        for param in self.params.values():
            param.clear()

    def reset_memory(self):
        self.fw_mem = {}

    def prepare(self):
        self.zerograds()
        self.reset_memory()

    def save_model(self, model_path):
        with open(model_path, 'wb') as fw:
            pickle.dump(self, fw)

    @classmethod
    def load_model(cls, model_path):
        with open(model_path, 'rb') as f:
            model = pickle.load(f)
        return model
