import json
import os
import hashlib
import pickle

if not  os.path.exists("/temp/cache_registry/"):
    os.mkdirs("/temp/cache_registry/")
    open("/temp/cache_registry/registry.json",'r').close()

CACHE_PATH = "/temp/cache_registry"

os.environ["PYTHONHASHSEED"] = "0"

print("CWD: ",os.getcwd())
class Cache:
    def __init__(self):
        self.init_cache_reg()

    def init_cache_reg(self):
        with open(CACHE_PATH + "/registry.json", "r") as f:
            self.reg = json.load(f)

    def isCached(self, hashed_args):
        if hashed_args in self.reg.keys():
            return True
        else:
            return False

    def load(self, hashed_args):
        if hashed_args in self.reg.keys():
            with open(self.reg[hashed_args], "rb") as f:
                data = pickle.load(f)
            return data
        else:
            raise Exception("File not present")

    def __sync_reg__(self):
        with open(CACHE_PATH + "/registry.json", "w") as f:
            json.dump(self.reg, f)

    def store(self, hash, data):
        with open(CACHE_PATH + "/" + str(hash) + ".p", "wb") as f:
            pickle.dump(data, f)
        self.reg[hash] = CACHE_PATH + "/" + str(hash) + ".p"
        self.__sync_reg__()


C = Cache()


def cache_it(func):
    def cache_wrappper(*args, **kwargs):
        key = json.dumps(kwargs)
        hash_object = hashlib.md5(key.encode())
        hex_dig = hash_object.hexdigest()
        if C.isCached(hex_dig):
            print("Loading Form Cache")
            return C.load(hex_dig)
        data = func(*args, **kwargs)
        C.store(hex_dig, data)
        return data

    return cache_wrappper
