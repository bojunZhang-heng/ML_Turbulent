import pickle

file_path="/Users/zhangbojun/Downloads/ShapeNet_dataset/vtk/pklFile/0.pkl"
with open(file_path, "rb") as f:
    data = pickle.load(f)

coor = data['coor']
features_6d = data['features_6d']

def describe(x, name="data", depth=0, max_depth=9):
    indent = "  " * depth
    if isinstance(x, dict):
        print(f"{indent}{name}: dict, {len(x)} keys -> {list(x.keys())[:10]}")
        if depth < max_depth:
            for k in list(x.keys())[:max_depth]:
                describe(x[k], f"[{k}]", depth + 1, max_depth)
    elif isinstance(x, (list, tuple)):
        print(f"{indent}{name}: {type(x).__name__}, len={len(x)}")
        if depth < max_depth and len(x) > 0:
            describe(x[0], f"{name}[0]", depth + 1, max_depth)
    elif hasattr(x, "shape"):
        print(f"{indent}{name}: {type(x).__name__}, shape={x.shape}, dtype={getattr(x, 'dtype', None)}")
    else:
        s = repr(x)
        print(f"{indent}{name}: {type(x).__name__} = {s[:100]}")

describe(data)

