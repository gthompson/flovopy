import scipy.io
import os

def matlab_struct_to_dict(mat_obj):
    """
    Recursively convert MATLAB structs to Python dictionaries.
    """
    result = {}
    for field_name in mat_obj._fieldnames:
        elem = getattr(mat_obj, field_name)
        if isinstance(elem, scipy.io.matlab.mio5_params.mat_struct):
            result[field_name] = matlab_struct_to_dict(elem)
        else:
            result[field_name] = elem
    return result

# Folder containing your .mat files
folder_path = '/data/Montserrat/AMPMAPDB'

# Find all .mat files
mat_files = sorted([f for f in os.listdir(folder_path) if f.endswith('.mat')])

for mat_file in mat_files:
    file_path = os.path.join(folder_path, mat_file)
    print(f"[INFO] Loading {mat_file}...")

    # Load .mat file
    mat_data = scipy.io.loadmat(file_path, struct_as_record=False, squeeze_me=True)

    # Remove meta fields
    mat_data = {k: v for k, v in mat_data.items() if not k.startswith('__')}

    # Convert structs inside to dicts
    for key in mat_data:
        if isinstance(mat_data[key], scipy.io.matlab.mio5_params.mat_struct):
            mat_data[key] = matlab_struct_to_dict(mat_data[key])

    print(f"[INFO] Loaded {len(mat_data)} top-level variables from {mat_file}")
    
    # You can now work with mat_data like a normal Python dictionary
    for varname, value in mat_data.items():
        print(f"  - Variable '{varname}': type={type(value)}, shape={getattr(value, 'shape', 'scalar')}")

    print(mat_data)
    print()
    # --- Optional: you could save to pickle or further process mat_data here
