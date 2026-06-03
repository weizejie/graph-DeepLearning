import sys
import os

# Add the parent directory to sys.path so we can find datanetAPI
project_root = r" c:\Users\weize jie\Desktop\test\graph DeepLearning\RouteNet-Fermi-main\
if project_root not in sys.path:
 sys.path.insert(0, project_root)

# Also add the real_traffic directory
real_traffic_dir = os.path.join(project_root, 'real_traffic')
if real_traffic_dir not in sys.path:
 sys.path.insert(0, real_traffic_dir)

# Change to topology_transfer directory
os.chdir(os.path.join(project_root, 'topology_transfer'))

# Now import and run the main module
import importlib.util
spec = importlib.util.spec_from_file_location(\main_module\, os.path.join(project_root, 'topology_transfer', 'main.py'))
main_module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(main_module)
main_module.main()