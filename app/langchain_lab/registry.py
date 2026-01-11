import pkgutil
import inspect
import importlib
from typing import List, Dict, Type
from app.langchain_lab.core import LabExperiment

class ExperimentRegistry:
    """
    Auto-discovers and registers LabExperiment classes.
    """
    _experiments: Dict[str, List[Type[LabExperiment]]] = {}

    @classmethod
    def scan(cls):
        """
        Scans the `app.langchain_lab.scenarios` package for modules.
        Inside each module, looks for subclasses of LabExperiment.
        """
        cls._experiments = {}
        
        # Define the base package to scan
        package_name = "app.langchain_lab.scenarios"
        try:
            package = importlib.import_module(package_name)
        except ImportError:
            # Package might not exist yet if empty
            return

        # Iterate over all modules in the package
        for _, name, _ in pkgutil.walk_packages(package.__path__, package.__name__ + "."):
            try:
                module = importlib.import_module(name)
                
                # Inspect module attributes
                for attr_name, obj in inspect.getmembers(module):
                    if (inspect.isclass(obj) 
                        and issubclass(obj, LabExperiment) 
                        and obj is not LabExperiment
                        and obj.__module__ == name): # Avoid importing base class from other modules
                        
                        # Instantiate to get properties (or make properties static? Instance is safer for running)
                        # We'll stick to registering CLASSES, instantiate when running.
                        # But to get category/name we might need to inspect the class or instantiate a dummy.
                        # For simplicity, we assume name/category are properties. We'll instantiate a temp one.
                        
                        try:
                            # Access class attributes directly
                            cat = obj.category
                            
                            if cat not in cls._experiments:
                                cls._experiments[cat] = []
                            cls._experiments[cat].append(obj)
                        except Exception:
                            continue
                            
            except Exception as e:
                print(f"Failed to load module {name}: {e}")

    @classmethod
    def get_categories(cls) -> List[str]:
        return sorted(list(cls._experiments.keys()))

    @classmethod
    def get_experiments_in_category(cls, category: str) -> List[Type[LabExperiment]]:
        return cls._experiments.get(category, [])
