import pkgutil
import importlib
from typing import List, Dict, Type, Optional
from app.langchain_lab.core import LabExperiment

class ExperimentRegistry:
    """
    Registry for Lab Experiments.
    Uses @register_experiment decorator to populate logic.
    """
    _experiments: Dict[str, List[Type[LabExperiment]]] = {}

    @classmethod
    def register(cls, experiment_cls: Type[LabExperiment]):
        """Register an experiment class."""
        try:
            # Instantiate temp to get category, or assume it's a property
            # Ideally category is a static property, but the current abstract base class 
            # defines it as a method or property. We'll access it directly assuming it's set on class.
            cat = getattr(experiment_cls, 'category', "Uncategorized")
            # If it's a property object, we might need to instantiate.
            if isinstance(cat, property):
                 cat = experiment_cls().category
            
            if cat not in cls._experiments:
                cls._experiments[cat] = []
            
            # Avoid duplicates
            if experiment_cls not in cls._experiments[cat]:
                cls._experiments[cat].append(experiment_cls)
        except Exception as e:
            print(f"Failed to register experiment {experiment_cls}: {e}")
            
    @classmethod
    def scan(cls):
        """
        Scans the `app.langchain_lab.scenarios` package by importing all modules.
        The @register_experiment decorator in those modules will trigger registration.
        """
        cls._experiments = {} # Reset? Or Keep? Let's reset to avoid stale.
        # But if we reset, we need to re-import? Python modules are cached.
        # If we re-import, the decorator runs again? No, decorator runs at definition time.
        # So if modules are already imported, we are stuck.
        # BUT: For a long-running app checking for *new* files, we might need `reload`.
        
        # Strategy:
        # 1. We keep a list of `discovered_classes` global in this module, 
        #    and the decorator appends to it?
        #    Then `scan` organizes them?
        #    That's safer for re-scans.
        pass # functionality moved to decorator
        
        # To support "Scan", we just ensure all modules are imported.
        package_name = "app.langchain_lab.scenarios"
        try:
            package = importlib.import_module(package_name)
            for _, name, _ in pkgutil.walk_packages(package.__path__, package.__name__ + "."):
                try:
                    importlib.import_module(name)
                except Exception as e:
                     print(f"Failed to import module {name}: {e}")
        except Exception:
            pass

    @classmethod
    def get_categories(cls) -> List[str]:
        return sorted(list(cls._experiments.keys()))

    @classmethod
    def get_experiments_in_category(cls, category: str) -> List[Type[LabExperiment]]:
        return cls._experiments.get(category, [])

# Decorator
def register_experiment(cls):
    """Decorator to register a LabExperiment."""
    ExperimentRegistry.register(cls)
    return cls
