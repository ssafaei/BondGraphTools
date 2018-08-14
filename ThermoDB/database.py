
class Molecule(object):
    def __init__(self):

        self.name = set()
        """set(str): list of names for this element"""

        self.formula = str()
        """str(): chemical formula of this element"""

        self.structure = str()
        """str() Description of the chemical structure"""

        self.charge = int()
        """int: Electrical charge"""

        self.mu = float()
        """float: Mean Formation Energy measured at pH 7, zero ionic strength, 
        298 kelvin"""

        self.std =float()
        """float: standard deviation of at physiological conditions."""

        self.sources = list()
        """list(dict)
        """
        self.metadata = {}

    def __eq__(self, other):
        if isinstance(other, Molecule) and \
                other.formula == self.formula and\
                other.charge == self.charge and \
                other.structure == self.structure:
            return True
        else:
            return False

    def to_dict(self):
        return {
            "name": list(self.name),
            "formula": self.formula,
            "structure": self.structure,
            "charge": self.charge,
            "mu": self.mu,
            "std":self.std,
            "sources": [s for s in self.sources],
            "metadata": self.metadata
        }

class Database(object):
    def __init__(self, citation, doi):
        self._db = []
        self.doi = doi
        self.citation = citation

    def add(self, item):
        self._db.append(item)

    def find_one(self, name=None, formula=None):
        if name:
            return next(e for e in self._db if name in e.name)
        elif formula:
            return next(e for e in self._db if formula == e.formula)
        else:
            raise NotImplementedError







