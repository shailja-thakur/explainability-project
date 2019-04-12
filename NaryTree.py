"""
N-ary Tree implementation
"""
import unittest
import weakref
class NaryTree(object):
    '''A generic N-ary tree implementations, that uses a list to store
    it's children.
    '''
    def __init__(self, key=None, item=None, children=None, parent=None):
        self.key = key
        self.item = item
        self.children = children or []
        self._parent = weakref.ref(parent) if parent else None

    @property
    def parent(self):
        if self._parent:
            return self._parent()

    def __getstate__(self):
        self._parent = None
        return self.__dict__

    def __setstate__(self, state):
        self.__dict__ = state
        for child in self.children:
            child._parent = weakref.ref(self)

    def __str__(self):
        return '{} : {}'.format(self.key, self.item)

    def is_leaf(self):
        return len(self.children) == 0

    def get_height(self):
        heights = [child.get_height() for child in self.children]
        return max(heights) + 1 if heights else 1

    def traversal(self, visit=None, *args, **kwargs):
        visit(self, *args, **kwargs)
        l = [self]
        for child in self.children:
            l += child.traversal(visit, *args, **kwargs)
        return l

    def __iter__(self):
        yield self
        for child in self.children:
            yield child

    def add_child(self, key=None, item=None):
        child = NaryTree(key=key, item=item, parent=self)
        self.children.append(child)
        return child



class TestNaryTree(unittest.TestCase):

    def setUp(self):
        self.tree = NaryTree(key='1')
        branch1 = self.tree.add_child(key='1.1', item=0)
        branch2 = self.tree.add_child(key='1.2', item=0)
        branch3 = self.tree.add_child(key='1.3', item=0)
        branch11 = branch1.add_child(key='1.1.1', item=0)
        branch12 = branch1.add_child(key='1.1.2', item=0)
        branch21 = branch2.add_child(key='1.2.1', item=0)
        branch22 = branch2.add_child(key='1.2.2', item=0)
        branch31 = branch3.add_child(key='1.3.1', item=0)
        self.all_items = [self.tree, branch1, branch2, branch3, branch11,
                          branch12, branch21, branch22, branch31]

    def test_traversal(self):
        def in_here(node):
            self.assertTrue(node in self.all_items)
        self.tree.traversal(in_here)

    def test_iterator(self):
        for node in self.tree:
            self.assertTrue(node in self.all_items)

    def test_height(self):
        self.assertEqual(self.tree.get_height(), 3)

    def test_leaf(self):
        self.assertFalse(self.tree.is_leaf())
        self.assertTrue(self.all_items[8].is_leaf())

if __name__ == '__main__':
    unittest.main()


    

