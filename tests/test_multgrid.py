"""
tests for multigrid class
"""

import unittest
from context import multigrids
from multigrids.multigrid import MultiGrid
import numpy as np
import os
import shutil

ROWS = 5
COLS = 10
N_GRIDS = 2
G_NAMES = ['first', 'second']

class TestMultiGridClass(unittest.TestCase):
    """test the DrainageGrid class"""
    def setUp(self):
        """setup class for tests 
        """
        # print("In method", self._testMethodName)
        os.mkdir('.TestMultiGridClass')

        self.init_data = np.ones([N_GRIDS,ROWS, COLS])
        self.init_data[1] += 1

        self.t1 = MultiGrid( ROWS, COLS, N_GRIDS)

        self.t2 = MultiGrid( ROWS, COLS, N_GRIDS, 
            grid_names = G_NAMES, 
            initial_data = self.init_data,
            data_model = 'array',
            mode = 'w',
            data_type = 'int',
            filename = '.test_multigrid.data'
        )

    def tearDown(self):
        shutil.rmtree('.TestMultiGridClass')

    def test_init_new (self):
        """
        Tests the creation of a new object
        """
        self.assertEqual(self.t1.real_shape, (N_GRIDS,ROWS, COLS))
        self.assertEqual(self.t1.memory_shape, (N_GRIDS,ROWS* COLS))
        self.assertEqual(self.t1.grid_shape, (ROWS, COLS))
        self.assertEqual(self.t1.num_grids, N_GRIDS)
        self.assertEqual(self.t1.data_type, 'float')
        self.assertEqual(self.t1.mode, 'r+') 
        self.assertEqual(self.t1.grid_name_map, {}) 
        self.assertEqual(self.t1.data_model, 'memmap') 
        self.assertEqual(self.t1.filename, None) 
        self.assertTrue((self.t1.grids == 0).all())

        self.assertEqual(self.t2.real_shape, (N_GRIDS,ROWS, COLS))
        self.assertEqual(self.t2.memory_shape, (N_GRIDS,ROWS* COLS))
        self.assertEqual(self.t2.grid_shape, (ROWS, COLS))
        self.assertEqual(self.t2.num_grids, N_GRIDS)
        self.assertEqual(self.t2.data_type, 'int')
        self.assertEqual(self.t2.mode, 'w') 
        self.assertEqual(set(self.t2.grid_name_map.keys()), set(G_NAMES)) 
        self.assertEqual(self.t2.data_model, 'array') 
        self.assertEqual(self.t2.filename, '.test_multigrid.data') 
        self.assertTrue(
            (self.t2.grids.flatten() == self.init_data.flatten()).all()
        )

        # array does not use the file so it does not need to be removed
        # os.remove('.test_multigrid.data')
        # self.assertNotEqual(t1, t2)

    def test_save_and_load (self):
        """ Test Save and load capabilities """
        self.t1.save('.TestMultiGridClass/test_init_load.yml', '.testmg')
        loaded = MultiGrid('.TestMultiGridClass/test_init_load.yml')
        self.assertEqual(loaded.real_shape, (N_GRIDS,ROWS, COLS))
        self.assertEqual(loaded.memory_shape, (N_GRIDS,ROWS* COLS))
        self.assertEqual(loaded.grid_shape, (ROWS, COLS))
        self.assertEqual(loaded.num_grids, N_GRIDS)
        self.assertEqual(loaded.data_type, 'float')
        self.assertEqual(loaded.mode, 'r+') 
        self.assertEqual(loaded.grid_name_map, {}) 
        self.assertEqual(loaded.data_model, 'memmap') 
        self.assertEqual(loaded.filename, 
            '.TestMultiGridClass/test_init_load.testmg'
        ) 
        self.assertTrue((loaded.grids == 0).all())

        self.t1.save('.TestMultiGridClass/.test_init_load.yml', '.testmg')

        loaded = MultiGrid('.TestMultiGridClass/.test_init_load.yml')
        self.assertEqual(loaded.filename, 
            '.TestMultiGridClass/.test_init_load.testmg'
        )         


    def test_getattr(self):
        """ Test Get attr functions """
        for c in self.t1.config:
            if c != 'mask':
                exec('self.assertEqual(self.t1.config[c], self.t1.' + c + ')' )
        
        self.t1.config['random spaced cfg item'] = 0
        self.assertEqual( 
            self.t1.config['random spaced cfg item'],
            self.t1.random_spaced_cfg_item
        )

    def test_getitem(self):
        """ test getitem """
        self.assertEqual( self.t1[0].shape, (ROWS,COLS) )
        self.assertEqual (self.t2['first'].shape, (ROWS,COLS) )

        self.assertTrue( (self.t1[0] == np.zeros((ROWS,COLS))).all() )
        self.assertTrue( (self.t2['first'] == self.init_data[0]).all() )


    def test_setitem(self):
        """ test setitem """
        data = np.ones((ROWS,COLS)) * 3

        self.t1[1] = data
        self.t2['second'] = data

        # these should be the same
        self.assertTrue( (self.t1[0] == np.zeros((ROWS,COLS))).all() )
        self.assertTrue( (self.t2['first'] == self.init_data[0]).all() )
        # these should have changed
        self.assertTrue( (self.t1[1] == data).all() )
        self.assertTrue( (self.t2['second'] == data).all() )


        ## test shortcut opperators 
        self.t1[1] *= data
        self.assertTrue( (self.t1[1] == data**2).all() )
        self.t1[1] /= data
        self.assertTrue( (self.t1[1] == data).all() )
        self.t1[1] += data
        self.assertTrue( (self.t1[1] == data*2).all() )
        self.t1[1] -= data
        self.assertTrue( (self.t1[1] == data).all() )
        self.t1[1] *= 2
        self.assertTrue( (self.t1[1] == data*2).all() )
        self.t1[1] /= 2
        self.assertTrue( (self.t1[1] == data).all() )
        self.t1[1] += 2
        self.assertTrue( (self.t1[1] == data+2).all() )
        self.t1[1] -= 2
        self.assertTrue( (self.t1[1] == data).all() )


    def test_equality(self):
        """ test equality """
        self.assertNotEqual(self.t1, self.t2)
        self.assertEqual(self.t1, self.t1)
        
        new = MultiGrid( ROWS, COLS, N_GRIDS)
        self.assertEqual(new, self.t1)
    

    def test_get_grid_number (self):
        """ test get grid number """
        self.assertIsInstance(self.t2.get_grid_number('first'), int)
        self.assertIsInstance(self.t2.get_grid_number(0), int)

        self.assertIsInstance(self.t1.get_grid_number(0), int)

        with self.assertRaises(KeyError):
            self.assertIsInstance(self.t1.get_grid_number('first'), int)
            self.assertIsInstance(self.t2.get_grid_number('third'), int)
            self.assertIsInstance(self.t1.get_grid_number(3), int)
            self.assertIsInstance(self.t2.get_grid_number(3), int)

    def test_get_grid(self):
        """ test get grid function """
        self.assertEqual( self.t1.get_grid(0).shape, (ROWS *COLS,) )
        self.assertEqual( self.t1.get_grid(0, False).shape, (ROWS,COLS) )
        self.assertEqual( self.t2.get_grid(0).shape, (ROWS *COLS,) )
        self.assertEqual( self.t2.get_grid(0, False).shape, (ROWS,COLS) )
        self.assertEqual( self.t2.get_grid('first').shape, (ROWS *COLS,) )
        self.assertEqual( self.t2.get_grid('first', False).shape, (ROWS,COLS) )

        self.assertTrue( (self.t1.get_grid(0) == np.zeros((ROWS*COLS))).all() )
        self.assertTrue( 
            (self.t2.get_grid('first') == self.init_data[0].flatten()).all() 
        )
        self.assertTrue( 
            (self.t1.get_grid(0, False) == np.zeros((ROWS,COLS))).all()
        )
        self.assertTrue( 
            (self.t2.get_grid('first',False) == self.init_data[0]).all() 
        )

    def test_set_grid (self):
        """ test set grid function """
        data = np.ones((ROWS,COLS)) * 3
        self.t1.set_grid(1, data) 
        self.t2.set_grid('second', data)

        # these should be the same
        self.assertTrue( (self.t1[0] == np.zeros((ROWS,COLS))).all() )
        self.assertTrue( (self.t2['first'] == self.init_data[0]).all() )
        # these should have changed
        self.assertTrue( (self.t1[1] == data).all() )
        self.assertTrue( (self.t2['second'] == data).all() )
        

if __name__ == '__main__':
    unittest.main()