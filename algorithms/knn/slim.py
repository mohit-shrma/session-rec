from SLIM import SLIM, SLIMatrix
from math import log10
import collections as col
import numpy as np
import pandas as pd
from scipy import sparse

class SLIMMethod:
    def __init__(self, l1r=0.01, l2r=0.01):
        self.params = {
            'dbglvl': 3,
            'algo': 'cd',
            'nthreads': 8,
            'l1r': l1r,
            'l2r': l2r,
            'optTol': 1e-7,
            'niters': 100
        }

        self.session = -1
        self.session_items = []



    def fit(self, data, test=None):
        '''
        Trains the predictor.
        
        Parameters
        --------
        data: pandas.DataFrame
            Training data. It contains the transactions of the sessions. It has one column for session IDs, one for item IDs and one for the timestamp of the events (unix timestamps).
            It must have a header. Column names are arbitrary, but must correspond to the ones you set during the initialization of the network (session_key, item_key, time_key properties).    
        '''
        session_items = data[['SessionId', 'ItemId']]
        train_data = session_items
        train_data['rating'] = 1
        train_data = train_data.rename(columns={
            'SessionId': 'userid',
            'ItemId': 'itemid'
        })
        train_mat = SLIMatrix(train_data)
        self.users = set(train_data['userid'])
        self.items = set(train_data['itemid'])
        self.model = SLIM()
        self.model.train(self.params, train_mat)
        model_csr, model_item_map =  self.model.to_csr(returnmap=True)
        data_id_2_model_ind = {}
        for i in range(len(model_item_map)):
            data_id_2_model_ind[model_item_map[i]] = i
        self.model_csr = model_csr
        self.data_id_2_model_ind = data_id_2_model_ind
        self.model_ind_2_data_id = model_item_map


    def predict_next(self,
                     session_id,
                     input_item_id,
                     predict_for_item_ids,
                     input_user_id=None,
                     skip=False,
                     type='view',
                     timestamp=0):
        '''
        Gives predicton scores for a selected set of items on how likely they be the next item in the session.
                
        Parameters
        --------
        session_id : int or string
            The session IDs of the event.
        input_item_id : int or string
            The item ID of the event.
        predict_for_item_ids : 1D array
            IDs of items for which the network should give prediction scores. Every ID must be in the set of item IDs of the training set.
            
        Returns
        --------
        out : pandas.Series
            Prediction scores for selected items on how likely to be the next item of this session. Indexed by the item IDs.
        
        '''
        if session_id != self.session:
            self.session_items = []
            self.session = session_id

        if type == 'view':
            self.session_items.append(input_item_id)

        if skip:
            return

        #Verify below on array of test items and slim model
        test_profile_rating = np.zeros(self.model_csr.shape[0])
        for item in self.session_items:
            if item not in self.data_id_2_model_ind:
                print('Item not found in model item map:', item)
            test_profile_rating[self.data_id_2_model_ind[item]] = 1
        all_item_preds = sparse.csr_matrix.dot(test_profile_rating, self.model_csr)

        predictions = []
        for item_id in predict_for_item_ids:
            predictions.append(all_item_preds[self.data_id_2_model_ind[item_id]])

        series = pd.Series(data = predictions, index=predict_for_item_ids)
        return series
    

    def clear(self):
        self.session = -1
        self.session_items = []
