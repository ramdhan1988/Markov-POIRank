import os, tempfile
from utils import *
from subprocess import PIPE, Popen

class Ranksvm:
    def __init__(self, bin_dir, useLinear=True, debug=False, city='Toro'):
        dir_ = bin_dir  # deal with environmental variables in path
        self.bin_dir = dir_[0]
        self.city = city
        self.bin_train = 'svm-train'
        self.bin_predict = 'svm-predict'
        self.bin_scale = 'svm-scale '
        if useLinear:
            self.bin_train = 'train'
            self.bin_predict = 'predict'

        assert(isinstance(debug, bool))
        self.debug = debug
        
        # create named tmp files for model and feature scaling parameters
        self.fmodel = None
        self.fscale = None
        with tempfile.NamedTemporaryFile(delete=False) as fd: 
            self.fmodel = fd.name
        with tempfile.NamedTemporaryFile(delete=False) as fd: 
            self.fscale = fd.name
        
        if self.debug:
            print('model file:', self.fmodel)
            print('feature scaling parameter file:', self.fscale)
    
    def train(self, train_df, cost=1):            
        # cost is parameter C in SVM
        # write train data to file
        ftrain = None
        with tempfile.NamedTemporaryFile(mode='w+t', delete=False) as fd: 
            ftrain = fd.name
            datastr = gen_data_str(train_df, DF_COLUMNS)
            fd.write(datastr)

        # feature scaling
        ftrain_scaled = None
        with tempfile.NamedTemporaryFile(mode='w+t', delete=False) as fd: 
            ftrain_scaled = fd.name

        cmd = '{0} -s "{1}" "{2}" > "{3}"'.format(self.bin_scale, self.fscale, ftrain, ftrain_scaled)
        # print(f'CMD: ', cmd)
        print(f'Scaling training data...')
        Popen(cmd, shell = True, stdout = PIPE).communicate()	
        # result = svm-scale -s self.fscale ftrain > ftrain_scaled

        if self.debug:
            print('cost:', cost)
            print('train data file:', ftrain)
            print('feature scaled train data file:', ftrain_scaled)

        cmd = '{0} -c "{1}" "{2}" "{3}"'.format(self.bin_train, cost, ftrain_scaled, self.fmodel)
        # print(f'CMD: ', cmd)
        print(f'Train training data...')
        Popen(cmd, shell = True, stdout = PIPE).communicate()	
        # result = !svm-train -c $cost $ftrain_scaled $self.fmodel

        if self.debug:
            print('Training finished.')

        # remove train data file
        if self.debug == False:
            os.unlink(ftrain)
            os.unlink(ftrain_scaled)        
    
    def predict(self, test_df):
        # predict ranking scores for the given feature matrix
        if self.fmodel is None or not os.path.exists(self.fmodel):
            print('Model should be trained before prediction')
            return
        
        # write test data to file
        ftest = None
        with tempfile.NamedTemporaryFile(mode='w+t', delete=False) as fd: 
            ftest = fd.name
            datastr = gen_data_str(test_df, DF_COLUMNS)
            fd.write(datastr)
                
        # feature scaling
        ftest_scaled = None
        with tempfile.NamedTemporaryFile(delete=False) as fd: 
            ftest_scaled = fd.name


        cmd = '{0} -r "{1}" "{2}" > "{3}"'.format(self.bin_scale, self.fscale, ftest, ftest_scaled)
        # print(f'CMD: ', cmd)
        print(f'Predicted scaling data...')
        Popen(cmd, shell = True, stdout = PIPE).communicate()	
        # result = !svm-scale -r $self.fscale $ftest > $ftest_scaled
            
        # generate prediction file
        fpredict = None
        with tempfile.NamedTemporaryFile(delete=False) as fd: 
            fpredict = fd.name
            
        if self.debug:
            print('test data file:', ftest)
            print('feature scaled test data file:', ftest_scaled)
            print('predict result file:', fpredict)
                
        cmd = '{0}  "{1}" "{2}" "{3}"'.format(self.bin_predict, ftest_scaled, self.fmodel, fpredict)
        # print(f'CMD: ', cmd)
        print(f'Predicted training data...')
        Popen(cmd, shell = True, stdout = PIPE).communicate()	
        # result = !$self.bin_dir\\$self.bin_predict $ftest_scaled $self.fmodel $fpredict
    
        # generate prediction DataFrame from prediction file
        poi_rank_df = pd.read_csv(fpredict, header=None)
        poi_rank_df.rename(columns={0:'rank'}, inplace=True)
        poi_rank_df['poiID'] = test_df['poiID'].astype(np.int32)
        poi_rank_df.set_index('poiID', inplace=True)
        poi_rank_df['probability'] = softmax(poi_rank_df['rank'])
        
        # Move model
        # cmd = 'cp "{0}" "{1}"'.format(self.fmodel, 'result/POIRank_model_'+ self.city)
        # print(f'Move model ...')
        # Popen(cmd, shell = True, stdout = PIPE).communicate()

        # remove test file and prediction file
        if self.debug == False:
            os.unlink(ftest)
            os.unlink(ftest_scaled)
            os.unlink(fpredict)

        return poi_rank_df
    
    def __del__(self):
        # remove tmp files
        if self.debug == False:
            if self.fmodel is not None and os.path.exists(self.fmodel):
                os.unlink(self.fmodel)
            if self.fscale is not None and os.path.exists(self.fscale):
                os.unlink(self.fscale)
