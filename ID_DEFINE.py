import os
dirName = os.path.dirname(os.path.abspath(__file__))
linearRegressionDataDir = os.path.join(dirName, 'data\\LinearRegressionData\\')
linearClassificationDataDir = os.path.join(dirName, 'data/LinearClassificationData\\')
nolinearRegressionDataDir = os.path.join(dirName, 'data\\NoLinearRegressionData\\')
othersDir = os.path.join(dirName, 'data\\Others\\')
modelDir = os.path.join(dirName, "Model\\")
tensorboardDir = os.path.join(dirName, "TensorBoard\\fit_log")
DATASET_PATH = os.path.join(dirName,'..\\..\\DataSet\\VOCtrainval_11-May-2012\\VOCdevkit\\VOC2012\\')