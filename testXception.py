#from Prediction import Prediction
from XceptionClass import Prediction
test = Prediction('/cluster/home/guillera/mode_3_medical/isu-chest-data/archive/test/2_IM-0652-1001.dcm.png')
print(test.predict_caption())
