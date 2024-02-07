import pandas as pd

data_path = '/c/Users/AGORASTOS/source/repos/Emerald/test_data.csv'
data = pd.read_csv(data_path)
dataframe = pd.DataFrame(data.values, columns=data.columns)
dataframe['CAD'] = data.CAD
x = dataframe.drop(['ID','female','CNN_Healthy','CNN_CAD','Doctor: CAD','HEALTHY','CAD'], axis=1)

doc_gen_rdnF_80_none = ['known CAD', 'previous PCI', 'Diabetes', 'Chronic Kindey Disease',
                        'ANGINA LIKE', 'RST ECG', 'male', 'u40', 'Doctor: Healthy']
sel_features = doc_gen_rdnF_80_none

##############
### CV-10 ####
##############
for feature in x.columns:
    if feature in sel_features:
        pass
    else:
        x = x.drop(feature, axis=1)

csv_file_path = '/c/Users/AGORASTOS/source/repos/Emerald/input/data.csv'
x.to_csv(csv_file_path, index=False)
