import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder # pylint: disable=import-error


def load_data():
    train = pd.read_csv('data/train.csv', encoding = "ISO-8859-1")
    test = pd.read_csv('data/test.csv')
    addresses = pd.read_csv('data/addresses.csv')
    latlons = pd.read_csv('data/latlons.csv')

    # pre-processing

    # drop all rows with Null compliance
    train = train[np.isfinite(train['compliance'])]
    # drop all rows not in the U.S
    train = train[train.country == 'USA']
    test = test[test.country == 'USA']
    # merge latlons and addresses with data
    train = pd.merge(train, pd.merge(addresses, latlons, on='address'), on='ticket_id')
    test = pd.merge(test, pd.merge(addresses, latlons, on='address'), on='ticket_id')
    # drop all unnecessary columns
    train.drop(['agency_name', 'inspector_name', 'violator_name', 'non_us_str_code', 'violation_description', 
                'grafitti_status', 'state_fee', 'admin_fee', 'ticket_issued_date', 'hearing_date',
                # columns not available in test
                'payment_amount', 'balance_due', 'payment_date', 'payment_status', 
                'collection_status', 'compliance_detail', 
                # address related columns
                'violation_zip_code', 'country', 'address', 'violation_street_number',
                'violation_street_name', 'mailing_address_str_number', 'mailing_address_str_name', 
                'city', 'state', 'zip_code', 'address'], axis=1, inplace=True)
    # discretizing relevant columns
    label_encoder = LabelEncoder()
    label_encoder.fit(train['disposition'].append(test['disposition'], ignore_index=True))
    train['disposition'] = label_encoder.transform(train['disposition'])
    test['disposition'] = label_encoder.transform(test['disposition'])
    label_encoder = LabelEncoder()
    label_encoder.fit(train['violation_code'].append(test['violation_code'], ignore_index=True))
    train['violation_code'] = label_encoder.transform(train['violation_code'])
    test['violation_code'] = label_encoder.transform(test['violation_code'])
    train['lat'] = train['lat'].fillna(train['lat'].mean())
    train['lon'] = train['lon'].fillna(train['lon'].mean())
    test['lat'] = test['lat'].fillna(test['lat'].mean())
    test['lon'] = test['lon'].fillna(test['lon'].mean())



    return train