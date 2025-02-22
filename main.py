'''   

Lending Club Proj.

'''


#import libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import cufflinks as cf
from plotly.offline import download_plotlyjs,plot,iplot,init_notebook_mode
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
import tensorflow
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense,Dropout
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.models import load_model
from sklearn.metrics import confusion_matrix,classification_report,explained_variance_score
cf.go_offline()
init_notebook_mode(connected=True)

#import_data
data_info = pd.read_csv('data\lending_club_info.csv',index_col='LoanStatNew')
df = pd.read_csv('data\lending_club_loan_two.csv')
#functions
def feat_info(col_name):
    print(data_info.loc[col_name]['Description'])

def get_sorted_grade(column):
    grades = []
    for grade in df[column].unique():
        grades.append(grade)
    grades.sort()
    return grades

def fill_mort(total_acc,mort_acc):
    if np.isnan(mort_acc):
        return total_acc_avg[total_acc]
    else:
        return mort_acc



 #check_description

#Data_Analysis

sns.countplot(data=df,x='loan_status')
plt.show()

#I did only one of iplot method due to massivity of the data 


plt.figure(figsize=(20,8))
df['loan_amnt'].plot.hist(bins=50,alpha=0.5)
plt.show()


plt.figure(figsize=(14,7))
sns.heatmap(df.corr(numeric_only=True),cmap='viridis',annot=True)
plt.tight_layout()
plt.show()


sns.scatterplot(x='installment',y='loan_amnt',data=df)
plt.show()

sns.boxplot(x='loan_status',y='loan_amnt',data=df)
plt.show()

df.groupby('loan_status').describe().transpose().head()

get_sorted_grade('grade') 
get_sorted_grade('sub_grade')

sns.countplot(x=df['grade'],data=df,hue='loan_status')
plt.show()

plt.figure(figsize=(12,4))
sns.countplot(x=df['sub_grade'].sort_values(),data=df,palette='coolwarm')
plt.show()

plt.figure(figsize=(12,4))
sns.countplot(x=df['sub_grade'].sort_values(),data=df,palette='coolwarm',hue='loan_status')
plt.show()

f_g = df[(df['grade']=='F')| (df['grade']=='G')]
plt.figure(figsize=(12,4))
sns.countplot(x=f_g['sub_grade'].sort_values(),data=df,palette='coolwarm',hue='loan_status')
plt.tight_layout()
plt.show()

df['loan_repaid'] = df['loan_status'].map({'Fully Paid':1,'Charged Off':0})
df[['loan_repaid','loan_status']]

df.corr(numeric_only=True)['loan_repaid'].sort_values().drop('loan_repaid').plot(kind='bar')
plt.show()


#Data_PreProcessing

sns.heatmap(df.isnull(),cmap='viridis',cbar=False)
plt.show()


df.isnull().sum()/len(df)*100 #percentage of the total DataFrame

df['emp_length']
df['emp_title'].nunique()
df['emp_title'].value_counts()
df = df.drop('emp_title',axis=1)
sort_order = sorted(df['emp_length'].dropna().unique())
sorted_order = ['< 1 year',
 '1 year',
 '2 years',
 '3 years',
 '4 years',
 '5 years',
 '6 years',
 '7 years',
 '8 years',
 '9 years',
 '10+ years']


plt.figure(figsize=(12,7))
sns.countplot(x=df['emp_length'],data=df,order=sorted_order)
plt.show()

plt.figure(figsize=(12,7))
sns.countplot(x=df['emp_length'],data=df,order=sorted_order,hue='loan_status')
plt.show()

emp_CO = df[df['loan_status'] == 'Charged Off'].groupby('emp_length').count()['loan_status']
emp_FP = df[df['loan_status'] == 'Fully Paid'].groupby('emp_length').count()['loan_status']
emp_len = emp_CO / (emp_FP + emp_CO)
emp_len.plot(kind='bar')
plt.show()

df = df.drop('emp_length',axis=1)

#check if there is repeated info

df = df.drop('title',axis=1)


#Get correlation between total_acc and mort_acc to fill the missing data in mort_acc


df.corrwith(df['mort_acc'],numeric_only=True).sort_values()
df['mort_acc'].sum()
total_acc_avg = df.groupby('total_acc')['mort_acc'].mean()
df['mort_acc'] = df.apply(lambda x: fill_mort(x['total_acc'],x['mort_acc']),axis=1)
df = df.dropna()
#check again for missing data

df.select_dtypes(['object']).columns #check for non integers


#Convert Objects to numeric values with get_dummies method 

df['term'] = df['term'].apply(lambda x: x[:3])
df = df.drop('grade',axis=1)
sub_grade_feat = pd.get_dummies(df['sub_grade'],drop_first=True)
df = df.drop('sub_grade',axis=1)
df = pd.concat([df,sub_grade_feat],axis=1)
verification_status_feat = pd.get_dummies(df['verification_status'],drop_first=True)
application_type_feat = pd.get_dummies(df['application_type'],drop_first=True)
initial_list_status_feat = pd.get_dummies(df['initial_list_status'],drop_first=True)
purpose_feat = pd.get_dummies(df['purpose'],drop_first=True)
df = df.drop(df[['verification_status','application_type','initial_list_status','purpose']],axis=1)
df = pd.concat([df,verification_status_feat,application_type_feat,initial_list_status_feat,purpose_feat],axis=1)

df['home_ownership'] = df['home_ownership'].replace(['NONE','ANY'],'OTHER') #Replacing NONE and ANY with OTHER
dummies = pd.get_dummies(df['home_ownership'],drop_first=True)
df = pd.concat([df.drop('home_ownership',axis=1),dummies],axis=1)
df['zip_code'] = df['address'].apply(lambda x:x[-5:])
df = df.drop('address',axis=1)
dummies = pd.get_dummies(df['zip_code'],drop_first=True)
df = pd.concat([df.drop('zip_code',axis=1),dummies],axis=1)
df = df.drop('issue_d',axis=1)
df['earliest_cr_line'] = pd.to_datetime(df['earliest_cr_line'])
df['earliest_cr_year'] = df['earliest_cr_line'].apply(lambda date: int(date.year))
df = df.drop('earliest_cr_line',axis=1)
df = df.drop('loan_status',axis=1)


''' Training and Testing Data '''


#Data distribution

X = df.drop('loan_repaid',axis=1).values
y = df['loan_repaid'].values


#Training & Testing Data

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=101)
X_train = np.asarray(X_train).astype(np.float32)
y_train = np.asarray(y_train).astype(np.float32)
X_test = np.asarray(X_test).astype(np.float32)
y_test = np.asarray(y_test).astype(np.float32)


#Data scaling

scaler = MinMaxScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)


''' Create the Model '''
#It can be dispensed with callback as I tested it already

Callback = EarlyStopping(monitor='val_loss',mode='min',patience=25,verbose=1)

model = Sequential()

model.add(Dense(80,activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(40,activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(20,activation='relu'))
model.add(Dropout(0.2))

model.add(Dense(1,activation='sigmoid'))
model.compile(optimizer='adam',loss='binary_crossentropy')


#Fitting Model

model.fit(x=X_train,y=y_train,batch_size=256,epochs=25,validation_data=(X_test,y_test),callbacks=[Callback])

#Data analysis

losses = pd.DataFrame(model.history.history)
losses.plot()
plt.show()


#Save Model

model.save('Myfirstmodel.h5')


#Predicitions

predictions = (model.predict(X_test) > 0.5).astype("int64") #predict_classes not exist in TF 12.


#Evaluating Model Performance

print('\n')
print(confusion_matrix(y_test,predictions))
print('\n')
print(classification_report(y_test,predictions))
print('\n')
print('Explained Variance Score: ',explained_variance_score(y_test,predictions))
print('\n')

''' Testing the Machine '''

test = True
while test:
    y_n_question = str(input('Test the Machine?(y/n): ')).lower()
    if y_n_question == 'y':
        while True:
            test_question = input('Please put the test number (0-{}): '.format(len(df)))
            ranger = map(str,range(0,len(df)+1))      
            
            if test_question in ranger:
                new_customer = df.drop('loan_repaid',axis=1).iloc[int(test_question)]
                customer_scaler = scaler.transform(new_customer.values.reshape(1,78))
                customer_prediction = (model.predict(customer_scaler) > 0.5).astype("int64")            
                
                if int(customer_prediction) == int(df['loan_repaid'][int(test_question)]) :
                    print("Predection is correct")
                    break
                else :
                    print("Predection is incorrect")
                    break            
            else:
                print("Invalid Entry, Please try again")
                break

    elif y_n_question == 'n':
        print('Until next time, See you soon')
        test = False
    else:
        print("Invalid Entry, Please try again")
        continue


input("prompt: ")