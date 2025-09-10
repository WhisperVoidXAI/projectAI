!mkdir asoc
!cd asoc
!python -m venv asoc
"asoc/Scripts/activate.bat"
!pip install asttokens
!pip install backcall
!pip install colorama
!pip install cycler
!pip install debugpy
!pip install decorator
!pip install entrypoints
!pip install executing
!pip install fonttools
!pip install ipykernel
!pip install ipython
!pip install jedi
!pip install joblib
!pip install jupyter-client
!pip install jupyter-core
!pip install kiwisolver
!pip install matplotlib
!pip install matplotlib-inline
!pip install mlxtend
!pip install nest-asyncio
!pip install numpy
!pip install packaging
!pip install pandas
!pip install parso
!pip install pickleshare
!pip install Pillow
!pip install prompt-toolkit
!pip install psutil
!pip install pure-eval
!pip install Pygments
!pip install pyparsing
!pip install python-dateutil
!pip install pytz
!pip install pywin32
!pip install pyzmq
!pip install scikit-learn
!pip install scipy
!pip install seaborn
!pip install six
!pip install stack-data
!pip install threadpoolctl
!pip install tornado
!pip install traitlets
!pip install wcwidth
transactions = [['Bread', 'Milk'],
 ['Bread', 'Diaper', 'Juice', 'Eggs'],
 ['Milk', 'Diaper', 'Juice', 'Coke' ],
 ['Bread', 'Milk', 'Diaper', 'Juice'],
 ['Bread', 'Milk', 'Diaper', 'Coke']]
from mlxtend.preprocessing import TransactionEncoder
te = TransactionEncoder()
te_model = te.fit(transactions)
rows=te_model.transform(transactions)
import pandas as pd
df = pd.DataFrame(rows, columns=te_model.columns_)
print(df)
from mlxtend.frequent_patterns import apriori
frequent_itemsets = apriori(df, min_support=0.4, use_colnames=True)
frequent_itemsets['length'] = frequent_itemsets['itemsets'].apply(lambda x: len(x))
print(frequent_itemsets)
from mlxtend.frequent_patterns import association_rules
rules = association_rules(frequent_itemsets,metric="confidence",min_threshold=0.6)
rules = rules.sort_values(['confidence'], ascending =[False])
print(rules)
from matplotlib import pyplot as plt
from pandas.plotting import parallel_coordinates
def rules_to_coordinates(rules):
    rules['antecedent'] = rules['antecedents'].apply(lambda antecedent: list(antecedent)[0])
    rules['consequent'] = rules['consequents'].apply(lambda consequent: list(consequent)[0])
    rules['rule'] = rules.index
    return rules[['antecedent','consequent','rule']]
coords = rules_to_coordinates(rules)
plt.figure(figsize=(4,8))
parallel_coordinates(coords,'rule')
plt.grid(True)
plt.show()
df = pd.read_csv('Groceries.csv',header=None)
df.head()
transactions =  df.T.apply(lambda x: x.dropna().tolist()).tolist()
print(transactions[1:10])
te = TransactionEncoder()
te_model = te.fit(transactions)
rows=te_model.transform(transactions)
df = pd.DataFrame(rows, columns=te_model.columns_)
print(df.shape)
frequent_itemsets = apriori(df, min_support=0.005, use_colnames=True)
frequent_itemsets['length'] = frequent_itemsets['itemsets'].apply(lambda x: len(x))
print(frequent_itemsets)
rules = association_rules(frequent_itemsets,metric='confidence',min_threshold=0.55)
rules = rules.sort_values(['confidence'], ascending =[False])
print(rules)
from pandas.plotting import parallel_coordinates
def rules_to_coordinates(rules):
    rules['antecedent'] = rules['antecedents'].apply(lambda antecedent: list(antecedent)[0])
    rules['consequent'] = rules['consequents'].apply(lambda consequent: list(consequent)[0])
    rules['rule'] = rules.index
    return rules[['antecedent','consequent','rule']]
coords = rules_to_coordinates(rules)
plt.figure(figsize=(4,8))
parallel_coordinates(coords,'rule')
plt.legend([])
plt.grid(True)
plt.show()
