import pandas as pd

df = pd.read_excel("/Users/tianchi.gzt/Desktop/__3.xls")

df.pop('部门')
df.pop('姓名.1')
df.pop('门禁授权')
df.pop('分部')
df = df.drop(0, axis=0)

df['员工号'] = df['员工号'].astype('str')
df['证件'] = df['证件'].astype('str')
df['证件'] = "legic1|" + df['证件'].str[:-2]

df = df[~((df['姓名'].notnull() & df['员工号'].notnull() & df['证件'].notnull() & (df['员工号'].str.len() == 18)))]

writer = pd.ExcelWriter("/Users/tianchi.gzt/Desktop/init_user_1.xls")
df.to_excel(writer, 'Sheet1', index=False)
writer.save()
