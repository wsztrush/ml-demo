from pandas import Series, DataFrame

s = Series(data=[1, 2, 3], index=['a', 'b', 'c'])

df = DataFrame(
    data={
        'c1': [1, 2, 3],
        'c2': [4, 5, 6],
        'c3': [7, 8, 9]
    },
    index=['a', 'b', 'c']
)

print(df['c1']['a'])

print(df[df.c2 < 6])
