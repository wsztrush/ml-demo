import csv

reader = csv.reader(open("../volume.csv", encoding="utf-8"))
for row in reader:
    print(row[2])
