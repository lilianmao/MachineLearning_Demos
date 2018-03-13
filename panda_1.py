import pandas as pd

city_names = pd.Series(['San Francisco', 'San Jose', 'Sacramento'])
population = pd.Series([852469, 1015785, 485199])

cities = pd.DataFrame({ 'City name': city_names, 'Population': population })
print cities
print "----------"

print type(cities['City name'])
print cities['City name']
print "----------"

print type(cities['City name'][1])
print cities['City name'][1]
print "----------"

print type(cities[0:2])
print cities[0:2]
print "----------"

cities['Area square miles'] = pd.Series([46.87, 176.53, 97.92])
cities['Population density'] = cities['Population'] / cities['Area square miles']
print cities
print "----------"

cities['Is wide and has saint name'] = (cities['Area square miles']>50) & cities['City name'].apply(lambda name: name.startswith('San'))
print cities
print "----------"

cities.reindex([2, 0, 1])
print cities
print "----------"