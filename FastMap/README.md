This is a sample code for FastMap algorithm.

This program implements FastMap to embed the objects in fastmap-data.txt into a 2D space. 

The first two columns in each line of the data file represent the IDs of the two objects; and the third column indicates the symmetric distance between them. If the furthest pair of objects is ambiguous, the one that includes the smallest object ID will be used. 

The objects listed in fastmap-data.txt are actually the words in fastmap-wordlist.txt (nth word in this list has an ID value of n) and the distances between each pair of objects are the Damerau–Levenshtein distances between them. The program will plot the words onto a 2D plane using the previous FastMap solution and see what it looks like.
