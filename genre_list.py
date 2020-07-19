import csv
import os, sys
import json

def get_genres():
    all_genres = {}
    data_file = '/home/clay/data/projects/deeporb/learning/MSDBGenre/msd_genre_dataset.txt'

    with open(data_file, newline='') as csvfile:
        dict_reader = csv.DictReader(csvfile)
        for row in dict_reader:
            # print(row['genre'], row['track_id'], row['artist_name'], row['title'])

            genre_name = row['genre'].replace(' ','_')
            artist_name = row['artist_name'].replace(' ','_')
        
            all_genres[genre_name] = {}
            all_genres[genre_name][artist_name] = {}
            all_genres[genre_name][artist_name][row['track_id']] = {'title':row['title']}
   
    return all_genres



def main(argv):
    all_genres = get_genres()
    for key in all_genres.keys():
        print(key)


    



if __name__ == "__main__":
    main(sys.argv[1:]) 