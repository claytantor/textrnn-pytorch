import os
import re
import sys
import time
import glob
import datetime
import sqlite3
import argparse

from operator import itemgetter

import numpy as np # get it at: http://numpy.scipy.org/
# path to the Million Song Dataset subset (uncompressed)
# CHANGE IT TO YOUR LOCAL CONFIGURATION
msd_subset_path='/home/clay/data/projects/deeporb/learning/MillionSongSubset'
msd_subset_data_path=os.path.join(msd_subset_path,'data')
msd_subset_addf_path=os.path.join(msd_subset_path,'AdditionalFiles')
assert os.path.isdir(msd_subset_path),'wrong path' # sanity check

# imports specific to the MSD
from msdb import hdf5_getters as GETTERS

# the following function simply gives us a nice string for
# a time lag in seconds
def strtimedelta(starttime,stoptime):
    return str(datetime.timedelta(seconds=stoptime-starttime))

# we define this very useful function to iterate the files
def apply_to_all_files(basedir,func=lambda x: x,ext='.h5'):
    """
    From a base directory, go through all subdirectories,
    find all files with the given extension, apply the
    given function 'func' to all of them.
    If no 'func' is passed, we do nothing except counting.
    INPUT
       basedir  - base directory of the dataset
       func     - function to apply to all filenames
       ext      - extension, .h5 by default
    RETURN
       number of files
    """
    cnt = 0
    # iterate over all files in all subdirectories
    for root, dirs, files in os.walk(basedir):
        files = glob.glob(os.path.join(root,'*'+ext))
        # count files
        cnt += len(files)
        # apply function to all files
        for f in files :
            func(f)       
    return cnt

# we define the function to apply to all files
def func_to_get_artist_name(filename, all_artist_names):
    """
    This function does 3 simple things:
    - open the song file
    - get artist ID and put it
    - close the file
    """
    h5 = GETTERS.open_h5_file_read(filename)
    artist_name = GETTERS.get_artist_name(h5)
    all_artist_names.add( artist_name )
    h5.close()

# we define the function to apply to all files
def func_to_get_song_title(filename, all_titles):
    """
    This function does 3 simple things:
    - open the song file
    - get artist ID and put it
    - close the file
    """
    h5 = GETTERS.open_h5_file_read(filename)
    songe_title = GETTERS.get_title(h5)
    all_artist_names.add(artist_name )
    h5.close()    

def search_song_titles(count=5000):
    conn = sqlite3.connect(os.path.join(msd_subset_addf_path,
                                        'subset_track_metadata.db'))
    # we build the SQL query
    q = "SELECT DISTINCT title FROM songs"
    # we query the database
    t1 = time.time()
    res = conn.execute(q)
    all_song_titles_sqlite = res.fetchall()
    t2 = time.time()
    print ('all titles names extracted (SQLite) in:',strtimedelta(t1,t2))
    # we close the connection to the database
    conn.close()
    # let's see some of the content
    all_song_titles = []
    for k in all_song_titles_sqlite:
        p = re.compile(r'\d+')
        song_title = k[0] 
        matches_count = p.findall(song_title)
        # re.match(pattern, string, flags=0)
        # print (all_song_titles_sqlite[k][0])
        if "version" not in song_title.lower() and len(matches_count)==0:
            all_song_titles.append(song_title
                .replace("(","").replace(")","").replace(" - "," ").replace("_"," "))
    
    return all_song_titles

def path_from_trackid(msddir,trackid):
    """
    Create a full path from the main MSD dir and a track id.
    Does not check if the file actually exists.
    """
    p = os.path.join(msddir,trackid[2])
    p = os.path.join(p,trackid[3])
    p = os.path.join(p,trackid[4])
    p = os.path.join(p,trackid.upper()+'.h5')
    return p


def feat_names():
    """ return the name of each feature return by the following function """
    # basic global info
    res =  ['track_id','artist_name','title','loudness','tempo','time_signature','key','mode','duration']
    # avg timbre, var timbre
    for k in range(1,13):
        res.append( 'avg_timbre'+str(k))
    for k in range(1,13):
        res.append( 'var_timbre'+str(k))
    # done
    return res


def feat_from_file(filename):
    """
    Extract a list of features in an array, already converted to string
    """
    feats = {}
    h5 = GETTERS.open_h5_file_read(filename)

    feats['track_title'] = str(GETTERS.get_title(h5).decode("utf-8"))
    feats['artist_name'] = str(GETTERS.get_artist_name(h5).decode("utf-8"))
    feats['track_id'] = str(GETTERS.get_track_id(h5).decode("utf-8"))
    
    h5.close()
    return feats

def search_song_titles_by_genres(genre_list):
    # genres_model = {}

     # open SQLite connections
    conn_tm = sqlite3.connect(os.path.join(msd_subset_addf_path,
                                        'subset_track_metadata.db')) 
    conn_at = sqlite3.connect(os.path.join(msd_subset_addf_path,
                                        'subset_artist_term.db'))

    # get top 50 most used musicbrainz tags
    # makes sure the ones we selected are in the top 50
    q = "SELECT mbtag,Count(artist_id) FROM artist_mbtag GROUP BY mbtag"
    res = conn_at.execute(q)
    top50mbtags = sorted(res.fetchall(),key=itemgetter(1),reverse=True)[:50]
    top50mbtags_names = map(lambda x: x[0], top50mbtags)

    
    for g in genre_list:
        assert g in top50mbtags_names,'Wrong or unrecognized genre: '+str(g)

    genre_artists = {}
    for genre in genre_list:
        genre_artists[genre] = {}
        q = "SELECT artist_id FROM artist_mbtag WHERE mbtag='"+genre+"'"
        res = conn_at.execute(q)
        artists = map(lambda x: x[0], res.fetchall())
        for a in artists:
            q = "SELECT mbtag FROM artist_mbtag WHERE artist_id='"+a+"'"
            res = conn_at.execute(q)
            mbtags = map(lambda x: x[0], res.fetchall())
            artist_is_safe = True
            for g2 in top50mbtags_names:
                if g2 != genre and g2 in mbtags:
                    #print 'artist:',a,'we got both',genre,'and',g2
                    artist_is_safe = False; break
            if artist_is_safe:
                genre_artists[genre][a] = {}
                # print(a)

    # iterate over all songs
    cnt_missing = 0 # debugging thing to limit printouts on my laptop
    
    for genre in genre_list:
        cnt = 0
        artists = genre_artists[genre]
        for artist in artists:
            q = "SELECT track_id FROM songs WHERE artist_id='"+artist+"'"
            res = conn_tm.execute(q)
            track_ids = map(lambda x: x[0], res.fetchall())
            for path in map(lambda x: path_from_trackid(msd_subset_data_path,x),track_ids):
                if not os.path.isfile(path):
                    cnt_missing += 1
                    if cnt_missing < 10:
                        print ('ERROR:',path,'does not exist.')
                    continue
                feats = feat_from_file(path)
                genre_artists[genre][artist][feats['track_id']] = feats
                # genre_artists[genre][artist]['artist_name'] = feats['artist_name']
                

    return genre_artists            

def print_song_titles(genre_name, songs_for_genre):
    for artist in songs_for_genre[genre_name].keys():
        # print(songs_for_genre[genre_name][artist])
        for track_id in songs_for_genre[genre_name][artist].keys():
            # print(track_id)
            track_info = songs_for_genre[genre_name][artist][track_id]
            if not isinstance(track_info,dict):
                print('not a dict:', track_info)
            print(track_info['track_title'])


def main(argv):
    parser = argparse.ArgumentParser()


    parser.add_argument("-g", "--genre", action="store",
                        required=True,  dest="genre", help="send to server")

    args = parser.parse_args()


    all_genres = args.genre.split(',')
    songs_for_genre = search_song_titles_by_genres(all_genres)
    for genre_name in all_genres:
        print_song_titles(genre_name, songs_for_genre) 




if __name__ == "__main__":
    main(sys.argv[1:]) 