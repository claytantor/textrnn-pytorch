import os
import re
import sys
import time
import glob
import datetime
import sqlite3
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
    all_artist_names.add( artist_name )
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
    

def main(argv):
    all_song_titles = search_song_titles()
    print(all_song_titles)



if __name__ == "__main__":
    main(sys.argv[1:]) 