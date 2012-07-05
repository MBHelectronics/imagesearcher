'''
Created on Jan 16, 2012
'''
from __future__ import division
import sys
import getopt
import threading
import time
import re
import Queue
import mechanize
import urllib
from urllib2 import HTTPError
from PIL import Image, ImageFilter
import scipy
from scipy import linalg
from scipy.spatial import KDTree

import socket
socket.setdefaulttimeout(10)

def main(argv):
    url = None
    string = None
    depth = 1
    k = 5
    try: 
        opts, args = getopt.getopt(argv, 'hu:s:d:k:',['url=','string=','depth=','k'])
    except getopt.GetoptError:
        usage()
        sys.exit(2)
    
    for opt, arg in opts:
        if opt in ['--help', '-h']:    
            usage()
            sys.exit(2)
        elif opt in ['--url', '-u']:
            url = arg
        elif opt in ['--string', '-s']:
            string = arg    
        elif opt in ['--depth', '-d']:
            arg = int(arg)
            if arg > 0:
                depth = arg
            else:
                sys.stderr.write('Search depth must be an integer greater than zero.\n')
                usage()
                sys.exit(2)
        else:
            sys.stderr.write('Unknown option "%s".\n' % opt)
            usage()
            sys.exit(2)
                
    if url == None or string == None:
        sys.stderr.write('An example URL and search string must be provided.\n')
        usage()
        sys.exit(2)
  
    image_analyzer(url, string, depth, k)
    
def usage():
    
    sys.stdout.write('-u -URL of image to analyze and search Google Images for similar examples\n')
    sys.stdout.write('-s -Search string to use (enclose in quotes for multi-word strings)\n')
    sys.stdout.write('-d -Depth of search: program will attempt to download 20 images per level (default=1)\n')
    sys.stdout.write('-k -Return resutls for the k closest matches (default=5)\n')
    sys.stdout.write('-h -This help text\n')

def image_analyzer(url, string, depth, k):
    
    try:
        example_image = download_image(url)
    except (HTTPError,IOError), e:
        urllib.urlcleanup()    
        sys.exit('Unable to download %s.\n' % url + 'Error: %s.\n' % e)
    except socket.timeout:
        urllib.urlcleanup()    
        sys.exit('HTTP timeout exception.  Unable to download example image %s.\n' % url)
    try:
        example_image = convert_image(example_image[0][0], example_image[1][0])
    except:
        urllib.urlcleanup()
        sys.exit('Unable to open image file associated with %s.\n' % url)
        
    start_time = time.time()    
    sys.stdout.write('Analysis started at %s\n' % time.strftime("%H:%M:%S", time.localtime()))
        
    example_vector = scipy.array(eigen_transform(example_image[0][0], example_image[1][0])[0][0])
  
    link_queue = Queue.Queue()
    image_queue = Queue.Queue()
    processeing_queue = Queue.Queue()
    clustering_queue = Queue.Queue()
    
    links = get_image_links(string, depth)
    
    if len(links) == 0:
        urllib.urlcleanup()    
        sys.exit('No image links found with supplied search term "%s."\n' % string)
    
    for link in links:
        link_queue.put(link)
    
    for image in xrange(len(links)):
        #create image downloading threads
        download_thread = ImageDownloadingThread(link_queue, image_queue)
        download_thread.setDaemon(True)
        download_thread.start()
    
    link_queue.join()    
    sys.stdout.write('Beginning processing.\n')
    batch_convert(image_queue, processeing_queue)

    for image in xrange(processeing_queue.qsize()):
        processing_thread = ImageProcessingThread(processeing_queue, clustering_queue)
        processing_thread.setDaemon(True)
        processing_thread.start()

    processeing_queue.join()
    
    eigenvalue_data_set = []
    data_set_dict = {}
    analyzed_count = 0
    
    for image in xrange(clustering_queue.qsize()):
        image_data = clustering_queue.get()
        eigenvalue_vector, image_url = image_data[0][0], image_data[1][0]
        eigenvalue_data_set.append(eigenvalue_vector)
        data_set_dict[image] = image_url
        analyzed_count += 1
                
    eigenvalue_data_set = scipy.array(eigenvalue_data_set)                
    tree = KDTree(eigenvalue_data_set, leafsize=eigenvalue_data_set.shape[1]+1)
    distances, k_nearest = tree.query(example_vector, k)
    sys.stdout.write(distances)
    
    for image_index in k_nearest:
        sys.stdout.write(data_set_dict[image_index])    
    
    urllib.urlcleanup()          
    sys.stdout.write('%i images analyzed.\n' % analyzed_count)
    sys.stdout.write('Processing time: %s seconds.\n' % str(round((time.time() - start_time),1)))
    
    
class ImageDownloadingThread(threading.Thread):
    def __init__(self, link_queue, image_queue):
        threading.Thread.__init__(self)
        self.link_queue = link_queue
        self.image_queue = image_queue
        
    def run(self):
        image_url = self.link_queue.get()
        sys.stdout.write('Downloading %s\n' % image_url)
        try:
            self.image_queue.put(download_image(image_url))
        except (HTTPError,IOError), e:
            sys.stderr.write('Unable to download %s,' % image_url + ' error: %s.' % e + ' Continuing...\n')
        except socket.timeout:
            sys.stderr.write('HTTP timeout exception caught. Continuing...\n')
        finally:
            self.link_queue.task_done()
        
        
class ImageProcessingThread(threading.Thread):    
    def __init__(self, processing_queue, clustering_queue):
        threading.Thread.__init__(self)
        self.processing_queue = processing_queue
        self.clustering_queue = clustering_queue
        self.image_eigenvalues = ([],[])

    def run(self):
        image_data = self.processing_queue.get()
        image_sections, image_url = image_data[0][0], image_data[1][0]
        sys.stdout.write('Starting eigen-transform of %s\n' % image_url)
        self.clustering_queue.put(eigen_transform(image_sections, image_url))
        sys.stdout.write('%s eigen-transform complete.\n' % image_url)
        self.processing_queue.task_done()
            
            
def download_image(image_url):
    # file trasfer progress callback    
    def file_data(*args):
        args = list(args)
        transferred_size = args[0]*args[1]
        if transferred_size > args[2]:
            transferred_size = args[2]
        if args[2] == -1:
            args[2] = 'unknown'
        else:
            args[2] = str(args[2])
        sys.stdout.write(args[3] + ': Transferred %i ' % transferred_size + 'bytes of %s ' % args[2] + 'total.\n')
    
    url_image_tuple = ([],[])
    (image_filename, headers) = urllib.urlretrieve(image_url, reporthook=lambda blocks, size, total: file_data(blocks, size, total, image_url))
    sys.stdout.write('%s download complete.\n' % image_url)
    url_image_tuple[0].append(image_filename)
    url_image_tuple[1].append(image_url)
    return url_image_tuple

def batch_convert(image_queue, processing_queue):
    for image in xrange(image_queue.qsize()):
        image_data = image_queue.get()
        image_file, image_url = image_data[0][0], image_data[1][0]
        sys.stdout.write('Converting %s\n' % image_url)
        try:
            processing_queue.put(convert_image(image_file,image_url))
        except IOError:
            sys.stderr.write('Error opening image file %s' % image_file + ' associated with URL %s' % image_url + '.  Continuing...\n')
        finally:
            image_queue.task_done()

def convert_image(image_file, image_url):
    horizontal_boxes = 10
    vertical_boxes = 10
    url_image_tuple = ([],[])
    image_sections = []
    #convert image to 600x600 size, enhance edges
    image = Image.open(image_file).convert("L").resize((600,600)).filter(ImageFilter.EDGE_ENHANCE)
    #section images into boxes for eignvalue extraction
    for i in xrange(horizontal_boxes):
        for j in xrange(vertical_boxes):
            box = (i*image.size[0]//horizontal_boxes, j*image.size[1]//vertical_boxes, 
                   (i*image.size[0]//horizontal_boxes) + image.size[0]//horizontal_boxes, 
                   (j*image.size[1]//vertical_boxes) + image.size[1]//vertical_boxes)
            pixels = image.crop(box).getdata()
            image_section_array = scipy.array(scipy.reshape(pixels,(image.size[0]//horizontal_boxes,image.size[1]//vertical_boxes)))
            image_sections.append(image_section_array)
    url_image_tuple[0].append(image_sections)
    url_image_tuple[1].append(image_url)
    return url_image_tuple

def eigen_transform(image_sections, image_url):
    processed_data = ([],[])
    eigenvalues = []
    for index1, image_section in enumerate(image_sections):
        eigenvalues.append(scipy.absolute(linalg.eig(image_section)[0]).tolist())
        eigenvalues[index1] = sorted(eigenvalues[index1]) #assort eigenvalue magnitudes in ascending order
        eigenvalues[index1].reverse()
        #compute "eigen-transform"
        eigen_transform = 0
        w = len(eigenvalues[index1])//3
        
        for index2, eigenvalue in enumerate(eigenvalues[index1]):
            if index2 > 1 and index2 <= w:
                eigen_transform += (1/(w - index2 + 1))*eigenvalue
            if index2 > w:
                break
        eigenvalues[index1] = eigen_transform    
        processed_data[0].append(eigenvalues)
    
    processed_data[1].append(image_url)   
    return processed_data

def get_image_links(string, depth):
    br = mechanize.Browser()
    br.set_handle_robots(False)
    br.addheaders = [('User-agent', 'Firefox')]
    br.open('http://google.com/imghp')
    br.select_form('f')
    br.form[ 'q' ] = string
    # Get the search results
    br.submit()
   
    raw_links = []
    processed_links = []
    for page in xrange(2,depth+2):  #number of pages
        try:
            br.find_link(text=str(page))
            req = br.click_link(text=str(page))
            br.open(req)
        except:
            return processed_links
        for link in br.links():
            imageMatch = re.compile('imgurl').search(link.url)
            if imageMatch:
                raw_links.append(link)
        for link in raw_links:
            processed_link = str(link).split('imgurl=')[1].split('&')[0]
            if processed_link not in processed_links:
                processed_links.append(str(link).split('imgurl=')[1].split('&')[0])         
    return processed_links 
    
if __name__ == '__main__':
    main(sys.argv[1:])
    