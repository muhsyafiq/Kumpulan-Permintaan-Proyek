from distutils.core import setup,Extension
import appdirs
import py2exe,sys
import matplotlib
sys.setrecursionlimit(20000)

##setup(
##        windows=['main_ui.py'],
##        options={
##                "py2exe":{
##                        "includes": ["play_ui","training_ui","classifier_settings"],
##                        'optimize': 1,
##                        "dll_excludes": ["MSVFW32.dll",
##                                         "AVIFIL32.dll",
##                                         "AVICAP32.dll",
##                                         "ADVAPI32.dll",
##                                         "CRYPT32.dll",
##                                         "WLDAP32.dll",
##                                         "MSVCP90.dll","libzmq.pyd","geos_c.dll","api-ms-win-core-string-l1-1-0.dll","api-ms-win-core-registry-l1-1-0.dll","api-ms-win-core-errorhandling-l1-1-1.dll","api-ms-win-core-string-l2-1-0.dll","api-ms-win-core-profile-l1-1-0.dll","api-ms-win*.dll","api-ms-win-core-processthreads-l1-1-2.dll","api-ms-win-core-libraryloader-l1-2-1.dll","api-ms-win-core-file-l1-2-1.dll","api-ms-win-security-base-l1-2-0.dll","api-ms-win-eventing-provider-l1-1-0.dll","api-ms-win-core-heap-l2-1-0.dll","api-ms-win-core-libraryloader-l1-2-0.dll","api-ms-win-core-localization-l1-2-1.dll","api-ms-win-core-sysinfo-l1-2-1.dll","api-ms-win-core-synch-l1-2-0.dll","api-ms-win-core-heap-l1-2-0.dll","api-ms-win-core-handle-l1-1-0.dll","api-ms-win-core-io-l1-1-1.dll","api-ms-win-core-com-l1-1-1.dll","api-ms-win-core-memory-l1-1-2.dll","api-ms-win-core-version-l1-1-1.dll","api-ms-win-core-version-l1-1-0.dll",
##                                         'api-ms-win-core-processthreads-l1-1-2.dll']
##                            
##                }
##        }
##)

##On/Off nomor baris
### Used successfully in Python2.5 with matplotlib 0.91.2 and PyQt4 (and Qt 4.3.3)
##from distutils.core import setup
##import py2exe

# We need to import the glob module to search for all files.
import glob

# We need to exclude matplotlib backends not being used by this executable.  You may find
# that you need different excludes to create a working executable with your chosen backend.
# We also need to include include various numerix libraries that the other functions call.

opts = {
    'py2exe': { 'packages':['sip','six','appdirs','packaging','h5py','scipy','numpy','sklearn','theano','PySide','keras','dlib','cv2','sknn','psutil','Queue','threading','future'],
                "includes" : ['sip','six','h5py','sklearn.utils.weight_vector','sklearn.utils.lgamma',"play_ui","training_ui","classifier_settings","cv2", "PySide",
                              "matplotlib.backends",  "matplotlib.backends.backend_qt4agg",
                               "matplotlib.figure","pylab", "numpy",'scipy', 'scipy.integrate', 'scipy.special.*','scipy.linalg.*',
                              "scipy.sparse.csgraph._validation",'theano',
                              'sklearn','keras','dlib','threading','Queue','pickle','psutil','sknn'],
                               'excludes': ['_gtkagg', '_tkagg', '_agg2', '_cairo', '_cocoaagg',
                             '_fltkagg', '_gtk', '_gtkcairo','skimage'],
                'bundle_files': 3, 'compressed': True,
                'dll_excludes': ['libgdk-win32-2.0-0.dll',
                                 'libgobject-2.0-0.dll',
                                 "MSVFW32.dll",
                                 "AVIFIL32.dll",
                                 "AVICAP32.dll",
                                 "ADVAPI32.dll",
                                 "CRYPT32.dll",
                                 "WLDAP32.dll",
                                 "MSVCP90.dll",
                                 "api-ms-win-core-processthreads-l1-1-2.dll",
                                 'api-ms-win-core-sysinfo-l1-2-1.dll',
                                 'api-ms-win-core-errorhandling-l1-1-1.dll',
                                 'MSVCP90.dll',
'IPHLPAPI.DLL',
'NSI.dll',
'WINNSI.DLL',
'WTSAPI32.dll',
'SHFOLDER.dll',
'PSAPI.dll',
'MSVCR120.dll',
'MSVCP120.dll',
'CRYPT32.dll',
'GDI32.dll',
'ADVAPI32.dll',
'CFGMGR32.dll',
'USER32.dll',
'POWRPROF.dll',
'MSIMG32.dll',
'WINSTA.dll',
'MSVCR90.dll',
'KERNEL32.dll',
'MPR.dll',
'Secur32.dll']
              }
       }

# Save matplotlib-data to mpl-data ( It is located in the matplotlib\mpl-data
# folder and the compiled programs will look for it in \mpl-data
# note: using matplotlib.get_mpldata_info
data_files = [(r'mpl-data', glob.glob(r'C:\Python25\Lib\site-packages\matplotlib\mpl-data\*.*')),
                    # Because matplotlibrc does not have an extension, glob does not find it (at least I think that's why)
                    # So add it manually here:
                  (r'mpl-data\images',glob.glob(r'C:\Python25\Lib\site-packages\matplotlib\mpl-data\images\*.*')),
                  (r'mpl-data\fonts',glob.glob(r'C:\Python25\Lib\site-packages\matplotlib\mpl-data\fonts\*.*'))]

data_files.append(('imageformats', [
                r'C:\Python27\Lib\site-packages\PyQt4\plugins\imageformats\qjpeg4.dll',
                r'C:\Python27\Lib\site-packages\PyQt4\plugins\imageformats\qico4.dll'
                ]))

data_files.extend(matplotlib.get_py2exe_datafiles())
# for console program use 'console = [{"script" : "scriptname.py"}]
setup(windows=[{"script" : "FaceRecognition.py",
                'uac_info': "requireAdministrator",
                'icon_resources':[(1,'Oxygen.ico')]}], options=opts, zipfile = None,data_files = data_files)
